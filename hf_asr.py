"""Shared logic for the HuggingFace-hosted Bengali corpora (shrutilipi,
indicvoices, kathbath, vaani).

They all ship as parquet with one audio column and one transcript column, so a
single builder covers them - each source module only supplies its repo id, its
config name(s) and its manifest prefix.

Column names are *not* consistent across these repos (`text` vs `transcript`
vs `transcription`, `audio` vs `audio_filepath`), and several are gated, so the
schema cannot be pinned in advance. Rather than hardcode a guess per repo we
detect the columns from the loaded dataset's features and fail loudly if
nothing matches.

These corpora all ship their own `duration` column and declare the Audio
feature's sampling rate, so when both are present every clip is written
straight to disk with no header read and no decode - which is what makes
preparing a few hundred thousand utterances tolerable.

Everything is folded into our train manifest: dev/test come from the
mcv/openslr sources, so there is no eval-contamination risk here.
"""

import json
import re
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000

TEXT_COLUMNS = (
    "text",
    "transcript",
    "transcription",
    "transcripts",
    "sentence",
    "normalized_text",
    "raw_text",
    "clean_text",
)
AUDIO_COLUMNS = ("audio", "audio_filepath", "wav", "speech", "file")
DURATION_COLUMNS = ("duration", "duration_seconds", "length")

NOISE_MARKERS = re.compile(r"<[^>]*>|\[[^\]]*\]")


def clean_text(raw: str) -> str:
    return " ".join(NOISE_MARKERS.sub(" ", raw or "").split())


def detect_columns(dataset) -> tuple:
    """Pick the audio and transcript columns out of a loaded dataset."""
    from datasets import Audio

    features = dataset.features

    audio_col = next(
        (name for name, feat in features.items() if isinstance(feat, Audio)), None
    )
    if audio_col is None:
        audio_col = next((c for c in AUDIO_COLUMNS if c in features), None)

    text_col = next((c for c in TEXT_COLUMNS if c in features), None)

    if audio_col is None or text_col is None:
        raise SystemExit(
            "could not identify the audio/transcript columns; "
            f"got audio={audio_col!r} text={text_col!r} from features {list(features)}"
        )
    return audio_col, text_col


def detect_duration_column(dataset):
    """The corpus's own duration column, if it publishes one."""
    return next((c for c in DURATION_COLUMNS if c in dataset.features), None)


def declared_sample_rate(dataset, audio_col: str):
    """The Audio feature's declared sampling rate, read *before* any
    `cast_column(..., Audio(decode=False))` - that cast builds a fresh Audio
    feature with `sampling_rate=None`, losing this."""
    return getattr(dataset.features.get(audio_col), "sampling_rate", None)


def _write_clip(raw_bytes: bytes, dst: Path, trusted_sample_rate: bool) -> None:
    """Write one clip to dst, resampling to 16 kHz mono unless the corpus has
    already declared it is 16 kHz."""
    if dst.exists():
        return
    dst.write_bytes(raw_bytes)
    if trusted_sample_rate:
        return
    info = sf.info(dst)
    if info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(dst)
        audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).export(
            dst, format="wav"
        )


def _clip_duration(dst: Path) -> float:
    info = sf.info(dst)
    return info.frames / info.samplerate


def write_manifest(
    dataset,
    audio_col: str,
    text_col: str,
    clips_out_dir: Path,
    manifest_path: Path,
    prefix: str,
    desc: str,
    duration_col=None,
    trusted_sample_rate: bool = False,
) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(manifest_path, "w") as out:
        for i, example in enumerate(tqdm(dataset, desc=desc)):
            text = clean_text(example[text_col])
            if not text:
                continue
            audio = example[audio_col]
            raw_bytes = audio["bytes"] if isinstance(audio, dict) else None
            if raw_bytes is None:
                continue
            dst = clips_out_dir / f"{prefix}{i:07d}.wav"
            try:
                _write_clip(raw_bytes, dst, trusted_sample_rate)
                duration = example[duration_col] if duration_col else None
                if not duration:
                    duration = _clip_duration(dst)
            except Exception as exc:
                print(f"  skipping {dst.name}: {type(exc).__name__}: {exc}")
                continue
            out.write(
                json.dumps(
                    {"audio_filepath": str(dst), "text": text, "duration": duration},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def prepare_hf_dataset(args, repo_id: str, default_configs, source_name: str):
    """Build a train manifest from a HuggingFace-hosted ASR corpus.

    `default_configs` is the list of config names to pull and concatenate;
    a source's config entry can override it with its own `configs:` list.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", f"{source_name}_")

    manifest_path = output_dir / f"{prefix}train_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            count = sum(1 for _ in f)
        print(
            f"train: manifest already exists -> {manifest_path} ({count} utterances), skipping"
        )
        return {"train": count}

    if args.skip_download:
        raise SystemExit(
            f"skip_download is set for the {source_name} source, but no manifest was found to skip to"
        )

    from datasets import Audio, concatenate_datasets, load_dataset

    configs = getattr(args, "configs", None) or default_configs
    if isinstance(configs, str):
        configs = [configs]

    parts, sample_rates = [], set()
    for config in configs:
        part = load_dataset(repo_id, config, split="train")
        audio_col, _ = detect_columns(part)
        sample_rates.add(declared_sample_rate(part, audio_col))
        parts.append(part.cast_column(audio_col, Audio(decode=False)))

    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    audio_col, text_col = detect_columns(dataset)
    duration_col = detect_duration_column(dataset)

    trusted_sample_rate = sample_rates == {TARGET_SAMPLE_RATE}

    print(
        f"{source_name}: audio column {audio_col!r}, text column {text_col!r}, "
        f"duration column {duration_col!r}, declared sample rate(s) {sorted(r for r in sample_rates if r)}"
    )
    if not trusted_sample_rate:
        print(f"{source_name}: sample rate not declared as 16 kHz, checking every clip")

    count = write_manifest(
        dataset,
        audio_col,
        text_col,
        output_dir / "wavs",
        manifest_path,
        prefix,
        desc=f"{source_name}-train",
        duration_col=duration_col,
        trusted_sample_rate=trusted_sample_rate,
    )
    print(f"train: wrote {count} utterances -> {manifest_path}")
    return {"train": count}
