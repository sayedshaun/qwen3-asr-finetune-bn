"""Core dataset logic: pull google/fleurs splits via the HuggingFace
`datasets` library and build a NeMo manifest.

Defaults to Bengali (`bn_in`). Set `locale` to any other FLEURS config to
build a replay manifest for one of Qwen3-ASR's pretrained languages; pair it
with `language` (the tag Qwen3-ASR itself uses, e.g. "Hindi") so the rows
train on their own language prefix instead of the run's Bengali one, and with
`max_utterances` to keep replay small.

Unlike mcv/openslr this source has no separate archive download/extract step -
`load_dataset` handles fetching and caching the audio+transcripts itself. All
of fleurs' own train/validation/test splits are folded into our train
manifest: our own dev/test held-out sets already come from the mcv/openslr
sources, so there's no eval-contamination risk in also training on fleurs'
dev/test.
"""

import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

DEFAULT_LOCALE = "bn_in"
EXPECTED_SAMPLE_RATE = 16000
FLEURS_SPLITS = ("train", "validation", "test")


def write_manifest(
    dataset,
    clips_out_dir: Path,
    manifest_path: Path,
    desc: str,
    clip_prefix: str = "fleurs",
    language: str = None,
    max_utterances: int = None,
) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(manifest_path, "w") as out:
        for i, example in enumerate(tqdm(dataset, desc=desc)):
            if max_utterances and written >= max_utterances:
                break
            text = example["transcription"].strip()
            if not text:
                continue
            dst = clips_out_dir / f"{clip_prefix}_{i:06d}.wav"
            if not dst.exists():
                dst.write_bytes(example["audio"]["bytes"])
            with sf.SoundFile(dst) as f:
                duration = len(f) / f.samplerate
            row = {"audio_filepath": str(dst), "text": text, "duration": duration}
            if language:
                row["language"] = language
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_fleurs_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "fleurs_")
    locale = getattr(args, "locale", None) or DEFAULT_LOCALE
    language = getattr(args, "language", None)
    max_utterances = getattr(args, "max_utterances", None)
    clip_prefix = f"fleurs{locale}" if locale != DEFAULT_LOCALE else "fleurs"

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
            "skip_download is set for the fleurs source, but no manifest was found to skip to"
        )

    from datasets import Audio, concatenate_datasets, load_dataset

    parts = [
        load_dataset("google/fleurs", locale, split=split)
        for split in FLEURS_SPLITS
    ]
    dataset = concatenate_datasets(parts)
    assert dataset.features["audio"].sampling_rate == EXPECTED_SAMPLE_RATE
    dataset = dataset.cast_column("audio", Audio(decode=False))

    count = write_manifest(
        dataset,
        output_dir / "wavs",
        manifest_path,
        desc=f"fleurs-{locale}-train",
        clip_prefix=clip_prefix,
        language=language,
        max_utterances=max_utterances,
    )
    print(f"train: wrote {count} utterances -> {manifest_path}")
    return {"train": count}
