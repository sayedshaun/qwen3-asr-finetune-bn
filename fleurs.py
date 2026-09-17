"""Core dataset logic: pull all of google/fleurs' Bengali splits via the
HuggingFace `datasets` library and build a NeMo manifest.

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

CONFIG_NAME = "bn_in"
EXPECTED_SAMPLE_RATE = 16000
FLEURS_SPLITS = ("train", "validation", "test")


def write_manifest(dataset, clips_out_dir: Path, manifest_path: Path, desc: str) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(manifest_path, "w") as out:
        for i, example in enumerate(tqdm(dataset, desc=desc)):
            text = example["transcription"].strip()
            if not text:
                continue
            dst = clips_out_dir / f"fleurs_{i:06d}.wav"
            if not dst.exists():
                dst.write_bytes(example["audio"]["bytes"])
            with sf.SoundFile(dst) as f:
                duration = len(f) / f.samplerate
            out.write(
                json.dumps(
                    {"audio_filepath": str(dst), "text": text, "duration": duration},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def prepare_fleurs_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "fleurs_")

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
        load_dataset("google/fleurs", CONFIG_NAME, split=split)
        for split in FLEURS_SPLITS
    ]
    dataset = concatenate_datasets(parts)
    assert dataset.features["audio"].sampling_rate == EXPECTED_SAMPLE_RATE
    dataset = dataset.cast_column("audio", Audio(decode=False))

    count = write_manifest(
        dataset, output_dir / "wavs", manifest_path, desc="fleurs-train"
    )
    print(f"train: wrote {count} utterances -> {manifest_path}")
    return {"train": count}
