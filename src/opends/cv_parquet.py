"""Core dataset logic: build NeMo manifests from Common Voice parquet shards
already sitting on disk, and skip the Mozilla Data Collective API entirely.

The `mcv` source downloads a tar.gz release and needs an MDC_API_KEY. The
HuggingFace mirrors of the same corpus ship as parquet shards
(`validated-00000-of-00008.parquet`) with the audio embedded per row, so when
you already have those files this source reads them directly.

Audio lands in the parquet as encoded bytes (mp3 for Common Voice), so each row
is written out and handed to the same `convert_clip` helper the other sources
use, which resamples to 16 kHz mono wav.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src.audio import convert_clip

TEXT_COLUMNS = ("sentence", "text", "transcription")
AUDIO_COLUMN = "audio"


def resolve_files(spec) -> list:
    """Accepts a glob, a single path, or a list of either."""
    patterns = spec if isinstance(spec, (list, tuple)) else [spec]
    files = []
    for pattern in patterns:
        matches = sorted(glob(str(pattern)))
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        files.extend(matches)
    if not files:
        raise SystemExit(f"cv_parquet: no parquet files matched {spec!r}")
    return files


def pick_text_column(names: list) -> str:
    for c in TEXT_COLUMNS:
        if c in names:
            return c
    raise SystemExit(
        f"cv_parquet: no transcript column found in {names}; expected one of {TEXT_COLUMNS}"
    )


def extract_rows(files: list, clips_dir: Path, locale: str = None) -> list:
    """Writes each row's encoded audio to disk, returning (src, dst, text)."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    jobs, index = [], 0
    for path in files:
        pf = pq.ParquetFile(path)
        text_col = pick_text_column(pf.schema_arrow.names)
        has_locale = "locale" in pf.schema_arrow.names
        stem = Path(path).stem
        for group in tqdm(range(pf.metadata.num_row_groups), desc=f"reading {stem}"):
            for row in pf.read_row_group(group).to_pylist():
                text = (row.get(text_col) or "").strip()
                audio = row.get(AUDIO_COLUMN) or {}
                data = audio.get("bytes") if isinstance(audio, dict) else audio
                if not text or not data:
                    continue
                if locale and has_locale and row.get("locale") != locale:
                    continue
                suffix = (
                    Path(
                        (audio.get("path") or "x.mp3")
                        if isinstance(audio, dict)
                        else "x.mp3"
                    ).suffix
                    or ".mp3"
                )
                src = clips_dir / f"cv_{index:07d}_raw{suffix}"
                if not src.exists():
                    src.write_bytes(data)
                jobs.append((src, clips_dir / f"cv_{index:07d}.wav", text))
                index += 1
    return jobs


def prepare_cv_parquet_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "cv_parquet_")
    manifest_path = output_dir / f"{prefix}train_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            count = sum(1 for _ in f)
        print(
            f"train: manifest already exists -> {manifest_path} ({count} utterances), skipping"
        )
        return {"train": count}

    files = resolve_files(args.files)
    print(f"cv_parquet: {len(files)} shard(s)")
    jobs = extract_rows(files, output_dir / "cv_clips", getattr(args, "locale", None))
    if not jobs:
        raise SystemExit("cv_parquet: every row was empty or filtered out")

    workers = getattr(args, "workers", 8)
    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(
        manifest_path, "w"
    ) as out:
        convert_jobs = [(src, dst) for src, dst, _ in jobs]
        for (_, dst, text), duration in zip(
            jobs,
            tqdm(
                pool.map(convert_clip, convert_jobs), total=len(jobs), desc="cv_parquet"
            ),
        ):
            out.write(
                json.dumps(
                    {"audio_filepath": str(dst), "text": text, "duration": duration},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    for src, _, _ in jobs:
        src.unlink(missing_ok=True)

    print(f"train: wrote {written} utterances -> {manifest_path}")
    return {"train": written}
