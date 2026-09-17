"""Core dataset logic: download OpenSLR-37 (High quality TTS data for Bengali)
and build a NeMo manifest.

Distinct corpus from the OpenSLR-53 source already in this pipeline: SLR-37 is
Google's manually quality-checked multi-speaker TTS data, shipped as two zips -
`bn_bd.zip` (Bangladesh Bengali) and `bn_in.zip` (Indian Bengali). Each holds
`wavs/*.wav` plus a `line_index.tsv` of `fileID<TAB>transcription`.

Small (~10 h) but clean, and it is the only studio-grade audio in the mix.
Everything is folded into our train manifest: dev/test come from the
mcv/openslr sources, so there is no eval-contamination risk.
"""

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.audio import convert_clip
from src.download import download_with_resume

BASE_URL = "https://openslr.trmal.net/resources/37"
ALL_LOCALES = ["bn_bd", "bn_in"]


def download_locale(locale: str, dest: Path) -> Path:
    archive_path = dest / f"{locale}.zip"
    return download_with_resume(f"{BASE_URL}/{locale}.zip", archive_path)


def extract_locale(locale: str, archive_path: Path, dest: Path) -> Path:
    corpus_dir = dest / locale
    marker = dest / f".extracted_{locale}"
    if not marker.exists():
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(corpus_dir)
        marker.touch()
    return corpus_dir


def collect_utterances(corpus_dir: Path, locale: str) -> list:
    """Pair each line_index.tsv row with its wav.

    The tsv is `fileID<TAB>transcription`; some releases pad the columns with
    surrounding whitespace, hence the strip on both fields.
    """
    index_paths = list(corpus_dir.rglob("line_index.tsv"))
    if not index_paths:
        raise SystemExit(f"no line_index.tsv found under {corpus_dir}")

    utterances = []
    for index_path in index_paths:
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                utt_id, text = parts[0].strip(), parts[-1].strip()
                if not utt_id or not text:
                    continue
                matches = list(index_path.parent.rglob(f"{utt_id}.wav"))
                if not matches:
                    continue
                utterances.append((f"{locale}_{utt_id}", matches[0], text))
    return utterances


def write_manifest(
    utterances: list, clips_out_dir: Path, manifest_path: Path, workers: int, desc: str
) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    jobs, rows = [], []
    for utt_id, src, text in utterances:
        dst = clips_out_dir / f"openslr37_{utt_id}.wav"
        jobs.append((src, dst))
        rows.append((dst, text))

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(
        manifest_path, "w"
    ) as out:
        for (dst, text), duration in zip(
            rows, tqdm(pool.map(convert_clip, jobs), total=len(jobs), desc=desc)
        ):
            out.write(
                json.dumps(
                    {"audio_filepath": str(dst), "text": text, "duration": duration},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def prepare_openslr37_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "openslr37_")

    manifest_path = output_dir / f"{prefix}train_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            count = sum(1 for _ in f)
        print(
            f"train: manifest already exists -> {manifest_path} ({count} utterances), skipping"
        )
        return {"train": count}

    raw_dir = output_dir / "openslr37"
    raw_dir.mkdir(parents=True, exist_ok=True)
    locales = getattr(args, "locales", None) or ALL_LOCALES
    if locales == "all":
        locales = ALL_LOCALES

    utterances = []
    for locale in locales:
        corpus_dir = raw_dir / locale
        if args.skip_download:
            if not corpus_dir.exists():
                raise SystemExit(
                    f"skip_download is set but no extracted corpus found at {corpus_dir}"
                )
        else:
            archive_path = download_locale(locale, raw_dir)
            corpus_dir = extract_locale(locale, archive_path, raw_dir)
        utterances.extend(collect_utterances(corpus_dir, locale))

    if not utterances:
        raise SystemExit(f"no openslr37 utterances found under {raw_dir}")

    count = write_manifest(
        utterances,
        output_dir / "wavs",
        manifest_path,
        args.workers,
        desc="openslr37-train",
    )
    print(f"train: wrote {count} utterances -> {manifest_path}")
    return {"train": count}
