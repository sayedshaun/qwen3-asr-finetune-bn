"""Core dataset logic: pull bengaliAI/Ben-10 (regional Bangla dialect speech)
from the HuggingFace hub and build a NeMo manifest.

Unlike fleurs this repo is not a `datasets`-loadable corpus: it is a plain file
tree of `train/folder_N/` and `valid/folder_N/` directories, each holding its
own `train.csv` (`file_name,transcripts,district`) alongside the wavs it
describes. So we snapshot the repo and walk those folders ourselves.

Everything is folded into our train manifest: dev/test come from the
mcv/openslr sources, so there is no eval-contamination risk in training on all
of Ben-10.

The transcripts carry inline annotation markers - `<>` for unintelligible
speech and `[...]`/`(...)` for annotator notes - which are stripped before the
text reaches the manifest.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.audio import convert_clip

REPO_ID = "bengaliAI/Ben-10"
NOISE_MARKERS = re.compile(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)")


def clean_text(raw: str) -> str:
    text = NOISE_MARKERS.sub(" ", raw)
    return " ".join(text.split())


def collect_utterances(corpus_dir: Path) -> list:
    """Walk every `*/folder_*/train.csv` and pair each row with its wav.

    Returns (utt_id, wav_path, text) triples. Rows whose wav is missing, or
    whose transcript is empty once annotation markers are stripped, are
    dropped.
    """
    import csv

    utterances = []
    for csv_path in sorted(corpus_dir.glob("*/folder_*/*.csv")):
        folder = csv_path.parent
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                file_name = (row.get("file_name") or "").strip()
                text = clean_text(row.get("transcripts") or "")
                if not file_name or not text:
                    continue
                wav_path = folder / file_name
                if not wav_path.exists():
                    continue
                utt_id = re.sub(
                    r"[^0-9A-Za-z]+",
                    "_",
                    f"{folder.parent.name}_{folder.name}_{wav_path.stem}",
                ).strip("_")
                utterances.append((utt_id, wav_path, text))
    return utterances


def write_manifest(
    utterances: list, clips_out_dir: Path, manifest_path: Path, workers: int, desc: str
) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    jobs, rows = [], []
    for utt_id, src, text in utterances:
        dst = clips_out_dir / f"ben10_{utt_id}.wav"
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


def prepare_ben10_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "ben10_")

    manifest_path = output_dir / f"{prefix}train_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            count = sum(1 for _ in f)
        print(
            f"train: manifest already exists -> {manifest_path} ({count} utterances), skipping"
        )
        return {"train": count}

    raw_dir = output_dir / "ben10"
    if args.skip_download:
        if not raw_dir.exists():
            raise SystemExit(
                f"skip_download is set for the ben10 source but no corpus found at {raw_dir}"
            )
        corpus_dir = raw_dir
    else:
        from huggingface_hub import snapshot_download

        corpus_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=raw_dir,
            )
        )

    utterances = collect_utterances(corpus_dir)
    if not utterances:
        raise SystemExit(f"no ben10 utterances found under {corpus_dir}")

    count = write_manifest(
        utterances, output_dir / "wavs", manifest_path, args.workers, desc="ben10-train"
    )
    print(f"train: wrote {count} utterances -> {manifest_path}")
    return {"train": count}
