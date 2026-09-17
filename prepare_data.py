"""CLI entrypoint for downloading one or more data sources and building NeMo
manifests.

All settings come from config.yaml's `data:` section. Each entry under
`data.sources` picks a `type` (see PREPARERS below) and its own options; every
source writes its own `{name}_{split}_manifest.json` files, which are then
merged into the final train/dev/test manifests. If `data.split_ratio` is set,
the merged pool is then reshuffled and resplit by that ratio instead of
keeping each source's own train/dev/test boundaries. The Mozilla Data
Collective API key comes from the MDC_API_KEY env var (.env), never from
config.yaml.
"""

import argparse
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_section
from src.opends.ben10 import prepare_ben10_dataset
from src.opends.cv_parquet import prepare_cv_parquet_dataset
from src.opends.fleurs import prepare_fleurs_dataset
from src.opends.indicvoices import prepare_indicvoices_dataset
from src.opends.kathbath import prepare_kathbath_dataset
from src.opends.mcv import prepare_mcv_dataset
from src.opends.openslr import prepare_openslr_dataset
from src.opends.openslr37 import prepare_openslr37_dataset
from src.opends.shrutilipi import prepare_shrutilipi_dataset
from src.opends.vaani import prepare_vaani_dataset

load_dotenv()

PREPARERS = {
    "mcv": prepare_mcv_dataset,
    "openslr": prepare_openslr_dataset,
    "openslr37": prepare_openslr37_dataset,
    "fleurs": prepare_fleurs_dataset,
    "shrutilipi": prepare_shrutilipi_dataset,
    "indicvoices": prepare_indicvoices_dataset,
    "kathbath": prepare_kathbath_dataset,
    "vaani": prepare_vaani_dataset,
    "ben10": prepare_ben10_dataset,
    "cv_parquet": prepare_cv_parquet_dataset,
}
SPLITS = ("train", "dev", "test")
SPLIT_RATIO_SEED = 42


def merge_manifests(output_dir: Path, source_names: list) -> None:
    for split in SPLITS:
        part_paths = [
            output_dir / f"{name}_{split}_manifest.json" for name in source_names
        ]
        part_paths = [p for p in part_paths if p.exists()]
        if not part_paths:
            continue
        with open(output_dir / f"{split}_manifest.json", "w") as out:
            for part_path in part_paths:
                with open(part_path) as f:
                    out.write(f.read())
        for part_path in part_paths:
            part_path.unlink()


def resplit_manifests(output_dir: Path, split_ratio: dict) -> None:
    """Pool every utterance across the already-merged train/dev/test
    manifests and redistribute them by split_ratio, ignoring whatever
    train/dev/test boundaries each source produced."""
    total_ratio = sum(split_ratio.get(split, 0) for split in SPLITS)
    if total_ratio <= 0:
        raise SystemExit(
            f"data.split_ratio must have a positive total, got {split_ratio}"
        )

    pool = []
    for split in SPLITS:
        manifest_path = output_dir / f"{split}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                pool.extend(f.readlines())

    random.Random(SPLIT_RATIO_SEED).shuffle(pool)

    counts = {}
    start = 0
    split_list = list(SPLITS)
    for split in split_list[:-1]:
        n = round(len(pool) * split_ratio.get(split, 0) / total_ratio)
        counts[split] = pool[start : start + n]
        start += n
    counts[split_list[-1]] = pool[start:]

    for split, lines in counts.items():
        with open(output_dir / f"{split}_manifest.json", "w") as out:
            out.writelines(lines)
        print(
            f"{split}: resplit to {len(lines)} utterances -> {output_dir / f'{split}_manifest.json'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument(
        "--dataset",
        help="Comma-separated source name/type to keep, e.g. --dataset fleurs "
        "(default: all of data.sources)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a data: config value, e.g. --set workers=16",
    )
    cli_args, _ = parser.parse_known_args()

    args = load_section(
        cli_args.config,
        "data",
        required=["output_dir", "sources"],
        overrides=cli_args.set,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources
    if cli_args.dataset:
        wanted = {name.strip() for name in cli_args.dataset.split(",")}
        sources = [src for src in sources if src.get("name", src.get("type")) in wanted]
        if not sources:
            raise SystemExit(
                f"--dataset {cli_args.dataset!r} matched none of data.sources' names/types"
            )

    source_names = []
    for source_cfg in sources:
        source_cfg = dict(source_cfg)
        source_type = source_cfg.pop("type", None)
        if source_type not in PREPARERS:
            raise SystemExit(
                f"Unknown source type {source_type!r}, expected one of {list(PREPARERS)}"
            )
        name = source_cfg.pop("name", source_type)
        source_names.append(name)

        source_cfg.setdefault("output_dir", args.output_dir)
        source_cfg.setdefault("workers", getattr(args, "workers", 8))
        source_cfg.setdefault("skip_download", getattr(args, "skip_download", False))
        source_cfg["manifest_prefix"] = f"{name}_"

        if source_type == "mcv":
            source_cfg.setdefault("splits", list(SPLITS))
            source_cfg["api_key"] = os.environ.get("MDC_API_KEY")
            if not source_cfg["api_key"] and not source_cfg["skip_download"]:
                raise SystemExit(
                    f"Set MDC_API_KEY in .env, or skip_download: true for the {name!r} source"
                )

        PREPARERS[source_type](argparse.Namespace(**source_cfg))

    merge_manifests(output_dir, source_names)

    split_ratio = getattr(args, "split_ratio", None)
    if split_ratio:
        resplit_manifests(output_dir, split_ratio)


if __name__ == "__main__":
    main()
