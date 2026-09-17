"""CLI entrypoint for scoring a checkpoint's WER/CER on a manifest."""

import argparse

from src.config import load_section
from src.evaluation import run_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an eval: config value, e.g. --set batch_size=4",
    )
    cli = p.parse_args()
    args = load_section(
        cli.config,
        "eval",
        required=["model_path", "manifest", "language_tag"],
        overrides=cli.set,
    )
    run_eval(args)


if __name__ == "__main__":
    main()
