"""CLI entrypoint for fine-tuning Qwen3-ASR on Bengali.

All settings come from config.yaml's `train:` section; --set overrides any of
them, matching prepare_data.py's interface.
"""

import argparse

from dotenv import load_dotenv

from src.config import load_section
from src.training import run_training

load_dotenv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a train: config value, e.g. --set learning_rate=2e-5",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume from the latest checkpoint in output_dir, or a given path",
    )
    cli = p.parse_args()
    args = load_section(
        cli.config,
        "train",
        required=["model_id", "train_manifest", "output_dir", "language_tag"],
        overrides=cli.set,
    )
    run_training(args, resume_from_checkpoint=cli.resume)


if __name__ == "__main__":
    main()
