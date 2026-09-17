"""Transcribe one or more audio files with a checkpoint (or the base model).

python infer.py audio.wav
python infer.py a.wav b.mp3 --model Qwen/Qwen3-ASR-0.6B-hf   # compare to base
"""

import argparse

import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from src.evaluation import parse_output, transcribe_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+")
    p.add_argument("--model", default="experiments/qwen3-asr-bn")
    p.add_argument(
        "--language",
        default="Bengali",
        help="forced language tag; pass '' to let the model detect",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    model = (
        Qwen3ASRForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    for path in a.audio:
        raw = transcribe_batch(model, processor, [path], a.language, a.max_new_tokens)[
            0
        ]
        print(f"{path}: {parse_output(raw)}")


if __name__ == "__main__":
    main()
