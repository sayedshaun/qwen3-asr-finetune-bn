"""WER/CER for a fine-tuned Qwen3-ASR checkpoint.

Scored with jiwer rather than NeMo's word_error_rate so this project does not
drag in the whole NeMo stack for one metric.
"""

import json
import re
import unicodedata

import jiwer
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from src.dataset import ASR_TEXT_MARKER, build_conversation, load_manifest

_PUNCT = re.compile(r"[।॥!-/:-@\[-`{-~‐-‟…]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip()


def parse_output(text: str) -> str:
    """Qwen3-ASR emits `language <NAME><asr_text><transcript>`."""
    return (
        text.split(ASR_TEXT_MARKER, 1)[1].strip()
        if ASR_TEXT_MARKER in text
        else text.strip()
    )


def transcribe_batch(model, processor, paths, language_tag, max_new_tokens):
    conversations = [build_conversation(p, "", language_tag) for p in paths]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        padding=True,
        continue_final_message=True,
    ).to(model.device, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1] :]
    return [
        t.strip() for t in processor.batch_decode(trimmed, skip_special_tokens=True)
    ]


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = (
        Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )

    entries = load_manifest(args.manifest)
    paths = [e["audio_filepath"] for e in entries]
    refs = [e["text"] for e in entries]

    hyps = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="transcribing"):
        batch = paths[i : i + args.batch_size]
        hyps.extend(
            parse_output(t)
            for t in transcribe_batch(
                model, processor, batch, args.language_tag, args.max_new_tokens
            )
        )

    scored_refs, scored_hyps = (refs, hyps)
    if getattr(args, "normalize", True):
        scored_refs = [normalize(r) for r in refs]
        scored_hyps = [normalize(h) for h in hyps]

    pairs = [(r, h) for r, h in zip(scored_refs, scored_hyps) if r]
    if not pairs:
        raise SystemExit("every reference was empty after normalization")
    scored_refs, scored_hyps = map(list, zip(*pairs))

    wer = jiwer.wer(scored_refs, scored_hyps)
    cer = jiwer.cer(scored_refs, scored_hyps)
    print(f"Utterances: {len(scored_refs)}")
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")

    if getattr(args, "output_predictions", None):
        with open(args.output_predictions, "w") as f:
            f.writelines(
                json.dumps(
                    {"audio_filepath": p, "reference": r, "prediction": h},
                    ensure_ascii=False,
                )
                + "\n"
                for p, r, h in zip(paths, refs, hyps)
            )
        print(f"Predictions written to {args.output_predictions}")
    return wer, cer
