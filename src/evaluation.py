"""WER/CER for a fine-tuned Qwen3-ASR checkpoint.

Scored with jiwer rather than NeMo's word_error_rate so this project does not
drag in the whole NeMo stack for one metric.
"""

import json
import re
import unicodedata
from pathlib import Path

import jiwer
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from src.dataset import ASR_TEXT_MARKER, build_conversation, load_manifest

_PUNCT = re.compile(r"[।॥!-/:-@\[-`{-~‐-‟…]")

# Text tokens per second of speech, used to size generation per batch from
# the longest clip in it. A single fixed cap (this used to be a flat 256)
# silently truncates anything longer than a short clip - the test manifest
# has utterances up to 26.6s, well past what 256 tokens covers - and a
# truncated hypothesis inflates WER with deletion errors that have nothing to
# do with transcription quality. generate() stops at EOS on its own in the
# normal case, so a generous rate costs little; it's a safety ceiling, not a
# target.
TOKENS_PER_SECOND = 32
MIN_NEW_TOKENS = 256
MAX_NEW_TOKENS_CEILING = 4096


def dynamic_max_new_tokens(durations, override=None):
    if override:
        return override
    longest = max(durations) if durations else 0
    return min(
        MAX_NEW_TOKENS_CEILING, max(MIN_NEW_TOKENS, int(longest * TOKENS_PER_SECOND) + 64)
    )


def load_model_and_processor(model_path, dtype, device):
    """Loads either a full model dir or a LoRA adapter dir (detected by
    adapter_config.json, whose base_model_name_or_path names the base to load
    it onto) - eval.py previously only handled the former, so it silently
    loaded raw base weights (or errored) against a LoRA checkpoint dir."""
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        base_model_id = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        processor = AutoProcessor.from_pretrained(base_model_id)
        base = Qwen3ASRForConditionalGeneration.from_pretrained(base_model_id, dtype=dtype)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(model_path, dtype=dtype)
    return processor, model.to(device).eval()


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
    processor, model = load_model_and_processor(args.model_path, torch.bfloat16, device)

    entries = load_manifest(args.manifest)
    paths = [e["audio_filepath"] for e in entries]
    refs = [e["text"] for e in entries]
    durations = [e["duration"] for e in entries]

    # Config's max_new_tokens is now an optional fixed override; left unset it
    # is sized per batch from that batch's longest clip (see
    # dynamic_max_new_tokens) instead of one number applied to every
    # utterance regardless of length.
    fixed_max_new_tokens = getattr(args, "max_new_tokens", None)

    hyps = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="transcribing"):
        batch = paths[i : i + args.batch_size]
        batch_max_new_tokens = dynamic_max_new_tokens(
            durations[i : i + args.batch_size], override=fixed_max_new_tokens
        )
        hyps.extend(
            parse_output(t)
            for t in transcribe_batch(
                model, processor, batch, args.language_tag, batch_max_new_tokens
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
