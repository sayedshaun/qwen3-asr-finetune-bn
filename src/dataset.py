"""Turns NeMo-style manifests into Qwen3-ASR training conversations.

The manifests are exactly what conformer-training-pipeline's prepare_data.py
emits — one JSON object per line with `audio_filepath`, `text` and `duration`.

Qwen3-ASR is trained on its own chat format: the audio goes in the user turn,
and the target goes in the assistant turn as `language <NAME><asr_text><text>`.
Keeping that prefix (rather than training on the bare transcript) preserves the
pretrained language-identification behaviour, which is what the model card
tells you to do.

That prefix is inside the loss, so what goes in it matters: train only on
Bengali rows and the model learns to answer "language Bengali" to every input,
whatever it hears. Rows may therefore carry a `language` field that overrides
the run's tag, which is how non-Bengali replay utterances keep their own.
"""

import json
from pathlib import Path

from torch.utils.data import Dataset

ASR_TEXT_MARKER = "<asr_text>"


def load_manifest(
    path: str, min_duration: float = 0.0, max_duration: float = 1e9
) -> list:
    """Reads a manifest, dropping utterances that are empty, out of the
    duration window, or whose audio has gone missing since preparation."""
    entries, skipped = [], {"empty": 0, "duration": 0, "missing": 0}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            text = (e.get("text") or "").strip()
            if not text:
                skipped["empty"] += 1
                continue
            dur = e.get("duration")
            if dur is not None and not (min_duration <= dur <= max_duration):
                skipped["duration"] += 1
                continue
            if not Path(e["audio_filepath"]).exists():
                skipped["missing"] += 1
                continue
            entries.append(
                {
                    "audio_filepath": e["audio_filepath"],
                    "text": text,
                    "duration": dur,
                    # Per-utterance override of the run's language tag. Bengali
                    # rows leave this unset; multilingual replay rows carry
                    # their own ("English", "Hindi", ...) so the language
                    # prefix they train on stays correct.
                    "language": e.get("language"),
                }
            )
    total_skipped = sum(skipped.values())
    print(
        f"{path}: {len(entries)} utterances"
        + (f" ({total_skipped} skipped: {skipped})" if total_skipped else "")
    )
    if not entries:
        raise SystemExit(
            f"{path} yielded no usable utterances — check paths and duration bounds"
        )
    return entries


def build_conversation(audio_path: str, text: str, language_tag: str) -> list:
    return [
        {"role": "user", "content": [{"type": "audio", "path": audio_path}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"language {language_tag}{ASR_TEXT_MARKER}{text}",
                }
            ],
        },
    ]


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        language_tag: str,
        min_duration: float = 0.0,
        max_duration: float = 1e9,
    ):
        self.entries = load_manifest(manifest_path, min_duration, max_duration)
        self.language_tag = language_tag

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> list:
        e = self.entries[i]
        return build_conversation(
            e["audio_filepath"], e["text"], e.get("language") or self.language_tag
        )


class ConversationCollator:
    """Batches conversations through the processor.

    `output_labels=True` makes the processor build the labels and mask out the
    audio and padding positions itself, so there is no hand-rolled label
    masking here to drift out of sync with the chat template.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, conversations: list) -> dict:
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"output_labels": True, "padding": True},
        )
