"""Core dataset logic: ARTPARK-IISc's Vaani via the shared HuggingFace ASR
builder.

Image-prompted natural speech from rural speakers, published one config per
language. Only Bengali is pulled by default; override with a `configs:`
list on the source entry.

Gated: accept the terms once at
https://huggingface.co/datasets/ARTPARK-IISc/Vaani and export HF_TOKEN.
"""

from src.opends.hf_asr import prepare_hf_dataset

REPO_ID = "ARTPARK-IISc/Vaani"
DEFAULT_CONFIGS = ["Bengali"]


def prepare_vaani_dataset(args):
    return prepare_hf_dataset(args, REPO_ID, DEFAULT_CONFIGS, "vaani")
