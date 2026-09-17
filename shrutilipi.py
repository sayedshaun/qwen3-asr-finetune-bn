"""Core dataset logic: AI4Bharat's Shrutilipi (Bengali) via the shared
HuggingFace ASR builder.

~440 h of All India Radio news bulletins, mined and force-aligned. Broadcast
read speech in Indian Bengali - a register and accent set that none of the
mcv/openslr/fleurs sources cover.

Gated: accept the terms once at
https://huggingface.co/datasets/ai4bharat/Shrutilipi and export HF_TOKEN.
"""

from src.opends.hf_asr import prepare_hf_dataset

REPO_ID = "ai4bharat/Shrutilipi"
DEFAULT_CONFIGS = ["bengali"]


def prepare_shrutilipi_dataset(args):
    return prepare_hf_dataset(args, REPO_ID, DEFAULT_CONFIGS, "shrutilipi")
