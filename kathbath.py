"""Core dataset logic: AI4Bharat's Kathbath (Bengali) via the shared
HuggingFace ASR builder.

Human-labelled read speech from contributors across 200+ Indian districts.
Same register as fleurs/mcv but a disjoint speaker pool.

Gated: accept the terms once at
https://huggingface.co/datasets/ai4bharat/Kathbath and export HF_TOKEN.
"""

from src.opends.hf_asr import prepare_hf_dataset

REPO_ID = "ai4bharat/Kathbath"
DEFAULT_CONFIGS = ["bengali"]


def prepare_kathbath_dataset(args):
    return prepare_hf_dataset(args, REPO_ID, DEFAULT_CONFIGS, "kathbath")
