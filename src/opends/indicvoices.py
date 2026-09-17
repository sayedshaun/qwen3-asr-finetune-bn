"""Core dataset logic: AI4Bharat's IndicVoices (Bengali) via the shared
HuggingFace ASR builder.

Spontaneous and extempore speech - the register missing from every other
source in this pipeline, which are all read prompts. The Bengali slice is
large (~96 parquet shards, tens of GB), so expect a long first download.

Gated: accept the terms once at
https://huggingface.co/datasets/ai4bharat/IndicVoices and export HF_TOKEN.
"""

from src.opends.hf_asr import prepare_hf_dataset

REPO_ID = "ai4bharat/IndicVoices"
DEFAULT_CONFIGS = ["bengali"]


def prepare_indicvoices_dataset(args):
    return prepare_hf_dataset(args, REPO_ID, DEFAULT_CONFIGS, "indicvoices")
