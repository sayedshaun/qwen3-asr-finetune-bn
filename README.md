# qwen3-asr-finetune-bn

Fine-tunes [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) for **Bengali (Bangla)**.

## Why

Qwen3-ASR ships with 30 languages — Chinese, English, Cantonese, Arabic, German, French,
Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese,
Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian,
Greek, Hungarian, Macedonian, Romanian. **Bengali is not among them.**

Measured on the base model with Bangla news audio:

| Setting | Output |
|---|---|
| Auto-detect | Identifies the audio as **Hindi** and transcribes into Devanagari |
| Forced `language Bengali` | Bangla script, but heavy word errors (`যোমোকাশমিট্টেকের` for জম্মু কাশ্মীর থেকে) |

So the model's audio encoder clearly hears Bangla phonetics — it just has no
Bengali text mapping. That is what this pipeline trains.

It is worth knowing what you are trading. Gemma-4-E4B (Q4_K_XL) transcribes the
same audio usably today at ~10–12 tok/s and ~6 GB. Qwen3-ASR-0.6B runs at
~50 tok/s in ~1 GB — **5× faster in a fifth the space** — which is the whole
reason to spend the training run.

## Layout

The data layer is reused **verbatim** from `conformer-training-pipeline`:
`prepare_data.py`, `src/config.py`, `src/audio.py` and all of `src/opends/`.
Manifests built there work here unchanged, and vice versa.

```
config.yaml       data / train / eval sections
prepare_data.py   download sources -> NeMo manifests   (reused)
train.py          fine-tune
eval.py           WER / CER on a manifest
infer.py          transcribe files with a checkpoint
src/dataset.py    manifest -> Qwen3-ASR chat format    (new)
src/training.py   model build + Trainer loop           (new)
src/evaluation.py generation + jiwer scoring           (new)
src/opends/       mcv, openslr, fleurs, ...            (reused)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add MDC_API_KEY for the mcv source
```

Qwen3-ASR needs **transformers ≥ 5.13.0**; earlier versions cannot load
`Qwen3ASRForConditionalGeneration`.

## Run

```bash
python prepare_data.py                      # all sources in config.yaml
python prepare_data.py --dataset fleurs     # just one, for a fast first pass
python train.py
python eval.py
python infer.py sample.wav
```

Every entrypoint takes `--set key=value` overrides against its config section:

```bash
python train.py --set learning_rate=2e-5 --set per_device_train_batch_size=4
python eval.py --set model_path=Qwen/Qwen3-ASR-0.6B-hf   # baseline before training
```

Get that baseline number before you train anything — it is the only way to know
whether the run helped.

## How the target is built

Qwen3-ASR is trained on its own chat format, with the transcript in the
assistant turn using the model's native output format:

```
language Bengali<asr_text>আজকের সংবাদ
```

Keeping the `language <NAME><asr_text>` prefix preserves the pretrained
language-identification behaviour; training on a bare transcript throws it
away. `processor.apply_chat_template(..., processor_kwargs={"output_labels": True})`
builds the labels and masks audio and padding positions itself, so this repo
does no hand-rolled label masking.

## Tuning

| Knob | Effect |
|---|---|
| `max_duration` | Audio is ~13 tokens/second, so this is the real memory knob |
| `freeze_audio_encoder` | Trains the decoder only. Cheaper, but usually worse when teaching a new script |
| `use_lora` | LoRA via peft instead of a full finetune. Needed under ~8 GB VRAM |
| `gradient_checkpointing` | On by default; trades compute for memory |

A full bf16 finetune of 0.6B needs roughly 7–8 GB for weights plus optimizer
states before activations, so it will not fit on a 4 GB card — use `use_lora: true`
there, or train on a larger GPU.

## Status

Verified end to end on an RTX 2050 (4 GB): LoRA training steps, the eval loop,
checkpoint saving, and the sample-transcription callback all run. What has
*not* run is a real dataset download or a full training run.

Environment used: torch 2.13.0+cu130, transformers 5.17.0, peft 0.21.0.
peft <0.21 fails against transformers 5.x with
`ImportError: cannot import name 'HybridCache'`.

### Measured on a 4 GB RTX 2050 (LoRA, r=32, bf16, gradient checkpointing)

23.3M trainable of 805.7M (2.89%); weights sit at ~1.58 GB before activations.

| Config | Step time | Peak VRAM |
|---|---|---|
| bs=1, 10s audio | 0.47s | 2325 MiB |
| bs=1, 15s audio | 0.56s | 2482 MiB |
| bs=1, 20s audio | 0.69s | 2647 MiB |
| bs=1, 30s audio | **OOM** | — |
| bs=2, 10s audio | **OOM** | — |

So a 4 GB card is limited to **batch size 1 and `max_duration` ≈ 20** — use
`gradient_accumulation_steps` for an effective batch. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; it measurably helps at this
margin. A full (non-LoRA) finetune does not fit.

### Two API traps, both already handled here

**`padding` and `output_labels` must both live inside `processor_kwargs`.**
Passing either as a top-level kwarg to `apply_chat_template` makes the
processor log a warning and then silently drop *both*, so the batch arrives
with no `labels`, the model returns `loss=None`, and training dies at
`loss.backward()` with `AttributeError: 'NoneType' object has no attribute
'backward'` — which reads like a model bug and is not.

**transformers 5.x removed `warmup_ratio`** (and `evaluation_strategy`). Use
`warmup_steps` and `eval_strategy`.

## Deploying the result

llama.cpp serves Qwen3-ASR through `ggml-org/Qwen3-ASR-0.6B-GGUF` (Q8_0 weights
plus a separate mmproj). Converting a fine-tuned checkpoint means running
llama.cpp's `convert_hf_to_gguf.py` and quantizing — the sibling
`llamacpp-inference-server` repo already has the serving side set up.
