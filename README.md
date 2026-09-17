# qwen3-asr-finetune-bn

Fine-tunes the **Qwen3-ASR family** — any checkpoint `transformers` can load as
`Qwen3ASRForConditionalGeneration`, including
[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) — for **Bengali
(Bangla)**. Bengali is the only target this repo is built and tuned for.

Bengali is not one of the model's 30 pretrained languages. Left to auto-detect it
identifies Bangla audio as Hindi and transcribes into Devanagari, so this pipeline
teaches it the language while keeping the model's native output format.

The checkpoint is a config value, so you can move between family members without a code
change. Larger checkpoints need proportionally more VRAM; the knobs in
[Tuning](#tuning) (`use_lora`, `max_duration`, `gradient_checkpointing`) are what make
that tractable.

```bash
python train.py --set model_id=Qwen/Qwen3-ASR-0.6B-hf
```

## Layout

```
config.yaml       data / train / eval sections
prepare_data.py   fetch sources -> NeMo manifests
train.py          fine-tune
eval.py           WER / CER on a manifest
infer.py          transcribe files with a checkpoint
src/dataset.py    manifest -> Qwen3-ASR chat format
src/training.py   model build + Trainer loop
src/evaluation.py generation + jiwer scoring
src/callbacks.py  periodic sample transcription during training
src/opends/       dataset preparers: mcv, openslr, fleurs, cv_parquet, ...
```

Manifests are NeMo-style JSONL, one object per line:

```json
{"audio_filepath": "data/wavs/clip_001.wav", "text": "বাংলা প্রতিলিপি", "duration": 8.4}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add MDC_API_KEY for the mcv source
```

Qwen3-ASR needs **transformers >= 5.13.0**; earlier versions cannot load
`Qwen3ASRForConditionalGeneration`. Use **peft >= 0.21**; older releases fail against
transformers 5.x with `ImportError: cannot import name 'HybridCache'`.

## Run

```bash
python prepare_data.py                      # every source in config.yaml
python prepare_data.py --dataset fleurs     # one source, for a fast first pass
python train.py
python eval.py
python infer.py sample.wav
```

Every entrypoint takes `--set key=value` overrides against its config section:

```bash
python train.py --set learning_rate=2e-5 --set per_device_train_batch_size=4
python train.py --resume                                  # latest checkpoint in output_dir
python eval.py --set model_path=Qwen/Qwen3-ASR-0.6B-hf    # score the base model
```

Score the base model before training. Without that number there is no way to tell
whether a run helped.

## Data sources

Add entries under `data.sources` in `config.yaml`. Each needs a `type` from
`prepare_data.py`'s `PREPARERS`, and writes its own manifest before they are merged
and re-split by `data.split_ratio`.

```yaml
data:
  sources:
    - type: fleurs
      name: fleurs
    - type: cv_parquet          # Common Voice parquet shards already on disk
      name: cv
      files: "/path/to/validated-*-of-*.parquet"
      locale: bn
```

To train on your own audio, write a manifest in the format above and point at it
directly, no code change:

```bash
python train.py --set train_manifest=data/my_train.json --set val_manifest=data/my_dev.json
```

Utterances with empty text, missing audio, or a duration outside
`min_duration`/`max_duration` are dropped, and the counts are reported.

## How the training target is built

The transcript goes in the assistant turn using the model's native output format:

```
language Bengali<asr_text>আজকের সংবাদ
```

Keeping the `language <NAME><asr_text>` prefix preserves the pretrained
language-identification behaviour; training on a bare transcript throws it away.

`processor.apply_chat_template(..., processor_kwargs={"output_labels": True, "padding": True})`
builds the labels and masks audio and padding positions itself, so this repo does no
hand-rolled label masking. Both keys must sit inside `processor_kwargs` — passing
either as a top-level kwarg makes the processor silently drop both, and training then
fails at `loss.backward()` with `loss=None`.

## Tuning

| Knob | Effect |
|---|---|
| `max_duration` | Audio dominates sequence length, so this is the main memory knob |
| `use_lora` | LoRA via peft instead of a full finetune |
| `gradient_checkpointing` | On by default; trades compute for memory |
| `freeze_audio_encoder` | Trains the decoder only; cheaper, usually worse for a new script |
| `gradient_accumulation_steps` | Raise this when batch size is memory-bound |
| `per_device_eval_batch_size` | Evaluation upcasts logits to fp32 over a 151936-token vocab; keep it low |
| `sample_audio` | Clip transcribed every `sample_every_steps`, to watch progress as text |

`transformers` 5.x removed `warmup_ratio` and `evaluation_strategy`; use
`warmup_steps` and `eval_strategy`.
