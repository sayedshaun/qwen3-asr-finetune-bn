"""Fine-tunes Qwen3-ASR on Bengali.

Bengali is not one of Qwen3-ASR's 30 pretrained languages — left alone the
model identifies Bangla audio as Hindi and emits Devanagari. Forcing the
language tag gets Bangla script out but with a high word error rate, so the
point of this run is to actually teach the language rather than coax it.
"""

from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.callbacks import SampleTranscriptionCallback
from src.dataset import ConversationCollator, ManifestDataset


def build_model(args):
    dtype = torch.bfloat16 if getattr(args, "bf16", True) else torch.float32
    model = Qwen3ASRForConditionalGeneration.from_pretrained(args.model_id, dtype=dtype)

    if getattr(args, "freeze_audio_encoder", False):
        frozen = 0
        for name, p in model.named_parameters():
            if "audio_tower" in name or "audio_encoder" in name:
                p.requires_grad = False
                frozen += p.numel()
        print(f"froze {frozen/1e6:.1f}M audio-encoder parameters")

    if getattr(args, "use_lora", False):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(args.lora_target_modules),
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {trainable/1e6:.1f}M")

    if getattr(args, "gradient_checkpointing", False):
        model.config.use_cache = False
    return model


def run_training(args, resume_from_checkpoint=None):
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = build_model(args)

    train_ds = ManifestDataset(
        args.train_manifest, args.language_tag, args.min_duration, args.max_duration
    )
    eval_ds = None
    if getattr(args, "val_manifest", None):
        eval_ds = ManifestDataset(
            args.val_manifest, args.language_tag, args.min_duration, args.max_duration
        )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "linear"),
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to=args.report_to,
        run_name=getattr(args, "run_name", None),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    callbacks = []
    sample_audio = getattr(args, "sample_audio", None)
    if sample_audio and not Path(sample_audio).exists():
        print(f"sample_audio not found, skipping sample transcriptions: {sample_audio}")
        sample_audio = None
    if sample_audio:
        callbacks.append(
            SampleTranscriptionCallback(
                processor,
                sample_audio,
                args.language_tag,
                every_n_steps=getattr(args, "sample_every_steps", 200),
                max_new_tokens=getattr(args, "sample_max_new_tokens", 256),
                max_seconds=getattr(args, "sample_max_seconds", 20.0),
                reference=getattr(args, "sample_reference", None),
            )
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ConversationCollator(processor),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")
