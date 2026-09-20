"""Transcribes a fixed sample clip during training, so you can watch the model
actually learn Bengali rather than infer it from the loss curve.

Loss alone is a poor signal here: the model starts out able to produce Bangla
script when the language is forced, just with the wrong words, so the
interesting change is qualitative. Printing the same clip every N steps makes
that visible.
"""

import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import TrainerCallback

from src.dataset import ASR_TEXT_MARKER, build_conversation


def to_wav16k(src: str, max_seconds: float) -> str:
    """Normalises any input to 16 kHz mono wav and trims it.

    Trimming matters on a small GPU: audio costs ~13 tokens/second and this
    runs while the optimizer states are still resident, so a long clip can OOM
    a training run that was otherwise fitting.
    """
    dst = (
        Path(tempfile.gettempdir())
        / f"qwen3asr_sample_{Path(src).stem[:32]}_{int(max_seconds)}s.wav"
    )
    if not dst.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-t",
                str(max_seconds),
                "-i",
                src,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            check=True,
        )
    return str(dst)


class SampleTranscriptionCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        audio_path: str,
        language_tag: str = "Bengali",
        every_n_steps: int = 200,
        max_new_tokens: int = 256,
        max_seconds: float = 20.0,
        reference: str = None,
    ):
        self.processor = processor
        self.audio = to_wav16k(audio_path, max_seconds)
        self.language_tag = language_tag
        self.every = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.reference = reference
        print(f"[sample] every {every_n_steps} steps on {self.audio}")

    def _transcribe(self, model) -> str:
        conv = [build_conversation(self.audio, "", self.language_tag)]
        inputs = self.processor.apply_chat_template(
            conv,
            tokenize=True,
            return_dict=True,
            continue_final_message=True,
            processor_kwargs={"padding": True},
        ).to(model.device, model.dtype)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return text.split(ASR_TEXT_MARKER, 1)[-1].strip()

    def _run(self, model, step):
        was_training = model.training
        cache_was = getattr(model.config, "use_cache", False)
        model.eval()
        model.config.use_cache = True
        try:
            print(f"\n[sample @ step {step}] {self._transcribe(model)}")
            if self.reference:
                print(f"[reference]          {self.reference}")
        except torch.OutOfMemoryError:
            print(
                f"\n[sample @ step {step}] skipped — OOM during generation; "
                f"lower eval.sample_max_seconds"
            )
            torch.cuda.empty_cache()
        finally:
            model.config.use_cache = cache_was
            if was_training:
                model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and self.every > 0 and state.global_step % self.every == 0:
            self._run(model, state.global_step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._run(model, 0)


class DevWerCallback(TrainerCallback):
    """Scores WER/CER on a dev manifest every N steps, during training.

    eval_loss tells you the run is converging but not whether the transcripts
    are any good, and cross-entropy on a 151936-token vocab is a poor proxy
    for word errors. This transcribes a fixed dev slice with the *live* model
    rather than loading a second copy - on a 16 GB card already holding the
    optimizer states there is no room for two.

    Generation is the expensive part, so keep `utterances` small: it runs
    inside the training loop and every second spent here is a second not
    spent on gradient steps.
    """

    def __init__(
        self,
        processor,
        manifest: str,
        language_tag: str = "Bengali",
        every_n_steps: int = 2000,
        utterances: int = 500,
        batch_size: int = 2,
        max_new_tokens: int = 128,
        max_duration: float = 20.0,
        normalize_text: bool = True,
        seed: int = 42,
    ):
        import random

        from src.dataset import load_manifest

        entries = load_manifest(manifest, 0.5, max_duration)
        if utterances and utterances < len(entries):
            entries = random.Random(seed).sample(entries, utterances)
        self.paths = [e["audio_filepath"] for e in entries]
        self.refs = [e["text"] for e in entries]
        self.processor = processor
        self.language_tag = language_tag
        self.every = every_n_steps
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.normalize_text = normalize_text
        print(
            f"[dev-wer] every {every_n_steps} steps on {len(self.paths)} utterances "
            f"from {manifest}"
        )

    def _score(self, model, step: int) -> None:
        import jiwer

        from src.evaluation import normalize, parse_output, transcribe_batch

        was_training = model.training
        cache_was = getattr(model.config, "use_cache", False)
        model.eval()
        model.config.use_cache = True
        try:
            hyps = []
            for i in range(0, len(self.paths), self.batch_size):
                hyps.extend(
                    transcribe_batch(
                        model,
                        self.processor,
                        self.paths[i : i + self.batch_size],
                        self.language_tag,
                        self.max_new_tokens,
                    )
                )
            hyps = [parse_output(h) for h in hyps]
            refs, preds = self.refs, hyps
            if self.normalize_text:
                refs = [normalize(r) for r in refs]
                preds = [normalize(h) for h in preds]
            # jiwer errors on an empty reference, and an empty hypothesis is a
            # real (total-deletion) error we must not silently drop.
            pairs = [(r, h) for r, h in zip(refs, preds) if r]
            wer = jiwer.wer([r for r, _ in pairs], [h or " " for _, h in pairs])
            cer = jiwer.cer([r for r, _ in pairs], [h or " " for _, h in pairs])
            print(
                f"\n[dev-wer @ step {step}] WER {wer*100:.2f}%  CER {cer*100:.2f}%  "
                f"({len(pairs)} utterances)"
            )
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {"eval/dev_wer": wer, "eval/dev_cer": cer}, step=step
                    )
            except ImportError:
                pass
        except torch.OutOfMemoryError:
            print(
                f"\n[dev-wer @ step {step}] skipped - OOM during generation; "
                f"lower train.dev_wer_batch_size"
            )
            torch.cuda.empty_cache()
        finally:
            model.config.use_cache = cache_was
            if was_training:
                model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and self.every > 0 and state.global_step % self.every == 0:
            self._score(model, state.global_step)
