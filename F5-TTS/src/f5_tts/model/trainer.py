from __future__ import annotations

from email.policy import strict
import gc
import json
import math
from operator import is_
import os
import logging
import string
from tabnanny import check

from numpy import isin
import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, LoRADynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        sample_per_updates=None,  # cadence for generating + scoring monitoring samples (RQ6)
        track_wer: bool = False,  # transcribe each sample with multilingual ASR and log WER
        asr_language: str | None = None,  # ISO code for the ASR (e.g. "nl"); None -> autodetect
        asr_model_name: str = "large-v3",  # faster-whisper multilingual model id/path
        last_per_updates=None,
        snapshot_per_updates=None,  # LoRA-only lightweight snapshot cadence (for RQ6 trajectory)
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        use_lora: bool = False
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            logging.info(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                logging.info(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.snapshot_per_updates = default(snapshot_per_updates, save_per_updates)

        # RQ6 monitoring samples: generate (and optionally WER-score) on their own
        # cadence, decoupled from the expensive full-checkpoint save cadence.
        self.sample_per_updates = default(sample_per_updates, save_per_updates)
        self.track_wer = track_wer
        self.asr_language = asr_language
        self.asr_model_name = asr_model_name
        self._asr_model = None  # lazily loaded in train() when log_samples & track_wer

        # LoRA accent-vector instrumentation (RQ6): the accent vector is exactly the
        # trainable (LoRA) params, so snapshots/norms are computed over those.
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")
        self.run_dir = os.path.dirname(os.path.abspath(self.checkpoint_path)) or "."
        self.stop_file = os.path.join(self.run_dir, "STOP")  # touch this for a graceful stop
        self._prev_lora_vec = None  # previous LoRA vector, for step-to-step cosine

        self.eval_valid = False
        self.best_valid_loss = float("inf")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        self.use_lora = use_lora
        parameter_with_grad = [p for p in model.parameters() if p.requires_grad]
        # weight_decay=0.0 makes AdamW mathematically equivalent to plain Adam, matching
        # the accent-vector paper (Lertpetchpun et al.: Adam, lr 3e-5). AdamW's default
        # (0.01) would apply decoupled decay that regularises the LoRA vector magnitude.
        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(parameter_with_grad, lr=learning_rate, weight_decay=0.0)
        else:
            self.optimizer = AdamW(parameter_with_grad, lr=learning_rate, weight_decay=0.0)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # Watch weight/gradient histograms in WandB ("params update" view).
        if self.logger == "wandb" and self.is_main:
            wandb.watch(self.model, log="all", log_freq=self.save_per_updates)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                logging.info(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        logging.info(f"Removed old checkpoint: {oldest_checkpoint}")

    def _lora_state(self):
        """Named trainable (LoRA) tensors of the unwrapped model, in a stable
        (name-sorted) order. These *are* the accent vector tau (paper Eq. 3)."""
        model = self.accelerator.unwrap_model(self.model)
        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        sd = model.state_dict()
        return names, sd

    def _lora_vector_flat(self):
        """The current LoRA weights flattened to one 1-D CPU float tensor."""
        names, sd = self._lora_state()
        if not names:
            return None
        return torch.cat([sd[n].detach().reshape(-1).float().cpu() for n in names])

    def save_lora_snapshot(self, update):
        """Write only the trainable LoRA tensors (tens of MB, vs ~1.3 GB full
        checkpoint) so RQ6 can trace the accent vector's trajectory cheaply."""
        if not self.is_main:
            return
        names, sd = self._lora_state()
        if not names:
            return
        snap_dir = os.path.join(self.checkpoint_path, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        lora_state = {n: sd[n].detach().cpu() for n in names}
        self.accelerator.save({"lora_state_dict": lora_state, "update": update},
                              os.path.join(snap_dir, f"lora_{update}.pt"))

    def log_lora_geometry(self, update):
        """Log ||tau_t|| and cos(tau_t, tau_{t-1}) so the accent direction can be
        watched stabilising live in WandB, not only post-hoc."""
        if not self.is_main:
            return
        vec = self._lora_vector_flat()
        if vec is None:
            return
        norm = float(vec.norm())
        cos_prev = float("nan")
        if self._prev_lora_vec is not None:
            denom = norm * float(self._prev_lora_vec.norm())
            if denom > 0:
                cos_prev = float(torch.dot(vec, self._prev_lora_vec) / denom)
        self._prev_lora_vec = vec
        self.accelerator.log({"lora_norm": norm, "lora_cos_prev": cos_prev}, step=update)
        if self.logger == "tensorboard":
            self.writer.add_scalar("lora_norm", norm, update)
            if not math.isnan(cos_prev):
                self.writer.add_scalar("lora_cos_prev", cos_prev, update)

    def _load_asr_model(self):
        """Lazily load a multilingual ASR model (faster-whisper) for WER tracking on
        generated samples. Fine-tuning targets non-English accents, so this must be a
        multilingual system rather than the English-only model used at eval time.
        Returns None (and disables WER) if the dependency or model is unavailable."""
        if not (self.track_wer and self.accelerator.is_local_main_process):
            return None
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logging.warning("track_wer=True but faster-whisper is not installed; skipping sample WER.")
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        try:
            model = WhisperModel(self.asr_model_name, device=device, compute_type=compute_type)
        except Exception as e:  # noqa: BLE001 - never let ASR setup crash training
            logging.warning(f"Could not load ASR model '{self.asr_model_name}' for WER tracking: {e}")
            return None
        logging.info(
            f"Loaded ASR model '{self.asr_model_name}' on {device} for sample WER "
            f"(language={self.asr_language or 'auto-detect'})."
        )
        return model

    @staticmethod
    def _truth_from_text_input(text_input):
        """Reconstruct the reference transcript string from a single collated text
        entry. For the char/pinyin tokenizer this is a list of character tokens; for
        pre-tokenized integer inputs the raw text is unrecoverable here, so WER is
        skipped (returns None)."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, list):
            if len(text_input) == 0 or isinstance(text_input[0], str):
                return "".join(text_input)
        return None

    def _sample_wer(self, gen_wav_path, truth):
        """Transcribe a generated sample with the multilingual ASR model and return
        (wer, hypothesis). Returns (None, hypo|None) when WER can't be computed."""
        if self._asr_model is None or not truth or not truth.strip():
            return None, None
        try:
            segments, _ = self._asr_model.transcribe(
                gen_wav_path, beam_size=5, language=self.asr_language
            )
            hypo = "".join(seg.text for seg in segments).strip()
        except Exception as e:  # noqa: BLE001 - transcription failure must not stop training
            logging.warning(f"ASR transcription failed for {gen_wav_path}: {e}")
            return None, None
        try:
            from jiwer import compute_measures
        except ImportError:
            logging.warning("track_wer=True but jiwer is not installed; skipping WER computation.")
            return None, hypo
        # Normalise like the eval pipeline: strip punctuation, lowercase, collapse spaces.
        punctuation_all = string.punctuation + "，。！？；：、（）【】《》「」『』“”‘’"
        ref, hyp = truth, hypo
        for x in punctuation_all:
            ref = ref.replace(x, " ")
            hyp = hyp.replace(x, " ")
        ref = " ".join(ref.lower().split())
        hyp = " ".join(hyp.lower().split())
        if not ref:
            return None, hypo
        wer = float(compute_measures(ref, hyp)["wer"])
        return wer, hypo

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            if self.use_lora:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
            else:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    logging.warning(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            if self.use_lora:
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, valid_dataset: Dataset = None, num_workers=16, resumable_with_seed: int = None):
        if valid_dataset is not None:
            self.eval_valid = True

        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

            # Multilingual ASR for per-sample WER (RQ6 intelligibility-vs-steps curve).
            self._asr_model = self._load_asr_model()
            wer_log_path = f"{log_samples_path}/wer_log.jsonl" if self._asr_model is not None else None

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=lambda batch: collate_fn(batch, use_lora=self.use_lora),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
            if self.eval_valid:
                dev_dataloader = DataLoader(
                    valid_dataset,
                    collate_fn=lambda batch: collate_fn(batch, use_lora=self.use_lora),
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    batch_size=self.batch_size_per_gpu,
                    shuffle=False,
                )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            if self.use_lora:
                batch_sampler = LoRADynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=resumable_with_seed,  # This enables reproducible shuffling
                    drop_residual=False,
                )
            else:
                batch_sampler = DynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=resumable_with_seed,  # This enables reproducible shuffling
                    drop_residual=False,
                )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=lambda batch: collate_fn(batch, use_lora=self.use_lora),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )

            if self.eval_valid:
                sampler = SequentialSampler(valid_dataset)
                if self.use_lora:
                    batch_sampler = LoRADynamicBatchSampler(
                        sampler,
                        self.batch_size_per_gpu,
                        max_samples=self.max_samples,
                        drop_residual=False,
                    )
                else:
                    batch_sampler = DynamicBatchSampler(
                        sampler,
                        self.batch_size_per_gpu,
                        max_samples=self.max_samples,
                        drop_residual=False,
                    )
                valid_dataloader = DataLoader(
                    valid_dataset,
                    collate_fn=lambda batch: collate_fn(batch, use_lora=self.use_lora),
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    batch_sampler=batch_sampler,
                )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        if self.eval_valid:
            train_dataloader, valid_dataloader, self.scheduler = self.accelerator.prepare(
                train_dataloader, valid_dataloader, self.scheduler
            )  # actual multi_gpu updates = single_gpu updates / gpu nums
        else:
            train_dataloader, self.scheduler = self.accelerator.prepare(
                train_dataloader, self.scheduler
            )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        stop_requested = False  # set by the sentinel-file (STOP) graceful stop
        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    lora_idx = batch.get("lora_idx", None)

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler, lora_idx=lora_idx
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    if self.eval_valid:
                        valid_loss = self.evaluate(valid_dataloader)
                        if self.is_main:
                            self.accelerator.log(
                                {"valid_loss": valid_loss}, step=global_update
                            )
                            if self.logger == "tensorboard":
                                self.writer.add_scalar("valid_loss", valid_loss, global_update)

                        if self.best_valid_loss > valid_loss:
                            self.best_valid_loss = valid_loss
                            self.save_checkpoint(global_update, last=True)
                            if self.is_main:
                                logging.info(f"New best valid loss: {self.best_valid_loss:.4f} at update {global_update}, saved checkpoint.")
                        else:
                            if self.is_main:
                                logging.info(f"Valid loss: {valid_loss:.4f} did not improve best loss: {self.best_valid_loss:.4f} at update {global_update}.")
                    else:
                        self.save_checkpoint(global_update, last=True)

                if global_update % self.sample_per_updates == 0 and self.accelerator.sync_gradients:
                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        if isinstance(text_inputs[0], list):
                            if isinstance(text_inputs[0][0], int):
                                infer_text = [text_inputs[0] + [0] + text_inputs[0]]
                            else:
                                infer_text = [text_inputs[0] + [" "] + text_inputs[0]]
                        else:
                            infer_text = [text_inputs[0] + " " + text_inputs[0]]
                        if self.use_lora:
                            if lora_idx is None or lora_idx.dim() == 1:
                                lora_idx_infer = lora_idx
                            else:
                                lora_idx_infer = lora_idx[0]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                lora_idx=lora_idx_infer if self.use_lora else None
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        gen_wav_path = f"{log_samples_path}/update_{global_update}_gen.wav"
                        torchaudio.save(gen_wav_path, gen_audio, target_sample_rate)
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )

                        # RQ6: score sample intelligibility with the multilingual ASR.
                        if self._asr_model is not None:
                            truth = self._truth_from_text_input(text_inputs[0])
                            wer, hypo = self._sample_wer(gen_wav_path, truth)
                            if wer is not None:
                                self.accelerator.log({"sample_wer": wer}, step=global_update)
                                if self.logger == "tensorboard":
                                    self.writer.add_scalar("sample_wer", wer, global_update)
                                logging.info(f"Sample WER at update {global_update}: {wer:.4f}")
                            with open(wer_log_path, "a") as f:
                                f.write(json.dumps({
                                    "update": global_update,
                                    "wer": wer,
                                    "truth": truth,
                                    "hypo": hypo,
                                }) + "\n")

                        self.model.train()

                # RQ6: cheap LoRA-only snapshot + live accent-vector geometry.
                if global_update % self.snapshot_per_updates == 0 and self.accelerator.sync_gradients:
                    self.log_lora_geometry(global_update)
                    self.save_lora_snapshot(global_update)

                # Graceful stop: `touch <run_dir>/STOP` to finish this step and exit cleanly.
                if self.accelerator.sync_gradients and os.path.exists(self.stop_file):
                    if self.is_main:
                        logging.info(
                            f"STOP file found at {self.stop_file}; saving model_last and stopping at update {global_update}."
                        )
                    self.save_checkpoint(global_update, last=True)
                    stop_requested = True
                    break

            if stop_requested:
                break

        if self.eval_valid:
            valid_loss = self.evaluate(valid_dataloader)
            if self.best_valid_loss  > valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint(global_update, last=True)
                if self.is_main:
                    logging.info(f"New best valid loss: {self.best_valid_loss:.4f} at update {global_update}, saved checkpoint.")
            else:
                if self.is_main:
                    logging.info(f"Valid loss: {valid_loss:.4f} did not improve best loss: {self.best_valid_loss:.4f} at update {global_update}.")
        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()

    def evaluate(self, valid_dataloader: DataLoader):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_samples = torch.tensor(0.0, device=self.accelerator.device)

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"evaluation on valid dataset", disable=not self.is_main):
                text_inputs = batch["text"]
                mel_spec = batch["mel"].permute(0, 2, 1)
                mel_lengths = batch["mel_lengths"]
                lora_idx = batch.get("lora_idx", None)

                loss, _, _ = self.model(
                    mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=None, lora_idx=lora_idx
                )

                bs = mel_spec.size(0)

                total_loss += loss.detach() * bs
                total_samples += bs

        total_loss = self.accelerator.reduce(total_loss, reduction="sum")
        total_samples = self.accelerator.reduce(total_samples, reduction="sum")
        
        avg_loss = total_loss / total_samples

        self.model.train()
        return avg_loss.item()

        