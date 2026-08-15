"""
Training loop for ARPM-KGC. Independent implementation (no `ours` import),
but follows the same general shape as a standard dual-encoder KGC trainer:
AdamW + linear/cosine warmup, gradient clipping, optional AMP, and two
alternative multi-GPU paths:

  - DDP (`config.distributed`), detected via torchrun's
    RANK/LOCAL_RANK/WORLD_SIZE env vars, exactly as `config.setup_distributed`
    does -- the standard multi-process approach.
  - `nn.DataParallel` (`config.data_parallel`), a single-process alternative
    that needs no external launcher (`torchrun`, `mp.spawn`, ...) and so is
    often the more convenient option in notebook environments (e.g. Kaggle).
    It splits each batch across every visible GPU automatically. If both are
    set, DDP takes priority.

Full filtered-ranking evaluation (Sec 7.2) is comparatively expensive (it
requires encoding every entity in the KG), so per-epoch / per-eval-step
"light" validation here uses in-batch top-1/top-3 accuracy on `S` (Sec
4.10's final score) purely to pick the best checkpoint; `evaluate.py` /
`predict.py` run the full filtered-ranking protocol on demand.
"""

import json
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .config import ARPMConfig
from .data.dataset import ARPMDataset, Collator, Tokenization
from .data.dict_hub import get_relation_vocab, get_train_triplet_dict, init_tokenizer
from .logging_utils import logger
from .losses import compute_loss
from .metrics import accuracy
from .model import ARPMKGCModel, to_model_output
from .utils import (
    AverageMeter,
    ProgressMeter,
    delete_old_checkpoints,
    get_model_obj,
    load_checkpoint,
    move_to_cuda,
    report_num_trainable_parameters,
    save_checkpoint,
)


class Trainer:
    def __init__(self, config: ARPMConfig):
        self.config = config
        self.best_metric = None
        self.start_epoch = 0

        init_tokenizer(config)
        self.tokenization = Tokenization(config)
        self.relation_vocab = get_relation_vocab(config)
        self.train_triplet_dict = get_train_triplet_dict(config)

        self._build_model()
        self._setup_device()
        self._init_optimizer()
        self._init_data_loaders()
        self._init_scheduler()
        self._init_amp()
        self._maybe_resume()

    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        if self.config.rank == 0:
            logger.info("Building ARPM-KGC model")
        self.model = ARPMKGCModel(self.config, num_relations=len(self.relation_vocab))

    def _setup_device(self) -> None:
        cfg = self.config
        if cfg.distributed:
            self.device = torch.device(f"cuda:{cfg.local_rank}")
            torch.cuda.set_device(self.device)
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cfg.gpu}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        if cfg.distributed:
            # find_unused_parameters=True: several ablation configs (e.g. A8,
            # A9, A10, and anchor_selection_mode in {'random', 'uniform'})
            # deliberately leave some always-instantiated submodules unused
            # on a given forward pass (see model.py's module docstring), so
            # DDP's default unused-parameter check would otherwise error out.
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank,
                broadcast_buffers=False, find_unused_parameters=True,
            )
        elif cfg.data_parallel and torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using nn.DataParallel across GPUs {device_ids} "
                        f"(effective per-GPU batch size ~= batch_size / {len(device_ids)})")
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        elif cfg.data_parallel:
            logger.warning("config.data_parallel=True but fewer than 2 GPUs are visible; "
                           "training on a single device.")

    def _init_optimizer(self) -> None:
        cfg = self.config
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        if cfg.rank == 0:
            report_num_trainable_parameters(get_model_obj(self.model))

    def _init_data_loaders(self) -> None:
        cfg = self.config
        self.train_dataset = ARPMDataset(cfg, cfg.train_path, tokenization=self.tokenization,
                                         seed=cfg.seed)
        collator = Collator(cfg, self.tokenization)

        self.train_sampler = None
        if cfg.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True,
            )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=cfg.batch_size,
            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
            collate_fn=collator, num_workers=cfg.workers, drop_last=True,
        )

        self.valid_loader = None
        if cfg.valid_path and cfg.rank == 0:
            valid_dataset = ARPMDataset(cfg, cfg.valid_path, tokenization=self.tokenization,
                                        seed=cfg.seed)
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=cfg.batch_size, shuffle=True,
                collate_fn=collator, num_workers=cfg.workers,
            )

    def _init_scheduler(self) -> None:
        cfg = self.config
        world_size = cfg.world_size if cfg.distributed else 1
        steps_per_epoch = len(self.train_dataset) // world_size // max(cfg.batch_size, 1)
        total_steps = cfg.epochs * steps_per_epoch
        cfg.warmup = min(cfg.warmup, max(total_steps // 10, 1))

        schedulers = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup}
        self.scheduler = schedulers[cfg.lr_scheduler](
            optimizer=self.optimizer, num_warmup_steps=cfg.warmup, num_training_steps=total_steps,
        )
        if cfg.rank == 0:
            logger.info(f"Total training steps: {total_steps}, warmup steps: {cfg.warmup}")

    def _init_amp(self) -> None:
        """Initializes the AMP gradient scaler (created before resume so its
        state can be restored). Uses the current `torch.amp` API
        (`torch.cuda.amp.GradScaler`/`autocast` are deprecated aliases as of
        recent PyTorch)."""
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

    def _maybe_resume(self) -> None:
        if not self.config.resume:
            return
        ckpt = load_checkpoint(self.config.resume_path, map_location=self.device)
        get_model_obj(self.model).load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.config.use_amp and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.best_metric = ckpt.get("best_metric")
        self.start_epoch = ckpt.get("epoch", -1) + 1
        if self.config.rank == 0:
            logger.info(f"Resumed from {self.config.resume_path}, starting at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    def train_loop(self) -> None:
        for epoch in range(self.start_epoch, self.config.epochs):
            get_model_obj(self.model).set_epoch(epoch)
            self.train_epoch(epoch)
            self._run_eval(epoch)

    def train_epoch(self, epoch: int) -> None:
        cfg = self.config
        if cfg.distributed:
            self.train_sampler.set_epoch(epoch)

        meters = {
            "loss": AverageMeter("Loss", ":.4f"),
            "final": AverageMeter("LFinal", ":.4f"),
            "proto": AverageMeter("LProto", ":.4f"),
            "retr": AverageMeter("LRetr", ":.4f"),
            "top1": AverageMeter("Acc@1", ":6.2f"),
            "top3": AverageMeter("Acc@3", ":6.2f"),
        }
        progress = ProgressMeter(
            len(self.train_loader),
            [meters["loss"], meters["final"], meters["proto"], meters["retr"], meters["top1"], meters["top3"]],
            prefix=f"Epoch [{epoch}]",
        )

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            batch = move_to_cuda(batch) if torch.cuda.is_available() else batch

            if cfg.use_amp:
                with torch.amp.autocast("cuda"):
                    output = to_model_output(self.model(batch))
                    loss_out = compute_loss(get_model_obj(self.model), output, batch, self.train_triplet_dict)
            else:
                output = to_model_output(self.model(batch))
                loss_out = compute_loss(get_model_obj(self.model), output, batch, self.train_triplet_dict)

            self._update_meters(meters, loss_out, cfg.batch_size)
            self._backward(loss_out.total)
            self.scheduler.step()

            if i % cfg.print_freq == 0 and cfg.rank == 0:
                progress.display(i)
            if (i + 1) % cfg.eval_every_n_step == 0:
                self._run_eval(epoch, step=i + 1)
                self.model.train()

        if cfg.rank == 0:
            logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]}")

    def _update_meters(self, meters: Dict[str, AverageMeter], loss_out, batch_size: int) -> None:
        # Reuses loss_out.scores (computed once, inside the same autocast()
        # context as the forward pass when AMP is on) instead of recomputing
        # scores separately -- recomputing outside that context mixes
        # float16/float32 tensors under AMP and crashes; it was also purely
        # redundant compute even without AMP.
        scores = loss_out.scores
        acc1, acc3 = accuracy(scores["S"], scores["target"], topk=(1, 3))

        meters["loss"].update(loss_out.total.item(), batch_size)
        meters["final"].update(loss_out.final.item(), batch_size)
        meters["proto"].update(loss_out.prototype.item(), batch_size)
        meters["retr"].update(loss_out.retrieval.item(), batch_size)
        meters["top1"].update(acc1.item(), batch_size)
        meters["top3"].update(acc3.item(), batch_size)

    def _backward(self, loss: torch.Tensor) -> None:
        cfg = self.config
        self.optimizer.zero_grad()
        if cfg.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_eval(self, epoch: int, step: int = 0) -> None:
        if self.config.rank == 0:
            metric_dict = self.eval_epoch(epoch)
            is_best = self._check_best(metric_dict)
            if is_best:
                self.best_metric = metric_dict
            self._save_checkpoint(epoch, step, is_best)
        if self.config.distributed:
            dist.barrier()

    @torch.no_grad()
    def eval_epoch(self, epoch: int) -> Dict:
        if not self.valid_loader:
            return {}
        self.model.eval()
        meters = {"loss": AverageMeter("Loss", ":.4f"), "top1": AverageMeter("Acc@1", ":6.2f"),
                  "top3": AverageMeter("Acc@3", ":6.2f")}

        for batch in self.valid_loader:
            batch = move_to_cuda(batch) if torch.cuda.is_available() else batch
            output = to_model_output(self.model(batch))
            loss_out = compute_loss(get_model_obj(self.model), output, batch, self.train_triplet_dict)

            scores = loss_out.scores
            acc1, acc3 = accuracy(scores["S"], scores["target"], topk=(1, 3))

            bs = self.config.batch_size
            meters["loss"].update(loss_out.total.item(), bs)
            meters["top1"].update(acc1.item(), bs)
            meters["top3"].update(acc3.item(), bs)

        metric_dict = {"Acc@1": round(meters["top1"].avg, 3), "Acc@3": round(meters["top3"].avg, 3),
                       "loss": round(meters["loss"].avg, 3)}
        logger.info(f"Epoch {epoch} valid metrics: {json.dumps(metric_dict)}")
        return metric_dict

    def _check_best(self, metric_dict: Dict) -> bool:
        if not self.valid_loader:
            return False
        if self.best_metric is None:
            return True
        return metric_dict.get("Acc@1", 0) > self.best_metric.get("Acc@1", 0)

    def _save_checkpoint(self, epoch: int, step: int, is_best: bool) -> None:
        cfg = self.config
        filename = f"{cfg.model_dir}/checkpoint_epoch{epoch}.mdl" if step == 0 \
            else f"{cfg.model_dir}/checkpoint_{epoch}_{step}.mdl"

        model_obj = get_model_obj(self.model)
        state_dict = model_obj.state_dict()

        full_state = {
            "epoch": epoch,
            "config": cfg.to_dict(),
            "num_relations": len(self.relation_vocab),
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if cfg.use_amp else None,
            "best_metric": self.best_metric,
        }
        eval_state = {
            "epoch": epoch,
            "config": cfg.to_dict(),
            "num_relations": len(self.relation_vocab),
            "state_dict": state_dict,
        }

        save_checkpoint(full_state, is_best=is_best, filename=filename, eval_state=eval_state)
        delete_old_checkpoints(path_pattern=f"{cfg.model_dir}/checkpoint_*.mdl", keep=cfg.max_to_keep)
