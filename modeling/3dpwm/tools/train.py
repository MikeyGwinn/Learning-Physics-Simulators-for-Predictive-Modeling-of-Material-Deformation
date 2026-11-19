"""Train the world model and export test rollouts after every epoch."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from data_loaders.data_utils import ptv3_move_to_gpu
from data_loaders.dataloader import Dataset, collate_fn
from torch.utils.data import DataLoader

from tools.builder import build_criterion, build_dynamics
from tools.evaluation import evaluate
from tools.runner import Runner
from tools.utils import get_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/config.yaml")
    parser.add_argument(
        "--epochs", type=int, help="Total epochs, including resumed epochs"
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = get_config(args.config_path)
    if args.epochs is not None:
        cfg["training"]["epoch"] = args.epochs
    os.environ["WANDB_MODE"] = "disabled"
    seed_everything(cfg.get("seed", 0), cfg["training"]["cpu_threads"])
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; check GPU access and the PyTorch installation."
        )

    train_data = Dataset("train", cfg)
    test_data = Dataset("test", cfg)
    workers = cfg["training"]["num_workers"]
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["training"]["batch_sz"],
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["testing"]["num_workers"],
        collate_fn=collate_fn,
    )
    model = build_dynamics(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["optimizer"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["step_size"],
        gamma=cfg["training"]["gamma"],
    )
    first_epoch = 1
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=cfg["device"], weights_only=False
        )
        model.load_state_dict(checkpoint["dynamics"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            raise ValueError(
                "Resume requires a release checkpoint with scheduler state."
            )
        first_epoch = checkpoint["epoch"] + 1
    runner = Runner(model, build_criterion(cfg), optimizer, cfg)
    experiment = Path(cfg["exp_dir"]) / cfg["exp_name"]
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(
        f"train_samples={len(train_data)} test_trajectories={len(test_data)} "
        f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"batch={cfg['training']['batch_sz']} workers={workers} "
        f"cpu_threads={cfg['training']['cpu_threads']}",
        flush=True,
    )

    for epoch in range(first_epoch, cfg["training"]["epoch"] + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        max_gradient_norm = 0.0
        torch.cuda.reset_peak_memory_stats()
        for index, batch in enumerate(train_loader):
            batch = ptv3_move_to_gpu(batch, cfg["device"])
            dynamics_loss = runner.loss_on_seq_ptv3(batch)
            gradient_norm = runner.optimize(dynamics_loss)
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            total_loss += dynamics_loss.item()
            if index % cfg["training"]["log_interval"] == 0:
                print(
                    f"epoch={epoch} batch={index}/{len(train_loader)} "
                    f"dynamics_loss={dynamics_loss.item():.6f} "
                    f"mean_loss={total_loss / (index + 1):.6f}",
                    flush=True,
                )
        seconds = time.perf_counter() - start
        scheduler.step()
        checkpoint_dir = experiment / f"epoch_{epoch:02d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pt"
        torch.save(
            {
                "epoch": epoch,
                "dynamics": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            },
            checkpoint_path.with_suffix(".tmp"),
        )
        checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
        metrics = {
            "epoch": epoch,
            "dynamics_loss": total_loss / len(train_loader),
            "training_seconds": seconds,
            "samples_per_second": len(train_data) / seconds,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "max_gradient_norm": max_gradient_norm,
        }
        metrics.update(
            evaluate(
                runner,
                test_loader,
                Path(cfg["output_dir"]) / f"epoch_{epoch}",
                max_sequences=cfg["testing"].get("max_sequences"),
                recordings=cfg["testing"]["recordings"],
            )
        )
        with (experiment / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
