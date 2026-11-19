"""Load a checkpoint and export autoregressive test-set predictions."""

import argparse
import json
import os
from pathlib import Path

from data_loaders.dataloader import Dataset, collate_fn
from torch.utils.data import DataLoader

from tools.builder import build_criterion, build_dynamics
from tools.evaluation import evaluate
from tools.runner import Runner
from tools.utils import get_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-sequences", type=int, help="Default: evaluate the full test split"
    )
    args = parser.parse_args()
    cfg = get_config(args.config_path)
    os.environ["WANDB_MODE"] = "disabled"
    seed_everything(cfg.get("seed", 0), cfg["training"]["cpu_threads"])
    data = Dataset("test", cfg)
    loader = DataLoader(
        data,
        batch_size=1,
        num_workers=cfg["testing"]["num_workers"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = build_dynamics(cfg, args.checkpoint)
    runner = Runner(model, build_criterion(cfg), None, cfg)
    output = args.output_dir or Path(cfg["output_dir"]) / "test"
    metrics = evaluate(
        runner,
        loader,
        output,
        max_sequences=args.max_sequences,
        recordings=cfg["testing"]["recordings"],
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
