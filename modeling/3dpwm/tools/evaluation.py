from pathlib import Path

import numpy as np
import torch
from data_loaders.data_utils import ptv3_move_to_gpu

from tools.recording import save_rollout


@torch.no_grad()
def evaluate(runner, loader, output_dir, max_sequences=None, recordings=2):
    runner.dynamics_model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_errors = []
    for index, batch in enumerate(loader):
        if max_sequences is not None and index >= max_sequences:
            break
        batch = ptv3_move_to_gpu(batch, runner.cfg["device"])
        predictions, targets = runner.predict(batch)
        if not all(np.isfinite(points).all() for points in predictions):
            raise FloatingPointError(
                f"Non-finite prediction on test trajectory {index}"
            )
        error = float(
            np.mean(
                [
                    np.abs(pred[:, :2] - target[:, :2]).mean() # Only xy coordinates
                    for pred, target in zip(predictions, targets)
                ]
            )
        )
        trajectory_errors.append(error)
        print(f"test trajectory={index} position_mae={error:.6f}", flush=True)
        if index < recordings:
            save_rollout(output_dir / f"test_{index}.rrd", predictions, targets)
    if not trajectory_errors:
        raise ValueError("No test trajectories evaluated")
    return {
        "position_mae": float(np.mean(trajectory_errors)),
        "trajectories": len(trajectory_errors),
    }
