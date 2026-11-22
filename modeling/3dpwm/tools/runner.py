"""Autoregressive world-model training and full-trajectory inference."""

import torch


class Runner:
    def __init__(self, model, criterion, optimizer, cfg):
        self.dynamics_model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg

    def loss_on_seq_ptv3(self, sequence):
        point = sequence[0]
        dynamics_losses = []
        for step in range(self.cfg["training"]["rollout_num"] - 1):
            encoded = self.dynamics_model(point)
            encoded = self.dynamics_model.dynamics_step(
                encoded, sequence[step]["action"]
            )
            delta_xy = self.dynamics_model.decode(encoded)
            delta_xyz = torch.cat(
                [delta_xy, torch.zeros_like(delta_xy[:, :1])],
                dim=1,
            )
            coords = delta_xyz + point["coord"]

            next_point = sequence[step + 1].copy()
            next_point["coord"] = coords
            next_point["feat"] = torch.cat(
                [delta_xyz, point["feat"][:, :3]], dim=-1
            )
            loss = self.criterion(delta_xy, sequence[step + 1]["feat"][:, :2]).sum() / len(
                delta_xy
            )
            dynamics_losses.append(loss)
            point = next_point
        return sum(dynamics_losses)

    def optimize(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.dynamics_model.parameters(), 1, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(norm)

    @torch.no_grad()
    def predict(self, sequence):
        predictions, targets = [], []
        # Match the original inference history and rollout endpoints.
        point = sequence[5].copy()
        for step in range(5, len(sequence) - 2):
            encoded = self.dynamics_model(point)
            encoded = self.dynamics_model.dynamics_step(
                encoded, sequence[step]["action"]
            )
            delta_xy = self.dynamics_model.decode(encoded)
            delta_xyz = torch.cat(
                [delta_xy, torch.zeros_like(delta_xy[:, :1])],
                dim=1,
            )
            coords = delta_xyz + point["coord"]
            point["feat"] = torch.cat(
                [delta_xyz, point["feat"][:, :3]], dim=-1
            )
            point["coord"] = coords
            predictions.append(coords.cpu().numpy())
            targets.append(sequence[step + 1]["coord"].cpu().numpy())
        if not predictions:
            raise ValueError("Test trajectory is too short for the inference history.")
        return predictions, targets
