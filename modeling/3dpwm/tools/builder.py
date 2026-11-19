import torch
from models.dynamics.model import Dynamics


def build_dynamics(cfg, checkpoint=None):
    model = Dynamics(cfg).to(cfg["device"])
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=cfg["device"], weights_only=False)
        model.load_state_dict(state["dynamics"])
    return model


def build_criterion(cfg):
    criterion = cfg["training"]["criterion"]
    if criterion["loss"] == "HuberLoss":
        return torch.nn.HuberLoss(reduction="none", delta=criterion["delta"])
    if criterion["loss"] == "L1Loss":
        return torch.nn.L1Loss(reduction="none")
    raise ValueError(f"Unsupported loss: {criterion['loss']}")
