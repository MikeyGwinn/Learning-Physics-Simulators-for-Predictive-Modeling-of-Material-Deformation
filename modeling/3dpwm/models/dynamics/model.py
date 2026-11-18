import torch.nn as nn

from models.nn.mlp.mlp import MLP
from models.nn.ptv3.model import PointTransformerV3


class Dynamics(nn.Module):
    """
    Dynamics model.
    Args:
        cfg: config
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pred_dim = 2 # change from 3 to 2
        self.dynamics_model = PointTransformerV3(
            cfg=cfg,
            in_channels=cfg["pointcloud"]["feat_dim"],
            action_channels=cfg["model"]["action_channels"],
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            **cfg["model"]["ptv3"],
        )

        # delta predictor: predict the delta positions of the point cloud
        self.delta_predictor = MLP(
            input_dim=cfg["model"]["ptv3"]["dec_channels"][0],
            hidden_dim=cfg["model"]["ptv3"]["dec_channels"][0] // 2,
            output_dim=self.pred_dim,
            num_layers=4,
            activation=nn.LeakyReLU(),
            norm=nn.LayerNorm,
            zero_init=True,
        )

    def forward(self, input):
        output = self.dynamics_model(input)
        return output

    # Keep in for now; maybe want to use function for our own action conditioning?
    def dynamics_step(self, point, action):
        #point = self.dynamics_model.next(point, action)
        return point

    def decode(self, point):
        point = self.dynamics_model.decode(point)
        return self.delta_predictor(point["feat"])
