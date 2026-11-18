import torch.nn as nn


class MLP(nn.Module):
    """
    Generic MLP with `num_layers` layers.
    Args:
        input_dim: input dimension
        hidden_dim: hidden dimension
        output_dim: output dimension
        num_layers: number of layers
        activation: activation function
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=4,
        activation=nn.LeakyReLU(),
        norm=nn.LayerNorm,
        zero_init=False,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm
        self.zero_init = zero_init

        self.fc_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
                if norm:
                    self.fc_layers.append(norm(hidden_dim))
                self.fc_layers.append(activation)
            elif i == num_layers - 1:
                self.fc_layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm:
                    self.fc_layers.append(norm(hidden_dim))
                self.fc_layers.append(activation)

        if zero_init:
            self.zero_init_last_layer()

    def zero_init_last_layer(self):
        """
        Zero initialize the last layer of the MLP.
        This is generally helpful for offset prediction biasing the
        deltas to be zero.
        """
        nn.init.zeros_(self.fc_layers[-1].weight)
        nn.init.zeros_(self.fc_layers[-1].bias)

    def forward(self, x):
        return self.fc_layers(x)
