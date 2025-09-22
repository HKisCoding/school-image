import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, architecture, input_dim):
        super(EmbeddingModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer in self.architecture:
            next_dim = layer
            self.layers.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
