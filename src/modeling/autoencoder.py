import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def build_model(input_dim: int, encoding_dim: int = 8) -> Autoencoder:
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Autoencoder — input_dim={input_dim}, "
                f"encoding_dim={encoding_dim}, params={total_params:,}")
    return model
