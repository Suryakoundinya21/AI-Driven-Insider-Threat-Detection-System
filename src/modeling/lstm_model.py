import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim   : int,
        hidden_dim  : int = 64,
        latent_dim  : int = 16,
        num_layers  : int = 2,
        dropout     : float = 0.2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder — Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = True,
        )

        # Bottleneck
        self.fc_enc = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)

        # Decoder — Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu         = nn.ReLU()

    def forward(self, x):
        B, T, F = x.shape

        # Encode
        enc_out, _ = self.encoder(x)
        # Use last timestep from both directions
        enc_last   = enc_out[:, -1, :]
        latent     = self.relu(self.fc_enc(enc_last))

        # Decode — repeat latent across timesteps
        dec_input  = self.relu(self.fc_dec(latent))
        dec_input  = dec_input.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_input)
        recon      = self.output_layer(dec_out)

        return recon

    def encode(self, x):
        enc_out, _ = self.encoder(x)
        enc_last   = enc_out[:, -1, :]
        return self.relu(self.fc_enc(enc_last))


def build_lstm_model(
    input_dim  : int,
    hidden_dim : int = 64,
    latent_dim : int = 16,
    num_layers : int = 2,
) -> LSTMAutoencoder:
    model  = LSTMAutoencoder(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
        num_layers = num_layers,
    )
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"LSTM Autoencoder built — input={input_dim}, "
                f"hidden={hidden_dim}, latent={latent_dim}, params={params:,}")
    return model
