"""Pure-Python GP-WAITER model implementation.

This module provides a source implementation of ``TModel`` so the project can run
on Python versions where the prebuilt extension module is unavailable.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TModel(nn.Module):
    """Hybrid weighted embedding + transformer regressor.

    Expected input shape: ``(batch, rows, cols)`` where ``rows`` and ``cols``
    match the SNP weight matrix shape.
    """

    def __init__(
        self,
        embed_size: int,
        w: torch.Tensor,
        param: Sequence[dict],
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if w.ndim != 2:
            raise ValueError("Expected weight matrix `w` with shape (rows, cols)")

        self.rows, self.cols = int(w.shape[0]), int(w.shape[1])
        self.register_buffer("weight_map", w.float())

        if not param:
            raise ValueError("`param` must contain at least one layer spec")

        d_model = int(param[0].get("embed_size1", embed_size))
        nhead = max(1, int(param[0].get("num_heads", 1)))

        # Ensure d_model is divisible by nhead for MHA.
        if d_model % nhead != 0:
            nhead = 1

        self.input_proj = nn.Linear(self.rows, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(64, d_model * 2),
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(embed_size, 8)),
            nn.GELU(),
            nn.Linear(max(embed_size, 8), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, rows, cols), got {tuple(x.shape)}")
        if x.shape[1] != self.rows or x.shape[2] != self.cols:
            raise ValueError(
                f"Input shape mismatch. Expected (_, {self.rows}, {self.cols}), got {tuple(x.shape)}"
            )

        # Apply SNP weighting element-wise.
        x = x * self.weight_map

        # Treat each SNP position (col) as a token; row values are token features.
        x = x.transpose(1, 2)  # (batch, cols, rows)
        x = self.input_proj(x)  # (batch, cols, d_model)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze(-1)
