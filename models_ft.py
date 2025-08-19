
import math
import torch
import torch.nn as nn
from typing import Optional

class FTTransformer(nn.Module):
    """A compact FT-Transformer for tabular data (open-weight).
    Reference concepts: Feature Tokenization + Transformer Encoder for tabular.
    """
    def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 2, n_heads: int = 4, p_dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = nn.Linear(1, d_token)  # per-feature token from scalar
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dropout=p_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, F)
        B, F = x.shape
        x = x.unsqueeze(-1)                              # (B, F, 1)
        tokens = self.tokenizer(x)                       # (B, F, d)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d)
        sequence = torch.cat([cls, tokens], dim=1)      # (B, 1+F, d)
        enc = self.encoder(sequence)[:, 0, :]           # (B, d) -> CLS
        logits = self.head(enc).squeeze(-1)             # (B,)
        return logits
