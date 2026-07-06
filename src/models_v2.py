"""
models_v2.py
============
Surface code icin Transformer tabanli decoder.

GIRDI: (B, T, H, W) sendrom tensoru (B=batch, T=rounds, H=grid_y, W=grid_x)
CIKTI: (B, 1) logit -> sigmoid -> P(logical_flip = 1)

MIMARI (Google AlphaQubit'in sadelestirilmis hali):
  1) Token embedding: her (t, y, x) hucresi -> d_model boyutlu vektor
  2) Positional encoding: hangi tur + hangi konum bilgisi (OGRENILEN)
  3) Transformer encoder: spatial + temporal attention (pre-norm, 4 katman)
  4) Aggregate: tum tokenleri bir vektore indir (CLS-style)
  5) MLP head: tek logit

NEDEN TRANSFORMER?
  * MLP'de parametre sayisi girdi boyutuyla dogrusal siser (d buyudukce
    ilk katman patlar). Transformer'da d_model sabittir; grid buyudugunde
    sadece kucuk positional embedding buyur -- aksine girdi boyutuna
    bagli olarak sismez.
  * Attention, uzak dedektorler arasindaki korelasyonlari (hata
    zincirleri, hook errors) dogrudan modelleyebilir.
  * Pre-norm (norm_first=True) siralamasi, warmup gerektirmeden kararli
    egitim saglar -- kisa SLURM islerinde onemli.

NOT: pos_weight (sinif dengesizligi) train_v2/train_v3'te BCE'ye eklenir.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """Spatial-temporal transformer for surface code syndrome decoding.

    Parameters
    ----------
    grid_shape : tuple (T, H, W)
        Sendrom tensoru boyutlari (dataset_v2'den gelir)
    d_model : int
        Token embedding boyutu (default 128)
    nhead : int
        Attention head sayisi
    num_layers : int
        Transformer encoder blok sayisi
    dim_feedforward : int
        Encoder icindeki FFN gizli boyutu (128 -> 256 -> 128)
    dropout : float
        Attention + FFN + head dropout orani
    """

    def __init__(self, grid_shape: tuple[int, int, int],
                 d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        T, H, W = grid_shape
        self.grid_shape = grid_shape
        self.num_tokens = T * H * W

        # 1) Token embedding: her hucredeki skaler bit -> d_model vektor.
        self.embed = nn.Linear(1, d_model)

        # 2) Ogrenilen positional embedding: (1, n_tok, d_model).
        #    Grid kucuk ve yapili oldugundan (n_tok <= ~512) sinusoidal
        #    yerine ogrenilen kodlama parametre butcesini asmadan daha
        #    esnek konum bilgisi tasir.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))

        # 3) [CLS] token: dizinin basina eklenir; attention araciligiyla
        #    tum sendromun global ozetini biriktirir (BERT/ViT tarzi).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 4) Pre-norm Transformer encoder (norm_first=True).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 5) MLP head: [CLS] embedding -> tek logit.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights(d_model)

    def _init_weights(self, d_model: int):
        # Kucuk std ile baslatma: pre-norm ile birlikte kararli baslangic.
        nn.init.normal_(self.pos_embed, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.cls_token, std=1.0 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) -> tokenlere ac: (B, n_tok, 1)
        B = x.size(0)
        tokens = x.reshape(B, self.num_tokens, 1)
        tokens = self.embed(tokens) + self.pos_embed      # (B, n_tok, d)

        # [CLS]'i basa ekle -> dizi uzunlugu 1 + n_tok
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d)
        seq = torch.cat([cls, tokens], dim=1)

        z = self.encoder(seq)                             # (B, 1+n_tok, d)
        return self.head(z[:, 0])                         # (B, 1) logit


def build_transformer(model_cfg: dict,
                      grid_shape: tuple[int, int, int]) -> nn.Module:
    """config'teki 'model' bolumune gore TransformerDecoder kurar."""
    return TransformerDecoder(
        grid_shape=grid_shape,
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("nhead", 8),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Hizli sanity check
    grid = (4, 4, 4)
    model = TransformerDecoder(grid_shape=grid)
    n = count_parameters(model)
    print(f"Grid: {grid}")
    print(f"Parametre sayisi: {n:,}")
    x = torch.randn(2, *grid)
    y = model(x)
    print(f"Girdi shape: {x.shape}")
    print(f"Cikti shape: {y.shape}  (beklenen: (2, 1))")
