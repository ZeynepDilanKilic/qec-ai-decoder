"""
models.py
=========
AI tabanlı decoder mimarileri.

PROBLEM TİPİ
------------
Girdi  : sendrom vektörü (num_detectors uzunlugunda 0/1 dizisi)
Çıktı  : tek bir logit -> sigmoid sonrası "mantıksal flip oldu" olasılığı
Yani bu bir İKİLİ SINIFLANDIRMA problemidir.

NEDEN bu iki mimari?
--------------------
1) MLPDecoder (tam bağlı ağ):
   En sade ve güçlü temel. Sendromdaki herhangi iki dedektör arasındaki
   ilişkiyi öğrenebilir (surface code'da hatalar uzak dedektörleri birlikte
   tetikleyebilir, bu yüzden "global" bağlantı faydalı).

2) Conv1dDecoder (1B evrişimli ağ):
   Parametre paylaşımı sayesinde daha az parametreyle yerel sendrom
   örüntülerini yakalar. Kod mesafesi büyüdükçe (d=7, 9, ...) MLP'nin
   parametre sayısı hızla şişer; CNN daha iyi ölçeklenir.

Her iki model de aynı arayüzü paylaşır: girdi (B, num_detectors),
çıktı (B, 1) logit. Böylece eğitim/değerlendirme kodu modelden bağımsız olur.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """Tam bağlı (fully-connected) decoder.

    Mimari: [Linear -> BatchNorm -> ReLU -> Dropout] x N  ->  Linear(.,1)

    - BatchNorm: eğitimi hızlandırır ve kararlı kılar.
    - ReLU: doğrusal olmayanlık; ağın karmaşık karar sınırları öğrenmesini sağlar.
    - Dropout: rastgele nöron söndürerek aşırı öğrenmeyi azaltır.
    """

    def __init__(self, num_detectors: int, hidden_dims: list[int],
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_detectors
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # son katman: tek logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_detectors) -> donus: (B, 1)
        return self.net(x)


class Conv1dDecoder(nn.Module):
    """1B evrişimli decoder.

    Sendrom vektörünü (num_detectors,) tek kanallı bir "sinyal" gibi
    görür: (B, 1, num_detectors). Ardışık Conv1d katmanları yerel
    örüntüleri çıkarır, global average pooling ile sabit boyuta indirir,
    sonra bir lineer kafa logit üretir.
    """

    def __init__(self, num_detectors: int, channels: list[int],
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        # Global average pooling: girdi uzunlugundan (num_detectors)
        # bagimsiz sabit boyutlu ozellik vektoru uretir.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_detectors) -> (B, 1, num_detectors)
        x = x.unsqueeze(1)
        x = self.features(x)          # (B, C, num_detectors)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.head(x)           # (B, 1)


def build_model(model_cfg: dict, num_detectors: int) -> nn.Module:
    """config.yaml'daki 'model' bölümüne göre doğru modeli kurar.

    Bu fabrika fonksiyonu sayesinde eğitim kodu hangi mimariyi
    kullandığını bilmek zorunda kalmaz; sadece config'i değiştirmek yeter.
    """
    mtype = model_cfg["type"].lower()
    if mtype == "mlp":
        return MLPDecoder(
            num_detectors=num_detectors,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=model_cfg["dropout"],
        )
    elif mtype == "conv":
        return Conv1dDecoder(
            num_detectors=num_detectors,
            channels=model_cfg["conv_channels"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Bilinmeyen model tipi: {mtype} (mlp|conv)")


def count_parameters(model: nn.Module) -> int:
    """Eğitilebilir parametre sayısı (raporlama için)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
