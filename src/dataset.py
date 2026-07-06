"""
dataset.py
==========
Diskteki .npz veri kümelerini PyTorch'un eğitim döngüsüne bağlayan katman.

NEDEN ayrı bir Dataset sınıfı?
------------------------------
PyTorch'un DataLoader'ı; mini-batch'leme, karıştırma (shuffle) ve paralel
okuma işlerini bizim için yapar. Tek yapmamız gereken `__len__` ve
`__getitem__` tanımlamak. Veri zaten bellekte tuttuğumuz kadar küçük
olduğundan (.npz tamamı RAM'e sığar), tüm tensörü bir kez yükleyip
GPU'ya beslemek en hızlı yoldur.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyndromeDataset(Dataset):
    """Tek bir (d, p, split) .npz dosyasını saran Dataset.

    Her örnek:
        x : float32 tensor, shape (num_detectors,)  -> sendrom bitleri
        y : float32 tensor, shape (1,)              -> mantiksal flip etiketi
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        # Bitleri float32'ye çeviriyoruz çünkü sinir ağı katmanları float
        # girdi bekler. Sendrom 0/1 olduğundan ek normalizasyon gerekmez.
        self.x = torch.from_numpy(data["detectors"].astype(np.float32))
        self.y = torch.from_numpy(data["observables"].astype(np.float32))
        # Meta veriyi sakla (model boyutu, raporlama için).
        self.distance = int(data["distance"])
        self.rounds = int(data["rounds"])
        self.p = float(data["p"])
        self.num_detectors = int(data["num_detectors"])
        self.num_observables = int(data["num_observables"])
        self.split = str(data["split"])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def positive_fraction(self) -> float:
        """Etiketlerin ne kadarının 1 olduğu (sınıf dengesizliği ölçüsü)."""
        return float(self.y.mean())


def compute_pos_weight(dataset: SyndromeDataset) -> float:
    """BCEWithLogitsLoss için pos_weight hesaplar.

    pos_weight = (negatif örnek sayısı) / (pozitif örnek sayısı)

    Düşük gürültüde mantıksal hata nadirdir (örn. etiketlerin %1'i 1).
    Bu ağırlık olmadan ağ "her zaman 0 tahmin et" diyerek yüksek doğruluk
    ama işe yaramaz bir decoder üretebilir. pos_weight nadir sınıfın
    kaybını büyüterek bunu engeller.
    """
    pos = float(dataset.y.sum())
    neg = float(len(dataset) - pos)
    if pos < 1:
        # Hiç pozitif yoksa (çok düşük p + az shot) dengeli kabul et.
        return 1.0
    return neg / pos
