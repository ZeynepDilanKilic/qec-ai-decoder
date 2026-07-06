"""
dataset_v2.py
=============
Duz sendrom vektorunu (num_detectors,) 3B spatiotemporal tensore
(T, H, W) cevirir. TransformerDecoder (models_v2) bu tensoru bekler.

NEDEN 3B TENSOR?
----------------
v1 modelleri sendromu yapisiz bir bit dizisi olarak gordu; oysa her
dedektorun fiziksel bir kimligi vardir: hangi TUR (t), gridde hangi
KONUM (y, x). stim her dedektor icin (x, y, t) koordinatini verir.
Bu koordinatlari kullanarak dedektorleri (T, H, W) seklinde bir
"video" karesine yerlestiririz: zaman ekseni olcum turlari, uzay
eksenleri stabilizer plaketleri. Boylece model komsuluk ve zaman
surekliligi bilgisini bedavaya alir.

ESLEME
------
stim koordinatlari (rotated surface code, memory_z):
  x, y in {0, 2, 4, ..., 2d}   (cift tam sayilar)
  t    in {0, 1, ..., rounds}
Her eksendeki BENZERSIZ degerler siralanip 0..N-1 indekslerine
sikistirilir (unique-value compression). d=3, rounds=3 icin bu
(T, H, W) = (4, 4, 4) verir; dedektor olmayan hucreler 0 kalir.

Devre gurultu seviyesinden bagimsiz olarak ayni koordinatlari
urettigi icin, esleme .npz'deki (distance, rounds) bilgisinden
yeniden kurulabilir -- hem v2 (p) hem v3 (p_iid, p_corr) dosyalari
ile calisir.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from qec_common import build_circuit


def _detector_grid_map(distance: int, rounds: int):
    """Her dedektor indeksini (t_idx, y_idx, x_idx)'e esler.

    Donus: (grid_shape, t_idx, y_idx, x_idx)
      grid_shape : (T, H, W)
      *_idx      : shape (num_detectors,) int dizileri
    """
    # Koordinatlar gurultuden bagimsizdir; kucuk sabit bir p yeterli.
    circuit = build_circuit(distance=distance, rounds=rounds, p=0.001)
    coords = circuit.get_detector_coordinates()
    arr = np.array([coords[i] for i in range(circuit.num_detectors)])

    ux = np.unique(arr[:, 0])
    uy = np.unique(arr[:, 1])
    ut = np.unique(arr[:, 2])

    x_idx = np.searchsorted(ux, arr[:, 0])
    y_idx = np.searchsorted(uy, arr[:, 1])
    t_idx = np.searchsorted(ut, arr[:, 2])

    grid_shape = (len(ut), len(uy), len(ux))  # (T, H, W)
    return grid_shape, t_idx, y_idx, x_idx


class SyndromeDataset3D(Dataset):
    """Bir .npz dosyasini (T, H, W) tensorlu Dataset olarak sarar.

    Her ornek:
        x : float32 tensor, shape (T, H, W)  -> sendrom "videosu"
        y : float32 tensor, shape (1,)       -> mantiksal flip etiketi
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.distance = int(data["distance"])
        self.rounds = int(data["rounds"])
        # v2 dosyalari "p", v3 dosyalari "p_iid"/"p_corr" saklar.
        if "p" in data:
            self.p = float(data["p"])
            self.p_corr = 0.0
        else:
            self.p = float(data["p_iid"])
            self.p_corr = float(data["p_corr"])
        self.num_detectors = int(data["num_detectors"])
        self.split = str(data["split"])

        detectors = data["detectors"].astype(np.float32)
        num_shots = detectors.shape[0]

        grid_shape, t_idx, y_idx, x_idx = _detector_grid_map(
            self.distance, self.rounds)
        self.grid_shape = grid_shape

        # Dedektor bitlerini 3B gride serpistir (bos hucreler 0).
        grid = np.zeros((num_shots, *grid_shape), dtype=np.float32)
        grid[:, t_idx, y_idx, x_idx] = detectors

        self.x = torch.from_numpy(grid)
        self.y = torch.from_numpy(data["observables"].astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def positive_fraction(self) -> float:
        """Etiketlerin ne kadari 1 (sinif dengesizligi olcusu)."""
        return float(self.y.mean())


def compute_pos_weight_v2(dataset: SyndromeDataset3D) -> float:
    """v1 ile ayni: BCE pos_weight = (negatif) / (pozitif)."""
    pos = float(dataset.y.sum())
    neg = float(len(dataset) - pos)
    if pos < 1:
        return 1.0
    return neg / pos


if __name__ == "__main__":
    # Hizli test
    import sys
    if len(sys.argv) > 1:
        ds = SyndromeDataset3D(sys.argv[1])
        print(f"distance={ds.distance}, rounds={ds.rounds}, p={ds.p}")
        print(f"grid_shape (T,H,W) = {ds.grid_shape}")
        print(f"num shots = {len(ds)}")
        print(f"x shape = {ds.x.shape}, dtype = {ds.x.dtype}")
        print(f"y shape = {ds.y.shape}")
        print(f"positive fraction = {ds.positive_fraction():.4f}")
        print(f"ornek tensor[0,0]:")
        print(ds.x[0, 0])
    else:
        print("Kullanim: python dataset_v2.py <npz_path>")
