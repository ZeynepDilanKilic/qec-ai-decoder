"""
evaluate_v3_blind.py
====================
ADIL KIYASLAMA: MWPM'i 'kor' tut.

Onceki evaluate_v3.py'de MWPM'e KORELASYONLU devrenin DEM'ini vermistik.
Yani MWPM 'gizli avantaj' elde etmisti: korelasyonu modelin parcasi olarak
gormus, ona gore tahmin etmisti. Bu adil degildi.

Gercek donanim senaryosunda:
  - MWPM tasarimi p_iid'i bilir, p_corr'u BILMEZ
  - Test verisi p_corr'lu sistemden gelir

Bu script:
  1) MWPM_naive : p_iid'li i.i.d. devreden DEM cikarir (p_corr=0 varsayar)
  2) MWPM_aware : p_iid + p_corr'lu devreden DEM cikarir (eski sonuc)
  3) Transformer : p_corr'lu veriyle egitilmis modeli kullanir

Her uc decoder'i p_corr'lu test verisinde kosup karsilastirir.
Eger Transformer >> MWPM_naive ise: korelasyon ogreniyor (tezin ana bulgusu)
Eger Transformer ~~ MWPM_aware ise: korelasyon ogrenmis, ama "gizli bilgi"
   ile esit. Daha kotu degil.

CIKTI: results_dir/summary_blind.csv + summary_blind.json
  delta_vs_naive = mwpm_naive_ler - trans_ler  (+ ise Transformer onde)
  delta_vs_aware = mwpm_aware_ler - trans_ler
"""
from __future__ import annotations
import sys, os, json, csv
import numpy as np
import torch
import yaml
sys.path.insert(0, 'src')
import _truba_fix  # noqa: F401

from dataset_v2 import SyndromeDataset3D
from models_v2 import build_transformer
from qec_common_v3 import build_circuit_correlated
from qec_common import build_circuit as build_circuit_iid
from qec_common import detector_error_model
import pymatching


def mwpm_decode(circuit, detectors, observables):
    """Verilen devreden DEM cikar, MWPM ile decode et."""
    dem = detector_error_model(circuit)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predicted = matching.decode_batch(detectors).astype(np.uint8)
    errors = np.any(predicted != observables, axis=1)
    num_errors = int(errors.sum())
    num_shots = int(len(errors))
    return {"logical_error_rate": num_errors / num_shots,
            "num_shots": num_shots, "num_errors": num_errors}


def binomial_stderr(rate: float, n: int) -> float:
    if n == 0:
        return 0.0
    return float(np.sqrt(max(rate, 0.0) * (1.0 - rate) / n))


@torch.no_grad()
def transformer_ler(model, dataset, device, batch_size: int = 4096) -> dict:
    model.eval()
    num_shots, num_errors = len(dataset), 0
    for i in range(0, num_shots, batch_size):
        x = dataset.x[i:i + batch_size].to(device)
        y = dataset.y[i:i + batch_size].to(device)
        with torch.amp.autocast(device_type='cuda',
                                enabled=device.type == 'cuda'):
            logits = model(x)
        pred = (logits > 0).float()
        num_errors += (pred != y).any(dim=1).sum().item()
    ler = num_errors / num_shots
    return {"logical_error_rate": ler, "num_shots": num_shots,
            "num_errors": int(num_errors)}


def load_model(results_dir, d, p_iid, p_corr, device):
    tag = f"d{d}_piid{p_iid:.4f}_pcorr{p_corr:.4f}"
    path = os.path.join(results_dir, f"model_{tag}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_transformer(ckpt["model_cfg"],
                              tuple(ckpt["grid_shape"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    results_dir = cfg["eval"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    with open(os.path.join(cfg["data"]["dir"], "manifest.json")) as f:
        manifest = json.load(f)
    test_entries = sorted(
        [e for e in manifest if e["split"] == "test"],
        key=lambda e: (e["distance"], e["p_iid"], e["p_corr"]))

    print("=" * 78)
    print("KOR/BILGILI MWPM + TRANSFORMER UCLU KIYASLAMASI")
    print("=" * 78)

    rows = []
    for e in test_entries:
        d, p_iid, p_corr = e["distance"], e["p_iid"], e["p_corr"]
        rounds = e["rounds"]
        # MWPM duz dedektor vektoru ister -> dogrudan npz'den oku;
        # Transformer icin ayni dosya SyndromeDataset3D ile 3B'ye acilir.
        raw = np.load(e["path"])
        detectors = raw["detectors"].astype(np.uint8)
        observables = raw["observables"].astype(np.uint8)
        ds = SyndromeDataset3D(e["path"])

        # 1) MWPM_naive: i.i.d. devre DEM'i (p_corr'u BILMEZ)
        c_naive = build_circuit_iid(distance=d, rounds=rounds, p=p_iid)
        naive = mwpm_decode(c_naive, detectors, observables)

        # 2) MWPM_aware: korelasyonlu devre DEM'i ("gizli bilgi")
        c_aware = build_circuit_correlated(distance=d, rounds=rounds,
                                           p_iid=p_iid, p_corr=p_corr)
        aware = mwpm_decode(c_aware, detectors, observables)

        # 3) Transformer (bu noktaya ozel egitilmis uzman model)
        model = load_model(results_dir, d, p_iid, p_corr, device)
        if model is not None:
            trans = transformer_ler(model, ds, device)
            t_ler = trans["logical_error_rate"]
            t_err = binomial_stderr(t_ler, trans["num_shots"])
        else:
            print(f"  UYARI: model yok (d={d}, p_iid={p_iid}, "
                  f"p_corr={p_corr}), atlaniyor.")
            t_ler, t_err = float("nan"), 0.0

        n = naive["num_shots"]
        row = {
            "distance": d, "p_iid": p_iid, "p_corr": p_corr,
            "num_shots": n,
            "mwpm_naive_ler": naive["logical_error_rate"],
            "mwpm_naive_stderr": binomial_stderr(
                naive["logical_error_rate"], n),
            "mwpm_aware_ler": aware["logical_error_rate"],
            "mwpm_aware_stderr": binomial_stderr(
                aware["logical_error_rate"], n),
            "trans_ler": t_ler,
            "trans_stderr": t_err,
            "delta_vs_naive": naive["logical_error_rate"] - t_ler,
            "delta_vs_aware": aware["logical_error_rate"] - t_ler,
        }
        rows.append(row)
        print(f"  d={d} p_iid={p_iid:.4f} p_corr={p_corr:.4f} | "
              f"naive={row['mwpm_naive_ler']:.5f} | "
              f"aware={row['mwpm_aware_ler']:.5f} | "
              f"trans={t_ler:.5f}")

    fieldnames = list(rows[0].keys())
    csv_path = os.path.join(results_dir, "summary_blind.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(results_dir, "summary_blind.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("=" * 78)
    print(f"Sonuclar -> {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
