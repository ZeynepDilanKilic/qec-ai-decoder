"""
evaluate_v3.py
==============
Korelasyonlu gurultu modelinde Transformer + MWPM kiyaslamasi.

Onemli detay: MWPM, KORELASYONLU devreden cikarilan DEM uzerinden kurulur.
Yani MWPM, p_corr'un VAR oldugunu bilir; ama yine de iki-bagimsiz hata
varsayimi yapar -> performansi dusebilir.

NOT (tarihsel): Bu script daha sonra evaluate_v3_blind.py ile
GENISLETILMISTIR. Buradaki tek MWPM referansi, blind script'teki
"MWPM_aware"e karsilik gelir; MWPM'e korelasyon bilgisini vermek
"gizli avantaj" oldugu icin nihai kiyaslamalar evaluate_v3_blind.py
ile yapilmistir. Bu dosya, ara sonuclarin yeniden uretilebilmesi
icin korunmustur.

CIKTI: results_dir/summary_v3.csv + summary_v3.json
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
from qec_common import detector_error_model
import pymatching


def binomial_stderr(rate: float, n: int) -> float:
    if n == 0:
        return 0.0
    return float(np.sqrt(max(rate, 0.0) * (1.0 - rate) / n))


def mwpm_decode(circuit, detectors, observables):
    dem = detector_error_model(circuit)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predicted = matching.decode_batch(detectors).astype(np.uint8)
    errors = np.any(predicted != observables, axis=1)
    num_errors = int(errors.sum())
    num_shots = int(len(errors))
    return {"logical_error_rate": num_errors / num_shots,
            "num_shots": num_shots, "num_errors": num_errors}


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

    with open(os.path.join(cfg["data"]["dir"], "manifest.json")) as f:
        manifest = json.load(f)
    test_entries = sorted(
        [e for e in manifest if e["split"] == "test"],
        key=lambda e: (e["distance"], e["p_iid"], e["p_corr"]))

    rows = []
    for e in test_entries:
        d, p_iid, p_corr = e["distance"], e["p_iid"], e["p_corr"]
        rounds = e["rounds"]
        raw = np.load(e["path"])
        detectors = raw["detectors"].astype(np.uint8)
        observables = raw["observables"].astype(np.uint8)
        ds = SyndromeDataset3D(e["path"])

        # MWPM: korelasyonlu devrenin DEM'i (= sonraki adlandirmayla "aware")
        c_corr = build_circuit_correlated(distance=d, rounds=rounds,
                                          p_iid=p_iid, p_corr=p_corr)
        mw = mwpm_decode(c_corr, detectors, observables)

        model = load_model(results_dir, d, p_iid, p_corr, device)
        if model is not None:
            tr = transformer_ler(model, ds, device)
            t_ler = tr["logical_error_rate"]
        else:
            print(f"  UYARI: model yok (d={d}, p_iid={p_iid}, "
                  f"p_corr={p_corr}), atlaniyor.")
            t_ler = float("nan")

        n = mw["num_shots"]
        row = {"distance": d, "p_iid": p_iid, "p_corr": p_corr,
               "num_shots": n,
               "mwpm_ler": mw["logical_error_rate"],
               "mwpm_stderr": binomial_stderr(mw["logical_error_rate"], n),
               "trans_ler": t_ler,
               "trans_stderr": binomial_stderr(t_ler, n)
               if not np.isnan(t_ler) else 0.0}
        rows.append(row)
        print(f"  d={d} p_iid={p_iid:.4f} p_corr={p_corr:.4f} | "
              f"MWPM={row['mwpm_ler']:.5f} | trans={t_ler:.5f}")

    csv_path = os.path.join(results_dir, "summary_v3.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(results_dir, "summary_v3.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Sonuclar -> {csv_path}")


if __name__ == "__main__":
    main()
