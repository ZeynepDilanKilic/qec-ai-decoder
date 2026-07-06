"""
generate_data_v3.py
===================
Korelasyonlu gurultu modeli icin veri uretici.

generate_data.py'ye benzer ama:
  - build_circuit_correlated() kullanir
  - p_iid ve p_corr olmak uzere IKI parametre saklar
  - Dosya adlandirma: d{d}_piid{x}_pcorr{y}_{split}.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from qec_common import sample_shots, circuit_metadata
from qec_common_v3 import build_circuit_correlated


def generate_one(distance, rounds, p_iid, p_corr, num_shots,
                 out_dir, seed, split):
    circuit = build_circuit_correlated(
        distance=distance, rounds=rounds, p_iid=p_iid, p_corr=p_corr)
    meta = circuit_metadata(circuit)

    t0 = time.time()
    detectors, observables = sample_shots(circuit, num_shots=num_shots,
                                          seed=seed)
    dt = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    fname = f"d{distance}_piid{p_iid:.4f}_pcorr{p_corr:.4f}_{split}.npz"
    fpath = os.path.join(out_dir, fname)
    np.savez_compressed(
        fpath,
        detectors=detectors,
        observables=observables,
        distance=distance,
        rounds=rounds,
        p_iid=p_iid,
        p_corr=p_corr,
        seed=seed,
        split=split,
        num_detectors=meta["num_detectors"],
        num_observables=meta["num_observables"],
    )
    base_rate = float(observables.mean())
    print(f"  [{split:5s}] d={distance} p_iid={p_iid:.4f} "
          f"p_corr={p_corr:.4f} | shots={num_shots} "
          f"| det={meta['num_detectors']} | flip={base_rate:.4f} "
          f"| {dt:.1f}s -> {fname}")
    return fpath


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--distance", type=int, default=None)
    p.add_argument("--p_iid", type=float, default=None)
    p.add_argument("--p_corr", type=float, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dc = cfg["data"]
    out_dir = dc["dir"]

    distances = [args.distance] if args.distance else dc["distances"]
    # noise_pairs: [(p_iid, p_corr), ...]  config'te liste-liste olarak verilir
    pairs = dc["noise_pairs"]
    if args.p_iid is not None and args.p_corr is not None:
        pairs = [[args.p_iid, args.p_corr]]

    splits = {
        "train": (dc["num_train"], 4000),
        "val":   (dc["num_val"],   5000),
        "test":  (dc["num_test"],  6000),
    }

    print("=" * 70)
    print("KORELASYONLU VERI URETIMI")
    print(f"  distances  = {distances}")
    print(f"  noise_pairs = {pairs}")
    print(f"  out_dir    = {out_dir}")
    print("=" * 70)

    manifest = []
    for d in distances:
        rounds = d
        for p_iid, p_corr in pairs:
            for split, (n, base) in splits.items():
                seed = base + d * 1000 + int(round(p_iid * 100000)) \
                       + int(round(p_corr * 100000)) * 7
                fpath = generate_one(d, rounds, p_iid, p_corr, n,
                                     out_dir, seed, split)
                manifest.append({
                    "distance": d, "rounds": rounds,
                    "p_iid": p_iid, "p_corr": p_corr,
                    "split": split, "num_shots": n,
                    "path": fpath, "seed": seed,
                })

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 70)
    print(f"TAMAMLANDI: {len(manifest)} dosya")


if __name__ == "__main__":
    main()
