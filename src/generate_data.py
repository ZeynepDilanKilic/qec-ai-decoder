"""
generate_data.py
================
AI decoder'ı eğitmek ve değerlendirmek için veri kümelerini üretir.

İŞ AKIŞI
--------
Her (kod mesafesi d, gürültü seviyesi p) kombinasyonu için:
  - bir stim devresi kurulur,
  - `num_shots` adet (sendrom, etiket) çifti örneklenir,
  - sıkıştırılmış bir .npz dosyasına yazılır.

NEDEN veriyi önceden üretip diske yazıyoruz?
--------------------------------------------
1) Tekrarlanabilirlik: Aynı veri kümesiyle farklı modelleri adil kıyaslarız.
2) TRUBA verimliliği: Veri üretimi CPU işidir, eğitim ise GPU işidir. İkisini
   ayrı SLURM işlerine bölersek pahalı GPU kuyruğunu sadece eğitim için
   kullanır, GPU'yu veri üretirken boşa harcamayız.
3) Veri üretimi "utanç verecek kadar paralel"dir (embarrassingly parallel):
   her (d, p) bağımsızdır, SLURM job array ile aynı anda çalıştırılabilir.

KULLANIM
--------
    python generate_data.py --config ../configs/config.yaml
    # veya tek bir (d, p) için (SLURM job array'den çağrılırken):
    python generate_data.py --config ../configs/config.yaml --distance 5 --p 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from qec_common import build_circuit, sample_shots, circuit_metadata


def generate_one(distance: int, rounds: int, p: float, num_shots: int,
                 out_dir: str, seed: int, split: str) -> str:
    """Tek bir (d, p, split) için veri kümesi üretip .npz olarak kaydeder.

    split : "train" | "val" | "test"
        Aynı (d, p) için train/val/test'e FARKLI seed verilir; böylece
        eğitim ve değerlendirme verisi birbirinden bağımsız olur (veri
        sızıntısı / data leakage olmaz).
    """
    circuit = build_circuit(distance=distance, rounds=rounds, p=p)
    meta = circuit_metadata(circuit)

    t0 = time.time()
    detectors, observables = sample_shots(circuit, num_shots=num_shots, seed=seed)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    fname = f"d{distance}_p{p:.4f}_{split}.npz"
    fpath = os.path.join(out_dir, fname)

    # np.savez_compressed: bit dizileri çok sıkışır, disk ve TRUBA kotası
    # açısından önemli.
    np.savez_compressed(
        fpath,
        detectors=detectors,
        observables=observables,
        distance=distance,
        rounds=rounds,
        p=p,
        seed=seed,
        split=split,
        num_detectors=meta["num_detectors"],
        num_observables=meta["num_observables"],
    )

    # Mantıksal hata "taban oranı": etiketlerin ne kadarı 1. Çok düşük p'de
    # bu oran çok küçük olabilir -> sınıf dengesizliği uyarısı için raporlanır.
    base_rate = float(observables.mean())
    print(f"  [{split:5s}] d={distance} p={p:.4f} | shots={num_shots} "
          f"| detectors={meta['num_detectors']} "
          f"| logical_flip_orani={base_rate:.4f} "
          f"| sure={elapsed:.1f}s | -> {fname}")
    return fpath


def main():
    parser = argparse.ArgumentParser(description="QEC veri kümesi üretici")
    parser.add_argument("--config", required=True, help="config.yaml yolu")
    parser.add_argument("--distance", type=int, default=None,
                        help="Sadece bu d icin uret (SLURM array icin)")
    parser.add_argument("--p", type=float, default=None,
                        help="Sadece bu p icin uret (SLURM array icin)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    out_dir = data_cfg["dir"]

    # Hangi (d, p) çiftleri üretilecek? CLI ile filtrelenebilir.
    distances = [args.distance] if args.distance is not None \
        else data_cfg["distances"]
    noise_levels = [args.p] if args.p is not None \
        else data_cfg["noise_levels"]

    # Her split icin shot sayisi ve seed taban degeri.
    splits = {
        "train": (data_cfg["num_train"], 1000),
        "val":   (data_cfg["num_val"],   2000),
        "test":  (data_cfg["num_test"],  3000),
    }

    print("=" * 70)
    print("VERI URETIMI BASLIYOR")
    print(f"  distances    = {distances}")
    print(f"  noise_levels = {noise_levels}")
    print(f"  cikti dizini = {out_dir}")
    print("=" * 70)

    manifest = []
    for d in distances:
        rounds = d  # dengeli space-time hacmi icin rounds = distance
        for p in noise_levels:
            for split, (n_shots, seed_base) in splits.items():
                # seed: split + d + p'ye gore deterministik ama essiz
                seed = seed_base + d * 100 + int(round(p * 100000))
                fpath = generate_one(d, rounds, p, n_shots, out_dir,
                                     seed, split)
                manifest.append({
                    "distance": d, "rounds": rounds, "p": p,
                    "split": split, "num_shots": n_shots,
                    "path": fpath, "seed": seed,
                })

    # Manifest: hangi dosyanin nerede oldugunu tutan indeks. Egitim ve
    # degerlendirme scriptleri bu dosyayi okuyarak dogru veriyi bulur.
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 70)
    print(f"TAMAMLANDI. {len(manifest)} dosya uretildi.")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
