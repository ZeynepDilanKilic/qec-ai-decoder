"""
evaluate.py
===========
Projenin değerlendirme katmanı: "gürültü arttıkça ne oluyor?" sorusunu
yanıtlar.

NE YAPAR?
---------
Her (kod mesafesi d, gürültü seviyesi p) test kümesi için:
  1) O (d, p) için eğitilmiş AI decoder'ı yükler ve çalıştırır,
  2) (opsiyonel) o (d, p) için MWPM referans decoder'ını çalıştırır,
  3) logical error rate (LER) ve doğruluk hesaplar,
  4) LER için istatistiksel hata payı (binom standart hatası) ekler,
sonra:
  - tüm sonuçları results/summary.csv ve summary.json olarak yazar,
  - "LER vs gürültü" ve "AI vs MWPM" grafiklerini results/ içine kaydeder.

DEĞERLENDİRME METRİĞİ: LOGICAL ERROR RATE (LER)
-----------------------------------------------
Bir shot'ta mantıksal hata = decoder'ın tahmin ettiği mantıksal flip,
gerçek observable'dan farklı. LER = (mantıksal hatalı shot) / (tüm shot).
QEC'te asıl önemli metrik budur: fiziksel hatalar olsa bile decoder bunları
düzeltip mantıksal bilgiyi koruyabiliyor mu?

BEKLENEN DAVRANIŞ (sağlık kontrolü)
-----------------------------------
  * p küçükken LER küçük; p büyüdükçe LER artar (monoton).
  * Eşiğin ALTINDA: d büyüdükçe LER düşer (kod işe yarıyor).
  * Eşiğin ÜSTÜNDE: d büyüdükçe LER artar (kod zarar veriyor).
  * Eğrilerin kesiştiği p civarı ~ hata eşiği.
  * İyi eğitilmiş bir AI decoder, MWPM eğrisine yakın seyreder; korelasyonlu
    gürültüde MWPM'i geçebilir.

KULLANIM
--------
    python src/evaluate.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")  # ekransiz (headless) TRUBA sunuculari icin sart
import matplotlib.pyplot as plt

from dataset import SyndromeDataset
from models import build_model
from baseline_mwpm import make_mwpm_decoder, mwpm_logical_error_rate


def binomial_stderr(error_rate: float, n: int) -> float:
    """LER bir oran; n shot'tan tahmin edildigi icin belirsizligi vardir.
    Binom standart hatasi: sqrt(p(1-p)/n). Grafikte hata cubugu olur."""
    if n == 0:
        return 0.0
    return float(np.sqrt(max(error_rate, 0.0) * (1.0 - error_rate) / n))


@torch.no_grad()
def ai_logical_error_rate(model, dataset: SyndromeDataset,
                          device, batch_size: int = 8192) -> dict:
    """Egitilmis AI decoder'ini bir test kumesi uzerinde calistirir.
    Mantıksal hata = (logit > 0) tahmini, gerçek etiketten farklıysa."""
    model.eval()
    x_all, y_all = dataset.x, dataset.y
    num_shots = len(dataset)
    num_errors = 0
    for i in range(0, num_shots, batch_size):
        x = x_all[i:i + batch_size].to(device)
        y = y_all[i:i + batch_size].to(device)
        pred = (model(x) > 0).float()
        num_errors += (pred != y).any(dim=1).sum().item()
    ler = num_errors / num_shots
    return {"logical_error_rate": ler, "accuracy": 1.0 - ler,
            "num_shots": num_shots, "num_errors": int(num_errors)}


def load_trained_model(results_dir: str, distance: int, p: float, device):
    """results/model_d{d}_p{p}.pt dosyasindan egitilmis modeli yukler."""
    path = os.path.join(results_dir, f"model_d{distance}_p{p:.4f}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"], ckpt["num_detectors"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="QEC decoder degerlendirme")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["eval"]["results_dir"]
    run_mwpm = cfg["eval"]["run_mwpm_baseline"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    with open(os.path.join(cfg["data"]["dir"], "manifest.json")) as f:
        manifest = json.load(f)

    # Sadece test bolunmesini degerlendiriyoruz.
    test_entries = sorted(
        [e for e in manifest if e["split"] == "test"],
        key=lambda e: (e["distance"], e["p"]),
    )
    distances = sorted({e["distance"] for e in test_entries})

    print("=" * 78)
    print("DEGERLENDIRME BASLIYOR")
    print("=" * 78)

    rows = []  # summary tablosu satirlari
    for entry in test_entries:
        d, p, rounds = entry["distance"], entry["p"], entry["rounds"]
        ds = SyndromeDataset(entry["path"])

        row = {
            "distance": d, "p": p, "rounds": rounds,
            "num_shots": len(ds),
            "logical_flip_base_rate": ds.positive_fraction(),
        }

        # --- AI decoder (bu (d,p) icin egitilmis uzman model) ---
        model = load_trained_model(results_dir, d, p, device)
        if model is not None:
            ai = ai_logical_error_rate(model, ds, device)
            row["ai_ler"] = ai["logical_error_rate"]
            row["ai_ler_stderr"] = binomial_stderr(ai["logical_error_rate"],
                                                   ai["num_shots"])
            row["ai_accuracy"] = ai["accuracy"]
        else:
            print(f"  UYARI: d={d} p={p:.4f} icin model yok, atlaniyor.")
            row["ai_ler"] = float("nan")
            row["ai_ler_stderr"] = 0.0
            row["ai_accuracy"] = float("nan")

        # --- MWPM referans decoder ---
        if run_mwpm:
            matching = make_mwpm_decoder(d, rounds, p)
            mw = mwpm_logical_error_rate(
                matching,
                ds.x.numpy().astype(np.uint8),
                ds.y.numpy().astype(np.uint8),
            )
            row["mwpm_ler"] = mw["logical_error_rate"]
            row["mwpm_ler_stderr"] = binomial_stderr(mw["logical_error_rate"],
                                                     mw["num_shots"])
        else:
            row["mwpm_ler"] = float("nan")
            row["mwpm_ler_stderr"] = 0.0

        rows.append(row)
        print(f"  d={d} p={p:.4f} | "
              f"AI LER={row['ai_ler']:.5f} | "
              f"MWPM LER={row['mwpm_ler']:.5f} | "
              f"shots={row['num_shots']}")

    # ---------------- Sonuc tablolarini kaydet ----------------
    csv_path = os.path.join(results_dir, "summary.csv")
    fieldnames = ["distance", "p", "rounds", "num_shots",
                  "logical_flip_base_rate",
                  "ai_ler", "ai_ler_stderr", "ai_accuracy",
                  "mwpm_ler", "mwpm_ler_stderr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSonuc tablosu -> {csv_path}")

    # ---------------- Grafikler ----------------
    plot_ler_vs_noise(rows, distances, results_dir)
    if run_mwpm:
        plot_ai_vs_mwpm(rows, distances, results_dir)
    print("=" * 78)
    print("DEGERLENDIRME TAMAMLANDI.")
    print("=" * 78)


def plot_ler_vs_noise(rows, distances, results_dir):
    """Grafik 1: her d icin AI decoder'in LER'i vs gurultu (log-log).
    Esik davranisi bu grafikte gorunur: dusuk p'de egriler ayrismali,
    yuksek p'de yer degistirebilir."""
    plt.figure(figsize=(8, 6))
    for d in distances:
        sub = sorted([r for r in rows if r["distance"] == d],
                     key=lambda r: r["p"])
        ps = [r["p"] for r in sub]
        ler = [r["ai_ler"] for r in sub]
        err = [r["ai_ler_stderr"] for r in sub]
        plt.errorbar(ps, ler, yerr=err, marker="o", capsize=3,
                     label=f"AI decoder, d={d}")
    # Referans cizgi: LER = p (decoder hic yokken kabaca beklenen mertebe).
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Fiziksel hata olasiligi  p")
    plt.ylabel("Logical error rate (LER)")
    plt.title("AI Decoder: Gurultu Arttikca Logical Error Rate")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    out = os.path.join(results_dir, "ler_vs_noise.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Grafik -> {out}")


def plot_ai_vs_mwpm(rows, distances, results_dir):
    """Grafik 2: her d icin AI decoder'i MWPM ile yan yana kiyaslar."""
    fig, axes = plt.subplots(1, len(distances),
                             figsize=(5 * len(distances), 5), squeeze=False)
    for ax, d in zip(axes[0], distances):
        sub = sorted([r for r in rows if r["distance"] == d],
                     key=lambda r: r["p"])
        ps = [r["p"] for r in sub]
        ax.errorbar(ps, [r["ai_ler"] for r in sub],
                    yerr=[r["ai_ler_stderr"] for r in sub],
                    marker="o", capsize=3, label="AI decoder")
        ax.errorbar(ps, [r["mwpm_ler"] for r in sub],
                    yerr=[r["mwpm_ler_stderr"] for r in sub],
                    marker="s", capsize=3, label="MWPM (referans)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Fiziksel hata olasiligi  p")
        ax.set_ylabel("Logical error rate")
        ax.set_title(f"d = {d}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("AI Decoder vs MWPM Referans Decoder")
    out = os.path.join(results_dir, "ai_vs_mwpm.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Grafik -> {out}")


if __name__ == "__main__":
    main()
