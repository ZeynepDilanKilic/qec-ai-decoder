"""
train.py
========
AI decoder'ı eğitir.

TASARIM KARARI: "(kod mesafesi, gürültü seviyesi) başına bir uzman model"
------------------------------------------------------------------------
Her (d, p) çifti için AYRI bir decoder eğitiyoruz. Yani d=3 ve p=0.01 için
bir model, d=3 ve p=0.02 için başka bir model, ...

Neden bu yaklaşım?
  * Adil kıyas: Referans decoder MWPM de her gürültü seviyesi için o p'nin
    gürültü modelini bilerek kurulur. AI decoder'a da aynı bilgiyi vermek
    (o p'nin verisiyle eğitmek) karşılaştırmayı dürüst kılar.
  * Temiz değerlendirme: "Gürültü arttıkça ne oluyor?" eğrisi, her noktada
    o gürültüye özel eğitilmiş bir modelle çizilir; karışıklık olmaz.
  * HPC'ye çok uygun: (d, p) çiftleri tamamen bağımsızdır. SLURM "job array"
    ile düzinelerce model aynı anda farklı GPU'larda eğitilir. Bu, TRUBA
    gibi bir kümeyi kullanmanın asıl kazancıdır.

NOT: Genelleyici (tek modelin tüm p'lere bakması) senaryosunu denemek
isterseniz, aynı kodu birden çok .npz'i birleştirerek çağıracak şekilde
genişletmek kolaydır; bu sürüm sadelik için uzman modele odaklanır.

KULLANIM
--------
    python src/train.py --config configs/config.yaml --distance 5 --p 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import SyndromeDataset, compute_pos_weight
from models import build_model, count_parameters


def set_seed(seed: int):
    """Tekrarlanabilirlik: aynı seed -> aynı sonuç (aynı donanımda)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_entry(manifest: list[dict], distance: int, p: float, split: str):
    """Manifest icinde belirli bir (d, p, split) kaydini bulur."""
    for entry in manifest:
        if (entry["distance"] == distance
                and abs(entry["p"] - p) < 1e-12
                and entry["split"] == split):
            return entry
    raise RuntimeError(
        f"Veri bulunamadi: d={distance}, p={p}, split={split}. "
        f"Once generate_data.py calistirildi mi?"
    )


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    """Bir veri kümesi üzerinde ortalama kayıp ve doğruluk hesaplar."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        # logit > 0  <=>  sigmoid > 0.5  <=>  tahmin = 1
        pred = (logits > 0).float()
        total_correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser(description="QEC AI decoder egitimi")
    parser.add_argument("--config", required=True)
    parser.add_argument("--distance", type=int, required=True,
                        help="Kod mesafesi d")
    parser.add_argument("--p", type=float, required=True,
                        help="Bu modelin egitilecegi gurultu seviyesi")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    results_dir = cfg["eval"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    set_seed(train_cfg["seed"])

    # GPU varsa GPU, yoksa CPU. TRUBA GPU kuyrugunda CUDA gorunur olur.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Manifest'i oku (generate_data.py uretmis olmali).
    with open(os.path.join(cfg["data"]["dir"], "manifest.json")) as f:
        manifest = json.load(f)

    d, p = args.distance, args.p
    print("=" * 70)
    print(f"EGITIM | kod mesafesi d = {d} | gurultu seviyesi p = {p}")
    print("=" * 70)

    # --- Veri yukleme: bu (d, p) icin train ve val ---
    train_ds = SyndromeDataset(find_entry(manifest, d, p, "train")["path"])
    val_ds = SyndromeDataset(find_entry(manifest, d, p, "val")["path"])
    num_detectors = train_ds.num_detectors
    print(f"  num_detectors (model girdi boyutu) = {num_detectors}")
    print(f"  egitim ornegi = {len(train_ds)}, dogrulama ornegi = {len(val_ds)}")
    print(f"  egitim setinde mantiksal flip orani = "
          f"{train_ds.positive_fraction():.4f}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # BatchNorm son kucuk batch'te patlamasin diye
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # --- Sinif dengesizligi: pos_weight ---
    # Dusuk p'de mantiksal hata nadirdir; pos_weight nadir "1" sinifinin
    # kaybini buyuterek agin "hep 0 de" tuzagina dusmesini engeller.
    if train_cfg["pos_weight"] == "auto":
        pw = compute_pos_weight(train_ds)
    else:
        pw = float(train_cfg["pos_weight"])
    print(f"  pos_weight = {pw:.2f}")
    pos_weight = torch.tensor([pw], device=device)

    # --- Model, kayip fonksiyonu, optimizasyon ---
    model = build_model(model_cfg, num_detectors).to(device)
    print(f"  model = {model_cfg['type']}, "
          f"egitilebilir parametre = {count_parameters(model):,}")

    # BCEWithLogitsLoss: sigmoid + binary cross-entropy'yi sayisal kararli
    # tek adimda hesaplar. pos_weight ile dengesizligi telafi eder.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    # val kaybi plato yaparsa ogrenme oranini yariya dusur.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Egitim dongusu + erken durdurma ---
    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    tag = f"d{d}_p{p:.4f}"
    model_path = os.path.join(results_dir, f"model_{tag}.pt")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()          # geri yayilim
            optimizer.step()         # agirlik guncelle
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running_loss / seen

        val_loss, val_acc = evaluate_loss_acc(model, val_loader,
                                              criterion, device)
        scheduler.step(val_loss)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": lr_now, "sec": dt,
        })
        print(f"  epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | lr={lr_now:.1e} | {dt:.1f}s")

        # En iyi modeli sakla (val kaybina gore -> overfit'i engeller).
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                print(f"  >> Erken durdurma: {train_cfg['patience']} epoch "
                      f"boyunca iyilesme yok.")
                break

    # En iyi agirliklari geri yukle ve diske kaydet.
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": model_cfg,
        "distance": d,
        "p": p,
        "num_detectors": num_detectors,
        "best_val_loss": best_val_loss,
    }, model_path)

    with open(os.path.join(results_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"TAMAMLANDI | {tag} | en iyi val_loss = {best_val_loss:.4f}")
    print(f"  model -> {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
