"""
train_v3.py
===========
Transformer decoder'i KORELASYONLU gurultu verisiyle GPU'da egitir.

train_v2'den tek fark: deney noktasi (d, p) yerine (d, p_iid, p_corr)
uclusu ile tanimlanir; manifest ve checkpoint adlari buna gore secilir.

Modern egitim ozellikleri:
  * Model: TransformerDecoder (3D girdi, spatial-temporal attention)
  * Dataset: SyndromeDataset3D ((T,H,W) tensor)
  * Optimizer: AdamW + cosine warm-restart scheduler
  * Mixed precision (AMP): V100'de 2x hizlanma, bellek 1/2
  * Gradient clipping (max_norm=1.0)
  * GPU device pinning + non-blocking transfers + persistent workers

KULLANIM
--------
    python src/train_v3.py --config configs/config_correlated.yaml \\
                           --distance 5 --p_iid 0.005 --p_corr 0.005
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

import _truba_fix  # noqa: F401

from dataset_v2 import SyndromeDataset3D, compute_pos_weight_v2
from models_v2 import build_transformer, count_parameters


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_entry(manifest, distance, p_iid, p_corr, split):
    for entry in manifest:
        if (entry["distance"] == distance
                and abs(entry["p_iid"] - p_iid) < 1e-12
                and abs(entry["p_corr"] - p_corr) < 1e-12
                and entry["split"] == split):
            return entry
    raise RuntimeError(f"Veri yok: d={distance}, p_iid={p_iid}, "
                       f"p_corr={p_corr}, split={split}")


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # AMP autocast cikarimda da hizlandirir
        with torch.amp.autocast(device_type='cuda',
                                enabled=device.type == 'cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = (logits > 0).float()
        total_correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--p_iid", type=float, required=True)
    parser.add_argument("--p_corr", type=float, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    results_dir = cfg["eval"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # sabit boyutlu girdi: autotune
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cihaz: {device}")

    with open(os.path.join(cfg["data"]["dir"], "manifest.json")) as f:
        manifest = json.load(f)

    d, p_iid, p_corr = args.distance, args.p_iid, args.p_corr
    tag = f"d{d}_piid{p_iid:.4f}_pcorr{p_corr:.4f}"
    print("=" * 70)
    print(f"EGITIM (v3) | d={d} | p_iid={p_iid} | p_corr={p_corr}")
    print("=" * 70)

    train_ds = SyndromeDataset3D(
        find_entry(manifest, d, p_iid, p_corr, "train")["path"])
    val_ds = SyndromeDataset3D(
        find_entry(manifest, d, p_iid, p_corr, "val")["path"])
    grid_shape = train_ds.grid_shape
    print(f"  grid_shape (T,H,W) = {grid_shape}")
    print(f"  egitim = {len(train_ds)}, dogrulama = {len(val_ds)}")

    loader_kw = dict(batch_size=train_cfg["batch_size"],
                     num_workers=train_cfg["num_workers"],
                     pin_memory=(device.type == "cuda"),
                     persistent_workers=(train_cfg["num_workers"] > 0))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True,
                              **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # --- pos_weight: sinif dengesizligi telafisi ---
    # Yuksek pozitif oranda (pf > 0.05) agirliklandirma gereksiz -> 1.0;
    # dusuk oranda neg/pos, ama 10 ile tavanlanir (asiri agirlik logitleri
    # sisirip kalibrasyonu bozuyor).
    pf = train_ds.positive_fraction()
    if train_cfg["pos_weight"] == "auto":
        pw = 1.0 if pf > 0.05 else min(compute_pos_weight_v2(train_ds), 10.0)
    else:
        pw = float(train_cfg["pos_weight"])
    print(f"  positive_fraction = {pf:.4f} -> pos_weight = {pw:.2f}")

    model = build_transformer(model_cfg, grid_shape).to(device)
    print(f"  parametre = {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    history, best_val_loss, best_state, no_improve = [], float("inf"), None, 0
    model_path = os.path.join(results_dir, f"model_{tag}.pt")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda',
                                    enabled=device.type == 'cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()
        train_loss = running / seen

        val_loss, val_acc = evaluate_loss_acc(model, val_loader,
                                              criterion, device)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "lr": lr_now, "sec": dt})
        print(f"  epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"acc={val_acc:.4f} | lr={lr_now:.1e} | {dt:.1f}s")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= train_cfg["patience"]:
                print(f"  >> Erken durdurma ({train_cfg['patience']} epoch).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                "distance": d, "p_iid": p_iid, "p_corr": p_corr,
                "grid_shape": grid_shape, "best_val_loss": best_val_loss},
               model_path)
    with open(os.path.join(results_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("=" * 70)
    print(f"TAMAMLANDI | {tag} | best val_loss = {best_val_loss:.4f}")
    print(f"  model -> {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
