"""
list_tasks.py
=============
config.yaml'daki (kod mesafesi, gürültü seviyesi) ızgarasını düz bir
liste halinde yazdırır.

NEDEN bu dosya var?
-------------------
SLURM "job array" kullanırken her dizi görevinin (array task) hangi (d, p)
çiftini işleyeceğini bilmesi gerekir. Deney ızgarasını TEK bir yerde
(config.yaml) tanımlı tutmak için, SLURM betikleri bu scripti çağırıp
SLURM_ARRAY_TASK_ID numaralı satırı okur. Böylece ızgarayı değiştirmek
istediğinizde sadece config.yaml'ı düzenlemeniz yeterli olur; SLURM
betiklerine dokunmazsınız (DRY ilkesi).

KULLANIM
--------
    python src/list_tasks.py --config configs/config.yaml
        -> her satir: "<distance> <p>"   (d x p kartezyen carpimi)

    python src/list_tasks.py --config configs/config.yaml --mode distances
        -> her satir: "<distance>"        (sadece kod mesafeleri)

    python src/list_tasks.py --config configs/config.yaml --count
        -> tek sayi: toplam gorev sayisi (SLURM --array ust siniri icin)
"""

from __future__ import annotations

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="SLURM job array gorev listesi")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pairs", "distances"],
                        default="pairs",
                        help="pairs: (d,p) ciftleri | distances: sadece d")
    parser.add_argument("--count", action="store_true",
                        help="Listeyi degil, sadece toplam gorev sayisini yaz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    distances = cfg["data"]["distances"]
    noise_levels = cfg["data"]["noise_levels"]

    if args.mode == "distances":
        lines = [f"{d}" for d in distances]
    else:  # pairs: d x p kartezyen carpimi
        lines = [f"{d} {p}" for d in distances for p in noise_levels]

    if args.count:
        print(len(lines))
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
