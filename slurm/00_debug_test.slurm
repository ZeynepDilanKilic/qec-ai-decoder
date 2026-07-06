#!/bin/bash
# =====================================================================
# 00_debug_test.slurm  --  Hatti GERCEK kuyruga gondermeden once dener
# =====================================================================
# TRUBA'da 'debug' kuyrugu KISA testler icindir (en fazla ~15 dakika).
# Buyuk bir is gondermeden once "kod calisiyor mu, ortam dogru mu,
# yollar tutuyor mu" sorularini burada UCUZA yanitlarsiniz. Saatlerce
# kuyrukta bekleyip sonra basit bir yol hatasiyla patlamaktan iyidir.
#
# Bu betik kucuk 'smoke_test' config'i ile TUM hatti (uret -> egit ->
# degerlendir) tek tek, minik boyutlarda kosturur.
#
# GONDERME:
#   sbatch slurm/00_debug_test.slurm
#   # bittikten sonra:  cat logs/debug_*.out
# =====================================================================

#SBATCH -p debug                       # TRUBA test kuyrugu (~15 dk siniri)
#SBATCH -A KULLANICI_ADINIZ            # <-- TRUBA hesap adinizla DEGISTIRIN
#SBATCH -J qec-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

set -e

PROJECT_DIR="$HOME/qec-ai-decoder"
cd "$PROJECT_DIR"
mkdir -p logs
CONFIG="configs/smoke_test.yaml"        # KUCUK test config'i

module purge
module load comp/python/ai-tools-kolyoz-1.0
source "$HOME/qec-venv/bin/activate"

echo "Sunucu   : $(hostname)"
echo "=== 1) Veri uretimi (kucuk) ==="
python src/generate_data.py --config "$CONFIG"

echo "=== 2) Egitim (sadece d=3, p=0.0100 ile hizli kontrol) ==="
python src/train.py --config "$CONFIG" --distance 3 --p 0.0100

echo "=== 3) Degerlendirme (sadece var olan model degerlendirilir) ==="
# Not: smoke config'te birden cok (d,p) var; sadece egittigimiz model
# degerlendirilecek, digerleri icin "model yok" uyarisi normaldir.
python src/evaluate.py --config "$CONFIG" || true

echo "=== DEBUG TESTI BITTI: yukarida hata yoksa gercek hatta gecebilirsiniz. ==="
