#!/bin/bash
# =====================================================================
# 03_evaluate.slurm  --  Tum modelleri degerlendirir + grafikleri uretir
# =====================================================================
# Bu adim tek bir is olarak calisir (JOB ARRAY DEGIL): evaluate.py tum
# (d, p) test kumelerini sirayla gezer, her biri icin AI decoder ve MWPM
# referansini calistirir, sonra ozet tabloyu ve grafikleri uretir.
#
# Tek GPU yeterli: degerlendirme (cikarim/inference) egitimden cok daha
# hafiftir. MWPM kismi CPU'da kosar. Isterseniz -p barbun ile tamamen
# CPU'da da calistirabilirsiniz (sadece biraz yavas olur).
#
# GONDERME (egitim bittikten sonra):
#   sbatch slurm/03_evaluate.slurm
#
# Egitime bagimli otomatik calistirmak icin run_all.sh betigine bakin.
# =====================================================================

#SBATCH -p akya-cuda                   # GPU kuyrugu (CPU icin: barbun)
#SBATCH -A KULLANICI_ADINIZ            # <-- TRUBA hesap adinizla DEGISTIRIN
#SBATCH -J qec-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out      # %j = is numarasi
#SBATCH --error=logs/eval_%j.err

set -e

# --- Proje dizini --------------------------------------------------------
PROJECT_DIR="$HOME/qec-ai-decoder"
cd "$PROJECT_DIR"
CONFIG="configs/config.yaml"

# --- Ortam ---------------------------------------------------------------
module purge
module load comp/python/ai-tools-kolyoz-1.0
source "$HOME/qec-venv/bin/activate"

echo "Sunucu       : $(hostname)"
echo "Baslangic     : $(date)"

# --- Degerlendirmeyi calistir -------------------------------------------
# Cikti: results/summary.csv , results/summary.json ,
#        results/ler_vs_noise.png , results/ai_vs_mwpm.png
python src/evaluate.py --config "$CONFIG"

echo "Bitis         : $(date)"
echo "Degerlendirme tamamlandi. Sonuclar: results/ dizini."
