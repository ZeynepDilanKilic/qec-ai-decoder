#!/bin/bash
# =====================================================================
# run_all.sh  --  Tum hatti (pipeline) tek komutla, bagimliliklarla kurar
# =====================================================================
# Bu betik bir SLURM isi DEGILDIR; login sunucusunda calistirilir:
#
#     bash slurm/run_all.sh
#
# NE YAPAR?
#   3 asamayi SLURM'e SIRAYLA BAGIMLI olarak gonderir:
#     1) veri uretimi  (CPU array)
#     2) egitim        (GPU array)  -- 1 BASARIYLA bitince baslar
#     3) degerlendirme (GPU tek is) -- 2 BASARIYLA bitince baslar
#
#   "--dependency=afterok:<isno>" sayesinde siz bilgisayar basinda
#   beklemeden, tum hat kendi kendine sira sira ilerler.
#
# NEDEN array boyutlarini burada hesapliyoruz?
#   config.yaml'daki izgara degisirse (-array=0-N) elle guncellemek
#   hataya acik. Bunun yerine list_tasks.py'den sayilari okuyup
#   "sbatch --array=..." ile gecici olarak override ediyoruz.
# =====================================================================
set -e

PROJECT_DIR="$HOME/qec-ai-decoder"
cd "$PROJECT_DIR"
CONFIG="configs/config.yaml"

# logs/ dizini yoksa olustur (SLURM cikti dosyalari buraya yazilir).
mkdir -p logs results

# --- Array boyutlarini config'ten otomatik hesapla ----------------------
module purge
module load comp/python/ai-tools-kolyoz-1.0 2>/dev/null || true
source "$HOME/qec-venv/bin/activate"

N_DIST=$(python src/list_tasks.py --config "$CONFIG" --mode distances | wc -l)
N_PAIRS=$(python src/list_tasks.py --config "$CONFIG" --count)
echo "Kod mesafesi sayisi  : $N_DIST   -> veri uretimi array=0-$((N_DIST-1))"
echo "(d,p) cifti sayisi   : $N_PAIRS  -> egitim array=0-$((N_PAIRS-1))"
echo "-------------------------------------------------------------------"

# --- 1) Veri uretimi -----------------------------------------------------
JOB1=$(sbatch --parsable --array=0-$((N_DIST-1)) \
       slurm/01_generate_data.slurm)
echo "[1] Veri uretimi gonderildi   : is no $JOB1"

# --- 2) Egitim (1 basariyla bitince) ------------------------------------
JOB2=$(sbatch --parsable --array=0-$((N_PAIRS-1)) \
       --dependency=afterok:"$JOB1" \
       slurm/02_train.slurm)
echo "[2] Egitim gonderildi         : is no $JOB2  (bagimli: $JOB1)"

# --- 3) Degerlendirme (2 basariyla bitince) -----------------------------
JOB3=$(sbatch --parsable \
       --dependency=afterok:"$JOB2" \
       slurm/03_evaluate.slurm)
echo "[3] Degerlendirme gonderildi  : is no $JOB3  (bagimli: $JOB2)"

echo "-------------------------------------------------------------------"
echo "Tum hat kuyruga eklendi. Durumu izlemek icin:"
echo "    squeue -u \$USER"
echo "Bitince sonuclar 'results/' dizininde olacak."
