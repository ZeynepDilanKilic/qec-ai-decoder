#!/bin/bash
# =====================================================================
# 01_generate_data.slurm  --  Sendrom veri kumelerini uretir (CPU isi)
# =====================================================================
# Veri uretimi bir CPU isidir (stim simulasyonu). GPU GEREKMEZ; bu yuzden
# pahali GPU kuyrugu yerine CPU kuyruguna gonderiyoruz.
#
# JOB ARRAY: Her kod mesafesi (d) BAGIMSIZ uretilebilir. "--array" ile
# her d ayri bir gorev olarak ayni anda farkli sunuculara dagitilir.
# Asagida --array=0-2  => 3 gorev (d=3, d=5, d=7 icin config'e gore).
#
# GONDERME:
#   sbatch slurm/01_generate_data.slurm
#
# Array boyutunu config'inizdeki kod mesafesi sayisina gore ayarlayin:
#   python src/list_tasks.py --config configs/config.yaml --mode distances | wc -l
# =====================================================================

#SBATCH -p barbun                      # CPU kuyrugu (TRUBA: barbun/mid2/long)
#SBATCH -A KULLANICI_ADINIZ            # <-- TRUBA hesap adinizla DEGISTIRIN
#SBATCH -J qec-gendata                 # is adi
#SBATCH --array=0-2                    # <-- kod mesafesi sayisi - 1
#SBATCH --nodes=1                      # gorev basina 1 sunucu
#SBATCH --ntasks=1                     # tek surec (stim ic paralelligi yeter)
#SBATCH --cpus-per-task=8              # stim ornekleme icin birkac cekirdek
#SBATCH --time=02:00:00                # ust sinir (gercekte cok daha kisa)
#SBATCH --output=logs/gendata_%A_%a.out  # %A=is no, %a=array gorev no
#SBATCH --error=logs/gendata_%A_%a.err

set -e

# --- Proje dizinine gec --------------------------------------------------
# Projeyi nereye kopyaladiysaniz orayi yazin. TRUBA'da yogun I/O isleri
# icin scratch alani onerilir (ev dizininden hizli).
PROJECT_DIR="$HOME/qec-ai-decoder"
cd "$PROJECT_DIR"
CONFIG="configs/config.yaml"

# --- Ortami hazirla ------------------------------------------------------
module purge
module load comp/python/ai-tools-kolyoz-1.0   # setup_env.sh ile ayni modul
source "$HOME/qec-venv/bin/activate"

# --- Bilgilendirme (debug icin faydali) ---------------------------------
echo "Sunucu       : $(hostname)"
echo "Array gorevi  : $SLURM_ARRAY_TASK_ID"
echo "Baslangic     : $(date)"

# --- Bu array gorevinin isleyecegi kod mesafesini sec --------------------
# list_tasks.py kod mesafelerini satir satir verir; SLURM_ARRAY_TASK_ID
# numarali satiri aliyoruz (satir numarasi 1-tabanli, array 0-tabanli).
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
DISTANCE=$(python src/list_tasks.py --config "$CONFIG" --mode distances \
           | sed -n "${LINE_NO}p")
echo "Kod mesafesi  : d = $DISTANCE"

# --- Veriyi uret ---------------------------------------------------------
# --distance verilince generate_data.py o d icin TUM gurultu seviyelerini
# ve train/val/test bolunmelerini uretir.
python src/generate_data.py --config "$CONFIG" --distance "$DISTANCE"

echo "Bitis         : $(date)"
echo "Array gorevi $SLURM_ARRAY_TASK_ID (d=$DISTANCE) tamamlandi."
