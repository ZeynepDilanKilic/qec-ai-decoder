#!/bin/bash
# =====================================================================
# 02_train.slurm  --  AI decoder'lari egitir (GPU isi, JOB ARRAY)
# =====================================================================
# Her (kod mesafesi d, gurultu seviyesi p) cifti icin AYRI bir uzman
# model egitilir. Bu ciftler birbirinden tamamen bagimsiz oldugundan
# SLURM "job array" ile hepsi ayni anda farkli GPU'larda kosabilir.
# Iste TRUBA gibi bir kumeyi kullanmanin asil kazanci budur: 27 modeli
# tek tek (seri) degil, kuyruk musait oldukca paralel egitirsiniz.
#
# ARRAY BOYUTU: (kod mesafesi sayisi) x (gurultu seviyesi sayisi).
# Tam sayiyi su komutla ogrenin:
#   python src/list_tasks.py --config configs/config.yaml --count
# Varsayilan config: 3 x 9 = 27 gorev  -> --array=0-26
#
# GONDERME:
#   sbatch slurm/02_train.slurm
#
# TRUBA GPU KUYRUGU NOTLARI:
#   - akya-cuda: sunucu basina 4x NVIDIA V100. Her GPU icin 10 cekirdek
#     talep edilmesi beklenir (--gres=gpu:1 -> --cpus-per-task=10).
#   - GPU kuyruklarina gonderilen islerin GERCEKTEN GPU kullanmasi
#     zorunludur (train.py CUDA'yi otomatik kullanir).
# =====================================================================

#SBATCH -p akya-cuda                   # GPU kuyrugu (V100)
#SBATCH -A KULLANICI_ADINIZ            # <-- TRUBA hesap adinizla DEGISTIRIN
#SBATCH -J qec-train                   # is adi
#SBATCH --array=0-26                   # <-- (d sayisi)x(p sayisi) - 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                   # gorev basina 1 GPU
#SBATCH --cpus-per-task=10             # akya-cuda: GPU basina 10 cekirdek
#SBATCH --time=04:00:00                # model basina ust sinir
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
# Istege bagli e-posta bildirimi (basina ## koyarak kapali):
##SBATCH --mail-user=eposta@adresiniz.edu.tr
##SBATCH --mail-type=END,FAIL

set -e

# --- Proje dizini --------------------------------------------------------
PROJECT_DIR="$HOME/qec-ai-decoder"
cd "$PROJECT_DIR"
CONFIG="configs/config.yaml"

# --- Ortam ---------------------------------------------------------------
module purge
module load comp/python/ai-tools-kolyoz-1.0
source "$HOME/qec-venv/bin/activate"

# --- Bilgilendirme -------------------------------------------------------
echo "Sunucu       : $(hostname)"
echo "Array gorevi  : $SLURM_ARRAY_TASK_ID"
echo "GPU bilgisi   :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Baslangic     : $(date)"

# --- Bu gorevin (d, p) ciftini sec --------------------------------------
# list_tasks.py her satira "d p" yazar; ilgili satiri okuyup ayikliyoruz.
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
TASK=$(python src/list_tasks.py --config "$CONFIG" | sed -n "${LINE_NO}p")
DISTANCE=$(echo "$TASK" | awk '{print $1}')
P=$(echo "$TASK" | awk '{print $2}')
echo "Egitilecek    : d = $DISTANCE , p = $P"

# --- Egitimi calistir ----------------------------------------------------
python src/train.py --config "$CONFIG" --distance "$DISTANCE" --p "$P"

echo "Bitis         : $(date)"
echo "Array gorevi $SLURM_ARRAY_TASK_ID (d=$DISTANCE, p=$P) tamamlandi."
