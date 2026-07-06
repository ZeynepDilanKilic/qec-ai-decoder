#!/bin/bash
# =====================================================================
# setup_env.sh  --  TRUBA'da Python ortamini BIR KEZ hazirlar
# =====================================================================
# Bu betik bir SLURM isi DEGILDIR. Kullanici arayuzu (login) sunucusunda
# DOGRUDAN calistirilir:
#
#     bash slurm/setup_env.sh
#
# NE YAPAR?
#   1) TRUBA'nin merkezi Python/AI modulunu yukler (icinde PyTorch hazir),
#   2) Uzerine hafif bir sanal ortam (venv) kurar,
#   3) Sadece QEC'e ozel paketleri (stim, pymatching) pip ile ekler.
#
# NEDEN BU YAKLASIM?
#   TRUBA dokumantasyonu /truba ve /arf dosya sistemlerine dogrudan
#   conda/pip ile binlerce kucuk dosya kurmayi ONERMEZ (dosya sistemi
#   performansini dusurur). Bu yuzden:
#     - Buyuk paketler (torch, numpy, ...) MERKEZI modulden gelir,
#     - venv "--system-site-packages" ile o moduldeki paketleri devralir,
#     - sadece stim + pymatching gibi kucuk, ozel paketler eklenir.
#   Boylece diske eklenen dosya sayisi minimumda kalir.
#
# ALTERNATIF (daha izole, TRUBA'nin de onerdigi): Apptainer/Singularity
# konteyneri. README'nin "Ortam Kurulumu" bolumune bakin.
# =====================================================================
set -e  # herhangi bir komut hata verirse betigi durdur

echo "=================================================================="
echo " TRUBA QEC-AI-Decoder ortam kurulumu"
echo "=================================================================="

# --- 1) Merkezi modulleri yukle --------------------------------------
# ONEMLI: Asagidaki modul adi TRUBA'da zamanla degisebilir.
# Once  'module avail'  cikti icinde python / ai-tools / cuda arayin
# ve gerekirse bu satiri guncelleyin.
module purge
# TRUBA'nin yapay zeka araclari modulu (PyTorch, CUDA dahil):
module load comp/python/ai-tools-kolyoz-1.0 || {
    echo "UYARI: 'comp/python/ai-tools-kolyoz-1.0' bulunamadi."
    echo "       'module avail' ile mevcut python/ai modulunu bulup"
    echo "       bu betikteki 'module load' satirini guncelleyin."
    exit 1
}
echo "[1/3] Merkezi modul yuklendi."
python --version

# --- 2) Sanal ortam olustur ------------------------------------------
# venv'i HOME altinda tutuyoruz (proje koku scratch'te bile olsa, ortam
# kalici olsun). --system-site-packages: torch vb. modulden devralinir.
VENV_DIR="$HOME/qec-venv"
if [ -d "$VENV_DIR" ]; then
    echo "[2/3] venv zaten var: $VENV_DIR (yeniden kullanilacak)"
else
    python -m venv --system-site-packages "$VENV_DIR"
    echo "[2/3] venv olusturuldu: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- 3) QEC'e ozel paketleri kur -------------------------------------
# stim       : Clifford devre / stabilizer kod simulatoru (sendrom uretimi)
# pymatching : MWPM referans decoder
# pyyaml     : config.yaml okumak icin (modulde yoksa)
# torch modulden gelmis olmali; gelmediyse asagidaki satira torch ekleyin.
pip install --upgrade pip
pip install stim pymatching pyyaml
# torch modulde yoksa (kontrol edin), su satiri acin:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "[3/3] Ozel paketler kuruldu."
echo "=================================================================="
echo " Dogrulama:"
python - <<'PYEOF'
import stim, pymatching, numpy, torch
print(f"  stim       {stim.__version__}")
print(f"  pymatching {pymatching.__version__}")
print(f"  numpy      {numpy.__version__}")
print(f"  torch      {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
PYEOF
echo "=================================================================="
echo " Kurulum tamam. SLURM betikleri bu venv'i otomatik aktive eder."
echo " (Not: torch.cuda login sunucusunda False gorunebilir; bu normaldir,"
echo "  GPU sadece hesaplama dugumlerinde gorunur.)"
echo "=================================================================="
