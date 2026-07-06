#!/bin/bash
#SBATCH -A egitimg23
#SBATCH -p akya-cuda
#SBATCH -J qec-tr-corr
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:30:00
#SBATCH --output=logs/train_corr_%A_%a.out
#SBATCH --error=logs/train_corr_%A_%a.err
set -e
module purge
module load comp/python/miniconda3
source $HOME/qec-venv-gpu/bin/activate
cd /arf/scratch/$USER/qec-ai-decoder

# Array task ID -> (d, p_iid, p_corr) cifti
TASK_ID=$SLURM_ARRAY_TASK_ID

# 8 gorev: 2 mesafe x 4 noise_pair
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/config_correlated.yaml'))
ds = cfg['data']['distances']
pairs = cfg['data']['noise_pairs']
tasks = [(d, pi, pc) for d in ds for pi, pc in pairs]
d, pi, pc = tasks[$TASK_ID]
print(f'{d} {pi} {pc}')
" > /tmp/task_$$.txt
read DISTANCE PIID PCORR < /tmp/task_$$.txt
rm /tmp/task_$$.txt

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Egitilecek: d=$DISTANCE, p_iid=$PIID, p_corr=$PCORR"

python3 src/train_v3.py --config configs/config_correlated.yaml \
                         --distance $DISTANCE --p_iid $PIID --p_corr $PCORR
