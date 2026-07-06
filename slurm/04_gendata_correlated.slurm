#!/bin/bash
#SBATCH -A egitimg23
#SBATCH -p debug
#SBATCH -J qec-gen-corr
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_corr_%A_%a.out
#SBATCH --error=logs/gen_corr_%A_%a.err
set -e
module purge
module load comp/python/miniconda3
source $HOME/qec-venv-gpu/bin/activate
cd /arf/scratch/$USER/qec-ai-decoder

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
DISTANCE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/config_correlated.yaml'))
print(cfg['data']['distances'][$SLURM_ARRAY_TASK_ID])
")
echo "Uretiliyor: d=$DISTANCE (tum noise_pairs)"
python3 src/generate_data_v3.py --config configs/config_correlated.yaml --distance $DISTANCE
