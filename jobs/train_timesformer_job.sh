#!/bin/bash
#SBATCH --job-name=timesformer-train
#SBATCH --output=timesformer-train-%j.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GH100:1

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load cuda/12.2
module load python/3.11.6-gcc-11.3.1-6nwylkz

source ~/tesi/venv_tesi/bin/activate
cd ~/tesi
mkdir -p runs_timesformer

python src/train_timesformer.py \
  --manifest      data/manifest.json \
  --output_dir    runs_timesformer \
  --weights_dir   model_weights/timesformer-hr \
  --epochs        50 \
  --batch_size    4 \
  --lr            5e-5 \
  --weight_decay  1e-3 \
  --pos_weight    5.0 \
  --patience      10 \
  --num_workers   4 \
  --seq_len       16 \
  --stride        8 \
  --freeze_layers 12 \
  --warmup_epochs 3 \
  --fc_dropout 0.3 \

echo "End: $(date)"