#!/bin/bash
#SBATCH --job-name=epilepsy-train
#SBATCH --output=epilepsy-train-%j.log
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
mkdir -p runs

python src/train.py \
  --manifest      data/manifest.json \
  --output_dir    runs \
  --epochs        80 \
  --batch_size    16 \
  --lr            1e-5 \
  --weight_decay  1e-3 \
  --pos_weight    0.4265 \
  --patience      15 \
  --num_workers   4 \
  --seq_len       30 \
  --stride        15 \
  --freeze_layers 14 \
  --weights_path  model_weights/mobilenet_v3_small_imagenet.pth

echo "End: $(date)"