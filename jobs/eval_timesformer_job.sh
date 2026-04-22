#!/bin/bash
#SBATCH --job-name=timesformer-eval
#SBATCH --output=timesformer-eval-%j.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GH100:1

echo "Start: $(date)"

module purge
module load cuda/12.2
module load python/3.11.6-gcc-11.3.1-6nwylkz

source ~/tesi/venv_tesi/bin/activate
cd ~/tesi

python src/evaluate_timesformer.py \
  --checkpoint  runs_timesformer/20260420_133708/best_model.pt \
  --manifest    data/manifest.json \
  --weights_dir model_weights/timesformer-hr \
  --batch_size  8 \
  --num_workers 4 \
  --seq_len     16 \
  --stride      8

echo "End: $(date)"