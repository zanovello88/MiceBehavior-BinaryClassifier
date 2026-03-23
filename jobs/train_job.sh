#!/bin/bash
#SBATCH --job-name=epilepsy_cnn_lstm
#SBATCH --output=runs/slurm_%j.log
#SBATCH --error=runs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GH100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=francesc.zanovello@edu.unife.it

#Ambiente 
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# carica moduli 
module load python/3.11.6-gcc-11.3.1-6nwylkz
module load cuda/12.2

# attiva ambiente virtuale 
source ~/tesi/venv/bin/activate

# cartella del progetto
cd ~/tesi

# crea cartella runs se non esiste
mkdir -p runs

#Training 
python src/train.py \
  --manifest     data/manifest.json \
  --output_dir   runs \
  --epochs       50 \
  --batch_size   16 \
  --lr           1e-4 \
  --weight_decay 1e-4 \
  --pos_weight   0.4265 \
  --patience     10 \
  --num_workers  4 \
  --seq_len      30 \
  --stride       15 \
  --freeze_layers 10

echo "End: $(date)"