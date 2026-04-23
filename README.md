# Epilepsy Seizure Detection in Mice — Deep Learning Comparison

Tesi magistrale in Ingegneria Informatica (specializzazione Intelligenza Artificiale)  
Università degli Studi di Ferrara

## Obiettivo

Confronto tra due architetture deep learning per il riconoscimento automatico di 
crisi epilettiche in topi da laboratorio tramite analisi video frame-by-frame:

1. **CNN+LSTM** (baseline): MobileNetV3 + LSTM bidirezionale
2. **TimeSformer-HR**: Vision Transformer pre-addestrato con Linear Probing

Il sistema analizza sequenze di frame, identifica automaticamente onset e offset 
degli eventi epilettici, e produce timestamp esportabili in CSV. Include 
un'interfaccia grafica per l'uso in laboratorio.

---

## Risultati

### Confronto modelli (test set)

| Metrica | CNN+LSTM | TimeSformer-HR | Δ |
|---|---|---|---|
| **F1-score** | **0.872** | 0.703 | -16.9% |
| **Recall** | **0.935** | 0.605 | -33.0% |
| **Precision** | 0.817 | **0.839** | +2.2% |
| **ROC-AUC** | 0.653 | **0.724** | +7.1% |
| Crisi non rilevate | 0/15 | — | — |
| Overlap medio | 0.831 | — | — |
| Detection delay mediano | 0.00s | — | — |

**Conclusione**: CNN+LSTM è il modello migliore per questo task, con recall 
significativamente superiore. TimeSformer ha ROC-AUC migliore ma è troppo 
conservativo (perde il 40% delle crisi).

### CNN+LSTM — Metriche dettagliate

| Categoria | N | % |
|---|---|---|
| Veri Positivi (TP) | 503 | 73.9% |
| Veri Negativi (TN) | 30 | 4.4% |
| Falsi Positivi (FP) | 113 | 16.6% |
| Falsi Negativi (FN) | 35 | 5.1% |

**Osservazione chiave**: 100% dei FN concentrati all'onset. Probabilità media 
FP=0.93 (≈ TP), suggerendo pattern pre-ictali visivamente simili alla crisi.

### TimeSformer-HR — Metriche dettagliate

| Categoria | N | % |
|---|---|---|
| Veri Positivi (TP) | 499 | 42.6% |
| Veri Negativi (TN) | 249 | 21.3% |
| Falsi Positivi (FP) | 96 | 8.2% |
| Falsi Negativi (FN) | 326 | 27.9% |

**Distribuzione temporale FN**: Onset 39%, Middle 24%, Offset 37% — errori 
distribuiti su tutta la crisi, non solo all'onset come CNN+LSTM.

---

## Dataset

- 101 video (~90s ciascuno, 210×210px, 30fps)
- Ogni video: 10s pre-ictal + fase ictal (45-80s) + 10s post-ictal
- Annotazioni in `mappa_labels.csv` (onset/offset frame per video)
- Distribuzione classi dopo subsampling a 10fps: ~70% crisi, ~30% non-crisi
- Split train/val/test: 70/15/15 video (split per video, no data leakage)

---

## Architettura

### CNN+LSTM (baseline)

Input [B, T, C, H, W]
→ MobileNetV3-Small (pre-trained ImageNet, freeze_layers=8)
→ Proiezione lineare (576 → 256)
→ LSTM (2 layer, hidden=256, dropout=0.3)
→ FC classifier (256 → 64 → 1)
→ Sigmoid → P(crisi)
→ Smoothing temporale (mediana, window=10)
→ Threshold (0.874, Youden's J)

- **Parametri**: 2.1M totali, 1.98M trainabili
- **Sequenze**: 60 frame (6s a 10fps), stride=15
- **Risoluzione**: 210×210 nativa (no resize)

### TimeSformer-HR

Input [B, T, C, H, W]
→ TimeSformer-HR backbone (pre-trained Kinetics-400, freeze_layers=12)
→ LayerNorm
→ Dropout (0.3)
→ Linear (768 → 256)
→ GELU
→ Dropout (0.42)
→ Linear (256 → 1)
→ Sigmoid → P(crisi)

- **Parametri**: 121.9M totali, 0.80M trainabili (Linear Probing)
- **Sequenze**: 16 frame (1.6s a 10fps), stride=8
- **Risoluzione**: 448×448 (resize da 210×210)

---

## Struttura del Progetto

```text
.
├── data/
│   ├── mappa_labels.csv              # Annotazioni (onset/offset per video)
│   ├── manifest.json                 # Generato da preprocessing.py
│   └── frames/                       # Frame estratti (escluso da git)
│
├── src/
│   ├── creazione_dataset.py          # Ritaglio video focalizzato sul topo
│   ├── preprocessing.py              # Estrazione frame + generazione manifest
│   ├── transforms.py                 # Augmentation per CNN+LSTM
│   ├── transforms_timesformer.py     # Augmentation per TimeSformer (resize 448)
│   ├── dataset.py                    # Dataset PyTorch con sliding window
│   ├── model.py                      # CNN+LSTM (MobileNetV3+LSTM)
│   ├── model_timesformer.py          # TimeSformer-HR binario
│   ├── train.py                      # Training CNN+LSTM
│   ├── train_timesformer.py          # Training TimeSformer
│   ├── evaluate.py                   # Valutazione CNN+LSTM + smoothing
│   ├── evaluate_timesformer.py       # Valutazione TimeSformer
│   ├── inference.py                  # Inferenza CLI real-time
│   ├── analyze_video.py              # Analisi video con ROI selection
│   └── gui.py                        # Interfaccia Tkinter per laboratorio
│
├── tools/
│   ├── inspect_manifest.py           # Verifica bilanciamento dataset
│   ├── inspect_dataset.py            # Debug split e shape tensori
│   ├── plot_thesis.py                # Grafici per tesi
│   ├── error_analysis.py             # Analisi FP/FN CNN+LSTM
│   ├── error_analysis_timesformer.py # Analisi FP/FN TimeSformer
│   └── compare_models.py             # Confronto CNN+LSTM vs TimeSformer
│
├── jobs/
│   ├── train_job.sh                  # SLURM CNN+LSTM
│   ├── train_timesformer_job.sh      # SLURM TimeSformer
│   └── eval_timesformer_job.sh       # SLURM valutazione TimeSformer
│
├── model_weights/
│   ├── mobilenet_v3_small_imagenet.pth   # Pesi CNN (escluso da git)
│   └── timesformer-hr/                   # Pesi Transformer (escluso da git)
│
├── runs/                             # Output CNN+LSTM (escluso da git)
│   └── 20260327_101301/              # BEST RUN
│       ├── best_model.pt
│       ├── history.json
│       ├── eval_results.json
│       └── eval_results_smoothed.json
│
├── runs_timesformer/                 # Output TimeSformer (escluso da git)
│   └── 20260421_171823/              # Run 3 (Linear Probing)
│       ├── best_model.pt
│       └── eval_results_timesformer.json
│
├── error_analysis/                   # Analisi errori (escluso da git)
│   ├── errors_per_video.png
│   ├── errors_per_mouse.png
│   ├── fn_position_in_crisis.png
│   ├── probability_distribution.png
│   ├── error_report.txt
│   ├── timesformer_errors_per_video.png
│   ├── timesformer_errors_per_mouse.png
│   ├── timesformer_fn_position_in_crisis.png
│   ├── timesformer_probability_distribution.png
│   └── timesformer_error_report.txt
│
├── thesis_plots/                     # Grafici comparativi tesi (escluso da git)
│   ├── models_comparison_metrics.png
│   ├── models_comparison_confusion.png
│   ├── models_comparison_roc.png
│   ├── models_comparison_fn_distribution.png
│   ├── models_comparison_probabilities.png
│   └── models_comparison_table.txt
│
├── build_config/                     # PyInstaller configs
│   ├── epilepsy_detector.spec        # macOS
│   └── epilepsy_detector_windows.spec # Windows
│
├── requirements.txt
└── README.md
```

--- 

## Installazione

```bash
git clone <url-repo>
cd Tesi
pip install -r requirements.txt
```

### Cluster universitario (SLURM + CUDA 12.2)

```bash
module purge
module load cuda/12.2
module load python/3.11.6-gcc-11.3.1-6nwylkz
python -m venv venv_tesi
source venv_tesi/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Pipeline di esecuzione

### 1. Preprocessing (Mac locale)

```bash
python src/preprocessing.py
python tools/inspect_manifest.py
python tools/inspect_dataset.py
```

### 2. Training (cluster SLURM)

#### CNN+LSTM

```bash
sbatch jobs/train_job.sh
```

#### TimeSformer-HR

```bash
sbatch jobs/train_timesformer_job.sh
```

Monitoraggio:

```bash
squeue -u $USER
tail -f epilepsy-train-<JOBID>.log
```

### 3. Valutazione (cluster)

#### CNN+LSTM

```bash
python src/evaluate.py \
  --checkpoint runs/<run_id>/best_model.pt \
  --manifest   data/manifest.json
```

#### TimeSformer-HR

```bash
sbatch jobs/eval_timesformer_job.sh
```

### 4. Error analysis (Mac locale)

```bash
# CNN+LSTM
python tools/error_analysis.py

# TimeSformer
python tools/error_analysis_timesformer.py

# Confronto modelli
python tools/compare_models.py
```

### 5. Interfaccia grafica (Mac/PC laboratorio)

```bash
python src/gui.py
```

### 6. Analisi video da CLI

```bash
python src/analyze_video.py \
  --video path/al/video.mp4 \
  --min_duration_sec 40.0 \
  --skip_seconds 35.0 \
  --confidence_window_sec 12.0 \
  --confidence_ratio 0.85 \
  --confirm_frames 12
```

---

## Parametri dei modelli finali

### CNN+LSTM

| Parametro | Valore | Motivazione |
|---|---|---|
| Frame rate analisi | 10fps | Subsampling da 30fps, riduce ridondanza |
| Seq length | 60 frame | 6 secondi, cattura dinamica convulsiva |
| Stride | 15 frame | Overlap 75%, massimizza sequenze |
| Batch size | 16 | Bilanciamento memoria/gradients |
| Learning rate | 3e-5 | Fine-tuning conservativo |
| pos_weight | 3.0 | Penalizza falsi negativi (recall critico) |
| freeze_layers | 8 | Sblocca layer CNN per adattamento dominio |
| fc_dropout | 0.6 | Regolarizzazione classificatore |
| Smoothing | median, w=10 | Riduce picchi isolati, aumenta overlap |
| Threshold | 0.874 | Ottimizzato con Youden's J sulla ROC |

### TimeSformer-HR (Run 3 — Linear Probing)

| Parametro | Valore | Motivazione |
|---|---|---|
| Seq length | 16 frame | Nativo TimeSformer |
| Stride | 8 frame | Overlap 50% |
| Batch size | 4 | Limite memoria H100 |
| Learning rate | 5e-5 | Fine-tuning conservativo |
| pos_weight | 5.0 | Spinge recall (modello troppo conservativo) |
| freeze_layers | 12 | Linear Probing puro (~800K param trainabili) |
| fc_dropout | 0.3 | Dropout ridotto (pochi parametri) |
| weight_decay | 1e-3 | Regolarizzazione moderata |
| Risoluzione | 448×448 | Resize da 210×210 (nativo TimeSformer) |

---

## Cluster universitario

- **Indirizzo**: fzanovello@192.167.219.42
- **GPU**: NVIDIA H100 NVL
- **Scheduler**: SLURM
- **Moduli**: `cuda/12.2`, `python/3.11.6-gcc-11.3.1-6nwylkz`
- **Tempo per run CNN+LSTM**: ~45 minuti (early stopping ~30 epoche)
- **Tempo per run TimeSformer**: ~2-3 ore (early stopping ~15-20 epoche)

---

## Error Analysis — risultati chiave

### CNN+LSTM

**Falsi negativi (FN=35, 5.1%)**:  
100% concentrati all'onset. Il modello non manca mai la crisi una volta sviluppata.

**Falsi positivi (FP=113, 16.6%)**:  
Probabilità media 0.93 (≈ TP=0.94). Pattern pre-ictali simili alla fase ictal.

**Variabilità inter-individuale**:

| Topo | Error rate |
|---|---|
| A1 | 25.0% |
| 78 | 24.2% |
| P95 | 21.5% |
| A4 | 21.4% |

### TimeSformer-HR

**Falsi negativi (FN=326, 27.9%)**:  
Distribuzione: Onset 39%, Middle 24%, Offset 37% — errori su tutta la crisi.

**Falsi positivi (FP=96, 8.2%)**:  
Modello molto conservativo (precision alta, recall basso).

**Gap probabilità TP-FN**: 0.048 (vs 0.036 CNN+LSTM) — modello poco calibrato.

---

## Confronto architetture — osservazioni

### Vantaggi CNN+LSTM

- **Recall superiore** (0.935 vs 0.605): cruciale in ambito clinico
- **Sequenze lunghe** (6s vs 1.6s): cattura meglio dinamiche temporali
- **Risoluzione nativa**: nessun artefatto da resize
- **Smoothing**: post-processing efficace per ridurre FP
- **Lightweight**: 1.98M parametri trainabili, adatto a dataset piccoli

### Vantaggi TimeSformer-HR

- **ROC-AUC superiore** (0.724 vs 0.653): migliore separabilità classi
- **Precision alta** (0.839): quando predice crisi, è affidabile
- **Pre-training robusto**: Kinetics-400 fornisce feature spazio-temporali generali
- **Linear Probing efficace**: riduce overfitting con freeze_layers=12

### Limiti TimeSformer-HR

- **Dataset troppo piccolo**: 101 video insufficienti per 121M parametri
- **Sequenze troppo corte**: 16 frame (1.6s) perdono contesto temporale
- **Resize artifacts**: 210×210 → 448×448 introduce distorsioni
- **Domain mismatch**: pre-training su azioni umane, non comportamento murino
- **Recall insufficiente**: perde 40% delle crisi (inaccettabile clinicamente)

---

## Distribuzione applicazione GUI

### macOS

```bash
pyinstaller build_config/epilepsy_detector.spec
# Output: dist/EpilepsyDetector.app (1.4GB)
```

### Windows

```bash
pyinstaller build_config/epilepsy_detector_windows.spec
# Output: dist/EpilepsyDetector/EpilepsyDetector.exe
```

Entrambe le build includono il modello CNN+LSTM e tutte le dipendenze.

---

## Possibili miglioramenti futuri

### Per CNN+LSTM

- **Attention mechanisms**: pesare differentemente i timestep LSTM
- **Self-supervised pre-training**: pre-addestrare CNN su tutti i frame
- **Leave-one-mouse-out validation**: split più rigoroso
- **Dataset aumentato**: più video per topi con error rate alto

### Per TimeSformer

- **Dataset più grande**: almeno 500-1000 video per evitare overfitting
- **Fine-tuning progressivo**: sbloccare layer gradualmente (freeze_layers 12→10→8)
- **Sequenze più lunghe**: 32-64 frame per catturare contesto temporale
- **Pre-training domain-specific**: fine-tune su dataset medici (es. animal behavior)
- **Hybrid architecture**: CNN per feature extraction + Transformer per temporal modeling

### Generali

- **3D CNN**: convoluzione spazio-temporale diretta (es. I3D, SlowFast)
- **Multi-task learning**: predire anche intensità/tipo crisi
- **Ensemble**: combinare CNN+LSTM e TimeSformer con voting

---

## Pubblicazioni e materiale correlato

Questa tesi si basa su e contribuisce a:

- **Dataset**: 101 video annotati di crisi epilettiche in topi (laboratorio di neuroscienze, UniPD)
- **GUI distribuita**: Applicazione standalone per uso clinico (PyInstaller, cross-platform)
- **Codice open-source**: Repository GitHub con pipeline completa preprocessing → training → deployment

---

## Autore

Francesco Zanovello  
Corso di Laurea Magistrale in Ingegneria Informatica (Intelligenza Artificiale)  
Università degli Studi di Ferrara  
Anno Accademico 2025/2026
