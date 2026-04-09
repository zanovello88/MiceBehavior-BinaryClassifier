# Epilepsy Seizure Detection in Mice — CNN+LSTM Classifier

Tesi magistrale in Ingegneria Informatica (specializzazione Intelligenza Artificiale)  
Università degli Studi di Ferrara

## Obiettivo

Classificatore binario per il riconoscimento automatico di episodi di crisi
epilettiche in topi da laboratorio, tramite analisi video con un modello CNN+LSTM.
Il sistema analizza sequenze di frame, estrae feature spaziali (CNN) e modella
la dinamica temporale (LSTM), producendo per ogni sequenza una probabilità
di crisi e identificando automaticamente onset e offset dell'evento.

Il progetto include inoltre un'interfaccia grafica per l'uso in laboratorio,
che permette di caricare un video, selezionare il topo di interesse e ottenere
i timestamp di inizio e fine crisi esportati in CSV.

---

## Risultati

### Metriche sul test set (con smoothing temporale)

| Metrica | Baseline | Modello finale | + Smoothing |
|---|---|---|---|
| F1-score | 0.613 | 0.765 | **0.872** |
| Recall | 0.491 | 0.687 | **0.935** |
| Precision | 0.755 | 0.862 | 0.817 |
| ROC-AUC | 0.585 | 0.695 | 0.653 |
| Crisi non rilevate | 7/15 | 0/15 | **0/15** |
| Overlap medio | 0.455 | 0.669 | **0.831** |
| Detection delay mediano | 0.00s | 0.50s | 0.00s |

### Error analysis (test set)

| Categoria | N | % |
|---|---|---|
| Veri Positivi (TP) | 503 | 73.9% |
| Veri Negativi (TN) | 30 | 4.4% |
| Falsi Positivi (FP) | 113 | 16.6% |
| Falsi Negativi (FN) | 35 | 5.1% |

**Osservazione chiave:** il 100% dei falsi negativi è concentrato nella fase
di onset della crisi — il modello non manca mai la crisi una volta pienamente
sviluppata. I falsi positivi hanno probabilità media 0.93, quasi identica ai
veri positivi, suggerendo che certi movimenti pre-ictali condividono
caratteristiche visive con la fase ictal.

---

## Dataset

- 101 video (~90s ciascuno, 210×210px, 30fps)
- Ogni video contiene: 10s pre-ictal + fase ictal (45-80s) + 10s post-ictal
- Annotazioni in `mappa_labels.csv` (onset/offset frame per ogni video)
- Distribuzione classi dopo subsampling a 10fps: ~70% crisi, ~30% non-crisi
- Split train/val/test: 70/15/15 video (split per video, no data leakage)

---

## Architettura

Input [B, T, C, H, W]
→ MobileNetV3-Small (pre-trained ImageNet, freeze_layers=8)
→ Proiezione lineare (576 → 256)
→ LSTM (2 layer, hidden=256, dropout=0.3)
→ FC classifier (256 → 64 → 1)
→ Sigmoid → P(crisi) per sequenza
→ Smoothing temporale (mediana, window=10)
→ Threshold (0.874, Youden's J)

- **CNN**: MobileNetV3-Small — 2.1M parametri totali, scelto per il basso
  numero di parametri che riduce il rischio di overfitting con dataset piccoli
- **LSTM**: 2 layer con hidden size 256 — cattura dipendenze temporali
  su finestre di 6 secondi (60 frame a 10fps)
- **Loss**: BCEWithLogitsLoss con pos_weight=3.0 per bilanciamento classi
- **Post-processing**: smoothing con mediana mobile (window=10) per
  ridurre i falsi positivi e aumentare la copertura della crisi

---

## Struttura del progetto
├── data/
│   ├── mappa_labels.csv              # annotazioni (onset/offset per video)
│   ├── manifest.json                 # generato da preprocessing.py
│   └── frames/                       # frame estratti (non in git)
│
├── src/
│   ├── creazione_dataset.py          # ritaglio video sul topo
│   ├── preprocessing.py              # estrazione frame + manifest JSON
│   ├── transforms.py                 # augmentation e normalizzazione
│   ├── dataset.py                    # Dataset PyTorch + sliding window
│   ├── model.py                      # architettura CNN+LSTM
│   ├── train.py                      # training loop con early stopping
│   ├── evaluate.py                   # metriche + smoothing + plot
│   ├── inference.py                  # inferenza real-time da CLI
│   ├── analyze_video.py              # analisi video da CLI con ROI
│   └── gui.py                        # interfaccia grafica Tkinter
│
├── tools/
│   ├── inspect_manifest.py           # verifica distribuzione classi
│   ├── inspect_dataset.py            # verifica split e shape tensori
│   ├── plot_thesis.py                # grafici comparativi per la tesi
│   └── error_analysis.py             # analisi FP/FN per video e topo
│
├── jobs/
│   └── train_job.sh                  # script SLURM per il cluster
│
├── model_weights/
│   └── mobilenet_v3_small_imagenet.pth   # pesi pre-addestrati (non in git)
│
├── runs/                             # output training (non in git)
│   └── <run_id>/
│       ├── best_model.pt
│       ├── history.json
│       ├── train.log
│       ├── eval_results.json
│       └── eval_results_smoothed.json
│
├── error_analysis/                   # output error analysis (non in git)
│   ├── errors_per_video.png
│   ├── errors_per_mouse.png
│   ├── fn_position_in_crisis.png
│   ├── probability_distribution.png
│   └── error_report.txt
│
├── thesis_plots/                     # grafici per la tesi (non in git)
├── inference_results/                # output inferenza CLI (non in git)
├── requirements.txt
└── README.md

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

```bash
# trasferimento dati
rsync -avz data/frames/ utente@192.167.219.42:~/tesi/data/frames/
rsync -avz data/manifest.json utente@192.167.219.42:~/tesi/data/
rsync -avz model_weights/ utente@192.167.219.42:~/tesi/model_weights/

# submit job
sbatch jobs/train_job.sh

# monitoraggio
squeue -u $USER
tail -f epilepsy-train-<JOBID>.log
```

### 3. Valutazione (cluster)

```bash
python src/evaluate.py \
  --checkpoint runs/<run_id>/best_model.pt \
  --manifest   data/manifest.json
```

### 4. Error analysis (Mac locale)

```bash
python tools/error_analysis.py
```

### 5. Interfaccia grafica (Mac locale o PC laboratorio)

```bash
python src/gui.py
```

### 6. Analisi video da CLI (Mac locale o PC laboratorio)

```bash
python src/analyze_video.py \
  --video path/al/video.mp4 \
  --min_duration_sec 40.0 \
  --skip_seconds 35.0 \
  --confidence_window_sec 12.0 \
  --confidence_ratio 0.85 \
  --confirm_frames 12
```

### 7. Grafici per la tesi (Mac locale)

```bash
python tools/plot_thesis.py
```

---

## Parametri del modello finale

| Parametro | Valore | Motivazione |
|---|---|---|
| Frame rate analisi | 10fps | Subsampling da 30fps, riduce ridondanza |
| Seq length | 60 frame | 6 secondi, cattura dinamica convulsiva |
| Stride | 15 frame | Overlap 75%, massimizza sequenze |
| Batch size | 16 | Bilanciamento memoria/gradients |
| Learning rate | 3e-5 | Fine-tuning conservativo |
| pos_weight | 3.0 | Penalizza falsi negativi (recall critico) |
| freeze_layers | 8 | Sblocca layer CNN per adattamento dominio |
| Smoothing | median, w=10 | Riduce picchi isolati, aumenta overlap |
| Threshold | 0.874 | Ottimizzato con Youden's J sulla ROC |

## Parametri interfaccia grafica (default consigliati)

| Parametro | Valore | Motivazione |
|---|---|---|
| Min durata crisi | 40s | Scarta falsi positivi brevi |
| Skip inizio video | 35s | Evita falsi onset pre-ictali |
| Finestra confidenza | 12s | Richiede pattern sostenuto |
| Ratio confidenza | 0.85 | 85% della finestra sopra threshold |
| Confirm frames | 12 | ~2.4s consecutivi per confermare |
| Frame step | 6 | 5fps, bilanciamento velocità/accuratezza |
| Seq length | 20 | Ottimizzato per CPU (291ms/inferenza) |

---

## Cluster universitario

- **Indirizzo**: 192.167.219.42
- **Nodo GPU**: gnode01
- **GPU**: NVIDIA H100 NVL
- **Scheduler**: SLURM
- **Moduli**: `cuda/12.2`, `python/3.11.6-gcc-11.3.1-6nwylkz`
- **Tempo per run**: ~45 minuti (80 epoche max, early stopping ~30 epoche)

---

## Error Analysis — risultati principali

**Falsi negativi (FN=35, 5.1%):**
Tutti concentrati nella fase di onset della crisi (100%). La fase di onset
è caratterizzata da pattern motori ancora incerti e graduali — il segnale
è ambiguo anche per un osservatore umano. Il modello non manca mai la
crisi una volta pienamente sviluppata.

**Falsi positivi (FP=113, 16.6%):**
Probabilità media 0.93 — quasi identica ai veri positivi (0.94). Il modello
è molto sicuro quando genera falsi allarmi, il che suggerisce che certi
movimenti pre-ictali condividono caratteristiche visive reali con la fase
ictal. Non è un problema di soglia ma di similarità intrinseca dei pattern.

**Variabilità inter-individuale:**

| Topo | Error rate |
|---|---|
| A1 | 25.0% |
| 78 | 24.2% |
| P95 | 21.5% |
| A4 | 21.4% |
| 83 | 20.9% |
| 81 | 19.2% |
| A7 | 18.9% |

---

## Possibili miglioramenti futuri

- **Attention mechanisms**: aggiungere self-attention sull'output LSTM
  per pesare differentemente i timestep più informativi
- **Vision Transformer**: sostituire MobileNetV3 con ViT pre-addestrato
  su dataset medici (es. BioViL)
- **3D CNN**: convoluzione spazio-temporale per catturare pattern
  di movimento direttamente
- **Self-supervised pre-training**: pre-addestrare la CNN su tutti i
  frame prima del fine-tuning supervisionato
- **Leave-one-mouse-out validation**: split più rigoroso per misurare
  la generalizzazione inter-individuale
- **Dataset aumentato**: più video per topo, specialmente per A1 e 78
  che mostrano error rate più alto
- **Interfaccia medici (Fase 2)**: riquadro trascinabile sul primo frame
  e semaforo visivo in tempo reale durante l'analisi

---

## Autore

Francesco Zanovello  
Corso di Laurea Magistrale in Ingegneria Informatica  
Università degli Studi di Ferrara