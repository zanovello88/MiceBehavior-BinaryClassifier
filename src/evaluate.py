"""
Scopo: valutare il modello addestrato sul test set e produrre tutte le
       metriche necessarie — sia a livello di sequenza (seq-level)
       che a livello di evento (event-level).

Metriche calcolate:
  Seq-level  (una predizione per sequenza):
    - Accuracy, Precision, Recall, F1-score
    - ROC-AUC
    - Confusion matrix
    - Curva ROC + curva Precision-Recall

  Event-level (onset/offset dell'intera crisi):
    - Detection delay: quanti secondi dopo l'onset reale il modello
      predice la crisi per la prima volta
    - Overlap: percentuale di frame ictal correttamente identificati
      rispetto alla durata totale della crisi

Motivazioni:
  - Le metriche seq-level sono standard ML e permettono confronto con
    letteratura. F1 è più informativo di accuracy con classi sbilanciate.
  - Le metriche event-level sono quelle clinicamente rilevanti: un
    sistema di allerta precoce deve rilevare la crisi in pochi secondi
    dall'onset, non solo classificare sequenze isolate.
  - Il threshold ottimale viene cercato sulla curva ROC invece di
    usare 0.5 fisso — con classi sbilanciate 0.5 è quasi sempre subottimale.
  - Tutti i plot vengono salvati come PNG nella cartella del run. 
"""

import json
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from dataset    import build_dataloaders, build_sequences, split_sequences
from model      import CNNLSTM
from transforms import eval_transforms


# Argomenti

def parse_args():
    p = argparse.ArgumentParser(description='Valutazione CNN+LSTM epilessia')
    p.add_argument('--checkpoint',   type=str, required=True,
                   help='Path al file best_model.pt')
    p.add_argument('--manifest',     type=str,
                   default='data/manifest.json')
    p.add_argument('--output_dir',   type=str, default=None,
                   help='Cartella output plot (default: stessa del checkpoint)')
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--seq_len',      type=int, default=30)
    p.add_argument('--stride',       type=int, default=15)
    return p.parse_args()


# Metriche event-level 

def compute_event_metrics(sequences, probs, threshold, fps=10.0):
    """
    Calcola metriche a livello di evento per ogni video nel test set.

    Per ogni video:
      - ricostruisce la sequenza temporale di predizioni
      - trova il primo frame predetto come crisi (onset predetto)
      - calcola il detection delay rispetto all'onset reale
      - calcola l'overlap tra predizioni positive e ground truth ictal

    Restituisce un dict con le metriche aggregate su tutti i video.
    fps=10.0 perché i frame sono campionati a 10fps.
    """
    from collections import defaultdict

    # raggruppa sequenze e predizioni per video
    video_data = defaultdict(lambda: {
        'frame_indices': [], 'gt_labels': [], 'pred_probs': [],
        'onset': None, 'offset': None
    })

    for seq, prob in zip(sequences, probs):
        v = seq['video_name']
        center_idx = seq['start_idx'] + len(seq['labels']) // 2
        video_data[v]['frame_indices'].append(center_idx)
        video_data[v]['gt_labels'].append(seq['seq_label'])
        video_data[v]['pred_probs'].append(prob)
        video_data[v]['onset']  = seq['onset_frame'] // 3   # converti a 10fps
        video_data[v]['offset'] = seq['offset_frame'] // 3

    delays  = []
    overlaps = []

    for v, data in video_data.items():
        # ordina per indice temporale
        sorted_idx = np.argsort(data['frame_indices'])
        indices    = np.array(data['frame_indices'])[sorted_idx]
        gt         = np.array(data['gt_labels'])[sorted_idx]
        probs_v    = np.array(data['pred_probs'])[sorted_idx]
        preds      = (probs_v >= threshold).astype(int)

        onset_real = data['onset']

        # detection delay: frames tra onset reale e prima predizione positiva
        crisis_preds = np.where(preds == 1)[0]
        if len(crisis_preds) > 0:
            first_pred_frame = indices[crisis_preds[0]]
            delay_frames     = max(0, first_pred_frame - onset_real)
            delay_sec        = delay_frames / fps
        else:
            delay_sec = float('inf')   # crisi non rilevata
        delays.append(delay_sec)

        # overlap: intersezione predizioni positive con zona ictal reale
        gt_positive   = np.sum(gt == 1)
        true_positive = np.sum((preds == 1) & (gt == 1))
        overlap       = true_positive / gt_positive if gt_positive > 0 else 0.0
        overlaps.append(overlap)

    # filtra i casi con delay infinito per il calcolo della media
    finite_delays = [d for d in delays if d != float('inf')]
    missed        = sum(1 for d in delays if d == float('inf'))

    return {
        'mean_delay_sec'   : np.mean(finite_delays) if finite_delays else float('inf'),
        'median_delay_sec' : np.median(finite_delays) if finite_delays else float('inf'),
        'mean_overlap'     : np.mean(overlaps),
        'missed_seizures'  : missed,
        'total_videos'     : len(video_data),
        'per_video_delays' : delays,
        'per_video_overlaps': overlaps,
    }


# Plot 

def plot_roc_curve(fpr, tpr, auc, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Test Set')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


def plot_pr_curve(precision, recall, ap, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Test Set')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im)
    classes = ['Non crisi', 'Crisi']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Label reale')
    ax.set_xlabel('Label predetta')
    ax.set_title('Confusion Matrix — Test Set')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


def plot_prediction_timeline(sequences, probs, threshold,
                              video_name, output_path):
    """
    Plotta per un singolo video le predizioni nel tempo
    confrontate con il ground truth — il più utile sarà poi discusso nella tesi.
    """
    video_seqs = [(s, p) for s, p in zip(sequences, probs)
                  if s['video_name'] == video_name]
    if not video_seqs:
        return

    video_seqs.sort(key=lambda x: x[0]['start_idx'])
    indices = [s['start_idx'] + 15 for s, _ in video_seqs]
    gt      = [s['seq_label'] for s, _ in video_seqs]
    preds_p = [p for _, p in video_seqs]

    time_sec = [i / 10.0 for i in indices]   # converti a secondi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ground truth
    ax1.fill_between(time_sec, gt, alpha=0.6, color='steelblue',
                     label='Ground truth')
    ax1.set_ylabel('Label reale')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Non crisi', 'Crisi'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # predizioni
    ax2.plot(time_sec, preds_p, color='darkorange', lw=1.5,
             label='P(crisi) predetta')
    ax2.axhline(y=threshold, color='red', linestyle='--', lw=1,
                label=f'Threshold = {threshold:.2f}')
    ax2.fill_between(time_sec, preds_p, alpha=0.3, color='darkorange')
    ax2.set_ylabel('Probabilità crisi')
    ax2.set_xlabel('Tempo (secondi)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Predizioni nel tempo — {video_name}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_path}")


# Main 

@torch.no_grad()
def main():
    args       = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    saved_args = checkpoint.get('args', {})

    output_dir = Path(args.output_dir) if args.output_dir \
                 else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {output_dir}\n")

    # Carica modello 
    weights_path = saved_args.get('weights_path',
                   'model_weights/mobilenet_v3_small_imagenet.pth')
    freeze_layers = saved_args.get('freeze_layers', 14)

    model = CNNLSTM(
        freeze_layers = freeze_layers,
        weights_path  = weights_path,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Modello caricato correttamente\n")

    # DataLoader test 
    seq_len = saved_args.get('seq_len', args.seq_len)
    stride  = saved_args.get('stride',  args.stride)

    with open(args.manifest) as f:
        manifest = json.load(f)

    sequences            = build_sequences(manifest, seq_len, stride)
    _, _, test_sequences = split_sequences(sequences)

    from dataset import EpilepsyDataset
    from torch.utils.data import DataLoader

    test_ds     = EpilepsyDataset(test_sequences, transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Inferenza 
    all_probs  = []
    all_labels = []

    for frames, labels in test_loader:
        frames = frames.to(device)
        logits = model(frames).squeeze(1)
        probs  = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Threshold ottimale dalla curva ROC
    # Youden's J = sensitivity + specificity - 1, massimizzato
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    youden_idx      = np.argmax(tpr - fpr)
    best_threshold  = thresholds[youden_idx]
    print(f"Threshold ottimale (Youden's J): {best_threshold:.4f}")

    # Metriche seq-level 
    preds = (all_probs >= best_threshold).astype(int)
    auc   = roc_auc_score(all_labels, all_probs)
    ap    = average_precision_score(all_labels, all_probs)
    cm    = confusion_matrix(all_labels, preds)

    print("\n" + "=" * 50)
    print("METRICHE SEQ-LEVEL (test set)")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy_score(all_labels, preds):.4f}")
    print(f"  Precision : {precision_score(all_labels, preds):.4f}")
    print(f"  Recall    : {recall_score(all_labels, preds):.4f}")
    print(f"  F1-score  : {f1_score(all_labels, preds):.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Avg Prec  : {ap:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Metriche event-level
    event_metrics = compute_event_metrics(
        test_sequences, all_probs.tolist(), best_threshold
    )

    print("\n" + "=" * 50)
    print("METRICHE EVENT-LEVEL (test set)")
    print("=" * 50)
    print(f"  Video nel test set       : {event_metrics['total_videos']}")
    print(f"  Crisi non rilevate       : {event_metrics['missed_seizures']}")
    print(f"  Detection delay medio    : {event_metrics['mean_delay_sec']:.2f}s")
    print(f"  Detection delay mediano  : {event_metrics['median_delay_sec']:.2f}s")
    print(f"  Overlap medio            : {event_metrics['mean_overlap']:.4f}")

    # Salva metriche in JSON
    results = {
        'threshold'       : float(best_threshold),
        'accuracy'        : float(accuracy_score(all_labels, preds)),
        'precision'       : float(precision_score(all_labels, preds)),
        'recall'          : float(recall_score(all_labels, preds)),
        'f1'              : float(f1_score(all_labels, preds)),
        'roc_auc'         : float(auc),
        'avg_precision'   : float(ap),
        'confusion_matrix': cm.tolist(),
        'event_metrics'   : event_metrics,
    }
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRisultati salvati in: {output_dir / 'eval_results.json'}")

    # Genera plot 
    print("\nGenerazione plot...")
    precision_curve, recall_curve, _ = precision_recall_curve(
        all_labels, all_probs
    )

    plot_roc_curve(fpr, tpr, auc,
                   output_dir / 'roc_curve.png')
    plot_pr_curve(precision_curve, recall_curve, ap,
                  output_dir / 'pr_curve.png')
    plot_confusion_matrix(cm,
                          output_dir / 'confusion_matrix.png')

    # timeline per i primi 3 video del test set
    test_videos = list({s['video_name'] for s in test_sequences})[:3]
    for vid in test_videos:
        safe_name = vid.replace('.mp4', '').replace('/', '_')
        plot_prediction_timeline(
            test_sequences, all_probs.tolist(), best_threshold,
            vid, output_dir / f'timeline_{safe_name}.png'
        )

    print("\nValutazione completata.")


if __name__ == '__main__':
    main()