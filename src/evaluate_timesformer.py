"""
Scopo: valutare il modello TimeSformer sul test set e produrre
       le stesse metriche di evaluate.py per il confronto diretto
       con CNN+LSTM. Da lanciare come job SLURM sulla GPU.
       
VERSIONE 2: Aggiunge salvataggio dei risultati per-sequence
            per permettere error analysis dettagliata.
"""

import json
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score
)

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from dataset               import build_sequences, split_sequences, EpilepsyDataset
from model_timesformer     import TimeSformerBinary
from transforms_timesformer import eval_transforms_timesformer
from torch.utils.data      import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    type=str, required=True)
    p.add_argument('--manifest',      type=str, default='data/manifest.json')
    p.add_argument('--weights_dir',   type=str,
                   default='model_weights/timesformer-hr')
    p.add_argument('--batch_size',    type=int, default=4)
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--seq_len',       type=int, default=16)
    p.add_argument('--stride',        type=int, default=8)
    p.add_argument('--freeze_layers', type=int, default=11)
    p.add_argument('--fc_dropout',    type=float, default=0.7)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Carica modello 
    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = TimeSformerBinary(
        weights_dir   = args.weights_dir,
        freeze_layers = args.freeze_layers,
        fc_dropout    = args.fc_dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")

    # Dataset
    with open(args.manifest) as f:
        manifest = json.load(f)

    seqs             = build_sequences(manifest, args.seq_len, args.stride)
    _, _, test_seqs  = split_sequences(seqs)
    print(f"Sequenze test: {len(test_seqs)}")

    ds     = EpilepsyDataset(test_seqs, transform=eval_transforms_timesformer)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Inferenza con raccolta dati per-sequence
    all_probs, all_labels = [], []
    seq_results = []  # NUOVO: salva dati per ogni sequenza
    
    seq_idx = 0
    with torch.no_grad():
        for i, (frames, labels) in enumerate(loader):
            frames = frames.to(device)
            logits = model(frames).squeeze(1)
            probs  = torch.sigmoid(logits)
            
            batch_probs  = probs.cpu().tolist()
            batch_labels = labels.tolist()
            
            all_probs.extend(batch_probs)
            all_labels.extend(batch_labels)
            
            # Salva info per ogni sequenza nel batch
            for prob, label in zip(batch_probs, batch_labels):
                seq = test_seqs[seq_idx]
                
                # Gestisci sia 'mouse_name' che 'topo' (legacy)
                mouse_name = seq.get('mouse_name') or seq.get('topo', 'unknown')
                
                seq_results.append({
                    'video_name'   : seq['video_name'],
                    'mouse_name'   : mouse_name,
                    'start_idx'    : seq['start_idx'],
                    'onset_frame'  : seq['onset_frame'],
                    'offset_frame' : seq['offset_frame'],
                    'ground_truth' : label,
                    'probability'  : float(prob),
                })
                seq_idx += 1
            
            if i % 10 == 0:
                print(f"  Batch {i}/{len(loader)}")

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    # Metriche
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_threshold       = thresholds[np.argmax(tpr - fpr)]
    preds                = (probs >= best_threshold).astype(int)
    cm                   = confusion_matrix(labels, preds)

    # Aggiungi prediction a ogni sequenza
    for i, pred in enumerate(preds):
        seq_results[i]['prediction'] = int(pred)

    print("\n" + "=" * 50)
    print("METRICHE TEST SET — TimeSformer-HR")
    print("=" * 50)
    print(f"  Threshold  : {best_threshold:.4f}")
    print(f"  Accuracy   : {(preds == labels).mean():.4f}")
    print(f"  Precision  : {precision_score(labels, preds):.4f}")
    print(f"  Recall     : {recall_score(labels, preds):.4f}")
    print(f"  F1-score   : {f1_score(labels, preds):.4f}")
    print(f"  ROC-AUC    : {roc_auc_score(labels, probs):.4f}")
    print(f"  Avg Prec   : {average_precision_score(labels, probs):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Salva risultati con dati per-sequence
    output_dir = Path(args.checkpoint).parent
    results = {
        'model'       : 'TimeSformer-HR',
        'checkpoint'  : args.checkpoint,
        'threshold'   : float(best_threshold),
        'accuracy'    : float((preds == labels).mean()),
        'precision'   : float(precision_score(labels, preds)),
        'recall'      : float(recall_score(labels, preds)),
        'f1'          : float(f1_score(labels, preds)),
        'roc_auc'     : float(roc_auc_score(labels, probs)),
        'avg_precision': float(average_precision_score(labels, probs)),
        'confusion_matrix': cm.tolist(),
        'per_sequence_results': seq_results,  # NUOVO
    }
    out_path = output_dir / 'eval_results_timesformer.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRisultati salvati in: {out_path}")


if __name__ == '__main__':
    main()