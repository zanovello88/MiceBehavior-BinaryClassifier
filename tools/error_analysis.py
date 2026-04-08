"""
error_analysis.py
=================
Scopo: analizzare in dettaglio gli errori del modello sul test set per
       capire DOVE e PERCHÉ sbaglia. Produce tabelle e grafici pronti
       per la sezione "Analisi degli errori" della tesi.

Analisi prodotte:
  1. Errori per video — quali video hanno più FP e FN
  2. Errori per topo — se certi individui vengono classificati peggio
  3. Posizione temporale degli errori — gli FN sono concentrati
     all'onset, nel mezzo o all'offset della crisi?
  4. Distribuzione delle probabilità — come si distribuiscono le
     probabilità per TP, TN, FP, FN
  5. Video più difficili — top 5 per numero di errori con spiegazione

Output:
  error_analysis/
  ├── errors_per_video.png
  ├── errors_per_mouse.png
  ├── fn_position_in_crisis.png
  ├── probability_distribution.png
  ├── error_summary.csv
  └── error_report.txt
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from dataset    import build_sequences, split_sequences
from evaluate   import apply_temporal_smoothing
from model      import CNNLSTM
from transforms import eval_transforms

import torch
from torch.utils.data import DataLoader
from dataset import EpilepsyDataset

# ── Configurazione ─────────────────────────────────────────────────────────────
MANIFEST_PATH  = Path('data/manifest.json')
CHECKPOINT     = Path('runs/20260327_101301/best_model.pt')
OUTPUT_DIR     = Path('error_analysis')
THRESHOLD      = 0.874
SMOOTH_WINDOW  = 10
SEQ_LEN        = 60
STRIDE         = 15

OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ── Carica modello e inferenza ─────────────────────────────────────────────────

def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = CNNLSTM(
        freeze_layers = checkpoint['args'].get('freeze_layers', 8),
        weights_path  = 'model_weights/mobilenet_v3_small_imagenet.pth',
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def get_predictions(model, test_sequences):
    """
    Esegue l'inferenza sul test set e applica lo smoothing.
    Restituisce array di probabilità smoothed e label reali.
    """
    ds     = EpilepsyDataset(test_sequences, transform=eval_transforms)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for frames, _ in loader:
            logits = model(frames).squeeze(1)
            probs  = torch.sigmoid(logits)
            all_probs.extend(probs.tolist())

    # applica smoothing
    smoothed   = apply_temporal_smoothing(
        test_sequences, all_probs, window_size=SMOOTH_WINDOW, method='median'
    )
    all_labels = [s['seq_label'] for s in test_sequences]

    return np.array(smoothed), np.array(all_labels), np.array(all_probs)


# ── Classificazione errori ─────────────────────────────────────────────────────

def classify_errors(sequences, probs, labels, threshold):
    """
    Classifica ogni sequenza in TP, TN, FP, FN.
    Restituisce un dict con le quattro liste di sequenze + probabilità.
    """
    preds = (probs >= threshold).astype(int)

    tp = [(s, p) for s, p, pred, gt in
          zip(sequences, probs, preds, labels) if pred==1 and gt==1]
    tn = [(s, p) for s, p, pred, gt in
          zip(sequences, probs, preds, labels) if pred==0 and gt==0]
    fp = [(s, p) for s, p, pred, gt in
          zip(sequences, probs, preds, labels) if pred==1 and gt==0]
    fn = [(s, p) for s, p, pred, gt in
          zip(sequences, probs, preds, labels) if pred==0 and gt==1]

    print(f"\nClassificazione sequenze test set:")
    print(f"  TP (crisi corrette)    : {len(tp):>4}")
    print(f"  TN (normale corrette)  : {len(tn):>4}")
    print(f"  FP (falsi positivi)    : {len(fp):>4}")
    print(f"  FN (falsi negativi)    : {len(fn):>4}")
    print(f"  Totale                 : {len(tp)+len(tn)+len(fp)+len(fn):>4}")

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


# ── Analisi 1: errori per video ────────────────────────────────────────────────

def analyze_errors_per_video(errors):
    """
    Conta FP e FN per ogni video nel test set.
    Identifica i video più difficili per il modello.
    """
    fp_per_video = defaultdict(int)
    fn_per_video = defaultdict(int)
    tp_per_video = defaultdict(int)
    tn_per_video = defaultdict(int)

    for seq, _ in errors['fp']:
        fp_per_video[seq['video_name']] += 1
    for seq, _ in errors['fn']:
        fn_per_video[seq['video_name']] += 1
    for seq, _ in errors['tp']:
        tp_per_video[seq['video_name']] += 1
    for seq, _ in errors['tn']:
        tn_per_video[seq['video_name']] += 1

    all_videos = sorted(set(
        list(fp_per_video.keys()) + list(fn_per_video.keys()) +
        list(tp_per_video.keys()) + list(tn_per_video.keys())
    ))

    # calcola F1 per video
    video_stats = []
    for v in all_videos:
        tp = tp_per_video[v]
        fp = fp_per_video[v]
        fn = fn_per_video[v]
        tn = tn_per_video[v]
        f1 = (2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0
        total_errors = fp + fn
        video_stats.append({
            'video': v,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': f1,
            'total_errors': total_errors,
        })

    # ordina per errori totali decrescenti
    video_stats.sort(key=lambda x: x['total_errors'], reverse=True)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    videos_short = [v.replace('.mp4', '').replace('riga', 'r') 
                    for v in [s['video'] for s in video_stats]]
    fp_vals = [s['fp'] for s in video_stats]
    fn_vals = [s['fn'] for s in video_stats]
    f1_vals = [s['f1'] for s in video_stats]

    x = np.arange(len(video_stats))

    # FP e FN per video
    axes[0].bar(x - 0.2, fp_vals, 0.4, label='Falsi Positivi',
                color='#E8944A', alpha=0.85)
    axes[0].bar(x + 0.2, fn_vals, 0.4, label='Falsi Negativi',
                color='#5B8DB8', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(videos_short, rotation=45, ha='right', fontsize=8)
    axes[0].set_title('Falsi Positivi e Negativi per video')
    axes[0].set_ylabel('Numero sequenze')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # F1 per video
    colors = ['#5BAB6F' if f >= 0.8 else '#E8944A' if f >= 0.6
              else '#E85555' for f in f1_vals]
    axes[1].bar(x, f1_vals, color=colors, alpha=0.85)
    axes[1].axhline(y=0.8, color='gray', linestyle='--',
                    lw=1, label='F1=0.80')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(videos_short, rotation=45, ha='right', fontsize=8)
    axes[1].set_title('F1-score per video')
    axes[1].set_ylabel('F1-score')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Analisi errori per video — test set', fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / 'errors_per_video.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")

    return video_stats


# ── Analisi 2: errori per topo ─────────────────────────────────────────────────

def analyze_errors_per_mouse(errors):
    """
    Aggrega FP e FN per nome del topo.
    Rivela se certi individui sono sistematicamente più difficili.
    """
    fp_per_mouse = defaultdict(int)
    fn_per_mouse = defaultdict(int)
    tot_per_mouse = defaultdict(int)

    for seq, _ in errors['fp']:
        fp_per_mouse[seq['mouse_name']] += 1
        tot_per_mouse[seq['mouse_name']] += 1
    for seq, _ in errors['fn']:
        fn_per_mouse[seq['mouse_name']] += 1
        tot_per_mouse[seq['mouse_name']] += 1
    for seq, _ in errors['tp']:
        tot_per_mouse[seq['mouse_name']] += 1
    for seq, _ in errors['tn']:
        tot_per_mouse[seq['mouse_name']] += 1

    mice = sorted(tot_per_mouse.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mice))

    fp_vals  = [fp_per_mouse[m]  for m in mice]
    fn_vals  = [fn_per_mouse[m]  for m in mice]
    tot_vals = [tot_per_mouse[m] for m in mice]

    # error rate percentuale
    err_rate = [(fp_per_mouse[m] + fn_per_mouse[m]) / tot_per_mouse[m] * 100
                for m in mice]

    ax2 = ax.twinx()
    ax.bar(x - 0.2, fp_vals, 0.35, label='FP', color='#E8944A', alpha=0.8)
    ax.bar(x + 0.2, fn_vals, 0.35, label='FN', color='#5B8DB8', alpha=0.8)
    ax2.plot(x, err_rate, 'o-', color='#E85555', lw=2,
             label='Error rate %', markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(mice, rotation=30, ha='right')
    ax.set_ylabel('Numero errori assoluti')
    ax2.set_ylabel('Error rate (%)')
    ax.set_title('Errori per topo — variabilità inter-individuale')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = OUTPUT_DIR / 'errors_per_mouse.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")

    # stampa tabella
    print(f"\n{'Topo':<12} {'FP':>5} {'FN':>5} {'Tot seq':>8} {'Err%':>7}")
    print("-" * 38)
    for m in sorted(mice, key=lambda m: err_rate[mice.index(m)], reverse=True):
        i   = mice.index(m)
        tot = tot_per_mouse[m]
        print(f"{m:<12} {fp_per_mouse[m]:>5} {fn_per_mouse[m]:>5} "
              f"{tot:>8} {err_rate[i]:>6.1f}%")


# ── Analisi 3: posizione temporale degli FN ───────────────────────────────────

def analyze_fn_position(errors):
    """
    Per ogni Falso Negativo calcola la sua posizione relativa
    all'interno della crisi: 0.0 = onset, 1.0 = offset.
    Risponde a: il modello sbaglia di più all'inizio, nel mezzo
    o alla fine della crisi?
    """
    positions = []

    for seq, prob in errors['fn']:
        onset  = seq['onset_frame']
        offset = seq['offset_frame']
        center = seq['start_idx'] + SEQ_LEN // 2

        # converti a frame originali (sampled_idx × 3)
        center_orig = center * 3

        if offset > onset:
            pos = (center_orig - onset) / (offset - onset)
            pos = max(0.0, min(1.0, pos))
            positions.append(pos)

    if not positions:
        print("Nessun FN da analizzare.")
        return

    positions = np.array(positions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # istogramma posizioni
    axes[0].hist(positions, bins=20, color='#5B8DB8', alpha=0.8,
                 edgecolor='white', lw=0.5)
    axes[0].axvline(x=0.33, color='gray', linestyle='--',
                    lw=1, label='Onset / Mezzo / Offset')
    axes[0].axvline(x=0.67, color='gray', linestyle='--', lw=1)
    axes[0].set_xlabel('Posizione relativa nella crisi\n(0=onset, 1=offset)')
    axes[0].set_ylabel('Numero FN')
    axes[0].set_title('Dove si concentrano i Falsi Negativi?')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # tre fasi
    onset_zone  = np.sum(positions < 0.33)
    middle_zone = np.sum((positions >= 0.33) & (positions < 0.67))
    offset_zone = np.sum(positions >= 0.67)
    total       = len(positions)

    labels = [f'Onset\n({onset_zone/total*100:.0f}%)',
              f'Medio\n({middle_zone/total*100:.0f}%)',
              f'Offset\n({offset_zone/total*100:.0f}%)']
    vals   = [onset_zone, middle_zone, offset_zone]
    colors = ['#E85555', '#E8944A', '#5BAB6F']

    axes[1].bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for i, (lbl, v) in enumerate(zip(labels, vals)):
        axes[1].text(i, v + 0.5, str(v), ha='center',
                     fontweight='bold', fontsize=12)
    axes[1].set_title('FN per fase della crisi')
    axes[1].set_ylabel('Numero FN')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Analisi posizione temporale dei Falsi Negativi', fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / 'fn_position_in_crisis.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")

    print(f"\nFN per fase: onset={onset_zone} ({onset_zone/total*100:.0f}%) | "
          f"medio={middle_zone} ({middle_zone/total*100:.0f}%) | "
          f"offset={offset_zone} ({offset_zone/total*100:.0f}%)")


# ── Analisi 4: distribuzione probabilità ──────────────────────────────────────

def analyze_probability_distribution(errors):
    """
    Mostra la distribuzione delle probabilità predette per
    TP, TN, FP, FN separatamente.
    Un buon modello ha TP con prob alta, TN con prob bassa,
    mentre FP e FN si concentrano vicino al threshold.
    """
    tp_probs = [p for _, p in errors['tp']]
    tn_probs = [p for _, p in errors['tn']]
    fp_probs = [p for _, p in errors['fp']]
    fn_probs = [p for _, p in errors['fn']]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    configs = [
        (tp_probs, 'TP — Veri Positivi', '#5BAB6F'),
        (tn_probs, 'TN — Veri Negativi', '#5B8DB8'),
        (fp_probs, 'FP — Falsi Positivi', '#E8944A'),
        (fn_probs, 'FN — Falsi Negativi', '#E85555'),
    ]

    for ax, (probs, title, color) in zip(axes, configs):
        if probs:
            ax.hist(probs, bins=25, color=color, alpha=0.8,
                    edgecolor='white', lw=0.5)
            ax.axvline(x=THRESHOLD, color='red', linestyle='--',
                       lw=1.5, label=f'Threshold={THRESHOLD}')
            ax.set_xlabel('P(crisi)')
            ax.set_ylabel('Conteggio')
            ax.set_title(f'{title} (n={len(probs)})')
            ax.set_xlim(0, 1)
            mean_p = np.mean(probs)
            ax.axvline(x=mean_p, color='navy', linestyle=':',
                       lw=1.5, label=f'Media={mean_p:.3f}')
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Distribuzione probabilità per categoria', fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / 'probability_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")

    # statistiche
    print(f"\nProbabilità medie per categoria:")
    print(f"  TP: {np.mean(tp_probs):.4f} ± {np.std(tp_probs):.4f}")
    print(f"  TN: {np.mean(tn_probs):.4f} ± {np.std(tn_probs):.4f}")
    print(f"  FP: {np.mean(fp_probs):.4f} ± {np.std(fp_probs):.4f}")
    print(f"  FN: {np.mean(fn_probs):.4f} ± {np.std(fn_probs):.4f}")
    print(f"\n  Gap TP-FN (prob media): "
          f"{np.mean(tp_probs) - np.mean(fn_probs):.4f}")
    print(f"  Gap TN-FP (prob media): "
          f"{np.mean(tn_probs) - np.mean(fp_probs):.4f}")


# ── Report testuale ────────────────────────────────────────────────────────────

def generate_report(errors, video_stats):
    """
    Genera un report testuale strutturato con le osservazioni principali.
    Pronto per essere usato come base per la sezione della tesi.
    """
    tp = len(errors['tp'])
    tn = len(errors['tn'])
    fp = len(errors['fp'])
    fn = len(errors['fn'])
    total = tp + tn + fp + fn

    top5_error = video_stats[:5]
    top5_f1    = sorted(video_stats, key=lambda x: x['f1'])[:5]

    lines = [
        "=" * 60,
        "ERROR ANALYSIS REPORT",
        f"Modello: {CHECKPOINT}",
        f"Data: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "1. DISTRIBUZIONE ERRORI",
        f"   Sequenze totali : {total}",
        f"   TP : {tp:>4}  ({100*tp/total:.1f}%)",
        f"   TN : {tn:>4}  ({100*tn/total:.1f}%)",
        f"   FP : {fp:>4}  ({100*fp/total:.1f}%)",
        f"   FN : {fn:>4}  ({100*fn/total:.1f}%)",
        f"   Precision : {tp/(tp+fp):.4f}",
        f"   Recall    : {tp/(tp+fn):.4f}",
        f"   F1-score  : {2*tp/(2*tp+fp+fn):.4f}",
        "",
        "2. VIDEO PIÙ DIFFICILI (top 5 per errori totali)",
    ]

    for s in top5_error:
        v = s['video'].replace('.mp4', '')
        lines.append(
            f"   {v:<25} FP={s['fp']:>3} FN={s['fn']:>3} "
            f"F1={s['f1']:.3f}"
        )

    lines += [
        "",
        "3. VIDEO CON F1 PIÙ BASSO (top 5)",
    ]
    for s in top5_f1:
        v = s['video'].replace('.mp4', '')
        lines.append(
            f"   {v:<25} F1={s['f1']:.3f} "
            f"(FP={s['fp']} FN={s['fn']})"
        )

    lines += [
        "",
        "4. POSSIBILI CAUSE DEGLI ERRORI",
        "   - FP: movimenti pre-ictali simili alla crisi nei primi secondi",
        "   - FN: crisi con pattern atipici o onset/offset graduali",
        "   - Variabilità inter-individuale tra topi di razze diverse",
        "   - Dataset limitato (100 video) — generalizzazione parziale",
        "",
        "5. STRATEGIE DI MIGLIORAMENTO",
        "   - Raccogliere più video per topi con F1 basso",
        "   - Aumentare il dataset con video da altri laboratori",
        "   - Validazione leave-one-mouse-out per misurare",
        "     la generalizzazione inter-individuale",
        "   - Attention mechanism sull'LSTM per pesare",
        "     dinamicamente i timestep più informativi",
        "=" * 60,
    ]

    report = "\n".join(lines)
    path   = OUTPUT_DIR / 'error_report.txt'
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nReport salvato in: {path}")
    print("\n" + report)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Caricamento modello e dataset...")

    model = load_model()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    sequences            = build_sequences(manifest, SEQ_LEN, STRIDE)
    _, _, test_sequences = split_sequences(sequences)

    print(f"Sequenze test: {len(test_sequences)}")
    print("Inferenza in corso (può richiedere qualche minuto su CPU)...")

    probs, labels, raw_probs = get_predictions(model, test_sequences)

    print("Classificazione errori...")
    errors = classify_errors(test_sequences, probs, labels, THRESHOLD)

    print("\n1. Analisi errori per video...")
    video_stats = analyze_errors_per_video(errors)

    print("\n2. Analisi errori per topo...")
    analyze_errors_per_mouse(errors)

    print("\n3. Posizione temporale FN...")
    analyze_fn_position(errors)

    print("\n4. Distribuzione probabilità...")
    analyze_probability_distribution(errors)

    print("\n5. Generazione report...")
    generate_report(errors, video_stats)

    print(f"\nTutti i file salvati in: {OUTPUT_DIR}/")