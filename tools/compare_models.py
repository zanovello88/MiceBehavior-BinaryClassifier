"""
Scopo: confrontare graficamente CNN+LSTM e TimeSformer-HR per la tesi.
       Produce grafici side-by-side e tabelle comparative pronte per
       l'inserimento nel capitolo "Risultati Sperimentali".

Confronti prodotti:
  1. Metriche principali (F1, Recall, Precision, ROC-AUC) — bar chart
  2. Confusion matrices affiancate
  3. ROC curves sovrapposte
  4. Distribuzione FN per fase (onset/middle/offset)
  5. Distribuzione probabilità TP vs FN
  6. Tabella riassuntiva LaTeX-ready

Output:
  thesis_plots/
  ├── models_comparison_metrics.png
  ├── models_comparison_confusion.png
  ├── models_comparison_roc.png
  ├── models_comparison_fn_distribution.png
  ├── models_comparison_probabilities.png
  └── models_comparison_table.txt (LaTeX table)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ── Configurazione ─────────────────────────────────────────────────────────────
CNNLSTM_RESULTS = Path('runs/20260327_101301/eval_results_smoothed.json')
TIMESFORMER_RESULTS = Path('runs_timesformer/20260421_171823/eval_results_timesformer.json')
OUTPUT_DIR = Path('thesis_plots')

OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'sans-serif',
})


# ── Carica risultati ──────────────────────────────────────────────────────────

def load_results():
    """
    Carica i risultati di entrambi i modelli.
    """
    with open(CNNLSTM_RESULTS) as f:
        cnn_lstm = json.load(f)
    
    with open(TIMESFORMER_RESULTS) as f:
        timesformer = json.load(f)
    
    print("Risultati caricati:")
    print(f"  CNN+LSTM    : {CNNLSTM_RESULTS}")
    print(f"  TimeSformer : {TIMESFORMER_RESULTS}")
    
    return cnn_lstm, timesformer


# ── Plot 1: Metriche principali ───────────────────────────────────────────────

def plot_metrics_comparison(cnn_lstm, timesformer):
    """
    Bar chart comparativo delle metriche principali.
    """
    metrics = ['F1-score', 'Recall', 'Precision', 'ROC-AUC']
    
    cnn_vals = [
        cnn_lstm['f1'],
        cnn_lstm['recall'],
        cnn_lstm['precision'],
        cnn_lstm['roc_auc'],
    ]
    
    tf_vals = [
        timesformer['f1'],
        timesformer['recall'],
        timesformer['precision'],
        timesformer['roc_auc'],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, cnn_vals, width, 
                   label='CNN+LSTM', color='#5B8DB8', alpha=0.9)
    bars2 = ax.bar(x + width/2, tf_vals, width,
                   label='TimeSformer-HR', color='#E8944A', alpha=0.9)
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('Confronto metriche — CNN+LSTM vs TimeSformer-HR')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    path = OUTPUT_DIR / 'models_comparison_metrics.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")


# ── Plot 2: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(cnn_lstm, timesformer):
    """
    Confusion matrices affiancate.
    """
    cm_cnn = np.array(cnn_lstm['confusion_matrix'])
    cm_tf  = np.array(timesformer['confusion_matrix'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CNN+LSTM
    im1 = axes[0].imshow(cm_cnn, cmap='Blues', alpha=0.8)
    axes[0].set_title('CNN+LSTM')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Normal', 'Seizure'])
    axes[0].set_yticklabels(['Normal', 'Seizure'])
    
    for i in range(2):
        for j in range(2):
            text = axes[0].text(j, i, str(cm_cnn[i, j]),
                              ha='center', va='center', fontsize=16,
                              color='white' if cm_cnn[i, j] > cm_cnn.max()/2 else 'black')
    
    # TimeSformer
    im2 = axes[1].imshow(cm_tf, cmap='Oranges', alpha=0.8)
    axes[1].set_title('TimeSformer-HR')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Normal', 'Seizure'])
    axes[1].set_yticklabels(['Normal', 'Seizure'])
    
    for i in range(2):
        for j in range(2):
            text = axes[1].text(j, i, str(cm_tf[i, j]),
                              ha='center', va='center', fontsize=16,
                              color='white' if cm_tf[i, j] > cm_tf.max()/2 else 'black')
    
    fig.suptitle('Confusion Matrices — Confronto modelli', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / 'models_comparison_confusion.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")


# ── Plot 3: ROC Curves ────────────────────────────────────────────────────────

def plot_roc_curves(cnn_lstm, timesformer):
    """
    ROC curves sovrapposte.
    Usa i punti (FPR, TPR) calcolati dalla confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
    
    # CNN+LSTM
    cm_cnn = np.array(cnn_lstm['confusion_matrix'])
    tn_cnn, fp_cnn, fn_cnn, tp_cnn = cm_cnn.ravel()
    tpr_cnn = tp_cnn / (tp_cnn + fn_cnn)
    fpr_cnn = fp_cnn / (fp_cnn + tn_cnn)
    auc_cnn = cnn_lstm['roc_auc']
    
    ax.plot(fpr_cnn, tpr_cnn, 'o', color='#5B8DB8', markersize=10,
            label=f'CNN+LSTM (AUC={auc_cnn:.3f})', zorder=5)
    
    # TimeSformer
    cm_tf = np.array(timesformer['confusion_matrix'])
    tn_tf, fp_tf, fn_tf, tp_tf = cm_tf.ravel()
    tpr_tf = tp_tf / (tp_tf + fn_tf)
    fpr_tf = fp_tf / (fp_tf + tn_tf)
    auc_tf = timesformer['roc_auc']
    
    ax.plot(fpr_tf, tpr_tf, 's', color='#E8944A', markersize=10,
            label=f'TimeSformer-HR (AUC={auc_tf:.3f})', zorder=5)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Confronto modelli')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    fig.tight_layout()
    path = OUTPUT_DIR / 'models_comparison_roc.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")


# ── Plot 4: Distribuzione FN per fase ─────────────────────────────────────────

def plot_fn_phase_distribution():
    """
    Confronta la distribuzione dei FN per fase (onset/middle/offset).
    Dati estratti dai report di error analysis.
    """
    # CNN+LSTM: 100% onset (dal report precedente)
    cnn_phases = [100, 0, 0]
    
    # TimeSformer: dal report appena generato
    tf_phases = [39, 24, 37]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    phases = ['Onset', 'Middle', 'Offset']
    colors = ['#E85555', '#E8944A', '#5BAB6F']
    
    # CNN+LSTM
    axes[0].bar(phases, cnn_phases, color=colors, alpha=0.85)
    axes[0].set_title('CNN+LSTM — FN per fase')
    axes[0].set_ylabel('Percentuale FN (%)')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(cnn_phases):
        axes[0].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # TimeSformer
    axes[1].bar(phases, tf_phases, color=colors, alpha=0.85)
    axes[1].set_title('TimeSformer-HR — FN per fase')
    axes[1].set_ylabel('Percentuale FN (%)')
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(tf_phases):
        axes[1].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Distribuzione temporale dei Falsi Negativi', fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / 'models_comparison_fn_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")


# ── Plot 5: Distribuzione probabilità TP vs FN ───────────────────────────────

def plot_probability_distributions():
    """
    Box plot delle probabilità per TP e FN.
    Dati estratti dai report di error analysis.
    """
    # CNN+LSTM - valori dal report error_analysis/error_report.txt
    # Aggiorna questi valori con quelli reali del tuo report
    cnn_tp_mean = 0.966
    cnn_fn_mean = 0.930
    
    # TimeSformer (dal report appena generato)
    tf_tp_mean = 0.9340
    tf_fn_mean = 0.8856
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = [0, 1, 3, 4]
    heights = [cnn_tp_mean, cnn_fn_mean, tf_tp_mean, tf_fn_mean]
    colors_list = ['#5B8DB8', '#5B8DB8', '#E8944A', '#E8944A']
    alphas = [0.9, 0.5, 0.9, 0.5]
    
    # Crea le barre individualmente con alpha diversi
    for i, (xi, h, col, a) in enumerate(zip(x, heights, colors_list, alphas)):
        ax.bar(xi, h, color=col, alpha=a, width=0.6)
    
    # Etichette
    labels = ['TP', 'FN', 'TP', 'FN']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Aggiungi valori sopra le barre
    for xi, h in zip(x, heights):
        ax.text(xi, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Linee verticali per separare i modelli
    ax.axvline(x=1.5, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Titoli dei modelli
    ax.text(0.5, 1.05, 'CNN+LSTM', ha='center', fontsize=12, 
            fontweight='bold')
    ax.text(3.5, 1.05, 'TimeSformer-HR', ha='center', fontsize=12,
            fontweight='bold')
    
    ax.set_ylabel('Probabilità media P(crisi)')
    ax.set_title('Gap probabilità TP vs FN — Confronto modelli')
    ax.set_ylim(0.8, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotazioni gap
    gap_cnn = cnn_tp_mean - cnn_fn_mean
    gap_tf  = tf_tp_mean - tf_fn_mean
    ax.text(0.5, 0.82, f'Gap: {gap_cnn:.3f}', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#5B8DB8', alpha=0.3))
    ax.text(3.5, 0.82, f'Gap: {gap_tf:.3f}', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#E8944A', alpha=0.3))
    
    fig.tight_layout()
    path = OUTPUT_DIR / 'models_comparison_probabilities.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvato: {path}")


# ── Tabella riassuntiva LaTeX ─────────────────────────────────────────────────

def generate_latex_table(cnn_lstm, timesformer):
    """
    Genera tabella LaTeX-ready per la tesi.
    """
    cm_cnn = np.array(cnn_lstm['confusion_matrix'])
    cm_tf  = np.array(timesformer['confusion_matrix'])
    
    tn_cnn, fp_cnn, fn_cnn, tp_cnn = cm_cnn.ravel()
    tn_tf, fp_tf, fn_tf, tp_tf = cm_tf.ravel()
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Confronto prestazioni CNN+LSTM vs TimeSformer-HR}
\label{tab:models_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metrica} & \textbf{CNN+LSTM} & \textbf{TimeSformer-HR} \\
\midrule
F1-score         & %.3f & %.3f \\
Recall           & %.3f & %.3f \\
Precision        & %.3f & %.3f \\
ROC-AUC          & %.3f & %.3f \\
\midrule
True Positives   & %d   & %d   \\
True Negatives   & %d   & %d   \\
False Positives  & %d   & %d   \\
False Negatives  & %d   & %d   \\
\midrule
Parametri train. & 1.98M & 0.80M \\
Seq. length      & 60 (6s) & 16 (1.6s) \\
Risoluzione      & 210×210 & 448×448 \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        cnn_lstm['f1'], timesformer['f1'],
        cnn_lstm['recall'], timesformer['recall'],
        cnn_lstm['precision'], timesformer['precision'],
        cnn_lstm['roc_auc'], timesformer['roc_auc'],
        tp_cnn, tp_tf,
        tn_cnn, tn_tf,
        fp_cnn, fp_tf,
        fn_cnn, fn_tf,
    )
    
    path = OUTPUT_DIR / 'models_comparison_table.txt'
    with open(path, 'w') as f:
        f.write(latex)
    
    print(f"\nTabella LaTeX salvata in: {path}")
    print(latex)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("CONFRONTO MODELLI: CNN+LSTM vs TimeSformer-HR")
    print("=" * 60)
    
    cnn_lstm, timesformer = load_results()
    
    print("\n1. Plot metriche principali...")
    plot_metrics_comparison(cnn_lstm, timesformer)
    
    print("2. Plot confusion matrices...")
    plot_confusion_matrices(cnn_lstm, timesformer)
    
    print("3. Plot ROC curves...")
    plot_roc_curves(cnn_lstm, timesformer)
    
    print("4. Plot distribuzione FN per fase...")
    plot_fn_phase_distribution()
    
    print("5. Plot distribuzione probabilità...")
    plot_probability_distributions()
    
    print("6. Generazione tabella LaTeX...")
    generate_latex_table(cnn_lstm, timesformer)
    
    print(f"\n{'='*60}")
    print(f"Tutti i grafici salvati in: {OUTPUT_DIR}/")
    print(f"{'='*60}")