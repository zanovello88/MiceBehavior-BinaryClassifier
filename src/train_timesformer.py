"""
Scopo: training loop per TimeSformer-HR su dataset crisi epilettiche.
       Struttura identica a train.py con adattamenti specifici per
       il Transformer:

Differenze rispetto a train.py:
  - Learning rate più basso (3e-5 → 1e-5): i Transformer sono molto più
    sensibili al LR rispetto alle CNN. Un LR troppo alto distrugge
    i pesi pre-addestrati nelle prime epoche.
  - Warmup lineare del LR per le prime 3 epoche: stabilizza il training
    iniziale evitando oscillazioni nei gradienti dell'attenzione.
  - Gradient clipping più aggressivo (1.0 → 0.5): i Transformer con
    attenzione globale possono avere gradienti molto più grandi delle LSTM.
  - batch_size=4 di default: TimeSformer-HR con 16 frame a 448x448 è molto
    più pesante in VRAM rispetto a CNN+LSTM. Su H100 con 80GB possiamo
    arrivare a batch_size=8.
  - seq_len=16, stride=8: configurazione nativa di TimeSformer-HR.
    stride=8 significa overlap del 50% tra sequenze consecutive.
"""

import json
import logging
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
import sys

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from dataset                  import build_dataloaders
from model_timesformer        import TimeSformerBinary, count_parameters_timesformer
from transforms_timesformer   import train_transforms_timesformer, \
                                     eval_transforms_timesformer


# Argomenti CLI

def parse_args():
    p = argparse.ArgumentParser(description='Training TimeSformer epilessia')
    p.add_argument('--manifest',      type=str,   default='data/manifest.json')
    p.add_argument('--output_dir',    type=str,   default='runs_timesformer')
    p.add_argument('--weights_dir',   type=str,
                   default='model_weights/timesformer-hr')
    p.add_argument('--epochs',        type=int,   default=30)
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--lr',            type=float, default=1e-5)
    p.add_argument('--weight_decay',  type=float, default=1e-2)
    p.add_argument('--pos_weight',    type=float, default=0.4265)
    p.add_argument('--patience',      type=int,   default=8)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seq_len',       type=int,   default=16)
    p.add_argument('--stride',        type=int,   default=8)
    p.add_argument('--freeze_layers', type=int,   default=8)
    p.add_argument('--warmup_epochs', type=int,   default=3)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


# Setup logging

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('train_timesformer')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%H:%M:%S')
    fh = logging.FileHandler(output_dir / 'train.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# Training e valutazione

def train_one_epoch(model, loader, optimizer, criterion,
                    scheduler, device, grad_clip=0.5):
    """
    Gradient clipping a 0.5 invece di 1.0 — i gradienti dell'attenzione
    globale di TimeSformer sono più instabili rispetto all'LSTM.
    Lo scheduler OneCycleLR viene aggiornato ad ogni batch (non epoca).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(frames).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()   # OneCycleLR si aggiorna per batch

        total_loss += loss.item() * frames.size(0)
        preds       = (torch.sigmoid(logits) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames).squeeze(1)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * frames.size(0)
        probs       = torch.sigmoid(logits)
        preds       = (probs >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_probs, all_labels


# Main

def main():
    args       = parse_args()
    run_id     = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / run_id
    log        = setup_logging(output_dir)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    log.info(f"Run ID: {run_id}")
    log.info(f"Args: {vars(args)}")

    # DataLoader
    log.info("Caricamento dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path   = Path(args.manifest),
        train_transform = train_transforms_timesformer,
        eval_transform  = eval_transforms_timesformer,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seq_len         = args.seq_len,
        stride          = args.stride,
    )
    log.info(f"Train: {len(train_loader.dataset):,} seq | "
             f"Val: {len(val_loader.dataset):,} seq | "
             f"Test: {len(test_loader.dataset):,} seq")

    # Modello
    log.info("Caricamento TimeSformer-HR...")
    model = TimeSformerBinary(
        weights_dir   = args.weights_dir,
        freeze_layers = args.freeze_layers,
    ).to(device)

    log.info("Parametri:")
    count_parameters_timesformer(model)

    # Loss
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    # solo parametri trainabili (quelli non congelati)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params,
                      lr           = args.lr,
                      weight_decay = args.weight_decay)

    # Scheduler: OneCycleLR con warmup 
    # OneCycleLR: sale da lr/10 a lr in warmup_epochs, poi scende cosine
    # È il scheduler più efficace per i Transformer con dataset piccoli
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        total_steps     = total_steps,
        pct_start       = args.warmup_epochs / args.epochs,
        anneal_strategy = 'cos',
    )

    # Training loop 
    best_val_loss = float('inf')
    epochs_no_imp = 0
    history       = []

    log.info("Inizio training...")
    log.info(f"{'Epoca':>5} | {'TrainLoss':>9} | {'TrainAcc':>8} | "
             f"{'ValLoss':>7} | {'ValAcc':>6} | {'LR':>8}")
    log.info("-" * 62)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"{epoch:>5} | {train_loss:>9.4f} | {train_acc:>7.3f}% | "
                 f"{val_loss:>7.4f} | {val_acc:>5.3f}% | {current_lr:>8.2e}")

        history.append({
            'epoch'     : epoch,
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'val_loss'  : val_loss,
            'val_acc'   : val_acc,
            'lr'        : current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_imp = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : val_loss,
                'val_acc'    : val_acc,
                'args'       : vars(args),
            }, output_dir / 'best_model.pt')
            log.info(f"        ✓ best model salvato (val_loss={val_loss:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= args.patience:
                log.info(f"Early stopping all'epoca {epoch}")
                break

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"Training completato. Best val_loss: {best_val_loss:.4f}")
    log.info(f"Checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()