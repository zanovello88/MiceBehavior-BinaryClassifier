"""
Scopo: definire il modello TimeSformer per classificazione binaria
       di sequenze video (crisi / non crisi).

Architettura:
  - Backbone: TimeSformer-HR (facebook/timesformer-hr-finetuned-k400)
    pre-addestrato su Kinetics-400 con 16 frame a 448x448 pixel.
    È il primo video Transformer puro — nessuna convoluzione, solo
    self-attention spazio-temporale con "divided attention":
    attenzione temporale e spaziale applicate separatamente in sequenza.
  - Testa di classificazione: il classificatore originale (400 classi
    Kinetics) viene rimosso e sostituito con una FC binaria.
  - Freeze parziale: i primi N layer del Transformer vengono congelati
    per evitare overfitting con dataset piccoli. I layer superiori
    vengono fine-tunati per adattarsi al dominio dei topi.

Differenze rispetto a CNN+LSTM:
  - CNN+LSTM processa i frame sequenzialmente: prima CNN su ogni frame
    poi LSTM per la dipendenza temporale.
  - TimeSformer processa tutti i 16 frame simultaneamente con
    self-attention globale — vede le relazioni tra tutti i frame
    in una sola forward pass, senza separare spazio e tempo.

Input atteso: [B, num_frames, C, H, W] con num_frames=16, H=W=448
  (il resize da 210→448 viene gestito internamente dalle trasformazioni)

Nota sui pesi: il modello viene caricato da file locale per evitare
  download da internet sul cluster HPC (senza accesso alla rete).
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import TimesformerModel


class TimeSformerBinary(nn.Module):
    """
    TimeSformer-HR adattato per classificazione binaria.

    Il backbone viene caricato dai pesi pre-addestrati salvati localmente.
    La testa di classificazione originale (400 classi) viene sostituita
    con una FC binaria con dropout per regolarizzazione.

    Parametri:
      weights_dir    : cartella contenente i pesi salvati con save_pretrained()
      freeze_layers  : numero di layer Transformer da congelare (0-11 per ViT-Base)
                       I layer congelati mantengono le feature generali di Kinetics,
                       quelli sbloccati si adattano al dominio specifico dei topi.
      hidden_size    : dimensione hidden del backbone (768 per ViT-Base)
      fc_dropout     : dropout prima del classificatore finale
    """

    def __init__(self,
                 weights_dir   : str   = 'model_weights/timesformer-hr',
                 freeze_layers : int   = 8,
                 hidden_size   : int   = 768,
                 fc_dropout    : float = 0.7):
        super().__init__()

        # ── Carica backbone da file locale ─────────────────────────────────────
        weights_path = Path(weights_dir)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pesi TimeSformer non trovati in: {weights_dir}\n"
                f"Esegui prima il download con:\n"
                f"  python3 -c \"from transformers import TimesformerModel; "
                f"TimesformerModel.from_pretrained("
                f"'facebook/timesformer-hr-finetuned-k400')"
                f".save_pretrained('{weights_dir}')\""
            )

        print(f"Caricamento TimeSformer-HR da: {weights_dir}")
        self.backbone = TimesformerModel.from_pretrained(str(weights_path))
        print("Backbone caricato.")

        # ── Congela i primi freeze_layers encoder layer ────────────────────────
        # TimeSformer-HR ha 12 layer Transformer (indici 0-11)
        # Congelare i primi 8 significa mantenere le feature generali
        # (patch embedding, posizione, feature spaziali di basso livello)
        # e allenare solo gli ultimi 4 layer per il dominio specifico
        encoder_layers = self.backbone.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # congela anche embeddings (patch + position + time)
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # ── Testa di classificazione binaria ───────────────────────────────────
        # hidden_size=768 per ViT-Base (TimeSformer-HR)
        # L'output del backbone è il CLS token: [B, hidden_size]
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=fc_dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=fc_dropout * 0.6),
            nn.Linear(256, 1),   # output scalare — sigmoid nella loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      x: [B, T, C, H, W] — T=16 frame, H=W=448
      output: [B, 1] — logit

      TimeSformer di HuggingFace si aspetta pixel_values di shape
      [B, C, T, H, W] internamente, ma il suo forward accetta
      [B, num_frames, C, H, W] e fa il reshape internamente.
      NON dobbiamo fare il permute — lo fa il modello.
      """
      # x è già [B, T, C, H, W] — TimeSformer lo gestisce direttamente
      outputs = self.backbone(pixel_values=x)

      # CLS token → [B, hidden_size]
      cls_output = outputs.last_hidden_state[:, 0, :]

      # classificatore binario
      logit = self.classifier(cls_output)   # [B, 1]
      return logit


# Utility 

def count_parameters_timesformer(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parametri totali     : {total:,}")
    print(f"  Parametri trainabili : {trainable:,}")
    print(f"  Parametri congelati  : {frozen:,}")
    print(f"  % trainabili         : {100*trainable/total:.1f}%")


# Test rapido

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = TimeSformerBinary(
        weights_dir   = 'model_weights/timesformer-hr',
        freeze_layers = 8,
    ).to(device)

    print("\nParametri:")
    count_parameters_timesformer(model)

    # forward pass con batch fittizio [B=2, T=16, C=3, H=448, W=448]
    print("\nForward pass...")
    dummy = torch.randn(2, 16, 3, 448, 448).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # atteso: [2, 1]
    print(f"Output values: {out.squeeze().tolist()}")