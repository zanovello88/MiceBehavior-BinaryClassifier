"""
Scopo: definire le trasformazioni specifiche per TimeSformer-HR.

Differenze rispetto a transforms.py (CNN+LSTM):
  - Resize a 448x448 invece di nessun resize (TimeSformer-HR richiede 448)
  - Stessa normalizzazione ImageNet — il backbone è pre-addestrato su ImageNet
  - Augmentation invariata — le stesse considerazioni sul dominio dei topi
    si applicano (no flip orizzontale, rotazioni minime)

Il resize da 210→448 introduce interpolazione ma è necessario per rispettare
il formato di input nativo del modello. L'upscaling da 210 a 448 non aggiunge
informazione ma permette al modello di applicare le patch di dimensione fissa
(16x16 pixel) per cui è stato addestrato, mantenendo la struttura dell'attenzione.
"""

from torchvision import transforms

IMG_SIZE = 448   # risoluzione nativa TimeSformer-HR

# Training
train_transforms_timesformer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(
        brightness = 0.2,
        contrast   = 0.2,
        saturation = 0.1,
    ),
    transforms.RandomAffine(
        degrees   = 3,
        translate = (0.02, 0.02),
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225],
    ),
])

# Validation / Test
eval_transforms_timesformer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225],
    ),
])