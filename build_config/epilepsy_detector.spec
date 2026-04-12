# File di configurazione PyInstaller per creare l'eseguibile
# dell'applicazione Epilepsy Seizure Detector.
#
# Per creare l'eseguibile eseguire:
#   pyinstaller build_config/epilepsy_detector.spec
#
# L'output sarà in dist/EpilepsyDetector/

import sys
from pathlib import Path

# root del progetto (due livelli sopra questo file)
ROOT = Path(SPECPATH).parent

# File da includere nell'eseguibile
# Formato: (path_sorgente, path_destinazione_nell_eseguibile)
added_files = [
    # pesi del modello CNN
    (str(ROOT / 'model_weights' /
         'mobilenet_v3_small_imagenet.pth'),
     'model_weights'),

    # checkpoint del modello addestrato
    (str(ROOT / 'runs' / '20260327_101301' / 'best_model.pt'),
     'runs/20260327_101301'),
]

# Analisi del codice sorgente
a = Analysis(
    # script principale da eseguire
    [str(ROOT / 'src' / 'gui.py')],

    # path dove cercare i moduli importati
    pathex=[str(ROOT / 'src')],

    # file binari aggiuntivi (nessuno in questo caso)
    binaries=[],

    # file di dati da includere
    datas=added_files,

    # import nascosti che PyInstaller non trova automaticamente
    # (librerie che vengono importate dinamicamente)
    hiddenimports=[
        'jaraco',
        'jaraco.text',
        'jaraco.functools',
        'jaraco.context',
        'jaraco.collections',
        'pkg_resources',
        'pkg_resources.extern',
        'torch',
        'torchvision',
        'torchvision.models',
        'torchvision.transforms',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'cv2',
        'sklearn',
        'sklearn.metrics',
        'numpy',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],

    # file da escludere per ridurre le dimensioni
    excludes=[
        'jupyter',
        'notebook',
        'IPython',
        'scipy',
        'pandas',
        'pytest',
        'setuptools',
    ],

    noarchive=False,
)

# Creazione PYZ (archivio moduli Python)
pyz = PYZ(a.pure)

# Creazione eseguibile
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,   # modalità cartella (non file singolo)
    name             = 'EpilepsyDetector',
    debug            = False,
    bootloader_ignore_signals = False,
    strip            = False,
    upx              = True,   # compressione UPX se disponibile
    console          = False,  # niente finestra terminale
    icon             = None,   # aggiungere icona .icns per macOS se disponibile
)

# Raccolta file in cartella finale
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip = False,
    upx   = True,
    name  = 'EpilepsyDetector',
)

# Bundle macOS .app
# Crea un bundle .app cliccabile per macOS
app = BUNDLE(
    coll,
    name    = 'EpilepsyDetector.app',
    icon    = None,
    bundle_identifier = 'it.unipd.epilepsy_detector',
    info_plist = {
        'NSHighResolutionCapable': True,
        'LSBackgroundOnly'       : False,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleName'           : 'Epilepsy Detector',
        'NSHumanReadableCopyright': 'Francesco Zanovello — Università di Padova',
    },
)