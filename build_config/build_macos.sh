#!/bin/bash
# Script per creare l'eseguibile macOS dell'Epilepsy Detector.
# Eseguire dalla ROOT del progetto:
#   bash build_config/build_macos.sh

set -e   # interrompi se un comando fallisce

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Root progetto: $ROOT"
cd "$ROOT"

# Verifica file necessari
echo ""
echo "Verifica file necessari..."

if [ ! -f "model_weights/mobilenet_v3_small_imagenet.pth" ]; then
    echo "ERRORE: model_weights/mobilenet_v3_small_imagenet.pth non trovato"
    exit 1
fi

if [ ! -f "runs/20260327_101301/best_model.pt" ]; then
    echo "ERRORE: runs/20260327_101301/best_model.pt non trovato"
    exit 1
fi

echo "File modello OK"

# Installa PyInstaller
echo ""
echo "Installazione PyInstaller..."
pip install pyinstaller --quiet

# Pulizia build precedente
echo ""
echo "Pulizia build precedente..."
rm -rf build/ dist/

# Build
echo ""
echo "Creazione eseguibile (può richiedere 5-10 minuti)..."
pyinstaller build_config/epilepsy_detector.spec \
    --noconfirm \
    --clean

# Verifica output
echo ""
if [ -d "dist/EpilepsyDetector.app" ]; then
    SIZE=$(du -sh dist/EpilepsyDetector.app | cut -f1)
    echo "BUILD COMPLETATA"
    echo "App: dist/EpilepsyDetector.app ($SIZE)"
    echo ""
    echo "Per testare:"
    echo "  open dist/EpilepsyDetector.app"
    echo ""
    echo "Per distribuire:"
    echo "  zip -r EpilepsyDetector_macOS.zip dist/EpilepsyDetector.app"
else
    echo "ERRORE: build fallita — controlla l'output sopra"
    exit 1
fi