#!/usr/bin/env bash
# Gera colab/medmonitor.zip com o código e os dados de demonstração, para enviar ao Google Colab.
# Uso (na raiz do projeto ou dentro de colab/):  bash colab/empacotar.sh
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f colab/medmonitor.zip
zip -qr colab/medmonitor.zip main.py config.yaml requirements.txt .env.example src \
    data/synthetic data/audio data/video \
    -x "*/__pycache__/*" "*.pyc"
echo "✓ colab/medmonitor.zip ($(du -h colab/medmonitor.zip | cut -f1))"
