#!/usr/bin/env bash
# Provisiona os recursos Azure usados pelo MedMonitor e grava as chaves no .env.
#
# Pré-requisitos: Azure CLI (https://learn.microsoft.com/cli/azure/install-azure-cli) e `az login`.
# Uso:  bash infra/azure_setup.sh [grupo] [região]
#
# Recursos criados (camada gratuita F0 quando disponível):
#   - Azure AI Speech     (kind SpeechServices) → Speech to Text / Text to Speech
#   - Azure AI Language   (kind TextAnalytics)  → sentimento, frases-chave, Text Analytics for Health
set -euo pipefail

RG="${1:-rg-medmonitor}"
LOCATION="${2:-brazilsouth}"
SUFFIX="$(openssl rand -hex 3)"
SPEECH="medmonitor-speech-${SUFFIX}"
LANGUAGE="medmonitor-language-${SUFFIX}"
SKU="${SKU:-F0}"   # use SKU=S0 se a cota gratuita já tiver sido usada na assinatura

echo "→ Grupo de recursos ${RG} (${LOCATION})"
az group create --name "$RG" --location "$LOCATION" --output none

echo "→ Azure AI Speech: ${SPEECH}"
az cognitiveservices account create --name "$SPEECH" --resource-group "$RG" --kind SpeechServices \
  --sku "$SKU" --location "$LOCATION" --yes --output none

echo "→ Azure AI Language: ${LANGUAGE}"
az cognitiveservices account create --name "$LANGUAGE" --resource-group "$RG" --kind TextAnalytics \
  --sku "$SKU" --location "$LOCATION" --custom-domain "$LANGUAGE" --yes --output none

SPEECH_KEY=$(az cognitiveservices account keys list -n "$SPEECH" -g "$RG" --query key1 -o tsv)
LANGUAGE_KEY=$(az cognitiveservices account keys list -n "$LANGUAGE" -g "$RG" --query key1 -o tsv)
LANGUAGE_ENDPOINT=$(az cognitiveservices account show -n "$LANGUAGE" -g "$RG" --query properties.endpoint -o tsv)

ENV_FILE="$(dirname "$0")/../.env"
[ -f "$ENV_FILE" ] || cp "$(dirname "$0")/../.env.example" "$ENV_FILE"
set_var() { grep -q "^$1=" "$ENV_FILE" && sed -i.bak "s#^$1=.*#$1=$2#" "$ENV_FILE" || echo "$1=$2" >> "$ENV_FILE"; }
set_var AZURE_SPEECH_KEY "$SPEECH_KEY"
set_var AZURE_SPEECH_REGION "$LOCATION"
set_var AZURE_LANGUAGE_KEY "$LANGUAGE_KEY"
set_var AZURE_LANGUAGE_ENDPOINT "$LANGUAGE_ENDPOINT"
rm -f "$ENV_FILE.bak"

echo "✓ Credenciais gravadas em .env. Teste com:  python main.py azure-check"
echo "  Para remover tudo depois:  az group delete --name ${RG} --yes --no-wait"
