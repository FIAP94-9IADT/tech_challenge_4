# Notebook — Google Colab / VS Code

`MedMonitor_Colab.ipynb` executa a solução passo a passo: sinais vitais, tempo real, prescrições, movimentação, áudio (Azure), vídeo (YOLOv8/OpenPose), PhysioNet e fusão multimodal. O arquivo já vem **executado**, com gráficos e vídeos anotados nas saídas.

## No VS Code

1. Abra `colab/MedMonitor_Colab.ipynb`.
2. Em **Select Kernel** (canto superior direito), escolha **MedMonitor (.venv)** ou o Python de `.venv/bin/python`.
3. **Run All**. O notebook detecta que está no projeto local e usa o código de `src/` diretamente.

## No Google Colab

1. Gere o pacote (se ainda não existir): `bash colab/empacotar.sh` → cria `colab/medmonitor.zip` (~7 MB).
2. Em [colab.research.google.com](https://colab.research.google.com), **Arquivo → Fazer upload de notebook** → `MedMonitor_Colab.ipynb`.
3. (Opcional) **Ambiente de execução → Alterar o tipo → GPU T4**, que deixa o vídeo bem mais rápido.
4. **Ambiente de execução → Executar tudo**. Quando aparecer o botão de upload na primeira célula, envie `medmonitor.zip`.

Outras formas de levar o código (variável `FONTE` na primeira célula):
- `"drive"`: coloque `medmonitor.zip` em `MyDrive/`;
- `"github"`: preencha `GITHUB_URL` com o repositório.

### Azure no Colab

Na barra lateral, clique em 🔑 **Secrets**, cadastre `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_LANGUAGE_KEY` e `AZURE_LANGUAGE_ENDPOINT`, e ative o acesso do notebook. Sem elas, tudo roda em modo local e a fonte aparece em cada resultado.
