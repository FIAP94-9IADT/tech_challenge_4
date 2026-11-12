FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
# Contêiner para o processamento em lote (sem interface gráfica). O aplicativo desktop
# (python main.py gui) roda na máquina do usuário, lendo a pasta outputs/ gerada aqui.
CMD ["python", "main.py", "demo"]
