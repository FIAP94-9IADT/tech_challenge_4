"""Prepara vídeos de demonstração a partir de fontes públicas (com licença registrada).

  P005_sentar_levantar.mp4   Teste de sentar e levantar em 30 s — CDC STEADI
                             (domínio público, Wikimedia Commons), trecho da paciente.
  P001_agachamento.mp4       "Squat and Frontal Raise" — Taco fleur, CC BY-SA 4.0.
  P005_queda_simulada.mp4    Derivado do anterior: os últimos quadros são girados
                             progressivamente em 90° para SIMULAR uma queda e validar o
                             detector. É um sinal de teste, não uma queda real.
  zona_critica.mp4           "solutions_ci_demo" — Ultralytics assets (AGPL-3.0),
                             pessoas circulando, usado para a área crítica.

Também é possível usar vídeos próprios: basta adicioná-los ao manifesto.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "medmonitor-academic-demo/1.0 (FIAP Tech Challenge)"}
SOURCES = {
    "chair_stand.webm": "https://upload.wikimedia.org/wikipedia/commons/d/d5/30-Second_Chair_Stand_Test.webm",
    "squat_frontal_raise.webm": "https://upload.wikimedia.org/wikipedia/commons/7/7f/Squat_and_Frontal_Raise.webm",
    "zone_people.mp4": "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4",
}


def _download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        with requests.get(url, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def _simulate_fall(src: Path, dest: Path, keep_s: float = 7.0, fall_s: float = 0.5, hold_s: float = 2.5) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while len(frames) < int(keep_s * fps):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    last = frames[-1]
    out = cv2.VideoWriter(str(dest.with_suffix(".tmp.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    # gira o último quadro em torno do ponto de apoio (pés) — 0° → 90° com aceleração
    pivot = (w * 0.55, h * 0.92)
    n_fall = int(fall_s * fps)
    for i in range(n_fall + int(hold_s * fps)):
        p = min(1.0, (i + 1) / n_fall) ** 2
        m = cv2.getRotationMatrix2D(pivot, -88 * p, 1.0)
        out.write(cv2.warpAffine(last, m, (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.release()
    _ffmpeg("-i", str(dest.with_suffix(".tmp.mp4")), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest))
    dest.with_suffix(".tmp.mp4").unlink(missing_ok=True)


def prepare_demo_videos(raw_dir: str | Path, out_dir: str | Path) -> Path:
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _download(raw)
    x264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-vf", "scale='min(960,iw)':-2"]
    _ffmpeg("-ss", "38", "-to", "64", "-i", str(raw / "chair_stand.webm"), *x264, str(out / "P005_sentar_levantar.mp4"))
    _ffmpeg("-i", str(raw / "squat_frontal_raise.webm"), *x264, str(out / "P001_agachamento.mp4"))
    _simulate_fall(out / "P001_agachamento.mp4", out / "P005_queda_simulada.mp4")
    # vídeo curto (2 s): repete 4x para ter duração útil
    _ffmpeg("-stream_loop", "3", "-i", str(raw / "zone_people.mp4"), *x264, str(out / "zona_critica.mp4"))

    rows = [
        ("P005_sentar_levantar.mp4", "P005", "sentar_levantar", "left", "2026-10-28T15:00:00",
         "Teste de sentar e levantar 30 s", "CDC STEADI — domínio público"),
        ("P001_agachamento.mp4", "P001", "agachamento", "largest", "2026-10-28T11:00:00",
         "Sessão de fisioterapia — agachamento", "Taco fleur — CC BY-SA 4.0 (Wikimedia Commons)"),
        ("P005_queda_simulada.mp4", "P005", "", "largest", "2026-10-29T02:59:00",
         "Queda simulada (rotação sintética)", "Derivado de Taco fleur — CC BY-SA 4.0"),
        ("zona_critica.mp4", "SALA-01", "", "largest", "2026-10-28T09:00:00",
         "Circulação em área crítica", "Ultralytics assets — AGPL-3.0"),
    ]
    rows = [(*r, r[0] == "zona_critica.mp4") for r in rows]
    manifest = out / "manifest.csv"
    pd.DataFrame(rows, columns=["file", "patient_id", "profile", "target", "timestamp", "description", "source", "zones"]).to_csv(
        manifest, index=False)
    return manifest
