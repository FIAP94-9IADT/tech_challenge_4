"""Gera áudios de consulta para demonstração.

A fala é sintetizada (Azure Neural TTS se houver chave; senão gTTS) frase a frase,
e efeitos de sinal simulam as condições que o módulo de áudio deve reconhecer:

  normal       fala fluente, pausas curtas
  fadiga       fala mais lenta e grave, intensidade decrescente, pausas longas
  dispneia     frases curtas entrecortadas por inspirações audíveis
  disartria    articulação lenta, tremor de amplitude/frequência, consoantes borradas

São sinais de teste para validar o pipeline — não representam pacientes reais.
Cada áudio acompanha a transcrição de referência (.txt) e o manifesto CSV indica
paciente e horário da consulta para a fusão multimodal.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

from ..config import azure_speech_enabled

log = logging.getLogger(__name__)
SR = 16000

SCRIPTS = {
    "P001": ("normal", "2026-10-28T10:30:00", [
        "Bom dia, doutora.",
        "Hoje eu estou me sentindo bem melhor.",
        "Dormi bem a noite toda e consegui caminhar pelo corredor com a fisioterapeuta.",
        "A dor da cirurgia diminuiu bastante e estou me alimentando normalmente.",
        "Não tenho falta de ar e estou bem animada para ir para casa.",
    ]),
    "P003": ("dispneia", "2026-10-28T22:40:00", [
        "Doutor, eu estou com falta de ar.",
        "Parece que o ar não entra direito.",
        "Estou muito sonolenta,",
        "e sinto um cansaço para respirar.",
        "Me deram um remédio para dor mais forte,",
        "e depois disso piorou.",
    ]),
    "P004": ("disartria", "2026-10-29T03:10:00", [
        "Doutor, a minha fala está enrolada.",
        "Senti um formigamento no braço direito,",
        "e o coração disparado de novo.",
        "Estou com tontura e a visão turva.",
    ]),
    "P005": ("fadiga", "2026-10-28T16:00:00", [
        "Estou muito cansada hoje.",
        "Sem energia nenhuma para levantar da cama.",
        "Sinto fraqueza nas pernas,",
        "e não consegui fazer os exercícios direito.",
        "Só quero ficar deitada.",
    ]),
}


def _tts(text: str, cfg: dict) -> np.ndarray:
    if azure_speech_enabled(cfg):
        try:
            import azure.cognitiveservices.speech as speechsdk
            sc = speechsdk.SpeechConfig(subscription=cfg["azure"]["speech_key"], region=cfg["azure"]["speech_region"])
            sc.speech_synthesis_voice_name = "pt-BR-FranciscaNeural"
            sc.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
            res = speechsdk.SpeechSynthesizer(speech_config=sc, audio_config=None).speak_text_async(text).get()
            if res.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                y, _ = sf.read(io.BytesIO(res.audio_data), dtype="float32")
                return y
        except Exception as exc:  # noqa: BLE001
            log.warning("Azure TTS falhou (%s); usando gTTS.", exc)
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text, lang="pt", tld="com.br").write_to_fp(buf)
    buf.seek(0)
    tmp = Path(__file__).with_name("_tts_tmp.mp3")
    tmp.write_bytes(buf.read())
    y, _ = librosa.load(tmp, sr=SR)
    tmp.unlink(missing_ok=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y


def _breath(rng, dur=0.45, level=0.12) -> np.ndarray:
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    noise = sosfilt(butter(4, [400, 3000], btype="band", fs=SR, output="sos"), noise)
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    return (noise / (np.max(np.abs(noise)) + 1e-9) * env * level).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def _tremor(y: np.ndarray, rate_hz=5.5, depth=0.45, rng=None) -> np.ndarray:
    t = np.arange(len(y)) / SR
    am = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t))
    # vibrato irregular: reamostragem com fator variável (instabilidade de F0)
    jitter = 1 + 0.035 * np.sin(2 * np.pi * 4.3 * t) + 0.02 * (rng.normal(0, 1, len(t)).cumsum() / np.sqrt(len(t)))
    idx = np.clip(np.cumsum(jitter), 0, len(y) - 1)
    warped = np.interp(idx, np.arange(len(y)), y)
    # micro-variações de amplitude ciclo a ciclo (shimmer)
    shimmer = 1 + 0.25 * rng.normal(0, 1, len(t) // 80 + 1).repeat(80)[: len(t)]
    return (warped * am * shimmer).astype(np.float32)


def _render(style: str, sentences: list[str], cfg: dict, rng) -> np.ndarray:
    parts = [_tts(s, cfg) for s in sentences]
    out: list[np.ndarray] = []
    if style == "normal":
        for p in parts:
            out += [p, _silence(0.35)]
    elif style == "fadiga":
        n = len(parts)
        for i, p in enumerate(parts):
            p = librosa.effects.time_stretch(p, rate=0.78)
            p = librosa.effects.pitch_shift(p, sr=SR, n_steps=-2.5)
            p = p * np.linspace(1.0 - 0.6 * i / n, 1.0 - 0.6 * (i + 1) / n, len(p))
            out += [p, _silence(rng.uniform(1.2, 1.9))]
    elif style == "dispneia":
        for p in parts:
            # quebra cada frase em fragmentos curtos intercalados por inspiração audível
            k = max(2, int(len(p) / SR / 0.7))
            for chunk in np.array_split(p, k):
                out += [chunk, _silence(0.08), _breath(rng, rng.uniform(0.35, 0.55)), _silence(0.08)]
    elif style == "disartria":
        lp = butter(6, 1800, btype="low", fs=SR, output="sos")
        for p in parts:
            p = librosa.effects.time_stretch(p, rate=0.62)
            p = sosfilt(lp, p).astype(np.float32)
            p = _tremor(p, rng=rng)
            out += [p, _silence(rng.uniform(0.6, 1.0))]
    y = np.concatenate(out)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


def generate_demo_audio(out_dir: str | Path, cfg: dict, seed: int = 3) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for pid, (style, ts, sentences) in SCRIPTS.items():
        wav = out / f"{pid}_consulta_{style}.wav"
        sf.write(wav, _render(style, sentences, cfg, rng), SR, subtype="PCM_16")
        wav.with_suffix(".txt").write_text(" ".join(sentences), encoding="utf-8")
        rows.append({"file": wav.name, "patient_id": pid, "timestamp": ts, "expected_condition": style})
    manifest = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
