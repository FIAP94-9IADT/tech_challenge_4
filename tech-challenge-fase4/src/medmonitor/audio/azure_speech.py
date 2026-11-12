"""Transcrição de áudio com Azure AI Speech (Speech to Text).

Usa reconhecimento contínuo (adequado a consultas longas) com timestamps por
palavra, que permitem calcular taxa de fala e pausas a partir da transcrição.

Sem credenciais Azure, usa a transcrição de referência ao lado do áudio
(`<arquivo>.txt`), e o resultado é marcado com `source="sidecar"` — útil para
rodar a demonstração offline; o relatório deixa claro qual fonte foi usada.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import azure_speech_enabled

log = logging.getLogger(__name__)


def _to_wav16k(y: np.ndarray, sr: int) -> str:
    """O SDK de fala espera WAV PCM 16 bits mono; gravamos um temporário a partir do sinal já carregado."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    return tmp.name


def transcribe_azure(y: np.ndarray, sr: int, cfg: dict) -> dict:
    import azure.cognitiveservices.speech as speechsdk

    az = cfg["azure"]
    speech_config = speechsdk.SpeechConfig(subscription=az["speech_key"], region=az["speech_region"])
    speech_config.speech_recognition_language = cfg["audio"]["language"]
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    wav = _to_wav16k(y, sr)
    audio_config = speechsdk.audio.AudioConfig(filename=wav)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    texts, words, confidences = [], [], []
    done = threading.Event()
    errors: list[str] = []

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            texts.append(evt.result.text)
            detail = json.loads(evt.result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult, "{}"))
            best = (detail.get("NBest") or [{}])[0]
            confidences.append(best.get("Confidence", 0.0))
            for w in best.get("Words", []):
                words.append({"word": w["Word"], "start_s": w["Offset"] / 1e7, "duration_s": w["Duration"] / 1e7})

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            errors.append(evt.cancellation_details.error_details)
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(lambda _: done.set())
    recognizer.canceled.connect(on_canceled)
    recognizer.start_continuous_recognition()
    done.wait(timeout=max(60, 3 * len(y) / sr))
    recognizer.stop_continuous_recognition()
    Path(wav).unlink(missing_ok=True)

    if errors:
        raise RuntimeError(f"Azure Speech: {errors[0]}")
    return {
        "text": " ".join(texts),
        "words": words,
        "confidence": round(float(np.mean(confidences)), 3) if confidences else None,
        "source": "azure-speech",
        "language": cfg["audio"]["language"],
    }


def transcribe(audio_path: str, y: np.ndarray, sr: int, cfg: dict) -> dict:
    if azure_speech_enabled(cfg):
        try:
            return transcribe_azure(y, sr, cfg)
        except Exception as exc:  # noqa: BLE001 — falha de rede/credencial não deve parar a análise
            log.warning("Azure Speech indisponível (%s); usando fallback.", exc)
    sidecar = Path(audio_path).with_suffix(".txt")
    if sidecar.exists():
        return {"text": sidecar.read_text(encoding="utf-8").strip(), "words": [], "confidence": None,
                "source": "sidecar", "language": cfg["audio"]["language"]}
    return {"text": "", "words": [], "confidence": None, "source": "indisponivel", "language": cfg["audio"]["language"]}


def word_timing_features(words: list[dict]) -> dict:
    """Taxa de fala e pausas a partir dos timestamps por palavra do Azure."""
    if len(words) < 3:
        return {}
    starts = np.array([w["start_s"] for w in words])
    ends = starts + np.array([w["duration_s"] for w in words])
    gaps = starts[1:] - ends[:-1]
    span = ends[-1] - starts[0]
    return {
        "words_per_min": round(len(words) / span * 60, 1),
        "word_gaps_over_0_5s": int(np.sum(gaps > 0.5)),
        "mean_word_gap_s": round(float(np.mean(gaps.clip(min=0))), 3),
    }
