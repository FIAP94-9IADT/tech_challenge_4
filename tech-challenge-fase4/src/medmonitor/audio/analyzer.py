"""Pipeline de áudio: sinal → biomarcadores vocais + transcrição (Azure) → texto (Azure) → alertas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..alerts import Alert, Severity
from .azure_speech import transcribe, word_timing_features
from .azure_text import analyze_text
from .features import extract_features, load_audio, vocal_indicators

INDICATOR_LABEL = {
    "fadiga": "Sinais vocais de fadiga",
    "dificuldade_respiratoria": "Padrão de fala sugestivo de dificuldade respiratória",
    "disartria": "Padrão de fala sugestivo de disartria",
}
# indicador vocal ↔ categoria de termos críticos que o corrobora
CORROBORATION = {"dificuldade_respiratoria": "respiratorio", "disartria": "neurologico", "fadiga": "geral"}
ACTION = {
    "fadiga": "Avaliar causas de fadiga (anemia, distúrbios metabólicos, infecção, efeito de sedativos).",
    "dificuldade_respiratoria": "Avaliar padrão respiratório e oximetria; considerar gasometria e ausculta.",
    "disartria": "Aplicar escala de AVC (NIHSS/BE-FAST) imediatamente e acionar protocolo de AVC se positivo.",
}


@dataclass
class AudioResult:
    file: str
    patient_id: str
    features: dict
    indicators: dict
    transcript: dict
    text_analysis: dict
    alerts: list[Alert] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        vocal = max(v for k, v in self.indicators.items() if not k.startswith("_"))
        terms = [t for t in self.text_analysis["critical_terms"] if not t["negated"]]
        return float(min(1.0, vocal + 0.1 * len({t["category"] for t in terms})))

    def to_dict(self) -> dict:
        return {"file": self.file, "patient_id": self.patient_id, "features": self.features,
                "indicators": self.indicators, "transcript": self.transcript,
                "text_analysis": self.text_analysis, "risk_score": round(self.risk_score, 2),
                "alerts": [a.to_dict() for a in self.alerts]}


def analyze_audio(path: str | Path, cfg: dict, patient_id: str = "desconhecido", timestamp: str | None = None) -> AudioResult:
    path = str(path)
    acfg = cfg["audio"]
    ts = timestamp or datetime.now().replace(microsecond=0).isoformat()
    y, sr = load_audio(path, acfg["sample_rate"])

    features = extract_features(y, sr)
    indicators = vocal_indicators(features, acfg["reference"])
    transcript = transcribe(path, y, sr, cfg)
    features.update(word_timing_features(transcript.get("words", [])))
    text = analyze_text(transcript["text"], cfg)

    alerts: list[Alert] = []
    affirmed = [t for t in text["critical_terms"] if not t["negated"]]
    categories = {t["category"] for t in affirmed}
    health_symptoms = [e["text"] for e in text["health_entities"]
                       if e["category"] in ("SymptomOrSign", "Diagnosis") and not e["negated"]]

    for key, value in indicators.items():
        if key.startswith("_") or value < 0.5:
            continue
        corroborated = CORROBORATION[key] in categories
        sev = Severity.CRITICAL if (value >= 0.75 and corroborated and key != "fadiga") else Severity.WARNING
        terms = [t["term"] for t in affirmed if t["category"] == CORROBORATION[key]]
        desc = f"Indicador vocal {value:.2f} (0-1)."
        if corroborated:
            desc += f" Corroborado pelo relato do paciente: {', '.join(sorted(set(terms)))}."
        alerts.append(Alert(patient_id, "audio", sev, INDICATOR_LABEL[key], desc, ts,
                            evidence={"arquivo": Path(path).name, "indicador": value,
                                      **{k: v for k, v in indicators["_componentes"].items() if v >= 0.5}},
                            recommended_action=ACTION[key]))

    reported = sorted({t["term"] for t in affirmed if t["category"] in ("respiratorio", "cardiaco", "neurologico", "dor")})
    if reported:
        sev = Severity.WARNING if categories & {"respiratorio", "cardiaco", "neurologico"} else Severity.INFO
        alerts.append(Alert(patient_id, "audio", sev, "Sintomas críticos relatados na consulta",
                            f"Termos identificados na transcrição: {', '.join(reported)}."
                            + (f" Entidades clínicas (Azure Health): {', '.join(health_symptoms[:6])}." if health_symptoms else ""),
                            ts, evidence={"arquivo": Path(path).name, "sentimento": text["sentiment"]["label"],
                                          "fonte_transcricao": transcript["source"], "fonte_texto": text["source"]},
                            recommended_action="Revisar a transcrição completa e reavaliar o paciente."))
    elif text["sentiment"]["label"] == "negative" and text["sentiment"]["scores"].get("negative", 0) >= 0.6:
        alerts.append(Alert(patient_id, "audio", Severity.INFO, "Sentimento predominantemente negativo na consulta",
                            "O paciente expressou desconforto/angústia na maior parte da fala.", ts,
                            evidence={"scores": text["sentiment"]["scores"]},
                            recommended_action="Avaliar dor, ansiedade e necessidade de suporte psicológico."))

    return AudioResult(Path(path).name, patient_id, features, indicators, transcript, text, alerts)
