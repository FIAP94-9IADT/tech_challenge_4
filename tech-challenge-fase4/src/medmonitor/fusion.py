"""Fusão multimodal: combina os achados de todas as modalidades por paciente.

Dois níveis de fusão (late fusion — cada modalidade é processada pelo seu modelo
especializado e a fusão ocorre no nível de decisão):

1. Escore de risco do paciente — "noisy-OR" ponderado dos escores (0-1) de cada
   modalidade:  risco = 1 − Π (1 − w_m · s_m).
   Uma modalidade muito alterada basta para elevar o risco; várias moderadas se somam.

2. Regras de correlação temporal — padrões clínicos que só aparecem cruzando
   fontes dentro de uma janela de tempo (ex.: aumento de opioide seguido de
   bradipneia). Geram alertas "fusion" com a evidência de cada modalidade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .alerts import Alert, Severity


@dataclass
class PatientFusion:
    patient_id: str
    modality_scores: dict[str, float]
    risk: float
    level: str
    alerts: list[Alert] = field(default_factory=list)
    correlations: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "risk": round(self.risk, 3), "level": self.level,
                "modality_scores": {k: round(v, 3) for k, v in self.modality_scores.items()},
                "n_alerts": len(self.alerts),
                "n_critical": sum(a.severity is Severity.CRITICAL for a in self.alerts + self.correlations),
                "correlations": [a.to_dict() for a in self.correlations]}


def risk_level(r: float) -> str:
    return "alto" if r >= 0.75 else "moderado" if r >= 0.45 else "baixo"


def noisy_or(scores: dict[str, float], weights: dict[str, float]) -> float:
    prod = 1.0
    for m, s in scores.items():
        prod *= 1 - min(1.0, weights.get(m, 0.5) * max(0.0, s))
    return 1 - prod


# ----------------------------------------------------------------------------- regras
def _ts(a: Alert) -> datetime:
    return datetime.fromisoformat(a.timestamp)


def _match(alerts, modality, *keywords, evidence_in: set | None = None):
    out = []
    for a in alerts:
        if a.modality != modality:
            continue
        text = f"{a.title} {a.description}".lower()
        if keywords and not any(k.lower() in text for k in keywords):
            continue
        if evidence_in is not None and str(a.evidence.get("medicamento", "")).lower() not in evidence_in:
            continue
        out.append(a)
    return out


def _pairs(first, second, window: timedelta, ordered: bool):
    for a in first:
        for b in second:
            dt = _ts(b) - _ts(a)
            if (0 <= dt.total_seconds() <= window.total_seconds()) if ordered else abs(dt) <= window:
                yield a, b


def _fused(pid, title, desc, parts: list[Alert], action, severity=Severity.CRITICAL) -> Alert:
    ts = max(_ts(p) for p in parts).isoformat()
    evidence = {f"{p.modality}@{p.timestamp[11:16]}": p.title for p in parts}
    return Alert(pid, "fusion", severity, title, desc, ts, evidence=evidence, recommended_action=action)


def correlate(pid: str, alerts: list[Alert], cfg: dict) -> list[Alert]:
    W = timedelta(hours=cfg["fusion"]["correlation_window_hours"])
    classes = cfg["prescriptions"]["therapeutic_classes"]
    sedatives = set(classes.get("opioide", [])) | set(classes.get("benzodiazepinico", []))
    out: list[Alert] = []

    def once(title):
        return not any(a.title == title for a in out)

    # R1 — opioide/sedativo ↑ → bradipneia/dessaturação
    rx = _match(alerts, "prescription", evidence_in=sedatives)
    resp = _match(alerts, "vitals", "bradipneia", "dessaturação")
    for a, b in _pairs(rx, resp, W, ordered=True):
        t = "Suspeita de depressão respiratória induzida por opioide/sedativo"
        if once(t):
            out.append(_fused(pid, t, f"{a.evidence.get('medicamento')} alterado às {a.timestamp[11:16]}; "
                              f"{b.title.lower()} às {b.timestamp[11:16]}.", [a, b],
                              "Suspender/reduzir opioide e sedativo, considerar naloxona, monitorização contínua de SpO2/FR."))

    # R2 — fala com padrão respiratório + dessaturação/taquipneia
    voice = _match(alerts, "audio", "respiratória")
    hyp = _match(alerts, "vitals", "dessaturação", "taquipneia", "bradipneia")
    for a, b in _pairs(voice, hyp, W, ordered=False):
        t = "Insuficiência respiratória confirmada por voz e sinais vitais"
        if once(t):
            out.append(_fused(pid, t, f"Padrão de fala com dispneia ({a.timestamp[11:16]}) e {b.title.lower()} "
                              f"({b.timestamp[11:16]}).", [a, b],
                              "Avaliação médica imediata, gasometria arterial, suporte ventilatório conforme necessidade."))

    # R3 — febre + instabilidade hemodinâmica/respiratória + agitação (qSOFA aproximado)
    sepsis = [a for a in _match(alerts, "vitals", "febre")
              if any(k in a.title for k in ("Taquicardia", "Hipotensão", "Taquipneia"))]
    agit = _match(alerts, "movement", "agitação")
    for a, b in _pairs(sepsis, agit, W * 2, ordered=False):
        t = "Suspeita de sepse com delirium (alteração do nível de consciência)"
        if once(t):
            out.append(_fused(pid, t, f"{a.title} às {a.timestamp[11:16]} e agitação noturna às {b.timestamp[11:16]}.",
                              [a, b], "Abrir protocolo de sepse (lactato, culturas, antibiótico na 1ª hora) e avaliar CAM-ICU."))
    if not agit and once("Suspeita de sepse"):
        for a in sepsis:
            if "Hipotensão" in a.title and "Febre" in a.title:
                out.append(_fused(pid, "Suspeita de sepse", f"{a.title} às {a.timestamp[11:16]}.", [a],
                                  "Abrir protocolo de sepse (lactato, culturas, antibiótico na 1ª hora)."))
                break

    # R4 — arritmia + disartria → AVC cardioembólico
    arr = _match(alerts, "vitals", "taquicardia", "bradicardia")
    dys = _match(alerts, "audio", "disartria")
    for a, b in _pairs(arr, dys, W, ordered=False):
        t = "Suspeita de AVC cardioembólico (arritmia + disartria)"
        if once(t):
            out.append(_fused(pid, t, f"{a.title} às {a.timestamp[11:16]} e fala disártrica às {b.timestamp[11:16]}.",
                              [a, b], "Acionar protocolo de AVC (código AVC), TC de crânio, avaliar janela para trombólise."))

    # R5 — queda confirmada por múltiplas fontes
    falls = _match(alerts, "movement", "queda") + _match(alerts, "video", "queda")
    after = _match(alerts, "vitals")
    if len({a.modality for a in falls}) >= 2 or (falls and any(_pairs(falls, after, timedelta(hours=1), True))):
        parts = falls + [b for _, b in _pairs(falls, after, timedelta(hours=1), True)][:1]
        out.append(_fused(pid, "Queda confirmada por múltiplas fontes",
                          f"Evidências: {', '.join(sorted({p.modality for p in parts}))}.", parts,
                          "Avaliar trauma (incluindo TCE), notificar evento adverso e revisar plano de prevenção de quedas."))

    # R6 — risco de queda: desempenho funcional ruim + imobilidade/fadiga
    func = _match(alerts, "video", "desempenho", "incompleta", "compensação")
    frail = _match(alerts, "movement", "imobilidade") + _match(alerts, "audio", "fadiga")
    if func and frail:
        out.append(_fused(pid, "Risco elevado de queda (funcional + fadiga/imobilidade)",
                          f"Execução motora alterada ({func[0].title.lower()}) e {frail[0].title.lower()}.",
                          [func[0], frail[0]], "Implementar medidas de prevenção de quedas e reavaliar fisioterapia.",
                          Severity.WARNING))

    # R7 — alteração de medicamento de alta vigilância logo após evento crítico
    critical = [a for a in alerts if a.severity is Severity.CRITICAL and a.modality != "prescription"]
    rx_hi = [a for a in alerts if a.modality == "prescription" and "alta vigilância" in a.title]
    for a, b in _pairs(critical, rx_hi, timedelta(hours=2), ordered=True):
        t = f"Alteração de alta vigilância após evento agudo ({b.evidence.get('medicamento')})"
        if once(t):
            out.append(_fused(pid, t, f"{b.title} às {b.timestamp[11:16]}, {a.title.lower()} às {a.timestamp[11:16]}.",
                              [a, b], "Confirmar indicação com o médico assistente antes de administrar.", Severity.WARNING))
    return out


def fuse_patient(pid: str, scores: dict[str, float], alerts: list[Alert], cfg: dict) -> PatientFusion:
    corr = correlate(pid, alerts, cfg)
    risk = noisy_or(scores, cfg["fusion"]["weights"])
    if any(a.severity is Severity.CRITICAL for a in corr):
        risk = max(risk, 0.9)  # padrão multimodal crítico confirmado
    return PatientFusion(pid, scores, risk, risk_level(risk), alerts, corr)
