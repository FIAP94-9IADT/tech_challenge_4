"""Padrões de movimentação do paciente durante a internação.

Entrada: série por minuto de atividade no leito (contagens de acelerômetro/sensor
de pressão do colchão ou energia de movimento extraída do vídeo pela pose).

Anomalias detectadas:
  * Imobilidade prolongada — sem reposicionamento por mais de N minutos
    (risco de lesão por pressão).
  * Agitação noturna — atividade sustentada muito acima da linha de base noturna
    do próprio paciente (sinal precoce de delirium).
  * Possível queda / saída do leito — pico extremo de atividade seguido de
    ausência total de sinal.
  * Escore horário de atipicidade — Isolation Forest sobre features horárias
    normalizadas por período (dia/noite), exibido no aplicativo como apoio visual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity
from .utils import mask_to_events, robust_z


@dataclass
class MovementResult:
    patient_id: str
    frame: pd.DataFrame
    events: list[dict] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    hourly: pd.DataFrame | None = None

    @property
    def risk_score(self) -> float:
        pts = {"queda": 1.0, "agitacao": 0.6, "imobilidade": 0.5}
        return max((pts[e["type"]] for e in self.events), default=0.0)


def _is_night(hours: pd.Series, night: list[int]) -> pd.Series:
    start, end = night
    return (hours >= start) | (hours < end) if start > end else (hours >= start) & (hours < end)


def detect_movement_anomalies(df: pd.DataFrame, cfg: dict, patient_id: str | None = None) -> MovementResult:
    mcfg = cfg["movement"]
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    pid = patient_id or str(df["patient_id"].iloc[0])
    act = df["activity"].astype(float)
    df["night"] = _is_night(df["timestamp"].dt.hour, mcfg["night_hours"])
    events: list[dict] = []

    # imobilidade prolongada
    immobile = act < mcfg["inactivity_threshold"]
    df["immobile"] = immobile
    for s, e in mask_to_events(immobile.to_numpy()):
        if e - s + 1 >= mcfg["max_immobile_minutes"]:
            events.append({"type": "imobilidade", "start": s, "end": e, "minutes": e - s + 1})

    # agitação noturna (linha de base robusta da própria noite do paciente)
    smooth = act.rolling(15, min_periods=1).mean()
    night_vals = smooth[df["night"]]
    med = night_vals.median()
    mad = (night_vals - med).abs().median()
    df["agitation_z"] = robust_z(smooth, med, mad, floor=2.0)
    agit = (df["agitation_z"] > mcfg["agitation_zscore"]) & df["night"]
    for s, e in mask_to_events(agit.to_numpy(), min_len=mcfg["agitation_min_minutes"], merge_gap=10):
        events.append({"type": "agitacao", "start": s, "end": e, "minutes": e - s + 1,
                       "peak": float(act.iloc[s:e + 1].max())})

    # possível queda: pico extremo seguido de silêncio total (paciente fora do leito)
    gmed = act.median()
    gmad = (act - gmed).abs().median()
    df["spike_z"] = robust_z(act, gmed, gmad, floor=3.0)
    still = mcfg["fall_stillness_minutes"]
    for i in np.where(df["spike_z"] > mcfg["fall_spike_zscore"])[0]:
        after = act.iloc[i + 1:i + 1 + still]
        if len(after) == still and (after <= 0.5).all():
            if not any(ev["type"] == "queda" and abs(ev["start"] - i) < 30 for ev in events):
                events.append({"type": "queda", "start": int(i), "end": int(i + still), "minutes": still,
                               "peak": float(act.iloc[i])})

    # Isolation Forest em janelas horárias — escore contínuo para o aplicativo
    # (não dispara alerta: com poucas horas por paciente gera falsos positivos)
    hourly = df.set_index("timestamp")["activity"].resample("1h")
    feats = pd.DataFrame({
        "mean": hourly.mean(), "std": hourly.std().fillna(0), "max": hourly.max(),
        "immobile_frac": df.set_index("timestamp")["immobile"].resample("1h").mean(),
    }).dropna()
    period = _is_night(pd.Series(feats.index.hour, index=feats.index), mcfg["night_hours"])
    feats["atypicality"] = 0.0
    if len(feats) >= 12:
        # normaliza dentro do período (dia x noite): noite calma não deve parecer "atípica"
        norm = feats.drop(columns="atypicality").groupby(period).transform(
            lambda c: (c - c.median()) / (c.std() + 1e-6))
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(norm)
        feats["atypicality"] = -iso.decision_function(norm)

    df["anomaly"] = False
    for ev in events:
        df.loc[ev["start"]:ev["end"], "anomaly"] = True
        ev["start_ts"] = str(df["timestamp"].iloc[ev["start"]])
        ev["end_ts"] = str(df["timestamp"].iloc[ev["end"]])

    events = sorted(events, key=lambda e: e["start"])
    return MovementResult(pid, df, events, _alerts(pid, events, df), hourly=feats)


_TEXT = {
    "imobilidade": (Severity.WARNING, "Imobilidade prolongada no leito",
                    "Sem reposicionamento por {minutes} min.",
                    "Reposicionar o paciente e reavaliar escala de Braden / pele."),
    "agitacao": (Severity.WARNING, "Agitação noturna",
                 "Atividade noturna sustentada {minutes} min acima da linha de base (pico {peak:.0f}).",
                 "Avaliar delirium (CAM-ICU), dor, hipóxia e retenção urinária."),
    "queda": (Severity.CRITICAL, "Possível queda / saída do leito",
              "Pico de movimento ({peak:.0f}) seguido de ausência de sinal no leito por {minutes} min.",
              "Verificar o paciente imediatamente no quarto."),
}


def _alerts(pid: str, events: list[dict], df: pd.DataFrame) -> list[Alert]:
    out = []
    for ev in events:
        sev, title, desc, action = _TEXT[ev["type"]]
        out.append(Alert(pid, "movement", sev, title, desc.format(**{"peak": 0, **ev}),
                         pd.Timestamp(df["timestamp"].iloc[ev["start"]]).isoformat(),
                         evidence={"inicio": ev["start_ts"], "fim": ev["end_ts"], "minutos": ev["minutes"]},
                         recommended_action=action))
    return out
