"""Detecção de anomalias em sinais vitais (FC, PA, SpO2, FR, temperatura).

Três detectores complementares + ensemble:
  1. Regras clínicas — escore NEWS2 (Royal College of Physicians, 2017) e limites absolutos.
  2. z-score robusto (mediana/MAD) — contra a janela móvel recente (mudanças abruptas)
     e contra a linha de base do próprio paciente (deriva lenta, ex.: febre).
  3. Isolation Forest multivariado — treinado na linha de base do paciente; capta
     combinações incomuns entre sinais mesmo quando cada um isoladamente parece normal.

O ensemble é uma votação por maioria (2 de 3 detectores) e aplica persistência mínima para filtrar artefatos de sensor (ex.: oxímetro deslocado).
Também é fornecido um monitor incremental (`StreamingVitalsMonitor`) para uso em tempo real.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity
from .utils import mask_to_events, persistent, robust_z, rolling_robust_z

VITALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
VITAL_NAMES = {"hr": "FC", "sbp": "PAS", "dbp": "PAD", "spo2": "SpO2", "rr": "FR", "temp": "Temp"}
VITAL_UNITS = {"hr": "bpm", "sbp": "mmHg", "dbp": "mmHg", "spo2": "%", "rr": "irpm", "temp": "°C"}
# piso da escala robusta: evita z enormes quando o sinal é quase constante
MAD_FLOOR = {"hr": 2.0, "sbp": 3.0, "dbp": 2.0, "spo2": 0.7, "rr": 1.0, "temp": 0.1}

# (nome, direção) -> rótulo clínico
CONDITION = {
    ("hr", "up"): "Taquicardia", ("hr", "down"): "Bradicardia",
    ("sbp", "up"): "Hipertensão", ("sbp", "down"): "Hipotensão",
    ("dbp", "up"): "Hipertensão diastólica", ("dbp", "down"): "Hipotensão diastólica",
    ("spo2", "up"): "SpO2 elevada", ("spo2", "down"): "Dessaturação",
    ("rr", "up"): "Taquipneia", ("rr", "down"): "Bradipneia",
    ("temp", "up"): "Febre", ("temp", "down"): "Hipotermia",
}


# --------------------------------------------------------------------------- NEWS2
def _band(value: float, bands: list[tuple[float, float, int]]) -> int:
    for lo, hi, pts in bands:
        if lo <= value <= hi:
            return pts
    return 0


NEWS2_BANDS = {
    "rr": [(-1e9, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 1e9, 3)],
    "spo2": [(-1e9, 91, 3), (92, 93, 2), (94, 95, 1), (96, 1e9, 0)],
    "sbp": [(-1e9, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 1e9, 3)],
    "hr": [(-1e9, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 1e9, 3)],
    "temp": [(-1e9, 35.0, 3), (35.05, 36.0, 1), (36.05, 38.0, 0), (38.05, 39.0, 1), (39.05, 1e9, 2)],
}


def news2_components(row) -> dict[str, int]:
    comps = {}
    for k, bands in NEWS2_BANDS.items():
        v = row.get(k)
        if v is not None and pd.notna(v):
            # arredonda como na prática clínica (inteiros, temperatura 1 casa)
            v = round(float(v), 1) if k == "temp" else round(float(v))
            comps[k] = _band(v, bands)
    return comps


def news2_score(row) -> tuple[int, int]:
    """Retorna (escore total, maior componente individual)."""
    c = news2_components(row)
    return sum(c.values()), max(c.values(), default=0)


def news2_risk(total: int, max_single: int) -> str:
    if total >= 7:
        return "alto"
    if total >= 5:
        return "médio"
    if max_single >= 3:
        return "baixo-médio"
    return "baixo"


# --------------------------------------------------------------------------- batch
@dataclass
class VitalsResult:
    patient_id: str
    frame: pd.DataFrame
    events: list[dict] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        """Escore 0-1 da modalidade (usado na fusão): maior NEWS2 dentro de eventos anômalos confirmados."""
        hit = self.frame.loc[self.frame["anomaly"], "news2"] if not self.frame.empty else []
        if len(hit) == 0:
            return 0.0
        n_events = len(self.events)
        return float(min(1.0, hit.max() / 9.0 + 0.05 * max(0, n_events - 1)))


def detect_vitals_anomalies(df: pd.DataFrame, cfg: dict, patient_id: str | None = None) -> VitalsResult:
    vcfg = cfg["vitals"]
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    pid = patient_id or str(df["patient_id"].iloc[0])
    cols = [c for c in VITALS if c in df.columns]
    df[cols] = df[cols].interpolate(limit_direction="both")

    n_base = max(30, int(vcfg["baseline_hours"] * 60 / vcfg["sampling_minutes"]))
    n_base = min(n_base, len(df) // 2)
    base = df.iloc[:n_base]

    # 1) regras clínicas
    scores = df.apply(news2_score, axis=1, result_type="expand")
    df["news2"], df["news2_max_single"] = scores[0], scores[1]
    hard = np.zeros(len(df), dtype=bool)
    for k, (lo, hi) in vcfg["hard_limits"].items():
        if k in df:
            hard |= (df[k] < lo) | (df[k] > hi)
    df["flag_hard_limit"] = hard
    df["flag_rules"] = hard | (df["news2"] >= 5) | (df["news2_max_single"] >= 3)

    # 2) z-score robusto (janela móvel e linha de base do paciente)
    zmax = np.zeros(len(df))
    for k in cols:
        zr = rolling_robust_z(df[k], vcfg["rolling_window"], MAD_FLOOR[k]).fillna(0)
        med = base[k].median()
        zb = robust_z(df[k], med, (base[k] - med).abs().median(), MAD_FLOOR[k])
        df[f"z_{k}"] = np.where(zr.abs() > zb.abs(), zr, zb)
        zmax = np.maximum(zmax, df[f"z_{k}"].abs().to_numpy())
    df["z_max"] = zmax
    df["flag_zscore"] = zmax > vcfg["zscore_threshold"]

    # 3) Isolation Forest multivariado (nível + variação de curto prazo)
    feats = df[cols].copy()
    for k in cols:
        feats[f"d_{k}"] = df[k].diff(5).fillna(0)
    model = IsolationForest(n_estimators=200, contamination=vcfg["iforest_contamination"], random_state=42)
    model.fit(feats.iloc[:n_base])
    df["iforest_score"] = -model.score_samples(feats)  # maior = mais anômalo
    df["flag_iforest"] = model.predict(feats) == -1

    # ensemble: maioria (2 de 3 detectores), com persistência mínima
    votes = df["flag_zscore"].astype(int) + df["flag_iforest"].astype(int) + df["flag_rules"].astype(int)
    df["votes"] = votes
    raw = votes >= 2
    df["anomaly"] = persistent(raw.to_numpy(), vcfg["min_persistence"])
    df["artifact_suspect"] = raw.to_numpy() & ~df["anomaly"].to_numpy()

    events, alerts = _build_events(df, base, cols, pid, vcfg)
    return VitalsResult(pid, df, events, alerts)


def _build_events(df, base, cols, pid, vcfg):
    events, alerts = [], []
    for s, e in mask_to_events(df["anomaly"].to_numpy(), merge_gap=vcfg["merge_gap"]):
        seg = df.iloc[s:e + 1]
        details = []
        for k in cols:
            if k == "dbp" and any(d["vital"] == "sbp" for d in details):
                continue  # PAD acompanha a PAS; evita rótulo redundante
            if seg[f"z_{k}"].abs().max() > vcfg["zscore_threshold"] or _violated(seg, k, vcfg):
                lo, hi = vcfg["hard_limits"].get(k, (-np.inf, np.inf))
                # o limite clínico violado define a direção (ex.: hipoxemia crônica com basal baixo)
                if (seg[k] < lo).any():
                    direction = "down"
                elif (seg[k] > hi).any():
                    direction = "up"
                else:
                    direction = "up" if seg[k].mean() > base[k].median() else "down"
                peak = seg[k].max() if direction == "up" else seg[k].min()
                details.append({"vital": k, "direction": direction, "peak": round(float(peak), 1),
                                "baseline": round(float(base[k].median()), 1)})
        labels = [CONDITION[(d["vital"], d["direction"])] for d in details] or ["Padrão multivariado atípico"]
        max_news = int(seg["news2"].max())
        hard_hit = bool(seg["flag_hard_limit"].any())
        severity = Severity.CRITICAL if (max_news >= 7 or (hard_hit and max_news >= 5)) else (
            Severity.WARNING if (max_news >= 5 or hard_hit or len(details) >= 2) else Severity.INFO)
        t0, t1 = seg["timestamp"].iloc[0], seg["timestamp"].iloc[-1]
        ev = {
            "patient_id": pid, "start": str(t0), "end": str(t1),
            "duration_min": round((t1 - t0).total_seconds() / 60 + 1, 1),
            "conditions": labels, "details": details, "max_news2": max_news,
            "news2_risk": news2_risk(max_news, int(seg["news2_max_single"].max())),
            "severity": severity.name,
        }
        events.append(ev)
        desc = "; ".join(
            f"{VITAL_NAMES[d['vital']]} {d['peak']} {VITAL_UNITS[d['vital']]} (basal {d['baseline']})" for d in details
        ) or "Combinação incomum de sinais vitais (Isolation Forest)."
        alerts.append(Alert(
            patient_id=pid, modality="vitals", severity=severity,
            title=" + ".join(labels),
            description=f"{desc}. NEWS2 máx. {max_news} (risco {ev['news2_risk']}), duração {ev['duration_min']} min.",
            timestamp=pd.Timestamp(t0).isoformat(),
            evidence={"inicio": str(t0), "fim": str(t1), "news2_max": max_news,
                      **{VITAL_NAMES[d["vital"]]: d["peak"] for d in details}},
            recommended_action=_action_for(labels, severity),
        ))
    return events, alerts


def _violated(seg, k, vcfg) -> bool:
    lim = vcfg["hard_limits"].get(k)
    return bool(lim and ((seg[k] < lim[0]) | (seg[k] > lim[1])).any())


def _action_for(labels: list[str], severity: Severity) -> str:
    joined = " ".join(labels)
    if "Dessaturação" in joined or "Bradipneia" in joined:
        return "Avaliar via aérea e padrão respiratório, checar posicionamento do oxímetro, considerar O2 e gasometria."
    if "Hipotensão" in joined:
        return "Reavaliar perfusão, balanço hídrico e medicações hipotensoras; considerar acionar time de resposta rápida."
    if "Febre" in joined:
        return "Investigar foco infeccioso (protocolo de sepse se critérios), coletar culturas conforme protocolo."
    if "Taquicardia" in joined or "Bradicardia" in joined:
        return "Realizar ECG de 12 derivações e avaliar causas (dor, febre, hipovolemia, arritmia)."
    if severity is Severity.CRITICAL:
        return "Acionar time de resposta rápida para avaliação à beira-leito."
    return "Reavaliar paciente e aumentar frequência de aferição dos sinais vitais."


# --------------------------------------------------------------------------- streaming
class StreamingVitalsMonitor:
    """Monitor incremental: recebe uma amostra por vez e emite alertas em tempo real.

    Usa, para cada sinal, z-score robusto contra a janela recente + linha de base
    aprendida no aquecimento, e NEWS2 por amostra. Um alerta só é emitido após
    `min_persistence` amostras consecutivas anômalas (reduz falsos alarmes).
    """

    def __init__(self, patient_id: str, cfg: dict):
        self.pid = patient_id
        self.vcfg = cfg["vitals"]
        self.window = {k: deque(maxlen=self.vcfg["rolling_window"]) for k in VITALS}
        self.warmup = max(30, int(self.vcfg["baseline_hours"] * 60 / self.vcfg["sampling_minutes"]))
        self.baseline: dict[str, tuple[float, float]] = {}
        self.base_buf: dict[str, list[float]] = {k: [] for k in VITALS}
        self.streak: dict[str, int] = {}
        self.n = 0

    def update(self, sample: dict) -> list[Alert]:
        self.n += 1
        alerts: list[Alert] = []
        ts = pd.Timestamp(sample["timestamp"]).isoformat()
        conditions: dict[str, tuple[str, float]] = {}

        for k in VITALS:
            v = sample.get(k)
            if v is None or pd.isna(v):
                continue
            v = float(v)
            if len(self.base_buf[k]) < self.warmup:
                self.base_buf[k].append(v)
                if len(self.base_buf[k]) == self.warmup:
                    arr = np.array(self.base_buf[k])
                    med = float(np.median(arr))
                    self.baseline[k] = (med, float(np.median(np.abs(arr - med))))
            z = 0.0
            if len(self.window[k]) >= 10:
                arr = np.array(self.window[k])
                med = float(np.median(arr))
                z = 0.6745 * (v - med) / max(float(np.median(np.abs(arr - med))), MAD_FLOOR[k])
            if k in self.baseline:
                bmed, bmad = self.baseline[k]
                zb = 0.6745 * (v - bmed) / max(bmad, MAD_FLOOR[k])
                z = zb if abs(zb) > abs(z) else z
            lim = self.vcfg["hard_limits"].get(k)
            out_of_limits = bool(lim and (v < lim[0] or v > lim[1]))
            if abs(z) > self.vcfg["zscore_threshold"] or out_of_limits:
                ref = self.baseline.get(k, (np.median(self.window[k]) if self.window[k] else v, 0))[0]
                conditions[k] = ("up" if v > ref else "down", v)
            self.window[k].append(v)

        total, single = news2_score({k: sample.get(k, np.nan) for k in NEWS2_BANDS})
        if total >= 5:
            conditions.setdefault("news2", ("up", total))

        for key in list(self.streak):
            if key not in conditions:
                self.streak[key] = 0
        for key, (direction, value) in conditions.items():
            self.streak[key] = self.streak.get(key, 0) + 1
            if self.streak[key] == self.vcfg["min_persistence"]:
                if key == "news2":
                    sev = Severity.CRITICAL if total >= 7 else Severity.WARNING
                    title, desc = f"NEWS2 elevado ({total})", f"Escore NEWS2 {total} (risco {news2_risk(total, single)})."
                else:
                    sev = Severity.CRITICAL if (total >= 7) else Severity.WARNING
                    title = CONDITION[(key, direction)]
                    desc = f"{VITAL_NAMES[key]} = {value:.1f} {VITAL_UNITS[key]} por {self.streak[key]} amostras consecutivas. NEWS2 atual {total}."
                alerts.append(Alert(self.pid, "vitals", sev, title, desc, ts,
                                    evidence={"news2": total, **{VITAL_NAMES.get(k, k): sample.get(k) for k in VITALS}},
                                    recommended_action=_action_for([title], sev)))
        return alerts
