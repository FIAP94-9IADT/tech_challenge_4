"""Detecção de alterações inesperadas na evolução das prescrições.

Cada evento de prescrição (início, alteração ou suspensão) é transformado em
features e avaliado por:
  * Regras farmacológicas — salto abrupto de dose (≥2x ou ≤½), dose diária acima
    do máximo de referência, duplicidade terapêutica, interação medicamentosa,
    excesso de alterações em 24h e mudanças noturnas em medicamentos de alta vigilância
    (lista ISMP).
  * Isolation Forest — escore de "atipicidade" do evento frente ao padrão de prescrição
    do serviço (complementa regras quando nenhuma é violada isoladamente).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity


@dataclass
class PrescriptionResult:
    frame: pd.DataFrame
    alerts: list[Alert] = field(default_factory=list)

    def risk_score(self, patient_id: str) -> float:
        sub = self.frame[self.frame["patient_id"] == patient_id]
        if sub.empty:
            return 0.0
        return float(min(1.0, sub["rule_points"].max() / 3.0))


def _class_of(drug: str, classes: dict) -> str | None:
    for cls, members in classes.items():
        if drug in members:
            return cls
    return None


def detect_prescription_anomalies(df: pd.DataFrame, cfg: dict) -> PrescriptionResult:
    pcfg = cfg["prescriptions"]
    high_alert = set(pcfg["high_alert_drugs"])
    max_daily = pcfg["max_daily_dose"]
    classes = pcfg["therapeutic_classes"]
    interactions = [tuple(p) for p in pcfg["interactions"]]
    n0, n1 = pcfg["night_hours"]

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True).copy()
    rows = []
    for pid, g in df.groupby("patient_id", sort=False):
        active: dict[str, dict] = {}
        history: dict[str, list[pd.Timestamp]] = {}
        for idx, r in g.iterrows():
            drug, ts = r["drug"], pd.Timestamp(r["timestamp"])
            daily = float(r["dose"]) * float(r["frequency_per_day"])
            unit = r.get("unit", "mg")
            prev = active.get(drug)
            ratio = daily / prev["daily"] if prev and prev["daily"] > 0 else 1.0
            history.setdefault(drug, []).append(ts)
            changes_24h = sum(1 for t in history[drug] if ts - t <= pd.Timedelta(hours=24))
            if r["action"] == "stop":
                active.pop(drug, None)
            else:
                active[drug] = {"daily": daily, "since": ts}

            findings, points = [], 0
            if r["action"] != "stop":
                lim = max_daily.get(drug)
                if lim and daily > lim:
                    findings.append(f"dose diária {daily:g} {unit} acima do máximo de referência ({lim:g} {unit})")
                    points += 3
                if prev and (ratio >= pcfg["dose_change_ratio"] or ratio <= 1 / pcfg["dose_change_ratio"]):
                    findings.append(f"alteração abrupta de dose ({prev['daily']:g} → {daily:g} {unit}/dia, {ratio:.1f}x)")
                    points += 2 if drug in high_alert else 1
                cls = _class_of(drug, classes)
                if cls:
                    dup = [d for d in active if d != drug and _class_of(d, classes) == cls]
                    if dup:
                        findings.append(f"duplicidade terapêutica ({cls}: {drug} + {', '.join(dup)})")
                        points += 2 if cls == "anticoagulante" else 1
                for a, b in interactions:
                    other = b if drug == a else a if drug == b else None
                    if other and other in active:
                        findings.append(f"interação potencial {drug} × {other}")
                        points += 1
            if changes_24h > pcfg["max_changes_24h"]:
                findings.append(f"{changes_24h} alterações de {drug} em 24h")
                points += 1
            is_night = n0 <= ts.hour < n1
            if is_night and drug in high_alert and r["action"] != "start" and findings:
                findings.append("alteração em horário noturno")

            rows.append({
                **r.to_dict(), "daily_dose": daily, "dose_ratio": round(ratio, 2),
                "changes_24h": changes_24h, "is_night": is_night, "high_alert": drug in high_alert,
                "frac_max_dose": daily / max_daily[drug] if drug in max_daily else 0.0,
                "findings": findings, "rule_points": points,
            })

    out = pd.DataFrame(rows)
    feats = pd.DataFrame({
        "log_ratio": np.abs(np.log(out["dose_ratio"].clip(lower=1e-3))),
        "changes_24h": out["changes_24h"],
        "is_night": out["is_night"].astype(int),
        "high_alert": out["high_alert"].astype(int),
        "frac_max_dose": out["frac_max_dose"],
    })
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42).fit(feats)
    out["iforest_score"] = -iso.score_samples(feats)
    out["flag_iforest"] = iso.predict(feats) == -1
    out["anomaly"] = (out["rule_points"] > 0) | (out["flag_iforest"] & out["high_alert"])

    alerts = []
    for _, r in out[out["anomaly"]].iterrows():
        pts = r["rule_points"]
        sev = Severity.CRITICAL if pts >= 3 or (pts >= 2 and r["high_alert"]) else (
            Severity.WARNING if pts >= 1 else Severity.INFO)
        findings = r["findings"] or [f"evento atípico frente ao padrão de prescrição (escore {r['iforest_score']:.2f})"]
        alerts.append(Alert(
            patient_id=r["patient_id"], modality="prescription", severity=sev,
            title=f"Prescrição atípica: {r['drug']}" + (" (alta vigilância)" if r["high_alert"] else ""),
            description=_sentence("; ".join(findings)) + f". Prescritor: {r.get('prescriber', 'n/d')}.",
            timestamp=pd.Timestamp(r["timestamp"]).isoformat(),
            evidence={"medicamento": r["drug"], "dose": f"{r['dose']:g} {r.get('unit', 'mg')}", "freq_dia": r["frequency_per_day"],
                      "dose_diaria": r["daily_dose"], "razao": r["dose_ratio"], "acao": r["action"]},
            recommended_action="Dupla checagem farmacêutica e confirmação com o prescritor antes da administração.",
        ))
    return PrescriptionResult(out, alerts)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]
