"""Gerador de dados clínicos sintéticos com anomalias rotuladas (ground truth).

Cinco pacientes, 24h de internação com amostragem por minuto. Cada um representa
um cenário clínico que exercita detectores diferentes e a fusão multimodal:

  P001  controle estável — só contém um artefato de oxímetro (não deve alertar)
  P002  sepse — febre, taquicardia, hipotensão, taquipneia; duplicidade de anticoagulante;
        agitação noturna (delirium)
  P003  depressão respiratória por opioide — morfina 5x + midazolam; bradipneia e
        dessaturação na hora seguinte; fala com dispneia na consulta
  P004  arritmia — episódios de taquicardia paroxística, KCl acima da dose máxima;
        fala disártrica (suspeita de AVC cardioembólico)
  P005  pós-operatório ortopédico — imobilidade prolongada, queda noturna com
        taquicardia de dor, insulina triplicada de madrugada; fala com fadiga
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-10-28 08:00")
MINUTES = 24 * 60

PATIENTS = {
    "P001": {"nome": "Ana Souza", "idade": 54, "leito": "UTI-01", "cenario": "Controle estável"},
    "P002": {"nome": "Bruno Lima", "idade": 71, "leito": "UTI-02", "cenario": "Sepse de foco pulmonar"},
    "P003": {"nome": "Carla Mendes", "idade": 63, "leito": "ENF-12", "cenario": "Pós-operatório abdominal em analgesia com opioide"},
    "P004": {"nome": "Diego Rocha", "idade": 68, "leito": "UCO-03", "cenario": "Fibrilação atrial paroxística"},
    "P005": {"nome": "Elisa Prado", "idade": 79, "leito": "ENF-07", "cenario": "Pós-operatório de artroplastia de quadril"},
}


def _ar1(rng, n, sd, phi=0.9):
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def _trapezoid(n, start, ramp, plateau, ramp_down=None):
    """Perfil 0→1→0 (início em minutos, rampa de subida, platô, rampa de descida)."""
    ramp_down = ramp if ramp_down is None else ramp_down
    p = np.zeros(n)
    t = np.arange(n)
    up = (t >= start) & (t < start + ramp)
    p[up] = (t[up] - start) / max(ramp, 1)
    flat = (t >= start + ramp) & (t < start + ramp + plateau)
    p[flat] = 1
    dn0 = start + ramp + plateau
    down = (t >= dn0) & (t < dn0 + ramp_down)
    p[down] = 1 - (t[down] - dn0) / max(ramp_down, 1)
    return p


def _baseline_vitals(rng, n):
    t = np.arange(n)
    hours = (START.hour + t / 60) % 24
    circ = np.sin(2 * np.pi * (hours - 10) / 24)  # pico à tarde, vale de madrugada
    return {
        "hr": 74 + 6 * circ + _ar1(rng, n, 2.5),
        "sbp": 122 + 6 * circ + _ar1(rng, n, 3.5),
        "dbp": 76 + 4 * circ + _ar1(rng, n, 2.5),
        "spo2": 97 + _ar1(rng, n, 0.6, 0.7),
        "rr": 16 + 1.0 * circ + _ar1(rng, n, 0.8, 0.8),
        "temp": 36.7 + 0.3 * circ + _ar1(rng, n, 0.06, 0.95),
    }


def _inject(v, label, name, profile, deltas: dict):
    for k, d in deltas.items():
        v[k] = v[k] + d * profile
    label[(profile > 0.3) & (label == "")] = name


def generate_vitals(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = MINUTES
    frames = []
    for pid in PATIENTS:
        v = _baseline_vitals(rng, n)
        label = np.array([""] * n, dtype=object)
        h = 60  # minutos por hora

        if pid == "P001":
            # artefato: oxímetro desconectado por 1 minuto (deve ser filtrado)
            v["spo2"][9 * h] = 62
            label[9 * h] = "artifact"
        elif pid == "P002":
            _inject(v, label, "febre_sepse", _trapezoid(n, 10 * h, 150, 300, 120), {"temp": 2.4, "hr": 28, "rr": 7})
            _inject(v, label, "hipotensao_sepse", _trapezoid(n, 13 * h, 45, 150, 60), {"sbp": -38, "dbp": -22, "hr": 12})
        elif pid == "P003":
            _inject(v, label, "depressao_respiratoria", _trapezoid(n, 14 * h, 40, 150, 60),
                    {"rr": -9, "spo2": -9, "hr": -10, "sbp": -12})
        elif pid == "P004":
            for start, dur in [(3 * h, 25), (11 * h + 20, 40), (18 * h, 30)]:
                prof = _trapezoid(n, start, 2, dur, 3)  # início/fim abruptos
                _inject(v, label, "taquicardia_paroxistica", prof, {"hr": 72})
                v["hr"] += rng.normal(0, 9, n) * prof  # ritmo irregular durante o episódio
            _inject(v, label, "bradicardia", _trapezoid(n, 21 * h, 10, 25, 10), {"hr": -30})
        elif pid == "P005":
            _inject(v, label, "dor_pos_queda", _trapezoid(n, 19 * h + 1, 3, 40, 30), {"hr": 32, "sbp": 25, "rr": 6})
            # deriva lenta de SpO2 por atelectasia (sutil, multivariada)
            _inject(v, label, "atelectasia", _trapezoid(n, 6 * h, 120, 120, 120), {"spo2": -5, "rr": 5, "temp": 0.6})

        df = pd.DataFrame({
            "timestamp": START + pd.to_timedelta(np.arange(n), unit="min"),
            "patient_id": pid,
            "hr": np.round(v["hr"]).astype(int),
            "sbp": np.round(v["sbp"]).astype(int),
            "dbp": np.round(v["dbp"]).astype(int),
            "spo2": np.clip(np.round(v["spo2"]), 50, 100).astype(int),
            "rr": np.round(v["rr"]).clip(4).astype(int),
            "temp": np.round(v["temp"], 1),
            "anomaly_type": label,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_prescriptions() -> pd.DataFrame:
    def at(hours, minutes=0):
        return START + pd.Timedelta(hours=hours, minutes=minutes)

    rows = [
        # pid, t, droga, dose, freq/dia, via, ação, prescritor, anomalia esperada
        ("P001", at(0, 30), "dipirona", 1000, 4, "IV", "start", "Dr. Alves", False),
        ("P001", at(0, 30), "enoxaparina", 40, 1, "SC", "start", "Dr. Alves", False),
        ("P001", at(6), "paracetamol", 750, 4, "VO", "start", "Dr. Alves", False),
        ("P001", at(12), "dipirona", 1000, 3, "IV", "change", "Dra. Reis", False),

        ("P002", at(0, 15), "dipirona", 1000, 4, "IV", "start", "Dr. Alves", False),
        ("P002", at(2), "enoxaparina", 40, 1, "SC", "start", "Dr. Alves", False),
        ("P002", at(10, 30), "vancomicina", 1000, 2, "IV", "start", "Dra. Reis", False),
        ("P002", at(11), "heparina", 5000, 3, "SC", "start", "Dr. Nunes", True),     # duplicidade anticoagulante
        ("P002", at(16), "vancomicina", 2000, 2, "IV", "change", "Dr. Nunes", True),  # dose dobrada

        ("P003", at(0), "morfina", 2, 6, "IV", "start", "Dra. Costa", False),
        ("P003", at(0), "dipirona", 1000, 4, "IV", "start", "Dra. Costa", False),
        ("P003", at(4), "ondansetrona", 4, 3, "IV", "start", "Dra. Costa", False),
        ("P003", at(13), "morfina", 10, 6, "IV", "change", "Dr. Nunes", True),       # 5x a dose
        ("P003", at(13, 30), "midazolam", 5, 4, "IV", "start", "Dr. Nunes", True),    # interação com opioide

        ("P004", at(0), "amiodarona", 200, 3, "VO", "start", "Dr. Faria", False),
        ("P004", at(0), "enoxaparina", 80, 2, "SC", "start", "Dr. Faria", False),
        ("P004", at(5), "cloreto de potassio", 1000, 3, "VO", "start", "Dr. Faria", False),
        ("P004", at(12), "cloreto de potassio", 2000, 4, "IV", "change", "Dr. Nunes", True),  # acima do máx.
        ("P004", at(12, 30), "amiodarona", 400, 3, "VO", "change", "Dr. Nunes", True),

        ("P005", at(0), "paracetamol", 750, 4, "VO", "start", "Dra. Reis", False),
        ("P005", at(0), "insulina", 10, 3, "SC", "start", "Dra. Reis", False),
        ("P005", at(0), "enoxaparina", 40, 1, "SC", "start", "Dra. Reis", False),
        ("P005", at(8), "tramadol", 50, 4, "VO", "start", "Dra. Reis", False),
        ("P005", at(19, 30), "insulina", 30, 3, "SC", "change", "Dr. Nunes", True),   # 3x de madrugada
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "timestamp", "drug", "dose", "frequency_per_day",
                                       "route", "action", "prescriber", "expected_anomaly"])
    df["unit"] = np.where(df["drug"].isin(["insulina", "heparina"]), "UI", "mg")
    return df


def generate_movement(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = MINUTES
    ts = START + pd.to_timedelta(np.arange(n), unit="min")
    hours = ts.hour.to_numpy()
    night = (hours >= 22) | (hours < 6)
    frames = []
    for pid in PATIENTS:
        lam = np.where(night, 2.5, 14.0)
        act = rng.poisson(lam).astype(float)
        # bursts de atividade diurna (refeições, banho, visitas)
        for s in rng.choice(np.where(~night)[0], 12, replace=False):
            act[s:s + 5] += rng.integers(25, 60)
        # reposicionamento pela enfermagem a cada ~2h durante a noite
        for s in np.where(night)[0][::110]:
            act[s:s + 3] += rng.integers(30, 60)
        label = np.array([""] * n, dtype=object)

        if pid == "P002":
            s, e = 17 * 60, 19 * 60 + 40  # 01:00–03:40: agitação (delirium)
            act[s:e] = rng.poisson(30, e - s) + rng.integers(0, 40, e - s)
            label[s:e] = "agitacao_noturna"
        elif pid == "P005":
            s, e = 2 * 60, 7 * 60 + 30  # 10:00–15:30: sem reposicionamento após cirurgia
            act[s:e] = rng.poisson(0.4, e - s)
            label[s:e] = "imobilidade"
            f = 19 * 60  # 03:00: queda
            act[f - 3:f] = [40, 90, 120]
            act[f] = 260
            act[f + 1:f + 26] = 0
            label[f - 3:f + 26] = "queda"
        frames.append(pd.DataFrame({"timestamp": ts, "patient_id": pid, "activity": act, "anomaly_type": label}))
    return pd.concat(frames, ignore_index=True)


def write_all(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "vitals": out / "vitals.csv",
        "prescriptions": out / "prescriptions.csv",
        "movement": out / "movement.csv",
        "patients": out / "patients.csv",
    }
    generate_vitals().to_csv(paths["vitals"], index=False)
    generate_prescriptions().to_csv(paths["prescriptions"], index=False)
    generate_movement().to_csv(paths["movement"], index=False)
    pd.DataFrame.from_dict(PATIENTS, orient="index").rename_axis("patient_id").to_csv(paths["patients"])
    return paths


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "anomaly_type" in df:
        df["anomaly_type"] = df["anomaly_type"].fillna("").astype(str)
    return df
