"""Carregador de dados reais do PhysioNet.

BIDMC PPG and Respiration Dataset (Pimentel et al., 2016) — 53 registros de 8 min de
pacientes críticos (MIMIC-II), com séries numéricas a 1 Hz de FC, pulso, FR e SpO2.
Acesso aberto (ODC-BY 1.0): https://physionet.org/content/bidmc/1.0.0/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DB = "bidmc/1.0.0"
COLUMN_MAP = {"HR": "hr", "RESP": "rr", "SpO2": "spo2", "PULSE": "pulse"}


def load_bidmc(record: int | str = 1, cache_dir: str | Path = "data/physionet",
               start: str = "2026-10-28 08:00") -> pd.DataFrame:
    import wfdb

    name = record if isinstance(record, str) else f"bidmc{int(record):02d}"
    cache = Path(cache_dir) / f"{name}n.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        return df
    rec = wfdb.rdrecord(f"{name}n", pn_dir=DB)
    df = pd.DataFrame(rec.p_signal, columns=[c.strip().rstrip(",") for c in rec.sig_name])
    df = df.rename(columns=COLUMN_MAP)[[c for c in COLUMN_MAP.values() if c in df.rename(columns=COLUMN_MAP)]]
    fs = rec.fs or 1
    df.insert(0, "timestamp", pd.Timestamp(start) + pd.to_timedelta(df.index / fs, unit="s"))
    df.insert(1, "patient_id", name.upper())
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def physionet_config(cfg: dict) -> dict:
    """Ajusta parâmetros temporais para séries a 1 Hz (8 minutos)."""
    c = {**cfg, "vitals": {**cfg["vitals"]}}
    c["vitals"].update(sampling_minutes=1 / 60, baseline_hours=2 / 60, rolling_window=60,
                       min_persistence=10, merge_gap=20)
    return c
