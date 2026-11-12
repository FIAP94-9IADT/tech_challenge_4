"""Funções auxiliares compartilhadas pelos detectores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mask_to_events(mask, min_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Converte uma máscara booleana em eventos (início, fim) inclusivos.

    Eventos separados por até `merge_gap` amostras são unidos; eventos com
    menos de `min_len` amostras positivas consecutivas são descartados.
    """
    m = np.asarray(mask, dtype=bool)
    events: list[list[int]] = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                if events and i - events[-1][1] - 1 <= merge_gap:
                    events[-1][1] = j
                else:
                    events.append([i, j])
            i = j + 1
        else:
            i += 1
    return [(s, e) for s, e in events]


def persistent(mask, min_len: int) -> np.ndarray:
    """Mantém apenas trechos positivos com pelo menos `min_len` amostras consecutivas."""
    out = np.zeros(len(mask), dtype=bool)
    for s, e in mask_to_events(mask, min_len=min_len):
        out[s:e + 1] = True
    return out


def robust_z(x: pd.Series, center: pd.Series | float, scale: pd.Series | float, floor: float) -> pd.Series:
    """z-score robusto (mediana/MAD) com piso de escala para séries muito estáveis."""
    if isinstance(scale, pd.Series):
        mad = scale.clip(lower=floor)
    else:
        mad = max(float(scale), floor)
    return 0.6745 * (x - center) / mad


def rolling_robust_z(x: pd.Series, window: int, floor: float) -> pd.Series:
    """z-score robusto contra a janela anterior (exclui a amostra atual)."""
    prev = x.shift(1)
    med = prev.rolling(window, min_periods=max(5, window // 3)).median()
    mad = (prev - med).abs().rolling(window, min_periods=max(5, window // 3)).median()
    return robust_z(x, med, mad, floor)


def event_overlap_metrics(pred_mask, true_mask, min_len: int = 1, merge_gap: int = 0) -> dict:
    """Métricas por evento: um evento real é detectado se algum evento previsto o sobrepõe."""
    pred_ev = mask_to_events(pred_mask, min_len=min_len, merge_gap=merge_gap)
    true_ev = mask_to_events(true_mask)

    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    tp_true = sum(any(overlaps(t, p) for p in pred_ev) for t in true_ev)
    tp_pred = sum(any(overlaps(p, t) for t in true_ev) for p in pred_ev)
    recall = tp_true / len(true_ev) if true_ev else float("nan")
    precision = tp_pred / len(pred_ev) if pred_ev else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if pred_ev and true_ev and (precision + recall) > 0 else float("nan"))

    # atraso médio de detecção (amostras entre início real e primeira detecção sobreposta)
    delays = []
    for t in true_ev:
        starts = [max(p[0], t[0]) for p in pred_ev if overlaps(t, p)]
        if starts:
            delays.append(min(starts) - t[0])
    pm, tm = np.asarray(pred_mask, bool), np.asarray(true_mask, bool)
    return {
        "eventos_reais": len(true_ev),
        "eventos_previstos": len(pred_ev),
        "eventos_detectados": tp_true,
        "recall_evento": round(recall, 3) if recall == recall else None,
        "precisao_evento": round(precision, 3) if precision == precision else None,
        "f1_evento": round(f1, 3) if f1 == f1 else None,
        "atraso_medio_amostras": round(float(np.mean(delays)), 1) if delays else None,
        "falsos_positivos_amostra": int((pm & ~tm).sum()),
    }
