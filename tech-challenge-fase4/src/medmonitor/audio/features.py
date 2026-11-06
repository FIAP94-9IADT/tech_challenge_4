"""Extração de biomarcadores vocais (acústicos) com librosa.

Os biomarcadores escolhidos seguem a literatura de análise de voz para triagem
(fadiga, dispneia, disartria). São aproximações em nível de quadro — não
substituem medidas clínicas (ex.: Praat) nem avaliação fonoaudiológica:

  Prosódia/ritmo   taxa de fala (sílabas/s), taxa de articulação, razão de pausas,
                   pausas por minuto, duração média de segmentos de fonação
  Fonte glótica    F0 mediana, variabilidade de F0 (semitons), jitter e shimmer
                   aproximados, razão harmônico/ruído (HNR) aproximada
  Energia          inclinação da intensidade ao longo da gravação (dB/min)
  Respiração       inspirações audíveis (trechos não vozeados energéticos >= 250 ms)
  Tremor           índice de tremor vocal (modulação de F0 em 3,5–7,5 Hz)
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import find_peaks

HOP = 160  # 10 ms em 16 kHz
FRAME = 640


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(y)) if y.size else 0
    return (y / peak * 0.95 if peak > 0 else y), sr


def _scale(x: float, lo: float, hi: float) -> float:
    """Mapeia x linearmente para 0-1 entre lo (normal) e hi (alterado)."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def extract_features(y: np.ndarray, sr: int) -> dict:
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # segmentos de fala x pausas
    intervals = librosa.effects.split(y, top_db=32, frame_length=FRAME, hop_length=HOP)
    seg_durs = (intervals[:, 1] - intervals[:, 0]) / sr if len(intervals) else np.array([])
    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sr if len(intervals) > 1 else np.array([])
    pauses = gaps[gaps >= 0.18]
    span = (intervals[-1, 1] - intervals[0, 0]) / sr if len(intervals) else duration
    phonation = float(seg_durs.sum())

    # F0 (pYIN) e medidas de perturbação aproximadas
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    f0v = f0[voiced & ~np.isnan(f0)]
    if f0v.size > 10:
        semis = 12 * np.log2(f0v / np.median(f0v))
        f0_std_st = float(np.std(semis))
        consecutive = voiced[1:] & voiced[:-1] & ~np.isnan(f0[1:]) & ~np.isnan(f0[:-1])
        periods = 1 / f0
        jitter = float(np.mean(np.abs(np.diff(periods))[consecutive]) / np.nanmean(periods[voiced]))
        amp = rms[: len(voiced)]
        n = min(len(amp), len(voiced)) - 1
        cons_amp = consecutive[:n]
        shimmer = float(np.mean(np.abs(np.diff(amp[: n + 1]))[cons_amp]) / (np.mean(amp[: n + 1][voiced[: n + 1]]) + 1e-9))
        f0_median = float(np.median(f0v))
    else:
        f0_std_st, jitter, shimmer, f0_median = 0.0, 0.0, 0.0, 0.0

    # HNR aproximada via separação harmônico/percussivo
    harm = librosa.effects.harmonic(y)
    noise = y - harm
    hnr = float(10 * np.log10((np.sum(harm ** 2) + 1e-9) / (np.sum(noise ** 2) + 1e-9)))

    # núcleos silábicos: picos de intensidade em trechos vozeados (de Jong & Wempe, 2009)
    env = librosa.feature.rms(y=librosa.effects.preemphasis(y), frame_length=FRAME, hop_length=HOP)[0]
    env_db = librosa.amplitude_to_db(env, ref=np.max)
    env_db = np.convolve(env_db, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(env_db, height=-28, prominence=4, distance=int(0.09 * sr / HOP))
    vmask = np.zeros(len(env_db), dtype=bool)
    vmask[: len(voiced)] = voiced[: len(env_db)]
    syllables = int(np.sum(vmask[peaks])) if peaks.size else 0

    # inclinação de energia (dB/min) nos trechos de fala
    speech_frames = rms_db > -35
    t = np.arange(len(rms_db)) * HOP / sr
    slope = float(np.polyfit(t[speech_frames], rms_db[speech_frames], 1)[0] * 60) if speech_frames.sum() > 20 else 0.0

    # respiração audível: trechos não vozeados com energia relevante e duração >= 0,25 s
    # (consoantes surdas duram tipicamente < 0,15 s; inspirações ruidosas, mais)
    nv = min(len(voiced), len(rms_db))
    breathy = (~voiced[:nv]) & (rms_db[:nv] > -40)
    breath_events = sum(1 for s_, e_ in _runs(breathy) if e_ - s_ + 1 >= int(0.25 * sr / HOP))

    # índice de tremor vocal: fração da potência da curva de F0 (semitons, sem tendência)
    # na faixa de 3,5–7,5 Hz, típica de tremor em disartrias neurológicas
    tremor_index = _f0_tremor_index(f0, voiced, sr / HOP)

    minutes = max(span / 60, 1e-6)
    return {
        "duration_s": round(duration, 2),
        "speech_rate_syl_s": round(syllables / max(span, 1e-6), 2),
        "articulation_rate_syl_s": round(syllables / max(phonation, 1e-6), 2),
        "pause_ratio": round(float(pauses.sum() / max(span, 1e-6)), 3),
        "pauses_per_min": round(len(pauses) / minutes, 1),
        "mean_pause_s": round(float(pauses.mean()), 2) if pauses.size else 0.0,
        "long_pauses": int(np.sum(pauses > 1.0)),
        "mean_phonation_segment_s": round(float(seg_durs.mean()), 2) if seg_durs.size else 0.0,
        "f0_median_hz": round(f0_median, 1),
        "f0_std_semitones": round(f0_std_st, 2),
        "jitter_approx": round(jitter, 4),
        "shimmer_approx": round(shimmer, 4),
        "hnr_approx_db": round(hnr, 2),
        "energy_slope_db_min": round(slope, 2),
        "breath_events_per_min": round(breath_events / minutes, 1),
        "f0_tremor_index": round(tremor_index, 3),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2] - 1))


def _f0_tremor_index(f0: np.ndarray, voiced: np.ndarray, fs: float) -> float:
    from scipy.signal import welch

    ok = voiced & ~np.isnan(f0)
    med = np.nanmedian(f0[ok]) if ok.any() else 0
    band, total = 0.0, 0.0
    for s, e in _runs(ok):
        if e - s + 1 < 30:  # trechos vozeados >= 300 ms
            continue
        seg = 12 * np.log2(f0[s:e + 1] / med)
        x = np.arange(len(seg))
        seg = seg - np.polyval(np.polyfit(x, seg, 2), x)
        fr, pw = welch(seg, fs=fs, nperseg=min(64, len(seg)))
        band += pw[(fr >= 3.5) & (fr <= 7.5)].sum()
        total += pw[fr >= 1].sum()
    return float(band / total) if total > 0 else 0.0


def vocal_indicators(feat: dict, ref: dict) -> dict:
    """Converte features em indicadores 0-1 (0 = compatível com fala típica)."""
    rate_lo, _ = ref["speech_rate_syl_s"]
    slow = _scale(-feat["speech_rate_syl_s"], -rate_lo, -rate_lo * 0.55)
    monotone = _scale(-feat["f0_std_semitones"], -ref["f0_std_semitones_min"], -ref["f0_std_semitones_min"] * 0.4)
    energy_decline = _scale(-feat["energy_slope_db_min"], 2, 12)
    long_pauses = _scale(feat["mean_pause_s"], 0.6, 1.5)

    frequent_pauses = _scale(feat["pauses_per_min"], 12, 30)
    short_phonation = _scale(-feat["mean_phonation_segment_s"], -0.65, -0.35)
    breath = _scale(feat["breath_events_per_min"], 4, 15)

    artic = _scale(-feat["articulation_rate_syl_s"], -rate_lo * 1.1, -rate_lo * 0.6)
    tremor = _scale(feat["f0_tremor_index"], 0.15, 0.27)
    jitter = _scale(feat["jitter_approx"], ref["jitter_max"], ref["jitter_max"] * 2.5)
    shimmer = _scale(feat["shimmer_approx"], ref["shimmer_max"], ref["shimmer_max"] * 2.5)
    low_hnr = _scale(-feat["hnr_approx_db"], -4, 0)

    return {
        "fadiga": round(float(np.mean([slow, monotone, energy_decline, long_pauses])), 2),
        "dificuldade_respiratoria": round(float(0.3 * frequent_pauses + 0.3 * short_phonation + 0.4 * breath), 2),
        "disartria": round(float(0.3 * artic + 0.3 * tremor + 0.2 * low_hnr + 0.1 * jitter + 0.1 * shimmer), 2),
        "_componentes": {
            "fala_lenta": round(slow, 2), "monotonia": round(monotone, 2), "queda_energia": round(energy_decline, 2),
            "pausas_longas": round(long_pauses, 2), "pausas_frequentes": round(frequent_pauses, 2),
            "fonacao_curta": round(short_phonation, 2), "respiracao_audivel": round(breath, 2),
            "articulacao_lenta": round(artic, 2), "tremor_vocal": round(tremor, 2), "jitter": round(jitter, 2), "shimmer": round(shimmer, 2),
            "hnr_baixa": round(low_hnr, 2),
        },
    }
