"""Testes da lógica de vídeo e áudio sem depender de modelos/arquivos grandes."""

import numpy as np
import pandas as pd

from medmonitor.audio.features import extract_features, vocal_indicators
from medmonitor.video.analyzer import ANGLE_COLS, RepCounter, VideoAnalyzer
from medmonitor.video.pose import angle, joint_angles


def test_angle():
    assert abs(angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) - 90) < 1e-6
    assert np.isnan(angle(None, np.array([0, 0]), np.array([0, 1])))


def test_joint_angles_standing_person():
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    for side, x in (("l", 0.45), ("r", 0.55)):
        idx = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11, "knee": 13, "ankle": 15}
        off = 0 if side == "l" else 1
        for name, y in (("shoulder", 0.2), ("elbow", 0.35), ("wrist", 0.5), ("hip", 0.5), ("knee", 0.7), ("ankle", 0.9)):
            k[idx[name] + off, :2] = (x * 100, y * 100)
    a = joint_angles(k)
    assert a["knee_l"] > 175 and a["trunk_lean"] < 1


def test_rep_counter_full_and_partial():
    rc = RepCounter(flexed_max=110, extended_min=155, min_rom=50)
    seq = list(np.linspace(170, 90, 20)) + list(np.linspace(90, 170, 20))   # completa
    seq += list(np.linspace(170, 130, 10)) + list(np.linspace(130, 170, 10))  # incompleta
    seq += list(np.linspace(170, 85, 20)) + list(np.linspace(85, 170, 20))   # completa
    for i, a in enumerate(seq):
        rc.update(i * 0.1, a)
    assert len(rc.reps) == 2 and len(rc.partials) == 1
    assert rc.reps[0]["min_angle"] <= 91


def _timeline(n=150, fps=15.0, fall_at=None):
    t = np.arange(n) / fps
    tl = pd.DataFrame({"t": t, "detected": True, "n_people": 1, "zone_people": 0, "zone_prohibited": "",
                       "zone_over_capacity": "", "instruments": 0, "conf_l": 0.9, "conf_r": 0.9})
    for c in ANGLE_COLS:
        tl[c] = 170.0
    tl["trunk_lean"], tl["hip_y"], tl["bbox_w"], tl["bbox_h"] = 5.0, 300.0, 120.0, 400.0
    if fall_at:
        i = int(fall_at * fps)
        tl.loc[i:i + 5, "hip_y"] = np.linspace(300, 700, 6)
        tl.loc[i + 5:, ["hip_y", "bbox_w", "bbox_h", "trunk_lean"]] = [700.0, 420.0, 130.0, 85.0]
    return tl


def _analyzer(cfg):
    va = VideoAnalyzer.__new__(VideoAnalyzer)  # sem carregar modelos
    va.cfg, va.vcfg = cfg, cfg["video"]
    return va


def test_fall_detection_rule(cfg):
    events, _, _ = _analyzer(cfg)._analyze_timeline(_timeline(fall_at=5.0), None, None, 15.0)
    kinds = [e["type"] for e in events]
    assert "queda" in kinds and "compensacao_tronco" not in kinds


def test_no_events_on_still_person(cfg):
    events, _, _ = _analyzer(cfg)._analyze_timeline(_timeline(), None, None, 15.0)
    assert [e for e in events if e["type"] != "padrao_atipico"] == []


def test_vocal_features_on_synthetic_vowel(cfg):
    sr = 16000
    t = np.arange(int(sr * 3)) / sr
    f0 = 140 * (1 + 0.03 * np.sin(2 * np.pi * 0.5 * t))
    y = sum(np.sin(2 * np.pi * h * np.cumsum(f0) / sr) / h for h in range(1, 8)).astype(np.float32)
    y[int(1.4 * sr):int(1.8 * sr)] = 0  # uma pausa
    feat = extract_features(y * 0.5, sr)
    assert 120 < feat["f0_median_hz"] < 160 and feat["pauses_per_min"] > 0
    ind = vocal_indicators(feat, cfg["audio"]["reference"])
    assert set(ind) >= {"fadiga", "dificuldade_respiratoria", "disartria"}
