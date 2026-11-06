import numpy as np
import pandas as pd
import pytest

from medmonitor.anomaly.movement import detect_movement_anomalies
from medmonitor.anomaly.prescriptions import detect_prescription_anomalies
from medmonitor.anomaly.utils import event_overlap_metrics, mask_to_events, persistent
from medmonitor.anomaly.vitals import StreamingVitalsMonitor, detect_vitals_anomalies, news2_score
from medmonitor.data import synthetic


@pytest.fixture(scope="module")
def vitals():
    return synthetic.generate_vitals()


def test_mask_to_events_merge_and_min_len():
    m = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1]
    assert mask_to_events(m) == [(1, 2), (4, 4), (8, 10)]
    assert mask_to_events(m, merge_gap=1) == [(1, 4), (8, 10)]
    assert mask_to_events(m, min_len=2) == [(1, 2), (8, 10)]
    assert persistent(np.array(m, bool), 3).tolist() == [False] * 8 + [True] * 3


def test_event_metrics():
    truth = np.zeros(100, bool)
    truth[10:20] = truth[50:60] = True
    pred = np.zeros(100, bool)
    pred[12:15] = pred[80:85] = True
    m = event_overlap_metrics(pred, truth)
    assert m["recall_evento"] == 0.5 and m["precisao_evento"] == 0.5


@pytest.mark.parametrize("row,expected", [
    ({"rr": 16, "spo2": 97, "sbp": 120, "hr": 75, "temp": 36.8}, 0),
    ({"rr": 7, "spo2": 88, "sbp": 95, "hr": 55, "temp": 36.5}, 3 + 3 + 2 + 0 + 0),
    ({"rr": 26, "spo2": 93, "sbp": 88, "hr": 135, "temp": 39.4}, 3 + 2 + 3 + 3 + 2),
])
def test_news2(row, expected):
    assert news2_score(row)[0] == expected


def test_control_patient_artifact_is_filtered(vitals, cfg):
    r = detect_vitals_anomalies(vitals[vitals.patient_id == "P001"], cfg)
    assert r.alerts == []
    artifact = r.frame["anomaly_type"] == "artifact"
    assert artifact.any() and not r.frame.loc[artifact, "anomaly"].any()


def test_respiratory_depression_is_critical(vitals, cfg):
    r = detect_vitals_anomalies(vitals[vitals.patient_id == "P003"], cfg)
    crit = [a for a in r.alerts if a.severity.name == "CRITICAL"]
    assert crit and "Bradipneia" in crit[0].title and "Dessaturação" in crit[0].title


def test_ensemble_recall_all_patients(vitals, cfg):
    for pid, g in vitals.groupby("patient_id"):
        r = detect_vitals_anomalies(g, cfg)
        truth = (r.frame["anomaly_type"].replace("artifact", "") != "").to_numpy()
        if truth.any():
            assert event_overlap_metrics(r.frame["anomaly"], truth, merge_gap=30)["recall_evento"] == 1.0, pid


def test_streaming_monitor_detects_tachycardia(vitals, cfg):
    g = vitals[vitals.patient_id == "P004"]
    mon = StreamingVitalsMonitor("P004", cfg)
    titles = [a.title for row in g.to_dict(orient="records") for a in mon.update(row)]
    assert titles.count("Taquicardia") >= 3


def test_missing_columns_are_tolerated(cfg):
    ts = pd.date_range("2026-01-01", periods=300, freq="1min")
    df = pd.DataFrame({"timestamp": ts, "patient_id": "X", "hr": 70 + np.random.default_rng(0).normal(0, 1, 300)})
    df.loc[250:280, "hr"] = 140
    r = detect_vitals_anomalies(df, cfg)
    assert any("Taquicardia" in a.title for a in r.alerts)


def test_prescriptions(cfg):
    r = detect_prescription_anomalies(synthetic.generate_prescriptions(), cfg)
    f = r.frame
    assert (f["anomaly"] == f["expected_anomaly"]).all()
    morphine = [a for a in r.alerts if "morfina" in a.title]
    assert morphine[0].severity.name == "CRITICAL" and "5.0x" in morphine[0].description
    assert any("duplicidade" in a.description.lower() for a in r.alerts if a.patient_id == "P002")


def test_movement(cfg):
    m = synthetic.generate_movement()
    types = {pid: {e["type"] for e in detect_movement_anomalies(g, cfg).events} for pid, g in m.groupby("patient_id")}
    assert types["P001"] == set() and types["P003"] == set()
    assert types["P002"] == {"agitacao"}
    assert types["P005"] == {"imobilidade", "queda"}
