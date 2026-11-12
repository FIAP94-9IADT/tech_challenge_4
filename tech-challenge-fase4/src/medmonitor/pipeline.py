"""Orquestração do fluxo multimodal completo.

    dados ──► detectores por modalidade ──► alertas por modalidade ─┐
                                                                    ├─► fusão ─► AlertManager ─► equipe
    (vitais, prescrições, movimentação, áudio, vídeo)  escores 0-1 ─┘

Também calcula as métricas de avaliação contra o ground truth sintético.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts import Alert, AlertManager
from .anomaly.movement import detect_movement_anomalies
from .anomaly.prescriptions import detect_prescription_anomalies
from .anomaly.utils import event_overlap_metrics
from .anomaly.vitals import StreamingVitalsMonitor, detect_vitals_anomalies
from .config import azure_language_enabled, azure_speech_enabled, resolve
from .data import synthetic
from .fusion import fuse_patient

log = logging.getLogger(__name__)


def ensure_data(cfg: dict, regenerate: bool = False) -> dict[str, Path]:
    d = resolve(cfg["paths"]["data_dir"]) / "synthetic"
    paths = {k: d / f"{k}.csv" for k in ("vitals", "prescriptions", "movement", "patients")}
    if regenerate or not all(p.exists() for p in paths.values()):
        paths = synthetic.write_all(d)
    return paths


def ensure_audio(cfg: dict) -> Path | None:
    manifest = resolve(cfg["paths"]["data_dir"]) / "audio" / "manifest.csv"
    if manifest.exists():
        return manifest
    try:
        from .data.demo_audio import generate_demo_audio
        return generate_demo_audio(manifest.parent, cfg)
    except Exception as exc:  # noqa: BLE001 — sem internet para TTS
        log.warning("Não foi possível gerar áudios de demonstração: %s", exc)
        return None


def ensure_video(cfg: dict) -> Path | None:
    root = resolve(cfg["paths"]["data_dir"])
    manifest = root / "video" / "manifest.csv"
    if manifest.exists():
        return manifest
    try:
        from .data.demo_video import prepare_demo_videos
        return prepare_demo_videos(root / "raw" / "video", root / "video")
    except Exception as exc:  # noqa: BLE001
        log.warning("Não foi possível preparar vídeos de demonstração: %s", exc)
        return None


def _video_alerts_from_cache(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def run_pipeline(cfg: dict, with_audio: bool = True, with_video: bool = True, reuse_video: bool = True,
                 regenerate: bool = False, quiet: bool = False) -> dict:
    out_dir = resolve(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts_log = resolve(cfg["paths"]["alerts_log"])
    alerts_log.unlink(missing_ok=True)

    paths = ensure_data(cfg, regenerate)
    vitals = synthetic.load_csv(paths["vitals"])
    movement = synthetic.load_csv(paths["movement"])
    rx = synthetic.load_csv(paths["prescriptions"])
    patients = pd.read_csv(paths["patients"]).set_index("patient_id").to_dict(orient="index")

    per_patient_alerts: dict[str, list[Alert]] = {pid: [] for pid in patients}
    scores: dict[str, dict[str, float]] = {pid: {} for pid in patients}
    results: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                     "azure": {"speech": azure_speech_enabled(cfg), "language": azure_language_enabled(cfg)},
                     "vitals_events": {}, "movement_events": {}, "audio": [], "video": []}

    # --- sinais vitais
    vit_results = {}
    for pid, g in vitals.groupby("patient_id"):
        r = detect_vitals_anomalies(g, cfg)
        vit_results[pid] = r
        per_patient_alerts[pid] += r.alerts
        scores[pid]["vitals"] = r.risk_score
        results["vitals_events"][pid] = r.events

    # --- movimentação
    mov_results = {}
    for pid, g in movement.groupby("patient_id"):
        r = detect_movement_anomalies(g, cfg)
        mov_results[pid] = r
        per_patient_alerts[pid] += r.alerts
        scores[pid]["movement"] = r.risk_score
        results["movement_events"][pid] = r.events

    # --- prescrições
    rx_res = detect_prescription_anomalies(rx, cfg)
    for a in rx_res.alerts:
        per_patient_alerts.setdefault(a.patient_id, []).append(a)
    for pid in patients:
        scores[pid]["prescription"] = rx_res.risk_score(pid)
    results["prescriptions"] = json.loads(rx_res.frame.to_json(orient="records", date_format="iso"))

    # --- áudio
    if with_audio and (manifest := ensure_audio(cfg)):
        from .audio.analyzer import analyze_audio
        m = pd.read_csv(manifest)
        for _, row in m.iterrows():
            ar = analyze_audio(manifest.parent / row["file"], cfg, row["patient_id"], row["timestamp"])
            d = ar.to_dict()
            d["expected_condition"] = row.get("expected_condition")
            results["audio"].append(d)
            per_patient_alerts.setdefault(ar.patient_id, []).extend(ar.alerts)
            scores.setdefault(ar.patient_id, {})["audio"] = max(scores.get(ar.patient_id, {}).get("audio", 0), ar.risk_score)

    # --- vídeo
    if with_video and (manifest := ensure_video(cfg)):
        m = pd.read_csv(manifest, keep_default_na=False)
        analyzer = None
        for _, row in m.iterrows():
            cache = out_dir / "video" / Path(row["file"]).stem / "result.json"
            d = _video_alerts_from_cache(cache) if reuse_video else None
            if d is None:
                from .video.analyzer import VideoAnalyzer
                analyzer = analyzer or VideoAnalyzer(cfg)
                vr = analyzer.analyze(manifest.parent / row["file"], row["patient_id"], row["profile"] or None,
                                      out_dir / "video", row["timestamp"], row["target"],
                                      use_zones=str(row.get("zones")) == "True")
                d = vr.to_dict()
            d["description"], d["source"] = row["description"], row["source"]
            results["video"].append(d)
            valerts = [Alert.from_dict(a) for a in d["alerts"]]
            per_patient_alerts.setdefault(row["patient_id"], []).extend(valerts)
            scores.setdefault(row["patient_id"], {})
            scores[row["patient_id"]]["video"] = max(scores[row["patient_id"]].get("video", 0), d["risk_score"])

    # --- fusão + despacho
    manager = AlertManager(cfg, alerts_log)
    manager.console = cfg["alerts"].get("console", True) and not quiet
    fusion = []
    for pid in per_patient_alerts:
        f = fuse_patient(pid, scores.get(pid, {}), per_patient_alerts[pid], cfg)
        fusion.append(f)
    all_alerts = [a for f in fusion for a in f.alerts + f.correlations]
    dispatched = manager.dispatch_many(all_alerts)

    results["patients"] = []
    for f in sorted(fusion, key=lambda f: -f.risk):
        info = patients.get(f.patient_id, {})
        results["patients"].append({**f.to_dict(), **{k: info.get(k) for k in ("nome", "idade", "leito", "cenario")}})
    results["alerts"] = {"generated": len(all_alerts), "dispatched": len(dispatched), "suppressed": manager.suppressed,
                         "by_severity": pd.Series([a.severity.name for a in dispatched]).value_counts().to_dict(),
                         "by_modality": pd.Series([a.modality for a in dispatched]).value_counts().to_dict()}
    results["metrics"] = evaluate(cfg, vitals, movement, rx_res, vit_results, mov_results, results["audio"])

    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))
    from .report import write_results_report
    write_results_report(cfg, results, vit_results, mov_results, out_dir)
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ----------------------------------------------------------------------------- avaliação
def evaluate(cfg, vitals, movement, rx_res, vit_results, mov_results, audio) -> dict:
    gap = cfg["vitals"]["merge_gap"]
    methods = {"Regras clínicas (NEWS2 + limites)": "flag_rules", "z-score robusto": "flag_zscore",
               "Isolation Forest": "flag_iforest", "Ensemble (2 de 3 + persistência)": "anomaly"}
    vit_rows = []
    totals = {m: {"true": 0, "det": 0, "pred": 0, "pred_ok": 0, "fp_samples": 0} for m in methods}
    for pid, r in vit_results.items():
        truth = (r.frame["anomaly_type"].replace("artifact", "") != "").to_numpy()
        artifact = (r.frame["anomaly_type"] == "artifact").to_numpy()
        for name, col in methods.items():
            mm = event_overlap_metrics(r.frame[col].to_numpy(), truth, merge_gap=gap)
            t = totals[name]
            t["true"] += mm["eventos_reais"]
            t["det"] += mm["eventos_detectados"]
            t["pred"] += mm["eventos_previstos"]
            t["pred_ok"] += round((mm["precisao_evento"] or 0) * mm["eventos_previstos"])
            t["fp_samples"] += mm["falsos_positivos_amostra"]
            if col == "anomaly":
                vit_rows.append({"paciente": pid, **mm,
                                 "artefato_filtrado": bool(artifact.any() and not r.frame["anomaly"].to_numpy()[artifact].any())
                                 if artifact.any() else None})
    vit_methods = []
    for name, t in totals.items():
        rec = t["det"] / t["true"] if t["true"] else None
        prec = t["pred_ok"] / t["pred"] if t["pred"] else None
        f1 = 2 * rec * prec / (rec + prec) if rec and prec else None
        vit_methods.append({"metodo": name, "eventos_reais": t["true"], "detectados": t["det"], "previstos": t["pred"],
                            "recall": _r(rec), "precisao": _r(prec), "f1": _r(f1), "amostras_falso_positivas": t["fp_samples"]})

    # monitor em tempo real: atraso até o primeiro alerta de cada evento verdadeiro
    stream_rows = []
    for pid, g in vitals.groupby("patient_id"):
        mon = StreamingVitalsMonitor(pid, cfg)
        g = g.reset_index(drop=True)
        alert_idx = []
        for i, row in enumerate(g.to_dict(orient="records")):
            if mon.update(row):
                alert_idx.append(i)
        truth = (g["anomaly_type"].replace("artifact", "") != "").to_numpy()
        from .anomaly.utils import mask_to_events
        for s, e in mask_to_events(truth):
            # aceita detecção durante a rampa que antecede o rótulo (atraso negativo = antecipada)
            hits = [i for i in alert_idx if s - 15 <= i <= e + 5]
            stream_rows.append({"paciente": pid, "evento": g["anomaly_type"].iloc[s], "inicio": str(g["timestamp"].iloc[s]),
                                "detectado": bool(hits), "atraso_min": (hits[0] - s) if hits else None})
        in_event = sum(1 for i in alert_idx if any(s - 15 <= i <= e + 5 for s, e in mask_to_events(truth)))
        stream_rows.append({"paciente": pid, "evento": "(alertas fora de eventos)", "inicio": "",
                            "detectado": None, "atraso_min": len(alert_idx) - in_event})

    # movimentação
    mov_rows = []
    for pid, r in mov_results.items():
        truth = r.frame["anomaly_type"].to_numpy()
        for kind in sorted(set(truth) - {""}):
            mask = truth == kind
            mov_rows.append({"paciente": pid, "evento_real": kind,
                             "detectado": bool(r.frame["anomaly"].to_numpy()[mask].any()),
                             "tipos_detectados": sorted({e["type"] for e in r.events})})
        mov_rows.append({"paciente": pid, "evento_real": "(falsos positivos)", "detectado": None,
                         "tipos_detectados": [e["type"] for e in r.events
                                              if not (truth[e["start"]:e["end"] + 1] != "").any()]})

    # prescrições
    f = rx_res.frame
    tp = int((f["anomaly"] & f["expected_anomaly"]).sum())
    fp = int((f["anomaly"] & ~f["expected_anomaly"]).sum())
    fn = int((~f["anomaly"] & f["expected_anomaly"]).sum())
    rx_metrics = {"eventos": len(f), "anomalias_esperadas": int(f["expected_anomaly"].sum()), "vp": tp, "fp": fp, "fn": fn,
                  "precisao": _r(tp / (tp + fp) if tp + fp else None), "recall": _r(tp / (tp + fn) if tp + fn else None)}

    # áudio: condição esperada × indicador dominante
    label = {"fadiga": "fadiga", "dificuldade_respiratoria": "dispneia", "disartria": "disartria"}
    audio_rows = []
    for a in audio:
        ind = {k: v for k, v in a["indicators"].items() if not k.startswith("_")}
        top = max(ind, key=ind.get)
        pred = label[top] if ind[top] >= 0.5 else "normal"
        audio_rows.append({"arquivo": a["file"], "esperado": a.get("expected_condition"), "previsto": pred,
                           "acerto": pred == a.get("expected_condition"), **ind})
    return {"vitals_methods": vit_methods, "vitals_by_patient": vit_rows, "streaming": stream_rows,
            "movement": mov_rows, "prescriptions": rx_metrics, "audio": audio_rows}


def _r(x):
    return round(x, 3) if x is not None else None
