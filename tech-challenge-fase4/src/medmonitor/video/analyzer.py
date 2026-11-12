"""Análise de vídeos clínicos (fisioterapia, mobilidade, áreas críticas de procedimento).

Fluxo em duas passagens:
  1. Inferência — para cada quadro amostrado: pose (YOLOv8-pose ou OpenPose),
     objetos (YOLOv8) e ocupação das áreas críticas; gera uma série temporal.
  2. Análise + renderização — detecta eventos sobre a série e grava o vídeo
     anotado (esqueleto, ângulos, zonas, contador de repetições e faixas de alerta).

Eventos detectados:
  * repetições (contagem, amplitude de movimento, duração) por perfil de exercício
  * repetição incompleta, hesitação (repetição muito lenta) e queda progressiva de
    amplitude (fadiga)
  * compensação de tronco e assimetria entre lados
  * queda (deslocamento vertical rápido do quadril + corpo horizontalizado)
  * padrão de movimento atípico (Isolation Forest sobre ângulos e velocidades)
  * paciente fora do campo de visão
  * objeto proibido / lotação em área crítica / discrepância na contagem de instrumentos
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert, Severity
from ..anomaly.utils import mask_to_events
from ..config import PROJECT_ROOT
from .objects import ObjectDetector, ZoneMonitor, draw_detections
from .pose import PersonPose, build_pose_backend, draw_pose, joint_angles

log = logging.getLogger(__name__)
ANGLE_COLS = [f"{j}_{s}" for j in ("elbow", "shoulder", "hip", "knee") for s in ("l", "r")]
MAX_WIDTH = 960
JOINT_PT = {"knee": "joelho", "hip": "quadril", "elbow": "cotovelo", "shoulder": "ombro"}

EVENT_META = {
    "queda": (Severity.CRITICAL, "Queda detectada", "Atender o paciente imediatamente e avaliar trauma."),
    "repeticao_incompleta": (Severity.WARNING, "Repetições com amplitude incompleta",
                             "Revisar execução com o fisioterapeuta; considerar dor ou limitação articular."),
    "hesitacao": (Severity.INFO, "Repetição com hesitação/lentidão", "Observar dor, medo de cair ou fadiga."),
    "fadiga_amplitude": (Severity.WARNING, "Queda progressiva de amplitude (fadiga)",
                         "Reduzir carga/volume e reavaliar tolerância ao esforço."),
    "compensacao_tronco": (Severity.WARNING, "Compensação de tronco", "Corrigir postura; avaliar fraqueza de membros inferiores."),
    "assimetria": (Severity.WARNING, "Assimetria entre lados", "Avaliar dor ou déficit motor unilateral."),
    "padrao_atipico": (Severity.INFO, "Padrão de movimento atípico", "Revisar o trecho do vídeo indicado."),
    "fora_do_campo": (Severity.INFO, "Paciente fora do campo de visão", "Verificar posicionamento da câmera/paciente."),
    "objeto_proibido": (Severity.WARNING, "Objeto proibido em área crítica", "Remover o objeto da área crítica."),
    "lotacao_zona": (Severity.WARNING, "Lotação acima do limite em área crítica", "Restringir circulação na área."),
    "instrumentos": (Severity.CRITICAL, "Discrepância na contagem de instrumentos",
                     "Realizar contagem cirúrgica antes do fechamento (protocolo de cirurgia segura)."),
    "desempenho_baixo": (Severity.WARNING, "Desempenho abaixo do esperado no teste funcional",
                         "Risco aumentado de queda: aplicar protocolo de prevenção de quedas (STEADI)."),
}


@dataclass
class VideoResult:
    file: str
    patient_id: str
    profile: str | None
    backend: str
    fps: float
    duration_s: float
    frames_processed: int
    timeline: pd.DataFrame
    events: list[dict] = field(default_factory=list)
    reps: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)
    annotated_path: str | None = None
    plot_path: str | None = None
    report_path: str | None = None

    @property
    def risk_score(self) -> float:
        w = {Severity.CRITICAL: 1.0, Severity.WARNING: 0.5, Severity.INFO: 0.15}
        return min(1.0, sum(w[a.severity] for a in self.alerts) / 2)

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "file": self.file, "patient_id": self.patient_id, "profile": self.profile, "backend": self.backend,
            "fps": self.fps, "duration_s": self.duration_s, "frames_processed": self.frames_processed,
            "events": self.events, "reps": self.reps, "summary": self.summary,
            "alerts": [a.to_dict() for a in self.alerts], "annotated_path": self.annotated_path,
            "plot_path": self.plot_path, "report_path": self.report_path, "risk_score": round(self.risk_score, 2),
        }.items()}


# ----------------------------------------------------------------------------- repetições
class RepCounter:
    """Máquina de estados com histerese: estendido → flexionado → estendido = 1 repetição."""

    def __init__(self, flexed_max: float, extended_min: float, min_rom: float):
        self.flexed_max, self.extended_min, self.min_rom = flexed_max, extended_min, min_rom
        self.state = None
        self.reps: list[dict] = []
        self.partials: list[dict] = []
        self._start = self._min = self._min_t = None

    def update(self, t: float, a: float) -> None:
        if np.isnan(a):
            return
        if self.state is None:
            self.state = "up" if a >= self.extended_min else "down"
            self._start, self._min, self._min_t = t, a, t
            return
        if self.state == "up":
            if a < self.extended_min - 10:  # iniciou descida
                if self._min is None or self._start is None:
                    self._start, self._min, self._min_t = t, a, t
                if a < self._min:
                    self._min, self._min_t = a, t
                if a <= self.flexed_max:
                    self.state = "down"
            elif self._min is not None and self._min < self.extended_min - 10:
                # voltou à extensão sem atingir a flexão alvo → repetição incompleta
                depth = self.extended_min - self._min
                if depth >= 0.3 * (self.extended_min - self.flexed_max):
                    self.partials.append({"start_s": round(self._start, 2), "bottom_s": round(self._min_t, 2),
                                          "end_s": round(t, 2), "min_angle": round(self._min, 1)})
                self._start = self._min = self._min_t = None
            else:
                self._start, self._min, self._min_t = t, a, t
        else:  # down
            if self._min is None or a < self._min:
                self._min, self._min_t = a, t
            if a >= self.extended_min and (self._min > self.flexed_max or a - self._min < self.min_rom):
                # estado inicial "flexionado" sem ciclo completo: apenas sincroniza
                self.state = "up"
                self._start = self._min = self._min_t = None
            elif a >= self.extended_min:
                self.reps.append({"rep": len(self.reps) + 1, "start_s": round(self._start or t, 2),
                                  "bottom_s": round(self._min_t, 2), "end_s": round(t, 2),
                                  "min_angle": round(self._min, 1), "max_angle": round(a, 1),
                                  "rom": round(a - self._min, 1),
                                  "duration_s": round(t - (self._start or t), 2)})
                self.state = "up"
                self._start = self._min = self._min_t = None


# ----------------------------------------------------------------------------- analisador
class VideoAnalyzer:
    def __init__(self, cfg: dict, with_objects: bool = True):
        self.cfg = cfg
        self.vcfg = cfg["video"]
        self.pose = build_pose_backend(self.vcfg, PROJECT_ROOT)
        self.detector = ObjectDetector(str(PROJECT_ROOT / self.vcfg["detect_model"]), self.vcfg["conf_threshold"]) if with_objects else None

    # -- seleção da pessoa monitorada (continuidade + preferência) ------------------
    @staticmethod
    def _select(poses: list[PersonPose], prev_bbox, target: str, width: int) -> PersonPose | None:
        if not poses:
            return None
        if prev_bbox is not None:
            def iou(b):
                x1, y1 = max(b[0], prev_bbox[0]), max(b[1], prev_bbox[1])
                x2, y2 = min(b[2], prev_bbox[2]), min(b[3], prev_bbox[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                union = (b[2] - b[0]) * (b[3] - b[1]) + (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]) - inter
                return inter / union if union > 0 else 0
            best = max(poses, key=lambda p: iou(p.bbox))
            if iou(best.bbox) > 0.2:
                return best
        cx = lambda p: (p.bbox[0] + p.bbox[2]) / 2 / width  # noqa: E731
        if target == "left":
            return min(poses, key=cx)
        if target == "right":
            return max(poses, key=cx)
        if target == "center":
            return min(poses, key=lambda p: abs(cx(p) - 0.5))
        return max(poses, key=lambda p: p.area)

    def analyze(self, path: str | Path, patient_id: str = "desconhecido", profile: str | None = None,
                output_dir: str | Path = "outputs/video", timestamp: str | None = None, target: str = "largest",
                start_s: float = 0.0, max_seconds: float | None = None, annotate: bool = True,
                use_zones: bool = False) -> VideoResult:
        path = Path(path)
        out_dir = Path(output_dir) / (path.stem + ("_openpose" if self.pose.name.startswith("OpenPose") else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        session_start = datetime.fromisoformat(timestamp) if timestamp else datetime.now().replace(microsecond=0)
        stride = max(1, int(self.vcfg["frame_stride"]))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        zones = None
        rows, store = [], []
        prev_bbox = None
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / fps
            if max_seconds and t > max_seconds:
                break
            if idx % stride:
                continue
            frame = _resize(frame)
            h, w = frame.shape[:2]
            if zones is None:
                zones = ZoneMonitor(self.vcfg.get("critical_zones") if use_zones else {}, (h, w))
            poses = self.pose.estimate(frame)
            main = self._select(poses, prev_bbox, target, w)
            prev_bbox = main.bbox if main else prev_bbox
            dets = self.detector.detect(frame) if self.detector else []
            row = {"t": round(t, 3), "n_people": len(poses), "detected": main is not None}
            if main is not None:
                row.update(joint_angles(main.keypoints))
                x1, y1, x2, y2 = main.bbox
                row.update(bbox_w=x2 - x1, bbox_h=y2 - y1, kp_conf=float(main.keypoints[:, 2].mean()))
                for side in ("l", "r"):  # confiança dos pontos usados na assimetria
                    row[f"conf_{side}"] = float(np.min(main.keypoints[[5 + (side == "r"), 11 + (side == "r"),
                                                                      13 + (side == "r")], 2]))
            zsum = zones.summarize(dets, self.vcfg["prohibited_in_zone"], self.vcfg["instrument_classes"])
            row["zone_people"] = max((z["people"] for z in zsum.values()), default=0)
            row["zone_over_capacity"] = ",".join(n for n, z in zsum.items() if z["people"] > zones.max_people[n])
            row["zone_prohibited"] = ",".join(sorted({o for z in zsum.values() for o in z["prohibited"]}))
            row["instruments"] = sum(d.cls in self.vcfg["instrument_classes"] for d in dets)
            rows.append(row)
            store.append((idx, main, [p for p in poses if p is not main], dets))
        cap.release()

        tl = pd.DataFrame(rows)
        if tl.empty:
            raise RuntimeError("Nenhum quadro processado.")
        for c in ANGLE_COLS + ["trunk_lean", "hip_y", "torso_len", "bbox_w", "bbox_h", "conf_l", "conf_r"]:
            if c not in tl:
                tl[c] = np.nan
        prof = self.vcfg["exercise_profiles"].get(profile) if profile else None
        events, reps, summary = self._analyze_timeline(tl, prof, profile, fps / stride)

        result = VideoResult(path.name, patient_id, profile, self.pose.name, round(fps, 2),
                             round(len(tl) * stride / fps, 2), len(tl), tl, events, reps, summary)
        result.alerts = self._alerts(result, session_start)
        result.plot_path = str(self._plot(result, prof, out_dir))
        if annotate:
            result.annotated_path = str(self._render(path, store, result, zones, prof, fps, stride, out_dir, start_s))
        tl.to_csv(out_dir / "timeline.csv", index=False)
        result.report_path = str(write_video_report(result, out_dir))
        (out_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return result

    # -- análise da série temporal ----------------------------------------------------
    def _analyze_timeline(self, tl: pd.DataFrame, prof: dict | None, profile: str | None, eff_fps: float):
        v = self.vcfg
        min_len = max(2, int(v["min_event_seconds"] * eff_fps))
        sm = tl[ANGLE_COLS + ["trunk_lean"]].interpolate(limit=int(eff_fps)).rolling(5, center=True, min_periods=1).median()
        events: list[dict] = []
        reps: list[dict] = []
        summary: dict = {"pessoa_visivel_pct": round(100 * tl["detected"].mean(), 1),
                         "max_pessoas_quadro": int(tl["n_people"].max())}

        def add(kind, s, e, **extra):
            events.append({"type": kind, "start_s": round(float(tl["t"].iloc[s]), 2),
                           "end_s": round(float(tl["t"].iloc[e]), 2), **extra})

        trunk_max = (prof or {}).get("trunk_max", v["trunk_lean_max_deg"])
        if prof:
            j = prof["joint"]
            primary = sm[[f"{j}_l", f"{j}_r"]].mean(axis=1)
            tl["primary_angle"] = primary
            rc = RepCounter(prof["flexed_max"], prof["extended_min"], prof["min_rom"])
            for t, a in zip(tl["t"], primary):
                rc.update(t, a)
            reps = rc.reps
            summary.update({"repeticoes": len(reps), "repeticoes_incompletas": len(rc.partials),
                            "amplitude_media": round(float(np.mean([r["rom"] for r in reps])), 1) if reps else None,
                            "angulo_minimo_medio": round(float(np.mean([r["min_angle"] for r in reps])), 1) if reps else None})
            for p in rc.partials:
                events.append({"type": "repeticao_incompleta", "start_s": p["start_s"], "end_s": p["end_s"],
                               "detail": f"ângulo mínimo {p['min_angle']}° (alvo ≤ {prof['flexed_max']}°)"})
            durs = np.array([r["duration_s"] for r in reps])
            if len(durs) >= 3:
                med = np.median(durs)
                summary["cadencia_s_por_rep"] = round(float(med), 2)
                for r in reps:
                    if r["duration_s"] > 2 * med and r["duration_s"] - med > 1.0:
                        events.append({"type": "hesitacao", "start_s": r["start_s"], "end_s": r["end_s"],
                                       "detail": f"repetição {r['rep']} levou {r['duration_s']}s (mediana {med:.1f}s)"})
            if len(reps) >= 4:
                roms = np.array([r["rom"] for r in reps])
                slope = np.polyfit(np.arange(len(roms)), roms, 1)[0]
                summary["tendencia_amplitude_graus_por_rep"] = round(float(slope), 2)
                if slope * len(roms) < -0.25 * roms[0]:
                    events.append({"type": "fadiga_amplitude", "start_s": reps[0]["start_s"], "end_s": reps[-1]["end_s"],
                                   "detail": f"amplitude caiu {roms[0]:.0f}° → {roms[-1]:.0f}°"})
            if profile == "sentar_levantar" and len(reps) >= 3:
                # Teste de sentar e levantar em 30 s (CDC STEADI): extrapola para 30 s a partir da cadência
                window = reps[-1]["end_s"] - reps[0]["start_s"]
                per30 = len(reps) * 30 / window if window > 0 else len(reps)
                summary["repeticoes_por_30s"] = round(per30, 1)
                ref = prof.get("min_reps_30s")
                if ref and per30 < ref:
                    events.append({"type": "desempenho_baixo", "start_s": reps[0]["start_s"], "end_s": reps[-1]["end_s"],
                                   "detail": f"{per30:.1f} repetições/30 s (referência ≥ {ref})"})

            # assimetria entre lados (somente quando ambos os lados estão bem visíveis)
            both = (tl["conf_l"] > 0.5) & (tl["conf_r"] > 0.5)
            asym = (sm[f"{j}_l"] - sm[f"{j}_r"]).abs()
            for s, e in mask_to_events((asym > v["asymmetry_max_deg"]) & both, min_len=min_len, merge_gap=min_len):
                add("assimetria", s, e, detail=f"diferença máx. {asym.iloc[s:e + 1].max():.0f}° no {j}")

        for s, e in mask_to_events(sm["trunk_lean"] > trunk_max, min_len=min_len, merge_gap=min_len):
            add("compensacao_tronco", s, e, detail=f"inclinação máx. {sm['trunk_lean'].iloc[s:e + 1].max():.0f}° (limite {trunk_max}°)")

        # queda: quadril desce > X alturas-de-corpo/s e o corpo fica horizontal logo depois
        ref_h = tl["bbox_h"].rolling(int(2 * eff_fps), min_periods=3).median().shift(1)
        vel = tl["hip_y"].interpolate(limit=3).diff() * eff_fps / ref_h
        tl["hip_velocity"] = vel
        aspect = tl["bbox_w"] / tl["bbox_h"]
        horizon = int(1.5 * eff_fps)
        for i in np.where(vel > v["fall_velocity"])[0]:
            after = slice(i, min(len(tl), i + horizon))
            if (aspect.iloc[after] > 1.0).any() or (tl["trunk_lean"].iloc[after] > 60).any():
                if not any(ev["type"] == "queda" and abs(ev["start_s"] - tl["t"].iloc[i]) < 3 for ev in events):
                    add("queda", i, min(len(tl) - 1, i + horizon), detail=f"velocidade do quadril {vel.iloc[i]:.1f} corpos/s")

        falls = [ev for ev in events if ev["type"] == "queda"]
        events = [ev for ev in events if not (ev["type"] == "compensacao_tronco" and any(
            ev["start_s"] <= f["end_s"] + 2 and f["start_s"] - 2 <= ev["end_s"] for f in falls))]

        # padrão atípico: Isolation Forest sobre ângulos + velocidades angulares
        feats = sm[ANGLE_COLS].copy()
        feats = feats.loc[:, feats.notna().mean() > 0.6].ffill().bfill()
        if feats.shape[1] >= 2 and len(feats) > 60:
            speeds = feats.diff().abs().fillna(0) * eff_fps
            X = pd.concat([feats, speeds.add_prefix("v_")], axis=1).to_numpy()
            iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42).fit(X)
            atyp = iso.predict(X) == -1
            tl["atypicality"] = -iso.score_samples(X)
            for s, e in mask_to_events(atyp, min_len=max(min_len, int(eff_fps)), merge_gap=min_len):
                if not any(ev["start_s"] <= tl["t"].iloc[e] and tl["t"].iloc[s] <= ev["end_s"] for ev in events):
                    add("padrao_atipico", s, e, detail="combinação incomum de ângulos/velocidades")

        for s, e in mask_to_events(~tl["detected"], min_len=int(2 * eff_fps)):
            add("fora_do_campo", s, e, detail=f"{tl['t'].iloc[e] - tl['t'].iloc[s]:.1f}s sem o paciente")

        # áreas críticas
        prohibited = tl["zone_prohibited"].astype(str).str.len() > 0
        for s, e in mask_to_events(prohibited, min_len=min_len, merge_gap=min_len):
            objs = sorted({o for x in tl["zone_prohibited"].iloc[s:e + 1] if x for o in str(x).split(",")})
            add("objeto_proibido", s, e, detail=", ".join(objs))
        crowded = tl["zone_over_capacity"].astype(str).str.len() > 0
        for s, e in mask_to_events(crowded, min_len=min_len, merge_gap=min_len):
            names = sorted({n for x in tl["zone_over_capacity"].iloc[s:e + 1] if x for n in str(x).split(",")})
            add("lotacao_zona", s, e, detail=f"{int(tl['zone_people'].iloc[s:e + 1].max())} pessoas em {', '.join(names)}")
        if tl["instruments"].max() > 0 and len(tl) > 20:
            k = max(3, len(tl) // 7)
            start_n, end_n = int(tl["instruments"].iloc[:k].median()), int(tl["instruments"].iloc[-k:].median())
            summary["instrumentos_inicio_fim"] = [start_n, end_n]
            if end_n < start_n:
                add("instrumentos", len(tl) - k, len(tl) - 1, detail=f"{start_n} instrumentos no início, {end_n} no fim")

        events.sort(key=lambda e: e["start_s"])
        summary["eventos"] = len(events)
        return events, reps, summary

    def _alerts(self, r: VideoResult, session_start: datetime) -> list[Alert]:
        alerts = []
        for ev in r.events:
            sev, title, action = EVENT_META[ev["type"]]
            if ev["type"] == "repeticao_incompleta" and r.summary.get("repeticoes_incompletas", 0) < 2:
                sev = Severity.INFO
            alerts.append(Alert(
                r.patient_id, "video", sev, title,
                f"{ev.get('detail', '')} — trecho {ev['start_s']:.1f}s a {ev['end_s']:.1f}s de {r.file}.",
                (session_start + timedelta(seconds=ev["start_s"])).isoformat(),
                evidence={"video": r.file, "inicio_s": ev["start_s"], "fim_s": ev["end_s"], "perfil": r.profile},
                recommended_action=action))
        return alerts

    # -- saídas ---------------------------------------------------------------------
    def _plot(self, r: VideoResult, prof: dict | None, out_dir: Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tl = r.timeline
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
        ax = axes[0]
        if prof and "primary_angle" in tl:
            ax.plot(tl["t"], tl["primary_angle"], color="#2a78d6", lw=1.6, label=f"{JOINT_PT[prof['joint']]} (média esq./dir.)")
            ax.axhline(prof["flexed_max"], color="#888", ls="--", lw=0.8, label="alvo de flexão")
            ax.axhline(prof["extended_min"], color="#888", ls=":", lw=0.8, label="alvo de extensão")
            for rep in r.reps:
                ax.annotate(str(rep["rep"]), (rep["bottom_s"], rep["min_angle"]), fontsize=8, ha="center", va="top")
        else:
            for c, color in (("knee_l", "#2a78d6"), ("knee_r", "#eb6834"), ("hip_l", "#1baf7a")):
                ax.plot(tl["t"], tl[c], lw=1, color=color, label=f"{JOINT_PT[c[:-2]]} {'esq.' if c.endswith('l') else 'dir.'}")
        ax.set_ylabel("ângulo (°)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{r.file} — {r.backend}  (faixas: eventos detectados)", loc="left", fontsize=10)
        axes[1].plot(tl["t"], tl["trunk_lean"], color="#2a78d6", lw=1.2)
        axes[1].set_ylabel("tronco (°)")
        if "hip_velocity" in tl:
            axes[2].plot(tl["t"], tl["hip_velocity"], color="#2a78d6", lw=1.0)
            axes[2].axhline(self.vcfg["fall_velocity"], color="#888", ls="--", lw=0.8)
        axes[2].set_ylabel("vel. quadril\n(corpos/s)")
        axes[2].set_xlabel("tempo (s)")
        colors = {Severity.CRITICAL: "#cf222e", Severity.WARNING: "#d29922", Severity.INFO: "#57606a"}
        for ev in r.events:
            for a in axes:
                a.axvspan(ev["start_s"], max(ev["end_s"], ev["start_s"] + 0.2),
                          color=colors[EVENT_META[ev["type"]][0]], alpha=0.15)
        fig.tight_layout()
        p = out_dir / "angles.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p

    def _render(self, path, store, r: VideoResult, zones, prof, fps, stride, out_dir: Path, start_s: float) -> Path:
        cap = cv2.VideoCapture(str(path))
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        raw = out_dir / "annotated_raw.mp4"
        writer = None
        by_idx = {s[0]: s for s in store}
        counted = {rep["end_s"]: rep["rep"] for rep in r.reps}
        rep_now = 0
        idx = -1
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx > store[-1][0]:
                break
            if idx % stride:
                continue
            t = idx / fps
            frame = _resize(frame)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
            _, main, others, dets = by_idx.get(idx, last or (idx, None, [], []))
            last = (idx, main, others, dets)
            active = [ev for ev in r.events if ev["start_s"] - 0.1 <= t <= ev["end_s"] + 1.5]
            hot_zone = set(zones.zones) if zones and any(ev["type"] in ("objeto_proibido", "lotacao_zona") for ev in active) else set()
            if zones:
                zones.draw(frame, hot_zone)
            for p in others:
                draw_pose(frame, p, color=(150, 150, 150))
            if main is not None:
                draw_pose(frame, main)
            draw_detections(frame, dets, set(self.vcfg["prohibited_in_zone"]) | set(self.vcfg["instrument_classes"]))
            for end_s, n in counted.items():
                if t >= end_s:
                    rep_now = max(rep_now, n)
            _hud(frame, t, main, prof, rep_now, active)
            writer.write(frame)
        cap.release()
        if writer:
            writer.release()
        final = out_dir / "annotated.mp4"
        if shutil.which("ffmpeg"):  # H.264: formato universal para reprodução
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(final)], check=False)
            if final.exists():
                raw.unlink(missing_ok=True)
                return final
        return raw


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= MAX_WIDTH:
        return frame
    return cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)


def _hud(frame, t, main, prof, rep_now, active):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (25, 25, 25), -1)
    txt = f"t={t:5.1f}s"
    if main is not None and prof:
        ang = joint_angles(main.keypoints)
        j = prof["joint"]
        vals = [ang.get(f"{j}_l"), ang.get(f"{j}_r")]
        vals = [x for x in vals if x is not None and not np.isnan(x)]
        if vals:
            txt += f" | {_ascii(JOINT_PT[j])}: {np.mean(vals):5.0f} graus"
    if prof:
        txt += f" | reps: {rep_now}"
    if main is not None:
        lean = joint_angles(main.keypoints)["trunk_lean"]
        if not np.isnan(lean):
            txt += f" | tronco: {lean:4.0f} graus"
    fs = 0.6 if w >= 900 else 0.45
    cv2.putText(frame, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    y = h - 12
    for ev in active[:3]:
        sev = EVENT_META[ev["type"]][0]
        color = {Severity.CRITICAL: (40, 40, 220), Severity.WARNING: (0, 160, 230), Severity.INFO: (120, 120, 120)}[sev]
        label = _ascii(f"{sev.name}: {EVENT_META[ev['type']][1]}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (8, y - th - 10), (8 + tw + 14, y + 4), color, -1)
        cv2.putText(frame, label, (15, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y -= th + 20


def _ascii(text: str) -> str:
    """OpenCV (Hershey) não renderiza acentos."""
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def write_video_report(r: VideoResult, out_dir: Path) -> Path:
    lines = [f"# Relatório de vídeo — {r.file}", "",
             f"- Paciente: **{r.patient_id}**  ",
             f"- Perfil de exercício: **{r.profile or 'livre'}**  ",
             f"- Modelo de pose: {r.backend}; detecção de objetos: YOLOv8  ",
             f"- Duração analisada: {r.duration_s}s ({r.frames_processed} quadros, {r.fps} fps)  ",
             f"- Gerado em {datetime.now():%d/%m/%Y %H:%M}", "", "## Resumo", ""]
    lines += [f"| Métrica | Valor |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r.summary.items()]
    if r.reps:
        lines += ["", "## Repetições", "", "| # | início (s) | fim (s) | ângulo mín. | amplitude | duração (s) |", "|---|---|---|---|---|---|"]
        lines += [f"| {x['rep']} | {x['start_s']} | {x['end_s']} | {x['min_angle']}° | {x['rom']}° | {x['duration_s']} |" for x in r.reps]
    lines += ["", "## Desvios e eventos detectados", ""]
    if r.events:
        lines += ["| Severidade | Evento | Trecho | Detalhe |", "|---|---|---|---|"]
        for ev in r.events:
            sev, title, _ = EVENT_META[ev["type"]]
            lines.append(f"| {sev.name} | {title} | {ev['start_s']:.1f}–{ev['end_s']:.1f}s | {ev.get('detail', '')} |")
    else:
        lines.append("Nenhum desvio detectado — execução dentro do padrão esperado.")
    lines += ["", "![Ângulos ao longo do tempo](angles.png)", "",
              "> Ferramenta de apoio à decisão. Os achados devem ser confirmados por profissional de saúde."]
    p = out_dir / "report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
