"""Estimativa de pose humana e cinemática articular.

Dois backends com a mesma interface, ambos retornando keypoints no layout COCO-17:

  * YoloPoseBackend  — Ultralytics YOLOv8-pose (padrão). Multi-pessoa, roda bem em CPU.
  * OpenPoseBackend  — modelo OpenPose (CMU, COCO-18) executado pelo módulo DNN do
    OpenCV; os 18 pontos são remapeados para o layout COCO-17. Requer os pesos
    Caffe originais (ver docs/RELATORIO_TECNICO.md).

A partir dos keypoints são calculados ângulos articulares (cotovelo, ombro,
quadril, joelho) e a inclinação do tronco em relação à vertical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
SKELETON = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
            ("l_hip", "r_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
            ("nose", "l_shoulder"), ("nose", "r_shoulder")]

# OpenPose COCO-18 → índice COCO-17 (o ponto 1 "pescoço" não existe no COCO-17)
OPENPOSE_TO_COCO = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16,
                    11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3}

# (ponto proximal, vértice, ponto distal)
JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "shoulder": ("hip", "shoulder", "elbow"),
    "hip": ("shoulder", "hip", "knee"),
    "knee": ("hip", "knee", "ankle"),
}


@dataclass
class PersonPose:
    keypoints: np.ndarray  # (17, 3): x, y, confiança
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class YoloPoseBackend:
    name = "YOLOv8-pose"

    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def estimate(self, frame: np.ndarray) -> list[PersonPose]:
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        return [PersonPose(np.concatenate([kxy[i], kconf[i][:, None]], axis=1), tuple(boxes[i]), float(scores[i]))
                for i in range(len(boxes))]


class OpenPoseBackend:
    """OpenPose (Cao et al., 2017) — modelo COCO via cv2.dnn, modo pessoa única."""

    name = "OpenPose (COCO, OpenCV DNN)"

    def __init__(self, proto: str, weights: str, conf: float = 0.1, size: int = 368):
        if not (Path(proto).exists() and Path(weights).exists()):
            raise FileNotFoundError(
                "Pesos do OpenPose não encontrados. Baixe pose_deploy_linevec.prototxt e "
                "pose_iter_440000.caffemodel para models/openpose/ (veja README).")
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        self.conf = conf
        self.size = size

    def estimate(self, frame: np.ndarray) -> list[PersonPose]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.size, self.size), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 57, H', W'): 18 mapas de calor + fundo + 38 PAFs
        kpts = np.zeros((17, 3), dtype=np.float32)
        for op_idx, coco_idx in OPENPOSE_TO_COCO.items():
            heat = out[0, op_idx]
            _, prob, _, (px, py) = cv2.minMaxLoc(heat)
            if prob >= self.conf:
                kpts[coco_idx] = (px * w / heat.shape[1], py * h / heat.shape[0], prob)
        valid = kpts[kpts[:, 2] > 0]
        if len(valid) < 4:
            return []
        x1, y1 = valid[:, :2].min(axis=0)
        x2, y2 = valid[:, :2].max(axis=0)
        return [PersonPose(kpts, (float(x1), float(y1), float(x2), float(y2)), float(valid[:, 2].mean()))]


def build_pose_backend(vcfg: dict, root: Path):
    if vcfg.get("pose_backend", "yolo") == "openpose":
        return OpenPoseBackend(str(root / vcfg["openpose_proto"]), str(root / vcfg["openpose_weights"]))
    return YoloPoseBackend(str(root / vcfg["pose_model"]), vcfg["conf_threshold"])


# ----------------------------------------------------------------------------- cinemática
def _pt(k: np.ndarray, name: str, min_conf: float = 0.3):
    x, y, c = k[KP[name]]
    return np.array([x, y]) if c >= min_conf else None


def angle(a, b, c) -> float:
    """Ângulo ABC em graus (vértice em B)."""
    if a is None or b is None or c is None:
        return np.nan
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def joint_angles(k: np.ndarray) -> dict[str, float]:
    out = {}
    for joint, (p, v, d) in JOINTS.items():
        for side in ("l", "r"):
            out[f"{joint}_{side}"] = angle(_pt(k, f"{side}_{p}"), _pt(k, f"{side}_{v}"), _pt(k, f"{side}_{d}"))
    ls, rs, lh, rh = (_pt(k, n) for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))
    sh = _mid(ls, rs)
    hp = _mid(lh, rh)
    if sh is not None and hp is not None:
        v = sh - hp
        out["trunk_lean"] = float(np.degrees(np.arctan2(abs(v[0]), -v[1]))) if np.linalg.norm(v) > 1 else np.nan
        out["hip_y"] = float(hp[1])
        out["torso_len"] = float(np.linalg.norm(v))
    else:
        out["trunk_lean"] = out["hip_y"] = out["torso_len"] = np.nan
    return out


def _mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


def draw_pose(frame: np.ndarray, person: PersonPose, color=(0, 255, 120)) -> None:
    k = person.keypoints
    for a, b in SKELETON:
        pa, pb = _pt(k, a), _pt(k, b)
        if pa is not None and pb is not None:
            cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb), color, 2, cv2.LINE_AA)
    for x, y, c in k:
        if c >= 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
