"""Detecção de objetos (YOLOv8) e monitoramento de áreas críticas.

As áreas críticas são polígonos normalizados (0-1) definidos em `config.yaml`
(ex.: campo estéril, mesa cirúrgica, região do leito). Para cada quadro, verifica:
  * pessoas dentro da área (ponto de apoio = base do bounding box)
  * objetos proibidos dentro da área (ex.: celular, copo, garrafa)
  * contagem de instrumentos (ex.: tesoura, bisturi) para conferência início × fim

Os pesos padrão são COCO (80 classes). Para cirurgia real, recomenda-se trocar
`detect_model` por pesos ajustados em instrumentais (ex.: m2cai16-tool-locations,
Cholec80) — a interface não muda.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]

    @property
    def anchor(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2) if self.cls == "person" else ((x1 + x2) / 2, (y1 + y2) / 2)


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        if res.boxes is None:
            return []
        out = []
        for box, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            out.append(Detection(self.names[int(k)], float(c), tuple(float(v) for v in box)))
        return out


class ZoneMonitor:
    def __init__(self, zones: dict[str, dict], frame_shape: tuple[int, int]):
        h, w = frame_shape
        zones = zones or {}
        self.zones = {name: np.array([[x * w, y * h] for x, y in z["polygon"]], dtype=np.float32)
                      for name, z in zones.items()}
        self.max_people = {name: int(z.get("max_people", 99)) for name, z in zones.items()}

    def inside(self, point: tuple[float, float]) -> list[str]:
        return [name for name, poly in self.zones.items()
                if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0]

    def summarize(self, detections: list[Detection], prohibited: list[str], instruments: list[str]) -> dict:
        summary = {}
        for name in self.zones:
            in_zone = [d for d in detections if name in self.inside(d.anchor)]
            summary[name] = {
                "people": sum(d.cls == "person" for d in in_zone),
                "prohibited": sorted({d.cls for d in in_zone if d.cls in prohibited}),
                "instruments": sum(d.cls in instruments for d in in_zone),
            }
        return summary

    def draw(self, frame: np.ndarray, alert_zones: set[str] = frozenset()) -> None:
        overlay = frame.copy()
        for name, poly in self.zones.items():
            color = (40, 40, 230) if name in alert_zones else (230, 160, 40)
            pts = poly.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            cv2.putText(frame, name, tuple(pts[0] + [6, 22]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_detections(frame: np.ndarray, detections: list[Detection], highlight: set[str] = frozenset()) -> None:
    for d in detections:
        if d.cls == "person":
            continue  # pessoas já recebem o esqueleto
        color = (40, 40, 230) if d.cls in highlight else (200, 200, 200)
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.cls} {d.conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
