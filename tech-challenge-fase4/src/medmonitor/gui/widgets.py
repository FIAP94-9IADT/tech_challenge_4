"""Componentes reutilizáveis da interface desktop."""

from __future__ import annotations

import traceback

import cv2
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

matplotlib.use("QtAgg")

SEV_COLOR = {"CRITICAL": "#e34948", "WARNING": "#eda100", "INFO": "#2a78d6"}
SEV_PT = {"CRITICAL": "CRÍTICO", "WARNING": "ATENÇÃO", "INFO": "INFORMATIVO"}
SERIES, DETECTED, MUTED = "#2a78d6", "#e34948", "#52514e"

STYLE = """
QMainWindow, QWidget { background: #fcfcfb; color: #1b1b1a; font-size: 13px; }
QTabWidget::pane { border: 1px solid #dcdbd6; border-radius: 6px; top: -1px; }
QTabBar::tab { padding: 8px 16px; border: 1px solid transparent; margin-right: 2px; color: #52514e; }
QTabBar::tab:selected { background: #ffffff; border: 1px solid #dcdbd6; border-bottom-color: #ffffff;
                        border-top-left-radius: 6px; border-top-right-radius: 6px; color: #0b0b0b; font-weight: 600; }
QPushButton { background: #2a78d6; color: white; border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; }
QPushButton:hover { background: #1f66bd; }
QPushButton:disabled { background: #b9c9de; }
QPushButton[secondary="true"] { background: #ecebe8; color: #1b1b1a; }
QTableWidget { background: #ffffff; border: 1px solid #dcdbd6; border-radius: 6px; gridline-color: #f0efec;
               selection-background-color: #dce9f8; selection-color: #0b0b0b; }
QHeaderView::section { background: #f4f3f0; padding: 6px; border: none; border-bottom: 1px solid #dcdbd6;
                       font-weight: 600; color: #52514e; }
QComboBox, QSpinBox, QLineEdit { background: #ffffff; border: 1px solid #c9c7c1; border-radius: 5px; padding: 4px 8px; }
QTextBrowser, QListWidget { background: #ffffff; border: 1px solid #dcdbd6; border-radius: 6px; }
QGroupBox { border: 1px solid #dcdbd6; border-radius: 6px; margin-top: 12px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QStatusBar { background: #f4f3f0; color: #52514e; }
"""


# ----------------------------------------------------------------------------- tarefas em segundo plano
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Executa uma função fora da thread da interface e devolve o resultado por sinal."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(*self.args, **self.kwargs))
        except Exception:  # noqa: BLE001 — a mensagem vai para a interface
            self.signals.failed.emit(traceback.format_exc(limit=4))


# ----------------------------------------------------------------------------- tabela
def make_table(headers: list[str], stretch_last: bool = True) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setAlternatingRowColors(False)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch_last:
        h.setStretchLastSection(True)
    return t


def fill_table(t: QTableWidget, rows: list[list], row_colors: list[str | None] | None = None,
               headers: list[str] | None = None) -> None:
    if headers is not None:
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            item = QTableWidgetItem("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)))
            if isinstance(v, (int, float)):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if row_colors and row_colors[i]:
                item.setBackground(QBrush(QColor(row_colors[i])))
            t.setItem(i, j, item)


def color_cell(t: QTableWidget, row: int, col: int, color: str, bold: bool = True) -> None:
    item = t.item(row, col)
    if item:
        item.setForeground(QBrush(QColor(color)))
        f = item.font()
        f.setBold(bold)
        item.setFont(f)


# ----------------------------------------------------------------------------- gráficos
class Canvas(FigureCanvasQTAgg):
    def __init__(self, height: float = 4.0):
        self.fig = Figure(figsize=(9, height), layout="constrained", facecolor="#fcfcfb")
        super().__init__(self.fig)

    def style(self, ax):
        ax.set_facecolor("#ffffff")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c7c1")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color="#ecebe8", lw=0.6)
        return ax


def shade(ax, x, mask, color=DETECTED, alpha=0.18):
    m = list(mask)
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            ax.axvspan(x.iloc[i], x.iloc[j], color=color, alpha=alpha, lw=0)
            i = j + 1
        else:
            i += 1


# ----------------------------------------------------------------------------- cartões de KPI
class KpiCard(QFrame):
    def __init__(self, title: str, color: str = "#0b0b0b"):
        super().__init__()
        self.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #dcdbd6; border-radius: 8px; }"
                           "QLabel { border: none; background: transparent; }")
        lay = QVBoxLayout(self)
        self.value = QLabel("—")
        self.value.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {color};")
        cap = QLabel(title)
        cap.setStyleSheet("color: #52514e;")
        lay.addWidget(self.value)
        lay.addWidget(cap)

    def set(self, v):
        self.value.setText(str(v))


# ----------------------------------------------------------------------------- player de vídeo
class VideoPlayer(QWidget):
    """Player baseado em OpenCV (independe dos codecs do backend multimídia do Qt)."""

    def __init__(self):
        super().__init__()
        self.cap = None
        self.fps = 25.0
        self.n = 0
        self.screen = QLabel("Nenhum vídeo carregado")
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setMinimumSize(480, 300)
        self.screen.setStyleSheet("background: #111; color: #aaa; border-radius: 6px;")
        self.btn = QPushButton("▶ Reproduzir")
        self.btn.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.seek)
        self.time = QLabel("0.0 s")
        ctr = QHBoxLayout()
        ctr.addWidget(self.btn)
        ctr.addWidget(self.slider, 1)
        ctr.addWidget(self.time)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.screen, 1)
        lay.addLayout(ctr)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next)

    def load(self, path: str) -> None:
        self.stop()
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, max(0, self.n - 1))
        self._next()

    def toggle(self):
        if not self.cap:
            return
        if self.timer.isActive():
            self.stop()
        else:
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.n - 1:
                self.seek(0)
            self.timer.start(int(1000 / self.fps))
            self.btn.setText("⏸ Pausar")

    def stop(self):
        self.timer.stop()
        self.btn.setText("▶ Reproduzir")

    def seek(self, frame: int):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self._next()

    def _next(self):
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok:
            self.stop()
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.screen.setPixmap(QPixmap.fromImage(img).scaled(self.screen.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.time.setText(f"{pos / self.fps:5.1f} s")


def scaled_pixmap(path: str, width: int) -> QPixmap:
    pm = QPixmap(path)
    return pm.scaledToWidth(width, Qt.SmoothTransformation) if not pm.isNull() else pm
