"""MedMonitor — aplicativo desktop da equipe médica.

Execute:  python main.py gui
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStyle, QSystemTrayIcon, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..alerts import Alert, AlertManager, Severity, load_alerts
from ..anomaly.movement import detect_movement_anomalies
from ..anomaly.vitals import VITAL_NAMES, VITAL_UNITS, StreamingVitalsMonitor, detect_vitals_anomalies, news2_score
from ..config import azure_language_enabled, azure_speech_enabled, load_config, resolve
from ..data.synthetic import load_csv
from .widgets import (DETECTED, MUTED, SERIES, SEV_COLOR, SEV_PT, STYLE, Canvas, KpiCard, Task, VideoPlayer,
                      color_cell, fill_table, make_table, scaled_pixmap, shade)

warnings.filterwarnings("ignore")
MOD_PT = {"vitals": "Sinais vitais", "prescription": "Prescrição", "movement": "Movimentação",
          "audio": "Áudio", "video": "Vídeo", "fusion": "Fusão multimodal"}


class State:
    """Resultados carregados do disco e caches compartilhados entre as abas."""

    def __init__(self):
        self.cfg = load_config()
        self.out = resolve(self.cfg["paths"]["output_dir"])
        self.data = resolve(self.cfg["paths"]["data_dir"])
        self.res: dict | None = None
        self.alerts: list[Alert] = []
        self._vit: dict = {}
        self._mov: dict = {}
        self.reload()

    def reload(self):
        p = self.out / "results.json"
        self.res = json.loads(p.read_text()) if p.exists() else None
        self.alerts = load_alerts(resolve(self.cfg["paths"]["alerts_log"]))
        self._vit.clear()
        self._mov.clear()

    @property
    def pids(self) -> list[str]:
        return sorted(p["patient_id"] for p in (self.res or {}).get("patients", []) if p["patient_id"].startswith("P"))

    def vitals(self, pid):
        if pid not in self._vit:
            df = load_csv(self.data / "synthetic" / "vitals.csv")
            self._vit[pid] = detect_vitals_anomalies(df[df["patient_id"] == pid], self.cfg).frame
        return self._vit[pid]

    def movement(self, pid):
        if pid not in self._mov:
            df = load_csv(self.data / "synthetic" / "movement.csv")
            self._mov[pid] = detect_movement_anomalies(df[df["patient_id"] == pid], self.cfg)
        return self._mov[pid]


def _hbox(*widgets, stretch_last=True):
    lay = QHBoxLayout()
    for w in widgets:
        lay.addWidget(w) if isinstance(w, QWidget) else lay.addLayout(w)
    if stretch_last:
        lay.addStretch(1)
    return lay


def _label(text, style=""):
    lb = QLabel(text)
    lb.setWordWrap(True)
    if style:
        lb.setStyleSheet(style)
    return lb


# ============================================================================= Central de alertas
class AlertsTab(QWidget):
    def __init__(self, st: State):
        super().__init__()
        self.st = st
        self.kpis = [KpiCard("Pacientes em risco alto", DETECTED), KpiCard("Alertas críticos", DETECTED),
                     KpiCard("Alertas de fusão", SERIES), KpiCard("Suprimidos (cooldown)", MUTED)]
        kp = QHBoxLayout()
        for k in self.kpis:
            kp.addWidget(k)
        self.risk = make_table(["Paciente", "Nome", "Leito", "Cenário", "Risco", "Nível", "Vitais", "Prescr.",
                                "Movim.", "Áudio", "Vídeo"])
        self.risk.setMaximumHeight(240)

        self.sev = {s: QCheckBox(SEV_PT[s]) for s in ("CRITICAL", "WARNING", "INFO")}
        for s, cb in self.sev.items():
            cb.setChecked(s != "INFO")
            cb.setStyleSheet(f"color: {SEV_COLOR[s]}; font-weight: 600;")
            cb.toggled.connect(self.fill_feed)
        self.mod = QComboBox()
        self.mod.addItem("Todas as modalidades", None)
        for k, v in MOD_PT.items():
            self.mod.addItem(v, k)
        self.pat = QComboBox()
        self.mod.currentIndexChanged.connect(self.fill_feed)
        self.pat.currentIndexChanged.connect(self.fill_feed)
        self.feed = make_table(["Horário", "Severidade", "Paciente", "Modalidade", "Alerta"])
        self.feed.itemSelectionChanged.connect(self.show_detail)
        self.detail = QTextBrowser()

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(_hbox(*self.sev.values(), self.mod, self.pat))
        ll.addWidget(self.feed)
        split.addWidget(left)
        split.addWidget(self.detail)
        split.setSizes([700, 420])

        lay = QVBoxLayout(self)
        lay.addLayout(kp)
        lay.addWidget(_label("<b>Risco multimodal por paciente</b>"))
        lay.addWidget(self.risk)
        lay.addWidget(_label("<b>Feed de alertas</b> — selecione um alerta para ver evidências e conduta"))
        lay.addWidget(split, 1)

    def refresh(self):
        res = self.st.res
        alerts = self.st.alerts
        pts = res["patients"]
        self.kpis[0].set(sum(p["level"] == "alto" for p in pts))
        self.kpis[1].set(sum(a.severity is Severity.CRITICAL for a in alerts))
        self.kpis[2].set(sum(a.modality == "fusion" for a in alerts))
        self.kpis[3].set(res["alerts"]["suppressed"])
        rows, colors = [], []
        for p in pts:
            ms = p["modality_scores"]
            rows.append([p["patient_id"], p.get("nome") or "—", p.get("leito") or "—", p.get("cenario") or "—",
                         round(p["risk"], 2), p["level"]] + [ms.get(m) for m in ("vitals", "prescription",
                                                                                   "movement", "audio", "video")])
        fill_table(self.risk, rows)
        level_color = {"alto": DETECTED, "moderado": "#c98500", "baixo": "#1baf7a"}
        for i, p in enumerate(pts):
            color_cell(self.risk, i, 4, level_color[p["level"]])
            color_cell(self.risk, i, 5, level_color[p["level"]])
        self.pat.blockSignals(True)
        self.pat.clear()
        self.pat.addItem("Todos os pacientes", None)
        for pid in sorted({a.patient_id for a in alerts}):
            self.pat.addItem(pid, pid)
        self.pat.blockSignals(False)
        self.fill_feed()

    def fill_feed(self):
        sev = {s for s, cb in self.sev.items() if cb.isChecked()}
        mod, pat = self.mod.currentData(), self.pat.currentData()
        self.shown = sorted([a for a in self.st.alerts if a.severity.name in sev and (not mod or a.modality == mod)
                             and (not pat or a.patient_id == pat)], key=lambda a: a.timestamp, reverse=True)
        fill_table(self.feed, [[a.timestamp[5:16].replace("T", " "), SEV_PT[a.severity.name], a.patient_id,
                                MOD_PT[a.modality], a.title] for a in self.shown],
                   ["#eef4fc" if a.modality == "fusion" else None for a in self.shown])
        for i, a in enumerate(self.shown):
            color_cell(self.feed, i, 1, SEV_COLOR[a.severity.name])
        if self.shown:
            self.feed.selectRow(0)
        else:
            self.detail.clear()

    def show_detail(self):
        rows = self.feed.selectionModel().selectedRows()
        if not rows:
            return
        a = self.shown[rows[0].row()]
        ev = "".join(f"<tr><td style='color:#52514e;padding-right:12px'>{k}</td><td>{v}</td></tr>" for k, v in a.evidence.items())
        self.detail.setHtml(
            f"<h3 style='color:{SEV_COLOR[a.severity.name]}'>{SEV_PT[a.severity.name]} — {a.title}</h3>"
            f"<p><b>Paciente:</b> {a.patient_id} &nbsp; <b>Modalidade:</b> {MOD_PT[a.modality]} &nbsp; "
            f"<b>Horário:</b> {a.timestamp.replace('T', ' ')}</p><p>{a.description}</p>"
            f"<p style='background:#eef4fc;padding:8px'><b>Conduta sugerida:</b> {a.recommended_action}</p>"
            f"<p><b>Evidências</b></p><table>{ev}</table>")


# ============================================================================= Sinais vitais
class VitalsTab(QWidget):
    def __init__(self, st: State, notify):
        super().__init__()
        self.st, self.notify = st, notify
        self.pid = QComboBox()
        self.pid.currentTextChanged.connect(self.draw)
        self.canvas = Canvas(7)
        self.events = make_table(["Início", "Duração (min)", "Condições", "NEWS2 máx.", "Severidade"])
        self.events.setMaximumHeight(150)

        # monitor em tempo real
        live = QGroupBox("Monitor em tempo real (replay das amostras, uma por vez)")
        self.start_h = QSpinBox()
        self.start_h.setRange(0, 23)
        self.start_h.setValue(12)
        self.start_h.setSuffix(" h após admissão")
        self.speed = QComboBox()
        for s in (5, 10, 30, 60, 120):
            self.speed.addItem(f"{s} amostras/s", s)
        self.speed.setCurrentIndex(2)
        self.btn = QPushButton("▶ Iniciar")
        self.btn.clicked.connect(self.toggle_live)
        self.vals = {k: QLabel("—") for k in ("hr", "sbp", "spo2", "rr", "temp", "news2", "clock")}
        vals_box = QWidget()
        vals_box.setMaximumHeight(70)
        grid = QGridLayout(vals_box)
        grid.setVerticalSpacing(2)
        for i, (k, name) in enumerate([("clock", "Horário"), ("hr", "FC"), ("sbp", "PA"), ("spo2", "SpO2"),
                                       ("rr", "FR"), ("temp", "Temp"), ("news2", "NEWS2")]):
            grid.addWidget(_label(name, f"color:{MUTED}"), 0, i)
            self.vals[k].setStyleSheet("font-size: 20px; font-weight: 700;")
            grid.addWidget(self.vals[k], 1, i)
        self.live_canvas = Canvas(2.6)
        self.live_alerts = QListWidget()
        self.live_alerts.setMaximumHeight(140)
        ll = QVBoxLayout(live)
        ll.addLayout(_hbox(self.start_h, self.speed, self.btn))
        ll.addWidget(vals_box)
        ll.addWidget(self.live_canvas, 1)
        ll.addWidget(self.live_alerts)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        hist = QWidget()
        hl = QVBoxLayout(hist)
        hl.addWidget(self.canvas, 1)
        hl.addWidget(self.events)
        sub = QTabWidget()
        sub.addTab(hist, "Histórico de 24 h")
        sub.addTab(live, "Tempo real")
        lay = QVBoxLayout(self)
        lay.addLayout(_hbox(_label("Paciente:"), self.pid))
        lay.addWidget(sub, 1)
        self.sub = sub

    def refresh(self):
        cur = self.pid.currentText()
        self.pid.blockSignals(True)
        self.pid.clear()
        self.pid.addItems(self.st.pids)
        if cur in self.st.pids:
            self.pid.setCurrentText(cur)
        self.pid.blockSignals(False)
        self.draw()

    def draw(self):
        pid = self.pid.currentText()
        if not pid:
            return
        df = self.st.vitals(pid)
        cols = [c for c in VITAL_NAMES if c in df]
        fig = self.canvas.fig
        fig.clear()
        axes = fig.subplots(len(cols), 1, sharex=True)
        for ax, c in zip(axes, cols):
            self.canvas.style(ax)
            ax.plot(df["timestamp"], df[c], color=SERIES, lw=0.9)
            shade(ax, df["timestamp"], df["anomaly"])
            ax.set_ylabel(f"{VITAL_NAMES[c]}\n{VITAL_UNITS[c]}", rotation=0, ha="right", va="center", fontsize=8, color=MUTED)
        axes[0].set_title(f"{pid} — faixas vermelhas: anomalias detectadas (ensemble)", loc="left", fontsize=9)
        fig.autofmt_xdate()
        self.canvas.draw_idle()
        ev = self.st.res["vitals_events"].get(pid, [])
        fill_table(self.events, [[e["start"][5:16], e["duration_min"], " + ".join(e["conditions"]), e["max_news2"],
                                  SEV_PT[e["severity"]]] for e in ev])
        for i, e in enumerate(ev):
            color_cell(self.events, i, 4, SEV_COLOR[e["severity"]])

    # -- replay em tempo real
    def toggle_live(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText("▶ Iniciar")
            return
        pid = self.pid.currentText()
        df = self.st.vitals(pid)
        warm = int(self.st.cfg["vitals"]["baseline_hours"] * 60)
        s = self.start_h.value() * 60
        self.mon = StreamingVitalsMonitor(pid, self.st.cfg)
        # mesmo cooldown do sistema: evita repetir o mesmo alerta (fadiga de alarmes)
        self.mgr = AlertManager({**self.st.cfg, "alerts": {**self.st.cfg["alerts"], "console": False, "webhook_url": ""}})
        for row in df.iloc[max(0, s - warm):s].to_dict(orient="records"):
            self.mon.update(row)  # aquecimento: aprende a linha de base sem exibir
        self.queue = df.iloc[s:].to_dict(orient="records")
        self.buf = []
        self.live_alerts.clear()
        self.timer.start(int(1000 / self.speed.currentData()))
        self.btn.setText("⏹ Parar")

    def tick(self):
        if not self.queue:
            self.toggle_live()
            return
        row = self.queue.pop(0)
        self.buf.append(row)
        n, _ = news2_score(row)
        self.vals["clock"].setText(pd.Timestamp(row["timestamp"]).strftime("%d/%m %H:%M"))
        self.vals["hr"].setText(str(row["hr"]))
        self.vals["sbp"].setText(f"{row['sbp']}/{row['dbp']}")
        self.vals["spo2"].setText(f"{row['spo2']}%")
        self.vals["rr"].setText(str(row["rr"]))
        self.vals["temp"].setText(f"{row['temp']:.1f}")
        self.vals["news2"].setText(str(n))
        self.vals["news2"].setStyleSheet(f"font-size:20px;font-weight:700;color:{DETECTED if n >= 5 else '#0b0b0b'}")
        for a in [a for a in self.mon.update(row) if self.mgr.dispatch(a)]:
            it = QListWidgetItem(f"{a.timestamp[11:16]}  {SEV_PT[a.severity.name]} — {a.title}: {a.description}")
            it.setForeground(QColor(SEV_COLOR[a.severity.name]))
            self.live_alerts.insertItem(0, it)
            self.notify(a)
        if len(self.buf) % 5 == 0:
            b = pd.DataFrame(self.buf[-240:])
            fig = self.live_canvas.fig
            fig.clear()
            axes = fig.subplots(1, 3)
            for ax, k in zip(axes, ("hr", "spo2", "rr")):
                self.live_canvas.style(ax)
                ax.plot(range(len(b)), b[k], color=SERIES, lw=1.2)
                ax.set_title(f"{VITAL_NAMES[k]} ({VITAL_UNITS[k]})", fontsize=9, loc="left")
                ax.set_xticks([])
            self.live_canvas.draw_idle()


# ============================================================================= Prescrições
class RxTab(QWidget):
    def __init__(self, st: State):
        super().__init__()
        self.st = st
        self.pid = QComboBox()
        self.pid.currentIndexChanged.connect(self.fill)
        self.table = make_table(["Paciente", "Horário", "Medicamento", "Dose", "Freq./dia", "Ação", "Razão",
                                 "Alta vigilância", "Escore IF", "Achados"])
        lay = QVBoxLayout(self)
        lay.addLayout(_hbox(_label("Paciente:"), self.pid,
                            _label("Linhas em vermelho: prescrição atípica", f"color:{DETECTED}")))
        lay.addWidget(self.table)

    def refresh(self):
        self.pid.blockSignals(True)
        self.pid.clear()
        self.pid.addItem("Todos", None)
        for p in self.st.pids:
            self.pid.addItem(p, p)
        self.pid.blockSignals(False)
        self.fill()

    def fill(self):
        pid = self.pid.currentData()
        rx = [r for r in self.st.res["prescriptions"] if not pid or r["patient_id"] == pid]
        fill_table(self.table, [[r["patient_id"], r["timestamp"][5:16].replace("T", " "), r["drug"],
                                 f"{r['dose']:g} {r['unit']}", r["frequency_per_day"], r["action"], r["dose_ratio"],
                                 "sim" if r["high_alert"] else "", round(r["iforest_score"], 3),
                                 "; ".join(r["findings"])] for r in rx],
                   ["#fbe4e4" if r["anomaly"] else None for r in rx])


# ============================================================================= Movimentação
class MovementTab(QWidget):
    def __init__(self, st: State):
        super().__init__()
        self.st = st
        self.pid = QComboBox()
        self.pid.currentTextChanged.connect(self.draw)
        self.canvas = Canvas(4.5)
        self.table = make_table(["Evento", "Início", "Fim", "Minutos"])
        lay = QVBoxLayout(self)
        lay.addLayout(_hbox(_label("Paciente:"), self.pid))
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.table)

    def refresh(self):
        self.pid.blockSignals(True)
        self.pid.clear()
        self.pid.addItems(self.st.pids)
        self.pid.blockSignals(False)
        self.draw()

    def draw(self):
        pid = self.pid.currentText()
        if not pid:
            return
        r = self.st.movement(pid)
        df = r.frame
        fig = self.canvas.fig
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [2.2, 1]})
        self.canvas.style(ax1)
        ax1.plot(df["timestamp"], df["activity"], color=SERIES, lw=0.6)
        shade(ax1, df["timestamp"], df["night"], color="#6250d6", alpha=0.05)
        shade(ax1, df["timestamp"], df["anomaly"])
        for e in r.events:
            ax1.annotate(e["type"], (df["timestamp"].iloc[e["start"]], ax1.get_ylim()[1] * 0.92), fontsize=8, color=MUTED)
        ax1.set_ylabel("atividade/min", fontsize=8, color=MUTED)
        ax1.set_title(f"{pid} — movimentação no leito (lilás = noite, vermelho = anomalia)", loc="left", fontsize=9)
        self.canvas.style(ax2)
        if r.hourly is not None:
            ax2.bar(r.hourly.index, r.hourly["atypicality"], width=0.03, color=SERIES)
        ax2.set_ylabel("atipicidade\n(IF, horária)", fontsize=8, color=MUTED)
        fig.autofmt_xdate()
        self.canvas.draw_idle()
        fill_table(self.table, [[e["type"], e["start_ts"][5:16], e["end_ts"][5:16], e["minutes"]] for e in r.events])


# ============================================================================= Áudio
class AudioTab(QWidget):
    def __init__(self, st: State, pool: QThreadPool, status):
        super().__init__()
        self.st, self.pool, self.status = st, pool, status
        self.items: list[dict] = []
        self.list = QListWidget()
        self.list.setMaximumWidth(280)
        self.list.currentRowChanged.connect(self.show_idx)
        self.open_btn = QPushButton("Abrir áudio…")
        self.open_btn.clicked.connect(self.open_file)
        self.play_btn = QPushButton("▶ Ouvir")
        self.play_btn.setProperty("secondary", True)
        self.play_btn.clicked.connect(self.play)
        self.canvas = Canvas(2.2)
        self.text = QTextBrowser()
        self.feats = make_table(["Biomarcador", "Valor"])
        self.player = None
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self.player, self.audio_out = QMediaPlayer(), QAudioOutput()
            self.player.setAudioOutput(self.audio_out)
        except Exception:  # noqa: BLE001 — sem backend multimídia, o botão abre o player do sistema
            pass

        left = QVBoxLayout()
        left.addWidget(_label("<b>Consultas</b>"))
        left.addWidget(self.list, 1)
        left.addWidget(self.open_btn)
        right = QVBoxLayout()
        right.addLayout(_hbox(self.play_btn))
        right.addWidget(self.canvas)
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self.text)
        sp.addWidget(self.feats)
        sp.setSizes([600, 300])
        right.addWidget(sp, 1)
        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

    def refresh(self):
        self.items = [dict(a, _path=str(self.st.data / "audio" / a["file"])) for a in self.st.res.get("audio", [])]
        self._fill_list()

    def _fill_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for a in self.items:
            self.list.addItem(f"{a['patient_id']} · {a['file']}")
        self.list.blockSignals(False)
        if self.items:
            self.list.setCurrentRow(len(self.items) - 1 if self.items[-1].get("_new") else 0)

    def show_idx(self, i):
        if 0 <= i < len(self.items):
            self.show_item(self.items[i])

    def show_item(self, a):
        self.current = a
        ind = {k: v for k, v in a["indicators"].items() if not k.startswith("_")}
        fig = self.canvas.fig
        fig.clear()
        ax = self.canvas.style(fig.subplots())
        names = [k.replace("_", " ") for k in ind]
        ax.barh(names, list(ind.values()), color=[DETECTED if v >= 0.5 else SERIES for v in ind.values()], height=0.5)
        for y, v in enumerate(ind.values()):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9, color=MUTED)
        ax.axvline(0.5, color="#9a9893", ls="--", lw=0.8)
        ax.set_xlim(0, 1.1)
        ax.set_title("Indicadores vocais (0–1) · tracejado = limiar de alerta", loc="left", fontsize=9)
        self.canvas.draw_idle()

        ta, tr = a["text_analysis"], a["transcript"]
        terms = ", ".join(f"<s>{t['term']}</s> (negado)" if t["negated"] else f"<b>{t['term']}</b> ({t['category']})"
                          for t in ta["critical_terms"]) or "—"
        alerts = "".join(f"<p style='color:{SEV_COLOR[x['severity']]}'><b>{SEV_PT[x['severity']]} — {x['title']}</b><br>"
                         f"{x['description']}</p>" for x in a["alerts"]) or "<p>Nenhum alerta.</p>"
        extra = ""
        if ta.get("key_phrases"):
            extra += f"<p><b>Frases-chave (Azure):</b> {', '.join(ta['key_phrases'])}</p>"
        if ta.get("health_entities"):
            extra += "<p><b>Entidades de saúde (Azure):</b> " + ", ".join(
                f"{e['text']} [{e['category']}]" for e in ta["health_entities"]) + "</p>"
        self.text.setHtml(f"<h3>{a['file']}</h3><p><b>Transcrição</b> <i>({tr['source']})</i>:<br><i>{tr['text'] or '—'}</i></p>"
                          f"<p><b>Sentimento</b> <i>({ta['source']})</i>: {ta['sentiment']['label']} "
                          f"{ta['sentiment']['scores']}</p><p><b>Termos críticos:</b> {terms}</p>{extra}<hr>{alerts}")
        fill_table(self.feats, [[k, v] for k, v in a["features"].items()])

    def play(self):
        path = getattr(self, "current", {}).get("_path")
        if not path:
            return
        if self.player:
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir áudio", str(Path.home()), "Áudio (*.wav *.mp3 *.m4a *.ogg *.flac)")
        if not path:
            return
        from ..audio.analyzer import analyze_audio
        self.open_btn.setEnabled(False)
        self.status("Analisando áudio (biomarcadores + transcrição + texto)…")
        task = Task(lambda: analyze_audio(path, self.st.cfg, "novo").to_dict())
        task.signals.done.connect(lambda d: self._added(d, path))
        task.signals.failed.connect(self._failed)
        self.pool.start(task)

    def _added(self, d, path):
        self.open_btn.setEnabled(True)
        self.items.append(dict(d, _path=path, _new=True))
        self._fill_list()
        self.status("Áudio analisado.")

    def _failed(self, msg):
        self.open_btn.setEnabled(True)
        QMessageBox.critical(self, "Erro", msg)


# ============================================================================= Vídeo
class VideoTab(QWidget):
    def __init__(self, st: State, pool: QThreadPool, status):
        super().__init__()
        self.st, self.pool, self.status = st, pool, status
        self.items: list[dict] = []
        self.list = QListWidget()
        self.list.setMaximumWidth(280)
        self.list.currentRowChanged.connect(self.show_idx)
        self.profile = QComboBox()
        self.profile.addItem("(livre)", None)
        for p in st.cfg["video"]["exercise_profiles"]:
            self.profile.addItem(p, p)
        self.target = QComboBox()
        self.target.addItems(["largest", "left", "right", "center"])
        self.zones = QCheckBox("Áreas críticas")
        self.openpose = QCheckBox("OpenPose")
        self.open_btn = QPushButton("Abrir vídeo…")
        self.open_btn.clicked.connect(self.open_file)
        self.player = VideoPlayer()
        self.info = QTextBrowser()
        self.plot = QLabel()
        self.plot.setAlignment(Qt.AlignCenter)
        self.reps = make_table(["#", "início (s)", "fim (s)", "ângulo mín.", "amplitude", "duração (s)"])

        left = QVBoxLayout()
        left.addWidget(_label("<b>Vídeos</b>"))
        left.addWidget(self.list, 1)
        form = QGridLayout()
        form.addWidget(_label("Perfil"), 0, 0)
        form.addWidget(self.profile, 0, 1)
        form.addWidget(_label("Pessoa"), 1, 0)
        form.addWidget(self.target, 1, 1)
        form.addWidget(self.zones, 2, 0)
        form.addWidget(self.openpose, 2, 1)
        left.addLayout(form)
        left.addWidget(self.open_btn)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.player)
        top.addWidget(self.info)
        top.setSizes([620, 380])
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.addWidget(self.plot)
        bl.addWidget(self.reps)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(bottom)
        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(scroll)
        right.setSizes([420, 380])
        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addWidget(right, 1)

    def refresh(self):
        self.items = list(self.st.res.get("video", []))
        self._fill_list()

    def _fill_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for v in self.items:
            self.list.addItem(f"{v['patient_id']} · {v['file']}")
        self.list.blockSignals(False)
        if self.items:
            self.list.setCurrentRow(len(self.items) - 1 if self.items[-1].get("_new") else 0)

    def show_idx(self, i):
        if not 0 <= i < len(self.items):
            return
        v = self.items[i]
        if v.get("annotated_path") and Path(v["annotated_path"]).exists():
            self.player.load(v["annotated_path"])
        summ = "".join(f"<tr><td style='color:#52514e;padding-right:12px'>{k}</td><td>{val}</td></tr>"
                       for k, val in v["summary"].items())
        alerts = "".join(f"<p style='color:{SEV_COLOR[a['severity']]}'><b>{SEV_PT[a['severity']]} — {a['title']}</b><br>"
                         f"{a['description']}</p>" for a in v["alerts"]) or "<p>Nenhum desvio detectado.</p>"
        self.info.setHtml(f"<h3>{v['file']}</h3><p style='color:#52514e'>{v.get('description', '')}<br>"
                          f"Fonte: {v.get('source', '—')} · Modelo: {v['backend']} · Perfil: {v['profile'] or 'livre'}</p>"
                          f"<table>{summ}</table><hr>{alerts}")
        if v.get("plot_path") and Path(v["plot_path"]).exists():
            self.plot.setPixmap(scaled_pixmap(v["plot_path"], 900))
        fill_table(self.reps, [[r["rep"], r["start_s"], r["end_s"], r["min_angle"], r["rom"], r["duration_s"]]
                               for r in v["reps"]])

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo", str(Path.home()), "Vídeo (*.mp4 *.mov *.avi *.webm *.mkv)")
        if not path:
            return
        from ..video.analyzer import VideoAnalyzer
        cfg = {**self.st.cfg, "video": {**self.st.cfg["video"],
                                        "pose_backend": "openpose" if self.openpose.isChecked() else "yolo"}}
        prof, target, zones = self.profile.currentData(), self.target.currentText(), self.zones.isChecked()
        self.open_btn.setEnabled(False)
        self.status("Estimando pose e detectando objetos (CPU) — pode levar alguns minutos…")
        task = Task(lambda: VideoAnalyzer(cfg).analyze(path, "novo", prof, self.st.out / "video",
                                                       target=target, use_zones=zones).to_dict())
        task.signals.done.connect(self._added)
        task.signals.failed.connect(self._failed)
        self.pool.start(task)

    def _added(self, d):
        self.open_btn.setEnabled(True)
        self.items.append(dict(d, _new=True))
        self._fill_list()
        self.status("Vídeo analisado.")

    def _failed(self, msg):
        self.open_btn.setEnabled(True)
        QMessageBox.critical(self, "Erro", msg)


# ============================================================================= Métricas
class MetricsTab(QWidget):
    def __init__(self, st: State):
        super().__init__()
        self.st = st
        self.tables = {k: make_table(["-"]) for k in ("vitals_methods", "vitals_by_patient", "streaming",
                                                      "movement", "audio", "prescriptions")}
        titles = {"vitals_methods": "Sinais vitais — comparação de detectores (por evento)",
                  "vitals_by_patient": "Sinais vitais — ensemble por paciente",
                  "streaming": "Monitor em tempo real — atraso até o 1º alerta (min; negativo = antecipado)",
                  "movement": "Movimentação", "audio": "Áudio — condição esperada × prevista",
                  "prescriptions": "Prescrições"}
        inner = QWidget()
        lay = QVBoxLayout(inner)
        for k, t in self.tables.items():
            t.setMinimumHeight(150)
            lay.addWidget(_label(f"<b>{titles[k]}</b>"))
            lay.addWidget(t)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        QVBoxLayout(self).addWidget(scroll)

    def refresh(self):
        m = self.st.res["metrics"]
        for k, t in self.tables.items():
            rows = m[k] if isinstance(m[k], list) else [m[k]]
            if rows:
                cols = list(rows[0].keys())
                fill_table(t, [[", ".join(r[c]) if isinstance(r.get(c), list) else r.get(c) for c in cols] for r in rows],
                           headers=cols)


# ============================================================================= janela principal
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.st = State()
        self.pool = QThreadPool.globalInstance()
        self.setWindowTitle("MedMonitor — Monitoramento Multimodal de Pacientes")
        self.resize(1360, 880)

        tb = self.addToolBar("Ações")
        tb.setMovable(False)
        self.run_act = QAction("▶ Executar pipeline", self)
        self.run_act.triggered.connect(self.run_pipeline)
        reload_act = QAction("⟳ Recarregar", self)
        reload_act.triggered.connect(self.reload)
        report_act = QAction("📄 Relatório de resultados", self)
        report_act.triggered.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.st.out / "relatorio_resultados.md"))))
        for a in (self.run_act, reload_act, report_act):
            tb.addAction(a)
        az = (f"Azure Speech: {'ativo' if azure_speech_enabled(self.st.cfg) else 'fallback local'}  ·  "
              f"Azure Language: {'ativo' if azure_language_enabled(self.st.cfg) else 'fallback local'}")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        az_label = QLabel(az + "   ")
        az_label.setStyleSheet(f"color:{MUTED}")
        tb.addWidget(az_label)

        self.banner = _label("", f"background:{DETECTED};color:white;font-weight:700;padding:8px;border-radius:4px")
        self.banner.hide()
        self.tabs = QTabWidget()
        self.alerts_tab = AlertsTab(self.st)
        self.vitals_tab = VitalsTab(self.st, self.notify)
        self.pages = [self.alerts_tab, self.vitals_tab, RxTab(self.st), MovementTab(self.st),
                      AudioTab(self.st, self.pool, self.status), VideoTab(self.st, self.pool, self.status),
                      MetricsTab(self.st)]
        for page, name in zip(self.pages, ["🚨 Central de alertas", "❤️ Sinais vitais", "💊 Prescrições",
                                            "🛏️ Movimentação", "🎙️ Áudio", "🎥 Vídeo", "📊 Métricas"]):
            self.tabs.addTab(page, name)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(self.banner)
        cl.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.SP_MessageBoxWarning), self)
            self.tray.show()
        self.reload()

    def status(self, msg: str):
        self.statusBar().showMessage(msg)

    def notify(self, a: Alert):
        """Alerta crítico em tempo real: faixa vermelha na janela + notificação do sistema."""
        if a.severity is not Severity.CRITICAL:
            return
        self.banner.setText(f"🔴 CRÍTICO {a.timestamp[11:16]} · {a.patient_id} · {a.title} — {a.recommended_action}")
        self.banner.show()
        QTimer.singleShot(12000, self.banner.hide)
        QApplication.beep()
        if self.tray:
            self.tray.showMessage(f"MedMonitor — {a.patient_id}", a.title, QSystemTrayIcon.Critical, 8000)

    def reload(self):
        self.st.reload()
        if not self.st.res:
            self.status("Nenhum resultado encontrado — clique em “Executar pipeline”.")
            return
        for p in self.pages:
            p.refresh()
        self.status(f"Resultados de {self.st.res['generated_at'].replace('T', ' ')} · "
                    f"{len(self.st.alerts)} alertas · {self.st.out}")
        crit = [a for a in self.st.alerts if a.modality == "fusion" and a.severity is Severity.CRITICAL]
        if crit:
            self.banner.setText(f"🔴 {len(crit)} alertas críticos de fusão multimodal — veja a Central de alertas")
            self.banner.show()

    def run_pipeline(self):
        from ..pipeline import run_pipeline
        self.run_act.setEnabled(False)
        self.status("Processando vídeo, áudio, sinais vitais, prescrições e movimentação…")
        task = Task(run_pipeline, self.st.cfg, quiet=True)
        task.signals.done.connect(lambda _: (self.run_act.setEnabled(True), self.reload()))
        task.signals.failed.connect(lambda m: (self.run_act.setEnabled(True), QMessageBox.critical(self, "Erro", m)))
        self.pool.start(task)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MedMonitor")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    return app.exec()
