"""Smoke test do aplicativo desktop em modo offscreen (sem monitor)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from medmonitor.config import load_config, resolve  # noqa: E402

pytestmark = pytest.mark.skipif(not (resolve(load_config()["paths"]["output_dir"]) / "results.json").exists(),
                                reason="rode `python main.py demo` antes")


def _pump(ms):
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_main_window_tabs_and_live_monitor(cfg):  # cfg desliga Telegram/webhook
    from PySide6.QtWidgets import QApplication

    from medmonitor.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = MainWindow()
    w.show()
    assert w.tabs.count() == 7
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i)
        _pump(100)
    assert w.alerts_tab.feed.rowCount() > 0 and w.alerts_tab.risk.rowCount() >= 5

    vt = w.vitals_tab
    vt.pid.setCurrentText("P003")
    vt.start_h.setValue(14)
    vt.speed.setCurrentIndex(4)  # 120 amostras/s
    vt.toggle_live()
    _pump(4000)
    vt.toggle_live()
    assert vt.live_alerts.count() > 0  # depressão respiratória do P003 surge ao vivo
    w.close()
