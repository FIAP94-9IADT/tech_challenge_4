"""Modelo de alerta clínico e despacho para a equipe médica.

Canais suportados:
  * console (colorido por severidade)
  * arquivo JSONL (consumido pelo aplicativo desktop e auditável)
  * webhook (Microsoft Teams Workflows / Slack / endpoint genérico)

Inclui supressão de alertas repetidos (cooldown) para reduzir fadiga de alarmes.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable

import requests

log = logging.getLogger(__name__)


class Severity(IntEnum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, value: "str | int | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.upper()]


SEVERITY_LABEL_PT = {Severity.INFO: "INFORMATIVO", Severity.WARNING: "ATENÇÃO", Severity.CRITICAL: "CRÍTICO"}
_ANSI = {Severity.INFO: "\033[36m", Severity.WARNING: "\033[33m", Severity.CRITICAL: "\033[1;31m"}


@dataclass
class Alert:
    patient_id: str
    modality: str  # vitals | prescription | movement | audio | video | fusion
    severity: Severity
    title: str
    description: str
    timestamp: str  # ISO 8601 do evento clínico (não do processamento)
    evidence: dict = field(default_factory=dict)
    recommended_action: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Alert":
        d = dict(d)
        d["severity"] = Severity.parse(d["severity"])
        return cls(**d)

    def short(self) -> str:
        return f"[{SEVERITY_LABEL_PT[self.severity]}] {self.patient_id} · {self.modality} · {self.title}"


class AlertManager:
    """Recebe alertas de todos os módulos, deduplica e despacha para os canais."""

    def __init__(self, cfg: dict, log_path: str | Path | None = None, extra_sinks: Iterable[Callable[[Alert], None]] = ()):
        acfg = cfg["alerts"]
        self.cooldown = timedelta(minutes=acfg.get("cooldown_minutes", 30))
        self.console = acfg.get("console", True)
        self.webhook_url = acfg.get("webhook_url", "")
        self.webhook_format = acfg.get("webhook_format", "teams")
        self.min_webhook = Severity.parse(acfg.get("min_severity_webhook", "WARNING"))
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.extra_sinks = list(extra_sinks)
        self._last_sent: dict[tuple, datetime] = {}
        self.history: list[Alert] = []
        self.suppressed = 0

    def _is_duplicate(self, alert: Alert) -> bool:
        key = (alert.patient_id, alert.modality, alert.title)
        ts = datetime.fromisoformat(alert.timestamp)
        last = self._last_sent.get(key)
        if last is not None and abs(ts - last) < self.cooldown:
            return True
        self._last_sent[key] = ts
        return False

    def dispatch(self, alert: Alert) -> bool:
        """Despacha um alerta. Retorna False se foi suprimido pelo cooldown."""
        if self._is_duplicate(alert):
            self.suppressed += 1
            return False
        self.history.append(alert)
        if self.console:
            print(f"{_ANSI[alert.severity]}{alert.short()}\033[0m  ({alert.timestamp})\n    {alert.description}")
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False, default=str) + "\n")
        if self.webhook_url and alert.severity >= self.min_webhook:
            self._send_webhook(alert)
        for sink in self.extra_sinks:
            sink(alert)
        return True

    def dispatch_many(self, alerts: Iterable[Alert]) -> list[Alert]:
        ordered = sorted(alerts, key=lambda a: a.timestamp)
        return [a for a in ordered if self.dispatch(a)]

    # ------------------------------------------------------------------ webhook
    def _send_webhook(self, alert: Alert) -> None:
        try:
            resp = requests.post(self.webhook_url, json=self._webhook_payload(alert), timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:  # alerta não pode derrubar o pipeline
            log.warning("Falha ao enviar webhook do alerta %s: %s", alert.id, exc)

    def _webhook_payload(self, alert: Alert) -> dict:
        label = SEVERITY_LABEL_PT[alert.severity]
        text = (f"**{label} — {alert.title}**\n\nPaciente: {alert.patient_id} | Modalidade: {alert.modality} | "
                f"Horário: {alert.timestamp}\n\n{alert.description}\n\nConduta sugerida: {alert.recommended_action}")
        if self.webhook_format == "slack":
            return {"text": text.replace("**", "*")}
        if self.webhook_format == "teams":
            color = {"INFO": "Accent", "WARNING": "Warning", "CRITICAL": "Attention"}[alert.severity.name]
            facts = [{"title": k, "value": str(v)} for k, v in list(alert.evidence.items())[:8]]
            card = {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "text": f"{label}: {alert.title}", "weight": "Bolder", "size": "Medium", "color": color},
                    {"type": "TextBlock", "text": f"Paciente {alert.patient_id} · {alert.modality} · {alert.timestamp}", "isSubtle": True},
                    {"type": "TextBlock", "text": alert.description, "wrap": True},
                    {"type": "FactSet", "facts": facts},
                    {"type": "TextBlock", "text": f"Conduta sugerida: {alert.recommended_action}", "wrap": True, "weight": "Bolder"},
                ],
            }
            return {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
        return alert.to_dict()


def load_alerts(path: str | Path) -> list[Alert]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [Alert.from_dict(json.loads(line)) for line in f if line.strip()]
