"""Carregamento de configuração (config.yaml + variáveis de ambiente/.env)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["azure"] = {
        "speech_key": os.getenv("AZURE_SPEECH_KEY", "").strip(),
        "speech_region": os.getenv("AZURE_SPEECH_REGION", "brazilsouth").strip(),
        "language_key": os.getenv("AZURE_LANGUAGE_KEY", "").strip(),
        "language_endpoint": os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip(),
    }
    cfg["alerts"]["webhook_url"] = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    cfg["alerts"]["webhook_format"] = os.getenv("ALERT_WEBHOOK_FORMAT", "teams").strip()
    cfg["alerts"].setdefault("telegram", {})
    cfg["alerts"]["telegram"].update(token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
                                     chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip())
    return cfg


def telegram_enabled(cfg: dict) -> bool:
    tg = cfg["alerts"].get("telegram", {})
    return bool(tg.get("token") and tg.get("chat_id"))


def resolve(path: str | Path) -> Path:
    """Resolve caminhos relativos à raiz do projeto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def azure_speech_enabled(cfg: dict) -> bool:
    return bool(cfg["azure"]["speech_key"] and cfg["azure"]["speech_region"])


def azure_language_enabled(cfg: dict) -> bool:
    return bool(cfg["azure"]["language_key"] and cfg["azure"]["language_endpoint"])
