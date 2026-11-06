import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medmonitor.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["alerts"]["webhook_url"] = ""  # testes nunca disparam webhook real
    return c
