from __future__ import annotations

import json
from pathlib import Path

import pytest

from damage_engine.registry import PhysicsRegistry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def target_data():
    return json.loads((ROOT / "examples" / "target.demo.json").read_text(encoding="utf-8"))


@pytest.fixture
def case_data():
    return json.loads((ROOT / "examples" / "case.demo.json").read_text(encoding="utf-8"))


@pytest.fixture
def registry():
    return PhysicsRegistry.from_file(ROOT / "examples" / "physics.demo.json")

