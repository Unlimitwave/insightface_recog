#!/usr/bin/env python3
"""Verify deploy app package imports (run before docker build)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import route modules the same way uvicorn does
from app.api.routes import health, identify, persons  # noqa: F401
from app.main import app

print("OK: imports successful")
print("app:", app.title)
