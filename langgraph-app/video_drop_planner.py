"""Compatibility shim for the pre-installed beatdrop_outfit extension.

The implementation lives in plugins/beatdrop_outfit so this specialized feature
can be disabled/removed without keeping the planner logic in AlphaRavis core.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins" / "beatdrop_outfit"
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

_module = importlib.import_module("beatdrop_outfit.planner")
sys.modules[__name__] = _module
