"""Backward-compatible import path for the shared restoring helper."""

from __future__ import annotations

if __package__:
    from .._shared.restoring import restoring_torque_body
else:
    # Preserve ``from restoring import ...`` for direct execution of the
    # existing tasks/docking/test_restoring.py script.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tasks._shared.restoring import restoring_torque_body

__all__ = ["restoring_torque_body"]
