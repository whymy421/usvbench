"""Pure JSON/preset CLI handling shared by condition-aware evaluators."""
import json
from pathlib import Path


def apply_condition_args(cfg, presets=(), config_path=None):
    """No args is a strict no-op; enabled flags are never silently inferred."""
    if not presets and config_path is None:
        return
    if int(getattr(cfg, "journal_contract_version", 0) or 0) == 2:
        # F7 / CONTEXT_R4 D4: journal cfgs take --suite (journal_config); a
        # legacy preset would desynchronise cfg.conditions and journal_conditions.
        raise ValueError("journal cfgs take --suite / --suite-file (F0 journal_config), not the legacy "
                         "condition presets")
    if not hasattr(cfg, "conditions") or not hasattr(cfg, "condition_presets"):
        raise ValueError("this task family is not wired to the shared condition layer")
    overrides = None
    if config_path is not None:
        overrides = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("condition config file must contain a JSON object")
    cfg.conditions = overrides
    cfg.condition_presets = tuple(presets)
