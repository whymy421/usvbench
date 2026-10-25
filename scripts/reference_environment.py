"""Reference environment RE-1 (CONTEXT_R4 D28): version stamp and the official-run rule (G6b).

The eight required versions live in ONE pinned file, scripts/reference_environment.json (part of
the code-tree hash: a change is a new release and a new protocol version RE-2, ...). This module
reads the versions the running process sees and compares them with that file:

    python      sys.version (its first token, e.g. "3.11.15")
    numpy, torch, isaacsim, hydra-core, omegaconf, skrl
                importlib.metadata distribution versions
    isaaclab    the VERSION file of the Isaac Lab checkout that provides the isaaclab package
                (found without importing it), else importlib.metadata. Order matters: a
                checkout installed with isaaclab.sh -i registers the distribution "isaaclab"
                with its EXTENSION version (source/isaaclab/config/extension.toml, "0.54.2"
                in Isaac Lab 2.3.2), not the release version in VERSION ("2.3.2", D28); a
                pip wheel install has no checkout VERSION file and carries the release version

``reference_block()`` returns the certificate block (cert_v3.runtime_metadata stamps it as
runtime.reference_environment): {"id", "required", "versions", "match", "mismatches"} with
mismatches as "name=found (required X)". ``require_official_match(registration, block)`` is the
D28 (b) rule used by eval_protocol's preflight: an official (non-diagnostic) run outside RE-1
is refused before the simulator starts; a diagnostic run (seed-registry purpose "diagnostic",
e.g. the DIAG42 smoke) is stamped and proceeds. No simulator import.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

REQUIRED_FILE = Path(__file__).resolve().with_name("reference_environment.json")
NAMES = ("python", "numpy", "torch", "isaacsim", "isaaclab", "hydra-core", "omegaconf", "skrl")
DIAGNOSTIC_PURPOSE = "diagnostic"


class ReferenceEnvironmentError(ValueError):
    """An official run outside the reference environment (D28 (b))."""


def load_required(path=REQUIRED_FILE):
    """(id, {name: version}) of the pinned file; exactly the eight D28 names."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    versions = document.get("versions")
    if not isinstance(document.get("id"), str) or not isinstance(versions, dict) or set(versions) != set(NAMES) \
            or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError("%s must hold an id and exactly the versions of %s" % (path, ", ".join(NAMES)))
    return document["id"], {name: versions[name] for name in NAMES}


def _distribution(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _isaaclab_version_file():
    """VERSION of the Isaac Lab checkout providing ``isaaclab`` (<root>/source/isaaclab/isaaclab)."""
    module = sys.modules.get("isaaclab")
    origin = getattr(module, "__file__", None)
    if origin is None:
        try:
            spec = importlib.util.find_spec("isaaclab")
        except (ImportError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None)
    if not origin:
        return None
    for parent in Path(origin).resolve().parents:
        candidate = parent / "VERSION"
        if candidate.is_file() and (parent / "source" / "isaaclab").is_dir():
            text = candidate.read_text(encoding="utf-8").strip()
            return text or None
    return None


def observed_versions():
    """{name: version or None} as this process sees them."""
    return {"python": sys.version.split()[0],
            "numpy": _distribution("numpy"),
            "torch": _distribution("torch"),
            "isaacsim": _distribution("isaacsim"),
            "isaaclab": _isaaclab_version_file() or _distribution("isaaclab"),
            "hydra-core": _distribution("hydra-core"),
            "omegaconf": _distribution("omegaconf"),
            "skrl": _distribution("skrl")}


def mismatches(versions, required):
    return ["%s=%s (required %s)" % (name, versions.get(name), required[name])
            for name in NAMES if versions.get(name) != required[name]]


def reference_block(versions=None, path=REQUIRED_FILE):
    """The runtime.reference_environment certificate block (``versions`` injects values: tests)."""
    env_id, required = load_required(path)
    seen = observed_versions() if versions is None else {name: versions.get(name) for name in NAMES}
    missing = mismatches(seen, required)
    return {"id": env_id, "required": dict(required), "versions": seen, "match": not missing,
            "mismatches": missing}


def is_diagnostic(registration):
    return isinstance(registration, dict) and registration.get("purpose") == DIAGNOSTIC_PURPOSE


def require_official_match(registration, block):
    """D28 (b): refuse an official run outside the reference environment (diagnostic runs pass)."""
    if block["match"] or is_diagnostic(registration):
        return block
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    raise ReferenceEnvironmentError(
        "official run (seed purpose %r) outside the reference environment %s (CONTEXT_R4 D28): %s; "
        "official certificates come only from %s hosts (a diagnostic run is allowed and stamped)"
        % (purpose, block["id"], "; ".join(block["mismatches"]), block["id"]))
