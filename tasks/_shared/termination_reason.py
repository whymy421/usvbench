"""Certificate v3 stop-cause telemetry, independent of Isaac Lab and torch.

The arguments are boolean vector objects supporting tensor-style ``long``,
``clone``, boolean operations and masked assignment. No input is modified.
Codes describe why a step ends an episode, not the episode's success score.
In particular a successful fixed-horizon hold still ends with TIMEOUT.

F0 (CONTEXT_R4 D9): journal branches do NOT recompute codes from masks.
They publish the task contract's reason-code tensor with
``publish_termination_codes(env, codes)``, which REPLACES the
``publish_termination_reason`` call in their ``_get_dones`` (same lifetime:
``env.termination_reason`` and a separate clone in ``env.extras``, not
cleared by the reset methods, replaced by the next ``_get_dones``). Codes are
validated against E5's single exported table
(``journal_contracts_v2.TERMINATION_TABLE``: C5 codes 0-5 plus the D4
validity codes 6-9), so the validity codes survive to the certificate. The
table is loaded lazily (never at import), from the package when available,
else by path from this folder.
"""

RUNNING = 0
SUCCESS = 1
COLLISION = 2
TIMEOUT = 3
OUT_OF_BOUNDS = 4
FAILURE = 5

TERMINATION_REASON_NAMES = {
    RUNNING: "running",
    SUCCESS: "success",
    COLLISION: "collision",
    TIMEOUT: "timeout",
    OUT_OF_BOUNDS: "out_of_bounds",
    FAILURE: "failure",
}


def termination_reason_codes(
    terminated, truncated, *, success=None, collision=None, out_of_bounds=None
):
    """Return integer stop causes without changing the actual done masks.

    A terminal cause wins over a simultaneous time limit. Within terminal
    causes the priority is out_of_bounds, collision, success, then failure.
    Masks for success/contact only matter where ``terminated`` is true, so a
    historical contact or an earlier latched success cannot relabel a timeout.
    """
    reason = terminated.long() * RUNNING
    reason[truncated & ~terminated] = TIMEOUT
    reason[terminated] = FAILURE
    if success is not None:
        reason[terminated & success] = SUCCESS
    if collision is not None:
        reason[terminated & collision] = COLLISION
    if out_of_bounds is not None:
        reason[terminated & out_of_bounds] = OUT_OF_BOUNDS
    return reason


def publish_termination_reason(env, terminated, truncated, **causes):
    """Expose this step's reason on the env and in extras before autoreset.

    Task reset methods deliberately do not clear this public telemetry. A
    caller can read it after ``env.step`` has reset a completed environment.
    The next ``_get_dones`` call replaces it, with RUNNING for active rows.
    ``extras`` receives its own clone so later writes cannot alias the buffer.
    """
    env.termination_reason = termination_reason_codes(terminated, truncated, **causes)
    env.extras["termination_reason"] = env.termination_reason.clone()


_TABLE = None


def _journal_contracts():
    """E5 journal_contracts_v2 (package import, else path load; numpy only)."""
    if __package__:
        from . import journal_contracts_v2
        return journal_contracts_v2
    import importlib.util
    import os
    import sys
    name = "_termination_reason_journal_contracts_v2"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal_contracts_v2.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def termination_table():
    """{code: TerminationRow} of E5's TERMINATION_TABLE (codes 0-9)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = {int(row.code): row for row in _journal_contracts().TERMINATION_TABLE}
    return _TABLE


def termination_code_name(code):
    """Name of one validated code (E5 table), e.g. 6 -> "capsize"."""
    table = termination_table()
    if isinstance(code, bool) or int(code) != code or int(code) not in table:
        raise ValueError(f"termination code {code!r} is not in journal_contracts_v2.TERMINATION_TABLE")
    return table[int(code)].name


def _codes_list(codes):
    if not hasattr(codes, "shape") or not hasattr(codes, "tolist") or not (
            hasattr(codes, "clone") or hasattr(codes, "copy")):
        raise TypeError("codes must be a tensor-like vector (torch tensor or numpy array)")
    shape = tuple(getattr(codes, "shape", ()))
    if len(shape) != 1:
        raise ValueError(f"codes must be a 1-D per-env vector, got shape {shape}")
    dtype = str(getattr(codes, "dtype", ""))
    if "int" not in dtype or "uint8" in dtype:
        raise ValueError(f"codes must be a signed integer vector, got dtype {dtype}")
    values = codes.detach().cpu().tolist() if hasattr(codes, "detach") else codes.tolist()
    return [int(v) for v in values]


def _mask_list(mask, n, label):
    if mask is None:
        return None
    shape = tuple(getattr(mask, "shape", ()))
    if shape != (n,):
        raise ValueError(f"{label} must have shape ({n},), got {shape}")
    values = mask.detach().cpu().tolist() if hasattr(mask, "detach") else mask.tolist()
    return [bool(v) for v in values]


def termination_masks_from_codes(codes):
    """(terminated, truncated) boolean vectors implied by the codes (E5 ends_as).

    RUNNING (0) -> neither; TIMEOUT (3) -> truncated; every other stop cause
    -> terminated. Works for torch tensors (returns tensors on the codes'
    device) and numpy arrays.
    """
    table = termination_table()
    values = _codes_list(codes)
    bad = sorted({v for v in values if v not in table})
    if bad:
        raise ValueError(f"termination codes {bad} are not in journal_contracts_v2.TERMINATION_TABLE")
    terminated = [table[v].ends_as == "terminated" for v in values]
    truncated = [table[v].ends_as == "truncated" for v in values]
    if hasattr(codes, "detach"):
        import torch
        return (torch.tensor(terminated, dtype=torch.bool, device=codes.device),
                torch.tensor(truncated, dtype=torch.bool, device=codes.device))
    import numpy as np
    return np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool)


def publish_termination_codes(env, codes, *, terminated=None, truncated=None):
    """Publish a contract's reason codes with publish_termination_reason's lifetime (D9).

    ``codes`` is the per-env integer vector of this ``_get_dones`` (0 for
    running rows; E5 codes 1-9 for rows that end now). Every value must be a
    code of ``journal_contracts_v2.TERMINATION_TABLE``. If the done masks are
    passed they must agree with the table (``ends_as``): a nonzero code on a
    row that is not done, a done row with code 0, or a timeout code on a
    terminated row is an error. The input is not modified; the env keeps a
    clone and ``extras`` receives its own clone.
    """
    table = termination_table()
    values = _codes_list(codes)
    bad = sorted({v for v in values if v not in table})
    if bad:
        raise ValueError(f"termination codes {bad} are not in journal_contracts_v2.TERMINATION_TABLE "
                         f"(valid: {sorted(table)})")
    n = len(values)
    term = _mask_list(terminated, n, "terminated")
    trunc = _mask_list(truncated, n, "truncated")
    if term is not None or trunc is not None:
        term = term if term is not None else [False] * n
        trunc = trunc if trunc is not None else [False] * n
        for i, code in enumerate(values):
            ends = table[code].ends_as
            if ends == "none" and (term[i] or trunc[i]):
                raise ValueError(f"env {i}: done row published with code 0 (running)")
            if ends == "terminated" and not term[i]:
                raise ValueError(f"env {i}: code {code} ({table[code].name}) needs terminated=True")
            if ends == "truncated" and (term[i] or not trunc[i]):
                raise ValueError(f"env {i}: code {code} ({table[code].name}) needs truncated and not terminated")
    published = codes.clone() if hasattr(codes, "clone") else codes.copy()
    env.termination_reason = published
    env.extras["termination_reason"] = published.clone() if hasattr(published, "clone") else published.copy()
    return published
