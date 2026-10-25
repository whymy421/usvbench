"""Round T D20 (e): the saved D16 fixtures equal a fresh run of the REAL family code. CPU only.

Runs ``scripts/make_d16_fixtures.py --check``: for each of the five families it re-runs the
family's journal env recipe in a child process (stubbed Isaac Lab, one thread, children one
at a time) and compares the not-admitted stamp, the D7 certificate-header block, the
scenario hashes and the row with scripts/fixtures_d16/<family>.json. A difference means the
family code changed what it emits: re-run ``python -B scripts/make_d16_fixtures.py --write``
and re-run scripts/test_d16_real_stamps.py and scripts/test_round_t_seams.py on the new
stamps. Changed provenance digests with an unchanged stamp are reported, not failed.

Run: python -B scripts/test_d16_fixtures_fresh.py
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def test_saved_fixtures_equal_a_fresh_run_of_the_family_code():
    command = [sys.executable, "-B", os.path.join(_SCRIPTS, "make_d16_fixtures.py"), "--check"]
    result = subprocess.run(command, cwd=os.path.dirname(_SCRIPTS), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=3000)
    print(result.stdout)
    print(result.stderr[-4000:])
    assert result.returncode == 0, "make_d16_fixtures.py --check failed (exit %d)" % result.returncode
    for family in ("hazard_nav", "path_hazard", "path_following", "docking", "harbor_mission"):
        assert ("OK " + family) in result.stdout, family


if __name__ == "__main__":
    try:
        test_saved_fixtures_equal_a_fresh_run_of_the_family_code()
    except AssertionError as exc:
        print("FAIL " + str(exc), flush=True)
        raise SystemExit(1)
    print("PASS test_saved_fixtures_equal_a_fresh_run_of_the_family_code", flush=True)
