"""G6b: the REAL cert_v3 journal header for all 24 journal ids (stubbed envs, REAL pinned pools).

P5 v3: journal station keeping raised in cert_v3 at construction ("journal env publishes no pinned
D7 pool manifest digest (D19)") on the GPU; the CPU suite never built a CertificateRun over a real
family env with its real pinned pool. cert_header_worker.py does, per family in its own process
(the family fixtures install different stub sets), and checks the D19 / D22 / D23 / D24 header
fields; this test runs it for every family with no suite and with the six D1 tokens.

Pools: station keeping, docking and path following ship in the release tree; hazard_nav,
path_hazard and harbour_mission come from the pool bundle ($USVBENCH_D7_POOL_DIR on the GPU host).
The bundle folder is $USVBENCH_TEST_D7_BUNDLE, else the G7 bundle of this workstation; without it
those three families are skipped with the reason (the other three always run).
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g6/test_g6b_cert_headers.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE / "cert_header_worker.py"
BUNDLE_CANDIDATES = (os.environ.get("USVBENCH_TEST_D7_BUNDLE"),
                     "D:/claude code/codex_tasks/20261001_g7_pool_validity/work/pool_bundle_g7")
BUNDLE = next((Path(c) for c in BUNDLE_CANDIDATES if c and (Path(c) / "BUNDLE_MANIFEST.txt").is_file()), None)
FAMILIES = {"hazard_nav": 14, "path_following": 2, "station_keeping": 2, "docking": 2, "path_hazard": 2,
            "harbor_mission": 2}
NEEDS_BUNDLE = ("hazard_nav", "path_hazard", "harbor_mission")
D1_TOKENS = ["waves=D1", "wind=D1", "current=D1", "actuator=D1", "perception=D1", "dynamic=D1"]


def run_family(family, tokens):
    env = {k: v for k, v in os.environ.items() if not k.startswith("USVBENCH_")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="")
    if BUNDLE is not None:
        env["USVBENCH_D7_POOL_DIR"] = str(BUNDLE)
    result = subprocess.run([sys.executable, "-B", str(WORKER), family, *tokens], capture_output=True, text=True,
                            env=env, timeout=3000)
    rows = [json.loads(line[len("G6B-HEADER "):]) for line in result.stdout.splitlines()
            if line.startswith("G6B-HEADER ")]
    return result, rows


@pytest.mark.parametrize("tokens", ([], D1_TOKENS), ids=("nosuite", "d1x6"))
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_journal_certificate_header_has_the_d19_d23_fields(family, tokens):
    if family in NEEDS_BUNDLE and BUNDLE is None:
        pytest.skip("pool bundle not found (set USVBENCH_TEST_D7_BUNDLE)")
    result, rows = run_family(family, tokens)
    assert len(rows) == FAMILIES[family], result.stdout[-2000:] + result.stderr[-3000:]
    bad = [(r["task_id"], r["errors"]) for r in rows if not r["ok"]]
    if family == "station_keeping" and "dynamic=D1" in tokens:
        # G23e: BlueBoat station keeping D1 is not applicable by design; the env refuses it at setup
        expected = [r for r in bad if r[0] == "Isaac-USV-StationKeep-BlueBoat-Journal-Direct-v1"
                    and any("not applicable by design" in e for e in r[1])]
        assert len(expected) == 1, bad
        bad = [r for r in bad if r not in expected]
        rows = [r for r in rows if r["ok"]]
        assert bad == [], bad
        assert len({r["pool_manifest_sha256"] for r in rows}) == 1
        return
    assert bad == [], bad
    assert result.returncode == 0
    assert len({r["pool_manifest_sha256"] for r in rows}) == 1      # one pinned manifest per family
