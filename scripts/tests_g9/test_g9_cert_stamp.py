"""G9: the mppi_implementation stamp in cert_v3.schema.json and validate_cert_v3.py.

* a journal MPPI certificate written by the REAL cert_v3 writer over the REAL F0 mixin (the
  round-T fake journal env of test_journal_evaluators), with the stamp written into the header
  exactly as eval_mppi writes it, validates (graph on CUDA with --mppi-impl auto, eager with
  --mppi-impl eager);
* required for journal MPPI certificates whose command carries --mppi-impl (schema and
  validator); journal MPPI certificates written before G9 (neither) stay valid; legacy-id MPPI
  certificates need no stamp;
* the schema refuses a malformed stamp; the validator refuses inconsistent stamps (eager with
  graphs, graph without graphs or with recaptures != captured_graphs - 1, graph on a CPU device
  or with the legacy noise stream, a mode that contradicts --mppi-impl) and a stamp on a
  non-MPPI certificate.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9_cert_stamp.py -q
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import g9_fixtures as fx  # noqa: E402
from g9_fixtures import torch  # noqa: E402

sys.path.insert(0, str(fx.SCRIPTS / "tests_journal_glue"))
import cert_v3  # noqa: E402
import test_journal_evaluators as tje  # noqa: E402
import validate_cert_v3  # noqa: E402

GRAPH = {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}
EAGER = {"mode": "eager", "captured_graphs": 0, "recaptures": 0}
KEYED = {"backend": "splitmix64-boxmuller-f64-v2", "keyed": True}
_DOC = {}


def mppi_document(stamp=GRAPH, mppi_impl="auto", device="cuda:0", noise=KEYED):
    """A 2 x 1 journal MPPI certificate from the REAL writer; stamp None = not written."""
    key = json.dumps([stamp, mppi_impl, device, noise], sort_keys=True)
    if key in _DOC:
        return copy.deepcopy(_DOC[key])
    env, cfg = tje.journal_env()
    provenance = {k: "a" * 64 for k in ("tasks_sha256", "scripts_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/x/env.py",
                      evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    provenance["tree_sha256"] = cert_v3.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {k: "test" for k in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                   "isaac_lab_version", "physx_version")}
    ticks = itertools.count()
    cli_args = {} if mppi_impl is None else {"mppi_impl": mppi_impl}
    run = cert_v3.CertificateRun(base=env, env_cfg=cfg, evaluator=__file__, task=tje.TASK_ID, controller="mppi",
                                 controller_config={"controller": "mppi", "mode": "goal", "device": device,
                                                    "sampling_noise": dict(noise)},
                                 checkpoint=None, eval_seed=42, level=0, episodes_requested=2, overrides={},
                                 argv=["eval_mppi.py"], cli_args=cli_args, provenance=provenance, runtime=runtime,
                                 synchronize=lambda: None, clock=lambda: next(ticks) / 1000.0)
    env.reset()
    run.after_reset()
    records, done_envs = [], set()
    while len(done_envs) < 2:
        with run.time_controller():
            action = torch.tensor([[0.6, 0.1], [0.5, -0.1]])
        with run.time_environment():
            _, terminated, truncated = env.step(action)
        run.after_step()
        for i in (terminated | truncated).nonzero().flatten().tolist():
            if i in done_envs:
                run.discard_episode(i)
                continue
            records.append(run.finish_episode(i, dict(env=i, ep=0, success=False, terminated=bool(terminated[i]),
                                                      truncated=bool(truncated[i]), tts_s=None,
                                                      path_length_m=1.0)))
            done_envs.add(i)
    if stamp is not None:
        # exactly as eval_mppi writes it before cert.build
        run.header["controller"]["configuration"]["mppi_implementation"] = dict(stamp)
    doc = run.build({"records": records})
    _DOC[key] = doc
    return copy.deepcopy(doc)


def check(doc):
    return validate_cert_v3.validate_certificate(doc, num_envs=2, episodes_per_env=1)


def test_g9_journal_graph_certificate_validates():
    doc = mppi_document()
    assert check(doc) == []
    header = doc["header"]
    assert header["controller"]["name"] == "mppi" and "journal" in header
    assert header["controller"]["configuration"]["mppi_implementation"] == GRAPH
    assert header["command"]["parsed_args"]["mppi_impl"] == "auto"


def test_g9_journal_eager_certificate_validates():
    assert check(mppi_document(EAGER, "eager")) == []
    assert check(mppi_document(EAGER, "auto", device="cpu")) == []
    assert check(mppi_document({"mode": "cuda_graph_v1", "captured_graphs": 3, "recaptures": 2}, "graph")) == []


def test_stamp_required_for_g9_journal_mppi_certificates():
    doc = mppi_document(stamp=None)
    errors = check(doc)
    assert errors and all(e.startswith("schema ") for e in errors)
    assert any("mppi_implementation" in e for e in errors), errors
    assert validate_cert_v3.mppi_implementation_errors(doc["header"]) == [
        "journal MPPI certificate lacks controller.configuration.mppi_implementation (G9)"]


def test_journal_mppi_certificate_written_before_g9_stays_valid():
    doc = mppi_document(stamp=None, mppi_impl=None)
    assert "mppi_impl" not in doc["header"]["command"]["parsed_args"]
    assert check(doc) == []


@pytest.mark.parametrize("bad", [
    {"mode": "graph", "captured_graphs": 1, "recaptures": 0},
    {"mode": "cuda_graph_v1", "captured_graphs": -1, "recaptures": 0},
    {"mode": "cuda_graph_v1", "captured_graphs": 1.5, "recaptures": 0},
    {"mode": "cuda_graph_v1", "captured_graphs": 1},
    {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0, "backend": "x"},
])
def test_schema_refuses_a_malformed_stamp(bad):
    doc = mppi_document()
    doc["header"]["controller"]["configuration"]["mppi_implementation"] = bad
    errors = check(doc)
    assert errors and all(e.startswith("schema header/controller/configuration/mppi_implementation") for e in errors)


@pytest.mark.parametrize("stamp,mppi_impl,device,noise,message", [
    ({"mode": "eager", "captured_graphs": 1, "recaptures": 0}, "eager", "cuda:0", KEYED, "eager run captures no"),
    ({"mode": "cuda_graph_v1", "captured_graphs": 0, "recaptures": 0}, "graph", "cuda:0", KEYED, "captured no graph"),
    ({"mode": "cuda_graph_v1", "captured_graphs": 2, "recaptures": 0}, "graph", "cuda:0", KEYED,
     "recaptures (0) must equal captured_graphs - 1 (1)"),
    (GRAPH, "graph", "cpu", KEYED, "cuda_graph_v1 on controller device 'cpu'"),
    (GRAPH, "graph", "cuda:0", {"backend": "legacy-shared-generator-v1"}, "legacy shared-generator"),
    (EAGER, "auto", "cuda:0", KEYED, "contradicts --mppi-impl auto (resolves to 'cuda_graph_v1'"),
    (GRAPH, "eager", "cuda:0", KEYED, "contradicts --mppi-impl eager"),
    (EAGER, "graph", "cuda:0", KEYED, "contradicts --mppi-impl graph"),
    (EAGER, "turbo", "cuda:0", KEYED, "unknown --mppi-impl 'turbo'"),
])
def test_validator_refuses_inconsistent_stamps(stamp, mppi_impl, device, noise, message):
    errors = check(mppi_document(stamp, mppi_impl, device, noise))
    assert any(message in e for e in errors), errors


def _header(name="mppi", journal=False, stamp=None, parsed=None, device="cuda:0"):
    configuration = {"device": device, "sampling_noise": dict(KEYED)}
    if stamp is not None:
        configuration["mppi_implementation"] = dict(stamp)
    header = {"controller": {"name": name, "configuration": configuration},
              "command": {"argv": ["x"], "parsed_args": dict(parsed or {})}}
    if journal:
        header["journal"] = {"task": "reach"}
    return header


def test_rules_for_legacy_ids_and_other_controllers():
    errors = validate_cert_v3.mppi_implementation_errors
    assert errors(_header(parsed={"mppi_impl": "auto"})) == []                      # legacy id: not required
    assert errors(_header(stamp=EAGER, parsed={"mppi_impl": "auto"})) == []         # legacy auto -> eager
    assert errors(_header(stamp=GRAPH, parsed={"mppi_impl": "graph"})) == []        # opt-in graph on a legacy id
    assert any("contradicts --mppi-impl auto" in e
               for e in errors(_header(stamp=GRAPH, parsed={"mppi_impl": "auto"})))
    assert errors(_header(name="planner_los_pid", journal=True, parsed={"mppi_impl": "auto"})) == []
    assert errors(_header(name="planner_los_pid", stamp=EAGER)) == [
        "controller.configuration.mppi_implementation on a 'planner_los_pid' certificate "
        "(only MPPI certificates stamp it)"]
    assert errors(_header(journal=True)) == []                                      # pre-G9 journal MPPI
