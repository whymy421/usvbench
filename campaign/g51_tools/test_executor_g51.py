"""G51: tests of the G51 executor copy (stdlib; run on Linux, e.g. WSL Ubuntu-22.04 venv ~/usvb_validate, python 3.11).

usage: python -B -I test_executor_g51.py <executor file> <record json> <plan dir with FINAL batches> <out dir for temp files>
Checks (nothing outside <out dir> is written; the executor runs only in --describe-only mode or as library calls):
  T1 describe-only of a classical/MPPI FINAL batch list: exit 0, manifest 'final:' line, every command runs run_cell.py with
     final_test / the batch block / 128 episodes / 64 envs / --python / --release-descriptor; docking cells pass d7_pool_dir;
  T2 a FINAL PPO batch (synthetic header fields): describe-only exit 0, commands carry --checkpoint/--training-seed/--gate-ledger;
  T3 refusals: PPO row in a DIAG42 batch; PPO and classical rows mixed; bad seed_block; FINAL grid 32 x 1; PPO checkpoint outside
     the data root; a PPO row expecting a name other than 'learned'; DIAG42 batch carrying seed_namespace;
  T4 certificate gate (report): a FINAL certificate with the right registration passes; a wrong block / purpose / namespace or
     a diagnostic registration is refused (SEED-REGISTRATION-MISMATCH); the G40 grid checks still hold;
  T5 a DIAG42 batch of the G43 c3 sweep is still accepted by read_batch unchanged (describe-only against the c3 record).
"""
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import types

exe_path, record_path, plan_dir, out_dir = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("exg51", exe_path)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
os.makedirs(out_dir, exist_ok=True)
RECSHA = hashlib.sha256(open(record_path, "rb").read()).hexdigest().upper()
DATA_ROOT = "/root/autodl-tmp"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail[:300]))


def write(path, text):
    with open(path, "wb") as handle:
        handle.write(text.encode("ascii"))
    return hashlib.sha256(text.encode("ascii")).hexdigest().upper()


def plan_list(batches, name):
    lines = ["format=usvbench-g12-planlist-v1"]
    for path in batches:
        lines.append("batch|%s|%s|1|1" % (os.path.basename(path), hashlib.sha256(open(path, "rb").read()).hexdigest().upper()))
    lp = os.path.join(out_dir, name)
    sha = write(lp, "\n".join(lines) + "\n")
    return lp, sha


def describe(list_path, list_sha, record=record_path, recsha=RECSHA):
    buf = io.StringIO()

    class Console(object):
        def write(self, text):
            buf.write(text)

        def flush(self):
            pass
    argv = ["--host-class", "RENTAL", "--host-key", "rental99", "--data-root", DATA_ROOT, "--record", record, "--record-sha256", recsha,
            "--slots", "2", "--slot", "0", "--plan-list", list_path, "--plan-list-sha256", list_sha, "--describe-only"]
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = ex.main(argv)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def copy_to_out(src, edit=None, new_name=None):
    text = open(src, "rb").read().decode("ascii")
    if edit:
        text = edit(text)
    dst = os.path.join(out_dir, new_name or os.path.basename(src))
    write(dst, text)
    return dst


plans = sorted(os.path.join(plan_dir, f) for f in os.listdir(plan_dir) if f.endswith(".plan"))
dk = [p for p in plans if "-DK_BB_CL_F1-b1" in p][0]
mp = [p for p in plans if "-SC_WAMV_MP_F2-b1" in p][0]
# T1
a, b = copy_to_out(dk), copy_to_out(mp)
lp, ls = plan_list([a, b], "t1.list")
rc, out = describe(lp, ls)
cmds = [l for l in out.splitlines() if l.startswith("command|")]
finals = [l for l in out.splitlines() if l.startswith("final: ")]
ok = (rc == 0 and len(finals) == 2 and cmds and all("/scripts/run_cell.py" in c and "--run-purpose final_test" in c and "--seed-namespace final_test" in c
                                                      and "--episodes 128" in c and "--num_envs 64" in c and "--python /root/autodl-tmp/usvb/env/bin/python" in c
                                                      and "--release-descriptor" in c for c in cmds))
check("T1 describe FINAL classical+MPPI", ok, "rc=%s commands=%d final_lines=%d" % (rc, len(cmds), len(finals)))
check("T1 block per batch", all(("--eval-seed-block final-01" in c) for c in cmds if "_f1 " in c or "_f1." in c)
      and all(("--eval-seed-block final-02" in c) for c in cmds if "_f2." in c), "")
check("T1 docking d7_pool_dir", all("--set d7_pool_dir=/root/autodl-tmp/usvb/pools/" in c for c in cmds if "fc_docking_" in c)
      and not any("d7_pool_dir" in c for c in cmds if "fc_scatter_" in c), "")
check("T1 MPPI controller", all("--controller mppi" in c for c in cmds if "_mp_" in c) and all("--controller classical" in c for c in cmds if "_cl_" in c), "")
check("T1 suite tokens", any("--suite waves=D1 >>" in c for c in cmds) and any("--suite waves=D1 wind=D1 current=D1 actuator=D1 perception=D1 dynamic=D1" in c for c in cmds), "")
print("SAMPLE " + (cmds[1] if len(cmds) > 1 else ""))

PPOHDR = ("run_purpose=final_test\nppo_training_seed=42\nppo_checkpoint=/root/autodl-tmp/usvb/ppo/R/formal/SC-WAMV-42/best/best_eval.pt\n"
          "ppo_checkpoint_sha256=" + "A" * 64 + "\ngate_ledger=/root/autodl-tmp/usvb/g51/ledgers/gate_ledger_v001.json\ngate_ledger_sha256=" + "B" * 64 + "\n")


def to_ppo(text, header=PPOHDR):
    text = text.replace("run_purpose=final_test\n", header)
    text = re.sub(r"\|mp\|", "|ppo|", text)
    text = text.replace("|mppi|", "|learned|")
    return text


# T2
p = copy_to_out(mp, to_ppo, "ppo_ok.plan")
lp, ls = plan_list([p], "t2.list")
rc, out = describe(lp, ls)
cmds = [l for l in out.splitlines() if l.startswith("command|")]
check("T2 describe FINAL PPO", rc == 0 and cmds and all("--controller ppo --run-purpose" not in c for c in cmds) is not None
      and all("--checkpoint /root/autodl-tmp/usvb/ppo/" in c and "--training-seed 42" in c and "--gate-ledger /root/autodl-tmp/usvb/g51/ledgers/" in c
              and "--controller ppo" in c for c in cmds), "rc=%s cmds=%d %s" % (rc, len(cmds), out.splitlines()[-1] if out else ""))


# T3 refusals
def refused(name, src, edit, want):
    path = copy_to_out(src, edit, name + ".plan")
    lp, ls = plan_list([path], name + ".list")
    rc, out = describe(lp, ls)
    last = [l for l in out.splitlines() if "REFUSED" in l]
    check("T3 " + name, rc != 0 and last and want in last[-1], last[-1] if last else "rc=%s %s" % (rc, out[-200:]))


refused("ppo_in_diag42", mp, lambda t: to_ppo(t).replace("mode=FINAL", "mode=DIAG42").replace("seed_namespace=final_test\n", "")
        .replace("seed_block=final-02\n", "").replace(PPOHDR, ""), "Unknown controller")
refused("ppo_mixed", mp, lambda t: to_ppo(t).replace("|ppo|", "|mp|", 1).replace("|learned|", "|mppi|", 1), "mixes PPO")
refused("bad_block", dk, lambda t: t.replace("seed_block=final-01", "seed_block=pilot-01"), "bad seed_block")
refused("grid_32x1", dk, lambda t: t.replace("num_envs=64", "num_envs=32").replace("episodes_per_env=2", "episodes_per_env=1"), "64 x 2")
refused("ckpt_outside", mp, lambda t: to_ppo(t, PPOHDR.replace("ppo_checkpoint=/root/autodl-tmp/usvb/", "ppo_checkpoint=/tmp/")), "safe path")
refused("ppo_name", mp, lambda t: to_ppo(t).replace("|learned|", "|mppi|", 1), "must expect controller 'learned'")
refused("diag_with_final_field", dk, lambda t: t.replace("mode=FINAL", "mode=DIAG42"), "FINAL field")

# T4 certificate gate
REC = json.load(open(record_path))


class FakeOs(object):
    def isfile(self, p):
        return os.path.isfile(p)

    def getsize(self, p):
        return os.path.getsize(p)

    def read_bytes(self, p):
        return open(p, "rb").read()


def cert(reg, envs=64, epe=2, ctrl="planner_los_pid_mem_v2"):
    recs = [{"env": e, "ep": q, "success": (e + q) % 3 == 0} for e in range(envs) for q in range(epe)]
    return {"header": {"task_id": "T", "controller": {"name": ctrl}, "eval_seed": 41041, "num_envs": envs, "episodes_per_env": epe,
                       "episodes_requested": envs * epe, "complete": True, "seed_namespace": reg,
                       "code": {"release_id": REC["release_id"], "code_tree_sha256": REC["code_tree_sha256"]}}, "records": recs}


def gate(doc, final=True):
    path = os.path.join(out_dir, "cert_tmp.json")
    write(path, json.dumps(doc))
    self = types.SimpleNamespace(os=FakeOs(), rec=REC)
    cell = types.SimpleNamespace(num_envs=64, episodes_per_env=2, journal_id="T", seed=41041, expected_controller="planner_los_pid_mem_v2",
                                 ctrl="cl", carrier="hazard_nav", name="c",
                                 final={"seed_namespace": "final_test", "run_purpose": "final_test", "seed_block": "final-01", "ppo": None} if final else None)
    try:
        return "OK " + ex.Executor.report(self, path, cell)
    except ex.Refusal as exc:
        return "REFUSED " + str(exc).split(":")[0]


good = {"namespace": "final_test", "purpose": "final_test", "seed": 41041, "block_id": "final-01", "registry_status": "frozen"}
check("T4 good registration", gate(cert(good)).startswith("OK"), gate(cert(good)))
for label, reg in (("wrong block", dict(good, block_id="final-02")), ("wrong purpose", dict(good, purpose="pilot")),
                   ("diagnostic", {"namespace": "diagnostic", "purpose": "diagnostic", "seed": 41041, "block_id": None}),
                   ("wrong seed", dict(good, seed=41047))):
    got = gate(cert(reg))
    check("T4 " + label, got.startswith("REFUSED SEED-REGISTRATION-MISMATCH"), got)
check("T4 grid still checked", gate(cert(good, 64, 1)).startswith("REFUSED GRID-MISMATCH"), gate(cert(good, 64, 1)))
check("T4 DIAG42 cell ignores registration", gate(cert({"namespace": "diagnostic"}), final=False).startswith("OK"), "")

# T5 DIAG42 c3 sweep batch against the c3 record
if len(sys.argv) > 6:
    c3_record, c3_batch = sys.argv[5], sys.argv[6]
    c3sha = hashlib.sha256(open(c3_record, "rb").read()).hexdigest().upper()
    path = copy_to_out(c3_batch)
    lp, ls = plan_list([path], "t5.list")
    rc, out = describe(lp, ls, c3_record, c3sha)
    cmds = [l for l in out.splitlines() if l.startswith("command|")]
    check("T5 DIAG42 sweep batch unchanged route", rc == 0 and cmds and all("/scripts/run_cell.py" not in c and re.search(r"--(eval-)?seed 42 ", c) for c in cmds)
          and not any(l.startswith("final: ") for l in out.splitlines()), "rc=%s cmds=%d" % (rc, len(cmds)))
fails = results.count(False)
print("G51-EXECUTOR-TEST %s passed=%d failed=%d" % ("PASS" if not fails else "FAIL", results.count(True), fails))
sys.exit(1 if fails else 0)
