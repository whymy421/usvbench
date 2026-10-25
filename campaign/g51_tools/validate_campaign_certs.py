"""G51 campaign certificate check (run on LINUX for rental-written certificates: WSL Ubuntu-22.04 venv ~/usvb_validate, python
3.11 + the release's own validate_cert_v3; the appearing-trigger cos recompute is platform-exact before v2.6.2's tolerance and
Linux is the platform that wrote them - G42/G43). Read-only.

usage: python -B validate_campaign_certs.py --release-root <extract> --manifest <release manifest> --plan <campaign plan json>
           --record <release record json> <pulled queue dir> [<pulled queue dir> ...]
For every <queue dir>/cells/<cell>.json of a plan cell (the executor moves a stage to that name only after its light gate):
  VALID   = validate_cert_v3 at 64 x 2 (require_verified False, code_root = the release, release manifest) has no error AND
            identity: header.task_id = plan jid, controller name = plan expected name, header.seed_namespace {namespace
            final_test, purpose final_test, block_id = plan seed block, seed = plan seed, registry_status frozen}, requested suite
            tokens = plan tokens, release id + code tree = the record's, 128 complete records;
  INVALID = anything else (reasons listed).
Output: one line per certificate 'VALID|INVALID <cell> <batch dir> sha256=<cert> success=<k>/<n> [reasons]' and a final
'G51-VALIDATE valid=<n> invalid=<m> missing=<cells of the batches without a certificate>' line. Exit 0 always (the caller
reads the lines)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("dirs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.release_root / "scripts"))
    spec = importlib.util.spec_from_file_location("g51_validator", str(a.release_root / "scripts" / "validate_cert_v3.py"))
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    plan = json.loads(a.plan.read_text(encoding="ascii"))
    rec = json.loads(a.record.read_text(encoding="utf-8"))
    cells = {c["name"]: c for c in plan["cells"]}
    batches = {b["id"]: b for b in plan["batches"]}
    nv = ni = 0
    missing = []
    for d in a.dirs:
        qname = d.name                                  # dose_<batch id>_a<attempt>
        bid = qname[len("dose_"):qname.rfind("_a")]
        for name in (batches[bid]["cells"] if bid in batches else []):
            path = d / "cells" / (name + ".json")
            if not path.is_file():
                missing.append("%s/%s" % (qname, name))
                continue
            cell = cells[name]
            raw = path.read_bytes()
            doc = json.loads(raw.decode("utf-8"))
            errors = [str(e) for e in v.validate_certificate(doc, num_envs=64, episodes_per_env=2, require_verified=False,
                                                              code_root=a.release_root, release_manifest=a.manifest)]
            h = doc.get("header") or {}
            if h.get("task_id") != cell["jid"]:
                errors.append("TASK %r" % h.get("task_id"))
            if (h.get("controller") or {}).get("name") != cell["expected_controller"]:
                errors.append("CONTROLLER %r" % (h.get("controller") or {}).get("name"))
            reg = h.get("seed_namespace") or {}
            want = {"namespace": "final_test", "purpose": "final_test", "block_id": cell["seed_block"], "seed": cell["seed"]}
            if any(reg.get(k) != x for k, x in want.items()) or reg.get("registry_status") != "frozen":
                errors.append("SEED-REGISTRATION %s" % json.dumps({k: reg.get(k) for k in list(want) + ["registry_status"]}, sort_keys=True))
            requested = list(((h.get("suite_presets") or {}).get("suite")) or [])
            if requested != list(cell["tokens"]):
                errors.append("SUITE requested=%r plan=%r" % (requested, cell["tokens"]))
            code = h.get("code") or {}
            if code.get("release_id") != rec["release_id"] or code.get("code_tree_sha256") != rec["code_tree_sha256"]:
                errors.append("RELEASE %r" % code.get("release_id"))
            records = doc.get("records") or []
            k = sum(1 for r in records if r.get("success") is True)
            if len(records) != 128 or h.get("complete") is not True:
                errors.append("GRID records=%d complete=%r" % (len(records), h.get("complete")))
            tag = "VALID" if not errors else "INVALID"
            nv += not errors
            ni += bool(errors)
            print("%s %s %s sha256=%s success=%d/%d %s" % (tag, name, qname, hashlib.sha256(raw).hexdigest().upper(), k, len(records),
                                                         ("; ".join(errors))[:600]))
            sys.stdout.flush()
    print("G51-VALIDATE valid=%d invalid=%d missing=%d %s" % (nv, ni, len(missing), " ".join(missing[:50])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
