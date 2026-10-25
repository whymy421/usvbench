#!/usr/bin/env python3
"""G51: build the campaign copy of the G40 Linux executor (FINAL mode = formal test cells through run_cell.py, PPO cells).

usage: python -B make_executor_g51.py            (writes work/executor_g51/usvb_dose_executor.py and r3_remote.py; never overwrites)
Source: G40 executor usvb_dose_executor_g40.py SHA-256 FF6B8627... (the executor G40/G41/G42/G43 ran on the rentals) and the
unchanged r3_remote.py of r3copy_g43 (byte copy). Every edit is an anchored replacement that must match exactly once.
What changes (nothing else):
  E1 header lines describing the G51 copy;
  E2 read_batch: mode FINAL besides DIAG42. FINAL requires seed_namespace=final_test, run_purpose=final_test,
     seed_block=final-NN, grid 64 x 2, level 0, and (only when the batch holds PPO rows) ppo_training_seed, ppo_checkpoint,
     ppo_checkpoint_sha256, gate_ledger, gate_ledger_sha256 (absolute safe paths under the data root); ctrl 'ppo' only in FINAL;
     a PPO row must expect controller 'learned'; one PPO training seed per batch;
  E3 build_arguments: a FINAL cell runs <release>/scripts/run_cell.py (the only route the release accepts for final_test
     cells: eval_protocol.py refuses a direct final_test evaluator call) with --run-purpose/--seed-namespace/--eval-seed-block,
     --python <RE-1 interpreter>, --release-descriptor, --out <stage>, PPO --checkpoint/--training-seed/--gate-ledger,
     docking --set d7_pool_dir=<bundle>, --suite tokens. run_cell writes <stage>.cell.json and starts the evaluator as its
     child (owned through the tagged ancestor);
  E4 before each FINAL PPO cell: the checkpoint and the gate ledger are hash-checked against the batch header;
  E5 report(): FINAL certificates must carry header.seed_namespace {namespace final_test, purpose final_test, block_id =
     the batch block, seed = eval_seed}; PPO certificates the controller name 'learned';
  E6 manifest: one 'final:' line (namespace, block, purpose, PPO fields) after the grid line.
The executor's own registered prediction lines (S1-S8 of the dose smoke) and PREDICTIONS_SHA256 are NOT changed; the
campaign predictions travel in each batch file's PREDICTION lines (as every G12/G32 batch did).
"""
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
TASKS = Path(r"D:\claude code\codex_tasks")
SRC = TASKS / "20261010_g40_rental_production" / "work" / "executor_g40" / "usvb_dose_executor_g40.py"
SRC_SHA = "ff6b8627ebe0d2f8b10dca2ff41160e25b97b75207f824af92d32b7384eab48d"
REMOTE = TASKS / "20261013_g43_c3_sweep" / "work" / "r3copy_g43" / "executor" / "r3_remote.py"
OUT = Path(__file__).resolve().parent.parent / "executor_g51"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


# E1 header
edit("#!/usr/bin/env python3\n# G40 COPY (2026-10-10)",
     "#!/usr/bin/env python3\n"
     "# G51 COPY (2026-10-14) of the G40 executor FF6B8627... for the v2.6.x formal campaign (built by G51 tools/make_executor_g51.py;\n"
     "# every change is listed there): mode FINAL = formal test cells (seed namespace final_test, 64 x 2, level 0) launched through\n"
     "# <release>/scripts/run_cell.py, the only route the release accepts for final_test cells; controller 'ppo' (FINAL only) with the\n"
     "# checkpoint, training seed and gate ledger of the batch header, hash-checked before every PPO cell; FINAL certificates must\n"
     "# carry the batch's seed registration. DIAG42 batches run exactly as in G40.\n"
     "# G40 COPY (2026-10-10)")

# E2 read_batch: mode FINAL
edit("""        if meta.get('mode') != 'DIAG42':
            raise Refusal('Only mode DIAG42 is supported (batch %s)' % meta.get('batch_id'))
""",
     """        if meta.get('mode') not in ('DIAG42', 'FINAL'):
            raise Refusal('Only modes DIAG42 and FINAL are supported (batch %s)' % meta.get('batch_id'))
""")
edit("""        if not predictions:
            raise Refusal('Batch %s registers no prediction' % meta['batch_id'])
""",
     """        if not predictions:
            raise Refusal('Batch %s registers no prediction' % meta['batch_id'])
        final = self.final_meta(meta, rows, path)            # G51: None for DIAG42
""")
edit("""            if row.ctrl != 'mp' and row.carrier not in CLASSICAL_BY_CARRIER:
                raise Refusal('Unknown carrier %s' % row.carrier)
            if not re.fullmatch(r'cl|mp', row.ctrl):
                raise Refusal('Unknown controller %s' % row.ctrl)
""",
     """            if row.ctrl == 'cl' and row.carrier not in CLASSICAL_BY_CARRIER:
                raise Refusal('Unknown carrier %s' % row.carrier)
            if not re.fullmatch(r'cl|mp|ppo', row.ctrl) or (row.ctrl == 'ppo' and final is None):
                raise Refusal('Unknown controller %s (ppo only in FINAL batches)' % row.ctrl)
            if row.ctrl == 'ppo' and row.expected_controller != 'learned':
                raise Refusal("Cell %s: a PPO cell must expect controller 'learned', not '%s'" % (row.name, row.expected_controller))
""")
edit("""        batch.episodes_per_env = int(meta['episodes_per_env'])
        batch.ceiling = ceilings[HOST_CLASS]
""",
     """        batch.episodes_per_env = int(meta['episodes_per_env'])
        batch.final = final
        batch.ceiling = ceilings[HOST_CLASS]
""")
# the final_meta helper, inserted before read_plan_list
edit("""    def read_plan_list(self):
""",
     """    def final_meta(self, meta, rows, path):
        # G51: the FINAL header (None for DIAG42). Values are checked here, before any cell runs.
        if meta.get('mode') != 'FINAL':
            for key in ('seed_namespace', 'seed_block', 'run_purpose', 'ppo_training_seed', 'ppo_checkpoint', 'gate_ledger'):
                if key in meta:
                    raise Refusal('DIAG42 batch %s carries the FINAL field %s' % (path, key))
            return None
        if meta.get('seed_namespace') != 'final_test' or meta.get('run_purpose') != 'final_test':
            raise Refusal('FINAL batch %s: seed_namespace and run_purpose must be final_test' % path)
        if not re.fullmatch(r'final-[0-9]{2}', meta.get('seed_block', '')):
            raise Refusal('FINAL batch %s: bad seed_block %r' % (path, meta.get('seed_block')))
        if (int(meta['num_envs']), int(meta['episodes_per_env']), int(meta['level'])) != (64, 2, 0):
            raise Refusal('FINAL batch %s: the formal grid is 64 x 2 at level 0' % path)
        final = {'seed_namespace': 'final_test', 'run_purpose': 'final_test', 'seed_block': meta['seed_block'], 'ppo': None}
        if any(row.ctrl == 'ppo' for row in rows):
            if not all(row.ctrl == 'ppo' for row in rows):
                raise Refusal('FINAL batch %s mixes PPO and other controllers (one PPO run per batch)' % path)
            ppo = {}
            for key in ('ppo_training_seed', 'ppo_checkpoint', 'ppo_checkpoint_sha256', 'gate_ledger', 'gate_ledger_sha256'):
                if not meta.get(key):
                    raise Refusal('FINAL PPO batch %s lacks %s' % (path, key))
                ppo[key] = meta[key]
            if not re.fullmatch(r'[0-9]{1,6}', ppo['ppo_training_seed']):
                raise Refusal('FINAL PPO batch %s: bad ppo_training_seed' % path)
            for key in ('ppo_checkpoint', 'gate_ledger'):
                if not SAFE_ROOT_RE.match(ppo[key]) or not ppo[key].startswith(self.root + '/') or '/../' in ppo[key]:
                    raise Refusal('FINAL PPO batch %s: %s must be an absolute safe path under %s/' % (path, key, self.root))
            for key in ('ppo_checkpoint_sha256', 'gate_ledger_sha256'):
                if not re.fullmatch(r'[0-9A-Fa-f]{64}', ppo[key]):
                    raise Refusal('FINAL PPO batch %s: bad %s' % (path, key))
            final['ppo'] = ppo
        elif any(key in meta for key in ('ppo_training_seed', 'ppo_checkpoint', 'gate_ledger')):
            raise Refusal('FINAL batch %s without PPO rows carries PPO fields' % path)
        return final

    def read_plan_list(self):
""")
# batch_cells: carry the FINAL header into every cell
edit("""            cell.num_envs, cell.seed, cell.level = batch.num_envs, batch.seed, batch.level
            cell.episodes_per_env = batch.episodes_per_env
            cells = root + '/cells/' + row.name
""",
     """            cell.num_envs, cell.seed, cell.level = batch.num_envs, batch.seed, batch.level
            cell.episodes_per_env = batch.episodes_per_env
            cell.final = getattr(batch, 'final', None)
            cells = root + '/cells/' + row.name
""")
# E3 build_arguments
edit("""    def build_arguments(self, cell):
        evaluator = MPPI_EVALUATOR if cell.ctrl == 'mp' else CLASSICAL_BY_CARRIER[cell.carrier]
""",
     """    def build_arguments(self, cell):
        if getattr(cell, 'final', None):
            return self.build_final_arguments(cell)
        evaluator = MPPI_EVALUATOR if cell.ctrl == 'mp' else CLASSICAL_BY_CARRIER[cell.carrier]
""")
edit("""    def build_command(self, cell):
""",
     """    def build_final_arguments(self, cell):
        # G51 FINAL: run_cell.py plans and launches the cell (final_test needs a frozen registry; run_cell refuses otherwise).
        final = cell.final
        controller = {'cl': 'classical', 'mp': 'mppi', 'ppo': 'ppo'}[cell.ctrl]
        tokens = ['-X', self.own_tag(), '-B', '%s/scripts/run_cell.py' % self.release_path,
                  '--task', cell.journal_id, '--vessel', cell.vessel, '--controller', controller,
                  '--run-purpose', final['run_purpose'], '--seed-namespace', final['seed_namespace'],
                  '--eval-seed-block', final['seed_block'], '--level', str(cell.level),
                  '--episodes', str(cell.num_envs * cell.episodes_per_env), '--num_envs', str(cell.num_envs),
                  '--device', 'cuda:0', '--python', self.python_path, '--release-descriptor', self.descriptor_path,
                  '--out', cell.stage]
        if cell.ctrl == 'ppo':
            ppo = final['ppo']
            tokens.extend(['--checkpoint', ppo['ppo_checkpoint'], '--training-seed', ppo['ppo_training_seed'],
                           '--gate-ledger', ppo['gate_ledger']])
        if cell.carrier == 'docking':
            tokens.extend(['--set', 'd7_pool_dir=%s' % self.bundle_path])
        if cell.suite:
            tokens.append('--suite')
            tokens.extend(cell.suite)
        for token in tokens:
            if UNSAFE_TOKEN_RE.search(token) or NONPRINT_TOKEN_RE.search(token):
                raise Refusal('Unsafe command token: %s' % token)
        return tokens

    def assert_ppo_inputs(self, cell):
        # G51 E4: the PPO checkpoint and the gate ledger must still be the files the batch pinned.
        ppo = cell.final['ppo']
        for key in ('ppo_checkpoint', 'gate_ledger'):
            if not self.os.isfile(ppo[key]):
                raise Refusal('PPO-INPUT-MISSING %s=%s' % (key, ppo[key]))
            have = self.os.sha256_upper(ppo[key])
            if have != ppo[key + '_sha256'].upper():
                raise Refusal('PPO-INPUT-HASH-MISMATCH %s=%s expected=%s actual=%s' % (key, ppo[key], ppo[key + '_sha256'].upper(), have))

    def build_command(self, cell):
""")
# E4 hash check before a FINAL PPO cell
edit("""        self.assert_inputs()
        self.assert_release_root('before %s' % cell.name)
        self.wait_own_python('before %s' % cell.name)
""",
     """        self.assert_inputs()
        if getattr(cell, 'final', None) and cell.ctrl == 'ppo':
            self.assert_ppo_inputs(cell)
        self.assert_release_root('before %s' % cell.name)
        self.wait_own_python('before %s' % cell.name)
""")
# E5 report: seed registration of FINAL certificates
edit("""        if header.get('complete') is not True:
            raise Refusal('INCOMPLETE: %s' % json_path)
""",
     """        if header.get('complete') is not True:
            raise Refusal('INCOMPLETE: %s' % json_path)
        final = getattr(cell, 'final', None)
        if final:
            reg = header.get('seed_namespace') if isinstance(header.get('seed_namespace'), dict) else {}
            if (reg.get('namespace') != final['seed_namespace'] or reg.get('purpose') != final['run_purpose']
                    or reg.get('block_id') != final['seed_block'] or reg.get('seed') != cell.seed):
                raise Refusal('SEED-REGISTRATION-MISMATCH: %s' % ps_string(json.dumps(reg, sort_keys=True)))
""")
# E6 manifest line
edit("""        lines.append('grid: num_envs=%d episodes=%d episodes_per_env=%d level=%d eval_seed=%d' % (batch.num_envs, batch.num_envs * batch.episodes_per_env, batch.episodes_per_env, batch.level, batch.seed))
""",
     """        lines.append('grid: num_envs=%d episodes=%d episodes_per_env=%d level=%d eval_seed=%d' % (batch.num_envs, batch.num_envs * batch.episodes_per_env, batch.episodes_per_env, batch.level, batch.seed))
        final = getattr(batch, 'final', None)
        if final:
            ppo = final['ppo'] or {}
            lines.append('final: route=run_cell.py seed_namespace=%s seed_block=%s run_purpose=%s ppo_training_seed=%s ppo_checkpoint=%s '
                         'ppo_checkpoint_sha256=%s gate_ledger=%s gate_ledger_sha256=%s'
                         % (final['seed_namespace'], final['seed_block'], final['run_purpose'], ppo.get('ppo_training_seed', '-'),
                            ppo.get('ppo_checkpoint', '-'), ppo.get('ppo_checkpoint_sha256', '-'), ppo.get('gate_ledger', '-'),
                            ppo.get('gate_ledger_sha256', '-')))
""")


def main():
    data = SRC.read_bytes()
    if hashlib.sha256(data).hexdigest() != SRC_SHA:
        raise SystemExit("source executor hash differs: %s" % SRC)
    text = data.decode("ascii")
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("anchor matches %d times: %r" % (n, old[:80]))
        text = text.replace(old, new)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in (("usvb_dose_executor.py", text.encode("ascii")), ("r3_remote.py", REMOTE.read_bytes())):
        path = OUT / name
        if path.exists():
            if path.read_bytes() == body:
                print("SAME %s" % path)
                continue
            raise SystemExit("refusing to overwrite %s" % path)
        with open(path, "xb") as handle:
            handle.write(body)
        print("WROTE %s sha256=%s" % (path, hashlib.sha256(body).hexdigest().upper()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
