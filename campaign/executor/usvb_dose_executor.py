#!/usr/bin/env python3
# G51 COPY (2026-10-14) of the G40 executor FF6B8627... for the v2.6.x formal campaign (built by G51 tools/make_executor_g51.py;
# every change is listed there): mode FINAL = formal test cells (seed namespace final_test, 64 x 2, level 0) launched through
# <release>/scripts/run_cell.py, the only route the release accepts for final_test cells; controller 'ppo' (FINAL only) with the
# checkpoint, training seed and gate ledger of the batch header, hash-checked before every PPO cell; FINAL certificates must
# carry the batch's seed registration. DIAG42 batches run exactly as in G40.
# G40 COPY (2026-10-10) of the R3 Linux executor 681C051C... mirroring the G32 v2.6 SWEEP executor
# work_v26/queue_sweep/dose_queue.ps1 (747AADFA...) against the G12 update-5 Windows executor 8C9B4846... it mirrored:
# (1) only usvbench-g12-batch-v3 batches (V2-BATCH-REFUSED); a cell without a declared expected name is refused;
# (2) episodes_per_env 1 or 2: --episodes num_envs x episodes_per_env, manifest grid line, light gate (header
# episodes_per_env / episodes_requested, record count, env < num_envs, ep < episodes_per_env, every env/ep pair once);
# (3) the v2.6 S6 prediction text and PREDICTIONS_SHA256 F77A648D... (the Windows sweep queue's).
# The Windows K cap (4080:2, 5070:3) has no RENTAL entry there; RENTAL K stays 1..4 (launcher). Nothing else changed.
# R3 LINUX EXECUTOR of the G12 dose smoke for rented RTX 4090 hosts (host class RENTAL): a port of the G12 Windows
# executor queue/dose_queue.ps1 (SHA-256 37BED64E8F900B69AF6188ECC334B61CE2FFB564EF25E041CA297220BFEECCD0, update 3).
# Python 3.11 standard library only (it runs inside the RE-1 venv of the rental, but never imports torch or Isaac).
# usage: <data root>/usvb/env/bin/python -X usvbexec=ds-<release_id>-k<K>s<s> -B -I usvb_dose_executor.py
#            --host-class RENTAL --host-key <rental host key> --data-root <data root> --record <record>
#            --record-sha256 <hash> --slots <K 1..4> --slot <0..K-1> --plan-list <list> --plan-list-sha256 <hash>
#            [--gpu-memory-wait-ms <ms>] [--describe-only]
# Same inputs as the Windows executor: the plan LIST (format usvbench-g12-planlist-v1, 'batch|<plan file next to the
# list>|<sha256>|<attempt>|<from_order>' lines) and the batch files (format usvbench-g12-batch-v2 with the affinity id,
# a ceiling per host class, the named timeouts of all three classes with reasons, and the cell rows). The executor uses
# the RENTAL ceiling and the RENTAL named timeouts and refuses what the Windows executor refuses (undeclared, reasonless
# or other-class timeouts; case-insensitive cell name collisions; non-ascending orders; non-ASCII or control text; a
# plan file whose SHA-256 differs from the list (edited predictions included); an existing queue directory).
# Layout (R2 rental kit): USVB_ROOT = <data root>/usvb replaces C:\usvb (releases/<id>, releases/_incoming/<id>,
# pools/<bundle id>, env/bin/python, IsaacLab/VERSION, usvb_env.sh, qualification/); queue directory
# <USVB_ROOT>/dose_<batch_id>_a<attempt> with the same files, names and progress lines as on Windows.
# Host identity: --host-key is the dispatch host identity (tools/dispatch.py --host, e.g. rental01). The computer name
# is this machine's hostname; it must equal the record's hosts.<host key>.computer_name when the record has that entry,
# otherwise the newest <USVB_ROOT>/qualification/qualification_<release_id>_<host key>_<stamp>.json written by the R2
# qualify.sh (format usvbench-r2-qualification-v1, verdict QUALIFIED, same release_id, record_sha256, host_key,
# data_root and hostname). The manifest's host_name is that computer name (Windows: COMPUTERNAME).
# K SLOTS PER GPU: slot s of K holds the flock <USVB_ROOT>/locks/slot<s>.lock; K = 1 holds slot0..slot3 and the
# whole-GPU lock <USVB_ROOT>/state/gpu.lock (the R2 kit's gpucheck and bench hold the same lock; the Windows single-slot
# mode holds the legacy G2 mutex); K >= 2 refuses while the whole-GPU lock is held. Ownership as on Windows: every python
# a slot starts carries '-X usvbslot=ds-<release_id>-k<K>s<s>'; a python is own / sibling when it or a live ancestor
# (created no later than its child: start-time guard against PID reuse, /proc/<pid>/stat field 22) carries the slot tag
# of this slot / of another slot of this K, otherwise foreign. A slot waits only for its own pythons (PYEXITWAITMS),
# never waits for, counts toward its wait or kills a sibling's, and refuses a foreign python that persists for
# PYEXITWAITMS. Python set on Linux: live processes whose executable is the RE-1 interpreter (realpath of
# <USVB_ROOT>/env/bin/python) or a python whose command line names <USVB_ROOT>/ (the R2 kit's py-wait scope; AutoDL
# images run JupyterLab on the system python). R3 tooling processes (this executor, the remote launch / status / abort
# tool) carry '-X usvbexec=<role>' and are never counted (on Windows the executor is powershell.exe, not a python).
# Each cell (and each verify-tree) runs from a script with /bin/bash in its own session and process group (setsid);
# its named timeout sends SIGTERM to the whole group and session, then SIGKILL after KILLGRACEMS, and the run ends only
# when no member is left; a member that survives EXITWAITMS after the kill aborts the batch with the PIDs
# (GROUP-SURVIVOR). As with the Windows job object, a cell lasts while any member of its group lives.
# GPU memory guard, release-root check before and after every cell, verify-tree at batch start and end, heartbeat file
# per cell, overall ceiling, manifest before the first cell, FINISHED / ABORTED markers, resume as a new list line with
# the next attempt and from_order = resume_from_order, DescribeOnly that writes nothing: as on Windows.
# SIGTERM / SIGINT to the executor (the abort command) stop the current cell's group and write the ABORTED marker.
# G12 UPDATE 4 mirrored (Windows executor SHA-256 FF9920EC4C902EA343D659983E93C67F7A04B010E626540AF2DFAD033DF8595E):
# (a) GPU memory threshold per cell class, chosen from the cell's controller and vessel by gpu_mem_class (same named
# constants as Windows; measured peak per process at 64 envs + about 1 GB headroom, R2 bench 2026-10-02 on the 5070 host):
#   GPU_MEM_MIN_FREE_MIB_CLASSICAL     3400  classical, both vessels: peak 2366 MiB
#   GPU_MEM_MIN_FREE_MIB_PPO           3400  PPO eval: peak 2366-2374 MiB (no PPO cell in the smoke; class kept for later plans)
#   GPU_MEM_MIN_FREE_MIB_MPPI_BLUEBOAT 3700  graph MPPI BlueBoat: peak 2686 MiB
#   GPU_MEM_MIN_FREE_MIB_MPPI_WAMV     4800  graph MPPI WAM-V: peak 3752 MiB
# every GPU-MEM-OK / GPU-MEM-LOW line carries class= and threshold_mib=, GPU-MEMORY-LOW names the constant, the manifest's
# slots line lists the four thresholds and each cell has a 'gpumem|<cell>|class=..|threshold_mib=..' line; an unknown
# class refuses the batch (GPU-MEM-CLASS-UNKNOWN). 4090 peaks are UNVERIFIED.
# (b) plan lists and batch files are read once as bytes: SHA-256 of those bytes (<PLAN|PLANLIST>-HASH-MISMATCH), then any
# byte > 0x7F or NUL refuses ('INPUT-NON-ASCII-BYTE <PLAN|PLANLIST> <file> offset=<n> byte=0x<XX>') before any parsing,
# DescribeOnly included; then the lines are split like ReadAllLines (CRLF, LF or CR).
# G12 UPDATE 5 mirrored (Windows executor SHA-256 8C9B4846F9FABEF10804F32EC9BFD7C27C9764461E52C3746AE814E6D7BCD6EB;
# release-aware classical controller names; manifest v4): a batch of format usvbench-g12-batch-v3 names the release it
# was generated for (release_id=, which must equal the record's, else BATCH-RELEASE-MISMATCH), the SHA-256 of that
# release's scripts/run_cell.py (controller_names_sha256=) and, in every cell row before the suite tokens, the expected
# certificate controller name (generated by G12 tools/plan_rerun_set.py); certificates are checked against that name and
# the manifest lists it (release_names: line, controller|<cell>|<name> per cell). A usvbench-g12-batch-v2 batch (the
# v2.4 smoke) is read exactly as before, names from the table below; its manifest differs from update 4 only in its
# first line ('G12 executor manifest v4').
# Predictions (the controller's, verbatim from the G12 brief, identical to the Windows executor header; the executor
# refuses to start if their SHA-256 differs from PREDICTIONS_SHA256; each batch file carries its registered
# predictions, which go into that batch's manifest):
# S1 (integrity) PASS: every cell exits 0 inside its named timeout and writes a certificate that passes the release's validate_cert_v3 at 32 x 1 with the identity, suite, combination and pool checks (a pool grid that covers the cell grid passes); every waves cell carries the green-water header and records; no termination code 9. FAIL: every cell that misses is listed with its class.
# S2 (gradient, success) For every group and every axis, success(D4) <= success(D0) and at least one of wind, current, waves, actuator.cap, perception.latency, dynamic shows success(D0) - success(D4) >= 0.25 on every group. FAIL: every group and axis where success rises by more than 0.10 from D0 to D4, and every group where none of those six axes drops by 0.25.
# S3 (wind on BlueBoat) PASS: on every BlueBoat group, success(wind D4) >= success(calm) - 0.10 (wind barely moves a 1.2 m low-profile boat), while the mean absolute cross-track or position error rises from calm to wind D4. FAIL: any BlueBoat group where wind D4 costs more than 0.10 of success, or where no error metric rises.
# S4 (wind on WAM-V) PASS: on at least half of the WAM-V groups, a graded metric (cross-track or position error, or success) gets worse from wind D2 to wind D4. FAIL: fewer than half.
# S5 (current) PASS: on BlueBoat station_keeping and docking, success(current D4 = 1.5 m/s) <= success(calm) - 0.50. FAIL otherwise.
# S6 (green water, v2.6 deck-edge probe) PASS when all four hold: (a) no WAM-V cell is excluded by the formula; (b) no WAM-V waves D4 cell is excluded by the simulated deck-edge share (> 5 %); (c) BlueBoat waves D4 (1.0 m) is excluded by the simulated share on at least one task; (d) BlueBoat waves D3 (0.5 m) is excluded on no task. FAIL: any statement wrong, listed per cell.
# S7 (release root clean) and S8 (python exit race): as in the G2 queues.
# REFERENCE (not a prediction; S7 and S8 are "as in the G2 queues", whose registered text is, p5v3_queue.ps1 lines 17-18):
# REFERENCE G2 P4 (release root stays clean) PASS: every RELEASE-ROOT check logs RELEASE-ROOT-CLEAN and the end verify-tree prints TREE-OK with the record's file count. FAIL: ABORT 'RELEASE-ROOT-DIRTY <entries>' right after the cell that wrote into the release, or the end verify-tree fails.
# REFERENCE G2 P5 (python exit race) PASS: every wait logs PY-CLEAR with waited_ms below PYEXITWAITMS. FAIL: ABORT 'python still running after PYEXITWAITMS' with every remaining PID and parent PID.
import argparse
import hashlib
import json
import os
import posixpath
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

PREDICTIONS_SHA256 = 'F77A648DA1A9DAB64F3DF4473EAE6D3B3CA891FCBD3C228097555DD95DB75866'
GPU_MEM_MIN_FREE_MIB_CLASSICAL = 3400
GPU_MEM_MIN_FREE_MIB_PPO = 3400
GPU_MEM_MIN_FREE_MIB_MPPI_BLUEBOAT = 3700
GPU_MEM_MIN_FREE_MIB_MPPI_WAMV = 4800
GPUMEMWAITMS_DEFAULT_K2 = 600000
GPUMEMWAITMS_DEFAULT_K3 = 7200000
CEILING_GPUMEM_ALLOWANCE_MS = 600000
GPUMEMPOLLMS = 30000
MAX_BATCHES = 128
# ---- queue core constants (G2 v2 queue core values, as in the Windows executor)
VERIFYMS = 600000
EXITWAITMS = 30000
ALIVEPROBEMS = 60000
PYEXITWAITMS = 120000
PYPOLLMS = 2000
HEARTBEATMS = 120000
# ---- Linux additions (named; logged in every manifest)
KILLGRACEMS = 10000        # SIGTERM -> SIGKILL grace inside EXITWAITMS (the Windows job is killed at once)
CELLPOLLMS = 500           # group poll period (the Windows helper polls every 100 ms)
PROBEMS = 600000           # version probe timeout (the Windows probe has none)
NVSMIMS = 60000            # nvidia-smi query timeout (Windows has none)
GPU_INDEX = '0'
HOST_CLASS = 'RENTAL'
ENV_FILE_MARKER = '# usvbench-rental-env-v2 (written by install.sh)'
QUALIFICATION_FORMAT = 'usvbench-r2-qualification-v1'
HOST_KEY_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
SAFE_ROOT_RE = re.compile(r'^/[A-Za-z0-9_./:+-]+$')
PY_NAME_RE = re.compile(r'^python[0-9]*(?:\.[0-9]+)*$')
NVIDIA_SMI_CANDIDATES = ('/usr/bin/nvidia-smi', '/usr/local/nvidia/bin/nvidia-smi', '/usr/local/bin/nvidia-smi')
ENV_PROBE_CODE = ("import importlib.metadata as m, sys, json; print('ENVJSON ' + json.dumps({'python': sys.version.split()[0], "
                  "'numpy': m.version('numpy'), 'torch': m.version('torch'), 'isaacsim': m.version('isaacsim'), "
                  "'hydra-core': m.version('hydra-core'), 'omegaconf': m.version('omegaconf'), 'skrl': m.version('skrl')}))")
# Windows refuses [\s"%^&|<>] and non-printable; bash adds ' $ ` \ ; ( ) * ? [ ] ~ # ! (braces stay allowed: the
# DescribeOnly token {ATTEMPT} has no comma, so bash never brace-expands it; real attempt ids are 32 hex digits).
UNSAFE_TOKEN_RE = re.compile(r'[\s"\'%^&|<>$`\\;()*?\[\]~#!]')
NONPRINT_TOKEN_RE = re.compile(r'[^\x21-\x7E]')

# The evaluator run_cell.py selects (run_cell.py:110-112 JOURNAL_CLASSICAL_BY_CARRIER; the docking classical controller
# is named docking_3phase_pid since v2.3, run_cell.py:77-81 DOCKING_BASELINE and :106 JOURNAL_DOCKING_BASELINE).
CLASSICAL_BY_CARRIER = {
    'hazard_nav': {'script': 'classical_planner_pid.py', 'seed_flag': '--eval-seed', 'extra': (), 'controller': 'planner_los_pid'},
    'path_hazard': {'script': 'classical_planner_pid.py', 'seed_flag': '--eval-seed', 'extra': (), 'controller': 'planner_los_pid'},
    'path_following': {'script': 'classical_baseline.py', 'seed_flag': '--seed', 'extra': ('--protocol', 'mission'), 'controller': 'los_pid'},
    'station_keeping': {'script': 'classical_baseline.py', 'seed_flag': '--seed', 'extra': ('--protocol', 'mission'), 'controller': 'los_pid'},
    'docking': {'script': 'classical_baseline.py', 'seed_flag': '--seed', 'extra': ('--protocol', 'mission'), 'controller': 'docking_3phase_pid'},
    'harbor_mission': {'script': 'classical_harbour.py', 'seed_flag': '--eval-seed', 'extra': (), 'controller': 'staged_harbour'},
}
MPPI_EVALUATOR = {'script': 'eval_mppi.py', 'seed_flag': '--eval-seed', 'extra': ()}


class Refusal(Exception):
    """A 'throw' of the Windows executor."""


class Interrupted(BaseException):
    """SIGTERM / SIGINT to the executor: abort the current batch (caught like a throw, never swallowed by inner handlers)."""


# ------------------------------------------------------------------ pure helpers (unit-tested on any platform)
class Proc(object):
    __slots__ = ('pid', 'ppid', 'start', 'state', 'argv', 'exe', 'pgid', 'sid', 'comm')

    def __init__(self, pid, ppid, start, state, argv, exe, pgid, sid, comm):
        self.pid, self.ppid, self.start, self.state = pid, ppid, start, state
        self.argv, self.exe, self.pgid, self.sid, self.comm = argv, exe, pgid, sid, comm


def parse_stat(text):
    """/proc/<pid>/stat -> (comm, state, ppid, pgrp, session, starttime); comm may hold spaces and parentheses."""
    open_at, close_at = text.find('('), text.rfind(')')
    if open_at < 0 or close_at < open_at:
        raise ValueError('malformed stat')
    tail = text[close_at + 2:].split()
    if len(tail) < 20:
        raise ValueError('short stat')
    return text[open_at + 1:close_at], tail[0], int(tail[1]), int(tail[2]), int(tail[3]), int(tail[19])


def parse_cmdline(data):
    """/proc/<pid>/cmdline bytes -> argv list, or None when empty (kernel thread, zombie, unreadable)."""
    if not data:
        return None
    if data.endswith(b'\0'):
        data = data[:-1]
    return [part.decode('utf-8', 'replace') for part in data.split(b'\0')]


def carries_tag(argv, wanted):
    """'-X <wanted>' as two consecutive arguments (the Windows rule '(?:^|\\s)-X\\s+<tag>(?:\\s|$)' on argv)."""
    if not argv or not wanted:
        return False
    for index in range(len(argv) - 1):
        if argv[index] == '-X' and argv[index + 1] == wanted:
            return True
    return False


def is_tooling(argv):
    """R3 tooling processes (executor, remote tool) carry '-X usvbexec=<role>' and are never counted as pythons."""
    if not argv:
        return False
    for index in range(len(argv) - 1):
        if argv[index] == '-X' and argv[index + 1].startswith('usvbexec='):
            return True
    return False


def is_python(proc, python_real, root_prefix):
    """The Linux python set (see header): the RE-1 interpreter, or a python naming the kit root on its command line."""
    if proc.exe is not None and python_real and proc.exe == python_real:
        return True
    names = []
    if proc.exe:
        names.append(posixpath.basename(proc.exe))
    if proc.argv:
        names.append(posixpath.basename(proc.argv[0]))
    if not any(PY_NAME_RE.match(name) for name in names):
        return False
    return bool(proc.argv) and any(root_prefix in part for part in proc.argv)


def python_owner(proc, by_id, own_tag, other_tags):
    """'own' / 'sibling' / 'foreign' (Windows PythonOwner): the process or a live ancestor created no later than its child
    carries the tag; a missing command line or a broken chain is foreign."""
    current, seen = proc, set()
    for _depth in range(16):
        if current is None or current.pid in seen:
            break
        seen.add(current.pid)
        if carries_tag(current.argv, own_tag):
            return 'own'
        for other in other_tags:
            if carries_tag(current.argv, other):
                return 'sibling'
        parent = by_id.get(current.ppid)
        if parent is None or parent.start is None or current.start is None or parent.start > current.start:
            break
        current = parent
    return 'foreign'


def classify_pythons(procs, own_tag, other_tags, python_real, root_prefix, self_pid):
    by_id = dict((p.pid, p) for p in procs)
    out = []
    for proc in procs:
        if proc.pid == self_pid or is_tooling(proc.argv) or not is_python(proc, python_real, root_prefix):
            continue
        out.append((proc, python_owner(proc, by_id, own_tag, other_tags)))
    return out


def describe_pythons(entries):
    parts = []
    for proc, owner in entries:
        name = posixpath.basename(proc.exe) if proc.exe else (proc.comm or '?')
        parts.append('pid=%d ppid=%d owner=%s name=%s' % (proc.pid, proc.ppid, owner, name))
    return '; '.join(parts)


def predictions_sha(texts):
    return hashlib.sha256('\n'.join(texts).encode('ascii')).hexdigest().upper()


def ascii_escape(text):
    """Non-ASCII characters as \\uXXXX (UTF-16 units, as the PowerShell regex replacement)."""
    out = []
    for char in text:
        code = ord(char)
        if code < 0x80:
            out.append(char)
        elif code <= 0xFFFF:
            out.append('\\u%04X' % code)
        else:
            code -= 0x10000
            out.append('\\u%04X\\u%04X' % (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)))
    return ''.join(out)


def one_line(text):
    return ascii_escape(re.sub(r'\r\n|\r|\n', ' | ', text))


def dotnet_sort_key(name):
    """Approximation of PowerShell's culture-aware Sort-Object for the timeout names (punctuation before digits before
    letters, case-insensitive); names of one batch differ only at CALM / SUITE, where any order rule agrees."""
    return ([(0 if c == '_' else (1 if c.isdigit() else 2), c.lower()) for c in name], name)


def sorted_unique(names):
    seen, out = set(), []
    for name in sorted(names, key=dotnet_sort_key):
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def ps_bool(value):
    return 'True' if value else 'False'


def ps_string(value):
    """[string] of a ConvertFrom-Json value (for the certificate gate)."""
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def is_finite_number(value):
    return type(value) in (int, float) and value == value and value not in (float('inf'), float('-inf'))


def split_lines(text):
    return re.split(r'\r\n|\r|\n', text)


def check_input_bytes(raw, path, kind):
    """G12 update 4 ReadInputLines, after the hash check: any byte > 0x7F or NUL refuses with its offset; then the lines
    are split like ReadAllLines (CRLF, LF or CR; no trailing empty line)."""
    for index, value in enumerate(bytearray(raw)):
        if value > 0x7F or value == 0:
            raise Refusal('INPUT-NON-ASCII-BYTE %s %s offset=%d byte=0x%02X' % (kind, path, index, value))
    text = raw.decode('ascii')
    if not text:
        return []
    lines = split_lines(text)
    if lines[-1] == '' and (text.endswith('\n') or text.endswith('\r')):
        lines.pop()
    return lines


def expected_controller_name(cell):
    """G12 update 5 ExpectedControllerName: a batch-v3 cell declares its expected name (release-aware); a v2 cell uses the
    v2.4 table (mppi for MPPI cells)."""
    if getattr(cell, 'expected_controller', ''):
        return cell.expected_controller
    # G40 (mirrors G32): v2 batches are refused in read_batch, so a cell always declares its name; no v2.4 table fallback.
    raise Refusal('Cell %s declares no expected controller name (G32 v2.6 executor)' % cell.name)


def gpu_mem_class(ctrl, vessel):
    """G12 update 4 GpuMemClass: (class, constant name, threshold MiB) of a cell from its controller and vessel."""
    if ctrl == 'cl':
        return 'classical', 'GPU_MEM_MIN_FREE_MIB_CLASSICAL', GPU_MEM_MIN_FREE_MIB_CLASSICAL
    if ctrl == 'ppo':
        return 'ppo', 'GPU_MEM_MIN_FREE_MIB_PPO', GPU_MEM_MIN_FREE_MIB_PPO
    if ctrl == 'mp' and vessel == 'blueboat':
        return 'mppi_blueboat', 'GPU_MEM_MIN_FREE_MIB_MPPI_BLUEBOAT', GPU_MEM_MIN_FREE_MIB_MPPI_BLUEBOAT
    if ctrl == 'mp' and vessel == 'wamv':
        return 'mppi_wamv', 'GPU_MEM_MIN_FREE_MIB_MPPI_WAMV', GPU_MEM_MIN_FREE_MIB_MPPI_WAMV
    raise Refusal('GPU-MEM-CLASS-UNKNOWN controller=%s vessel=%s' % (ctrl, vessel))


# ------------------------------------------------------------------ the real Linux OS layer (UNVERIFIED here: no Linux
# on the controller PC; r3_selftest_linux.py exercises it on the rental before the first launch)
class LinuxOs(object):
    is_real = True

    def __init__(self, proc_root='/proc'):
        self.proc_root = proc_root

    # ---- time and identity
    def now_ms(self):
        return int(time.monotonic() * 1000)

    def sleep_ms(self, ms):
        time.sleep(ms / 1000.0)

    def local_stamp(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def utc_stamp(self):
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def hostname(self):
        return socket.gethostname()

    def self_pid(self):
        return os.getpid()

    def executor_tag(self):
        return getattr(sys, '_xoptions', {}).get('usvbexec')

    def new_attempt_id(self):
        return uuid.uuid4().hex

    # ---- processes
    def processes(self):
        out = []
        try:
            names = os.listdir(self.proc_root)
        except OSError:
            raise Refusal('process table unreadable: %s' % self.proc_root)
        for name in names:
            if not name.isdigit():
                continue
            base = os.path.join(self.proc_root, name)
            try:
                with open(os.path.join(base, 'stat'), 'rb') as handle:
                    comm, state, ppid, pgrp, session, start = parse_stat(handle.read().decode('ascii', 'replace'))
            except (OSError, ValueError):
                continue
            if state in ('Z', 'X', 'x'):
                continue
            try:
                with open(os.path.join(base, 'cmdline'), 'rb') as handle:
                    argv = parse_cmdline(handle.read())
            except OSError:
                argv = None
            try:
                exe = os.path.realpath(os.readlink(os.path.join(base, 'exe')))
            except OSError:
                exe = None
            out.append(Proc(int(name), ppid, start, state, argv, exe, pgrp, session, comm))
        return out

    def spawn_group(self, script, log_path, cwd, info):
        log_handle = open(log_path, 'ab')
        try:
            proc = subprocess.Popen(['/bin/bash', '--noprofile', '--norc', script], cwd=cwd, stdin=subprocess.DEVNULL,
                                    stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        finally:
            log_handle.close()
        proc.r3_pid = proc.pid
        return proc

    def leader_poll(self, handle):
        return handle.poll()

    def group_members(self, handle):
        members = []
        for proc in self.processes():
            if proc.pgid == handle.r3_pid or proc.sid == handle.r3_pid:
                members.append(proc.pid)
        return sorted(members)

    def signal_group(self, handle, sig_name):
        sig = signal.SIGTERM if sig_name == 'TERM' else signal.SIGKILL
        try:
            os.killpg(handle.r3_pid, sig)
        except ProcessLookupError:
            pass
        for proc in self.processes():
            if proc.sid == handle.r3_pid and proc.pgid != handle.r3_pid:
                try:
                    os.kill(proc.pid, sig)
                except ProcessLookupError:
                    pass

    def run_capture(self, argv, timeout_ms):
        try:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=timeout_ms / 1000.0)
        except subprocess.TimeoutExpired:
            return None, 'TIMEOUT after %d ms' % timeout_ms
        except OSError as exc:
            return None, 'cannot start: %s' % exc
        return done.returncode, done.stdout.decode('utf-8', 'replace')

    # ---- locks (flock; the descriptor is close-on-exec, so no child inherits a lock)
    def lock_try(self, path):
        import fcntl
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0), 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    def lock_release(self, token):
        import fcntl
        try:
            fcntl.flock(token, fcntl.LOCK_UN)
        finally:
            os.close(token)

    # ---- files
    def realpath(self, path):
        return os.path.realpath(path)

    def abspath(self, path):
        return os.path.abspath(path)

    def is_exec_file(self, path):
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def exists(self, path):
        return os.path.lexists(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def isdir(self, path):
        return os.path.isdir(path)

    def mkdir(self, path):
        os.mkdir(path)

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def listdir_entries(self, path):
        with os.scandir(path) as entries:
            return [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries]

    def read_bytes(self, path):
        with open(path, 'rb') as handle:
            return handle.read()

    def write_new(self, path, data):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()

    def open_new_writer(self, path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        return os.fdopen(fd, 'wb')

    def append_bytes(self, path, data):
        with open(path, 'ab') as handle:
            handle.write(data)

    def move_new(self, src, dst):
        """[IO.File]::Move: never overwrites (os.rename would)."""
        if os.path.lexists(dst):
            raise Refusal('destination exists: %s' % dst)
        try:
            os.link(src, dst)
        except OSError:
            if os.path.lexists(dst):
                raise
            os.rename(src, dst)
            return
        os.unlink(src)

    def getsize(self, path):
        return os.path.getsize(path)

    def getmtime(self, path):
        return os.path.getmtime(path)

    def sha256_upper(self, path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest().upper()


# ------------------------------------------------------------------ the executor
class Batch(object):
    pass


class Cell(object):
    pass


class Executor(object):
    def __init__(self, args, layer, console=None, header_path=None):
        self.args = args
        self.os = layer
        self.console = console if console is not None else sys.stdout
        self.header_path = header_path or os.path.abspath(__file__)
        self.progress = None
        self.progress_owned = False
        self.interrupted = False
        self.outcomes = []
        self.out_root = ''
        self.attempt_id = ''
        self.locks = []
        self.computer_name = ''
        self.identity = 'not-checked'

    # ---------------------------------------------------------------- output
    def console_line(self, text):
        self.console.write(text + '\n')
        self.console.flush()

    def log(self, message):
        # Before the progress file exists the line goes to the console only (captured by the wrapper's console log).
        line = '%s %s' % (self.os.local_stamp(), one_line(message))
        self.console_line(line)
        if not self.progress_owned:
            return
        self.progress.write((line + '\r\n').encode('ascii'))
        self.progress.flush()

    def write_new_ascii(self, path, lines, newline='\r\n'):
        text = newline.join(lines) + newline
        if re.search(r'[^\x00-\x7F]', text):
            raise Refusal('Non-ASCII text: %s' % path)
        self.os.write_new(path, text.encode('ascii'))

    # ---------------------------------------------------------------- record and paths
    def read_record(self):
        given = self.args.record_sha256
        if not re.fullmatch(r'[0-9A-Fa-f]{64}', given or ''):
            raise Refusal('RecordSha256 must be 64 hex characters')
        measured = self.os.sha256_upper(self.args.record)
        if measured != given.upper():
            raise Refusal('RECORD-HASH-MISMATCH %s measured=%s given=%s' % (self.args.record, measured, given))
        parsed = json.loads(self.os.read_bytes(self.args.record).decode('utf-8-sig'))
        if not isinstance(parsed, dict) or parsed.get('format') != 'usvbench-release-record-v2':
            raise Refusal('Unexpected record format: %s' % (parsed.get('format') if isinstance(parsed, dict) else None))
        if not re.fullmatch(r'[0-9a-f]{12}-[0-9a-f]{12}', str(parsed.get('release_id') or '')):
            raise Refusal('Malformed release_id in the record')
        if not re.fullmatch(r'g[0-9a-z]+pool-[0-9a-f]{12}', str((parsed.get('bundle') or {}).get('bundle_id') or '')):
            raise Refusal('Malformed bundle_id in the record')
        return parsed

    def record_file(self, role):
        found = [f for f in ((self.rec.get('files') or {}).get('release') or []) if f.get('role') == role]
        if len(found) != 1:
            raise Refusal('Record has %d files with role %s' % (len(found), role))
        return found[0]

    def resolve_paths(self):
        root = self.root
        rid, bid = self.rec['release_id'], self.rec['bundle']['bundle_id']
        self.release_path = root + '/releases/' + rid
        self.incoming_path = root + '/releases/_incoming/' + rid
        self.bundle_path = root + '/pools/' + bid
        self.python_path = root + '/env/bin/python'
        self.env_file = root + '/usvb_env.sh'
        self.version_file = root + '/IsaacLab/VERSION'
        self.descriptor_path = self.incoming_path + '/' + self.record_file('descriptor')['name']
        self.manifest_file_path = self.incoming_path + '/' + self.record_file('code_manifest')['name']
        self.assets_manifest_path = self.incoming_path + '/' + self.record_file('assets_manifest')['name']
        self.kit_path = self.incoming_path + '/' + self.record_file('kit')['name']
        self.hash_checks = [
            (self.descriptor_path, self.record_file('descriptor')['sha256'].upper()),
            (self.manifest_file_path, self.record_file('code_manifest')['sha256'].upper()),
            (self.assets_manifest_path, self.record_file('assets_manifest')['sha256'].upper()),
            (self.kit_path, self.record_file('kit')['sha256'].upper()),
            (self.bundle_path + '/' + self.rec['bundle']['manifest_name'], self.rec['bundle']['manifest_sha256'].upper()),
        ]
        # The R2 kit deploys by the record's remote_layout with C:/usvb replaced by <data root>/usvb: refuse a record
        # whose layout would put the release elsewhere than the paths above.
        layout = self.rec.get('remote_layout')
        if isinstance(layout, dict):
            want = {'incoming': self.incoming_path, 'release': self.release_path, 'bundle': self.bundle_path}
            for key, path in want.items():
                template = layout.get(key)
                if not isinstance(template, str) or not template.startswith('C:/usvb/'):
                    raise Refusal('remote_layout.%s is not under C:/usvb: %r' % (key, template))
                mapped = root + template[len('C:/usvb'):].replace('<release_id>', rid).replace('<bundle_id>', bid)
                if mapped != path:
                    raise Refusal('remote_layout.%s maps to %s, the executor expects %s' % (key, mapped, path))

    # ---------------------------------------------------------------- slots, tags, pythons
    def slot_tag_for(self, tag_slot):
        return 'usvbslot=ds-%s-k%ds%d' % (self.rec['release_id'], self.args.slots, tag_slot)

    def own_tag(self):
        return self.slot_tag_for(self.args.slot)

    def other_tags(self):
        return [self.slot_tag_for(other) for other in range(self.args.slots) if other != self.args.slot]

    def classify(self):
        return classify_pythons(self.os.processes(), self.own_tag(), self.other_tags(), self.python_real,
                                self.root + '/', self.os.self_pid())

    def assert_no_foreign_python(self, reason):
        # A foreign python refuses only when it persists for PYEXITWAITMS (the Windows rule: exited pythons can linger
        # in the process table after their tagged parent is gone).
        start = self.os.now_ms()
        foreign = [e for e in self.classify() if e[1] == 'foreign']
        if foreign:
            self.log('PY-FOREIGN-WAIT %s %s' % (reason, describe_pythons(foreign)))
        while foreign and self.os.now_ms() - start < PYEXITWAITMS:
            self.os.sleep_ms(PYPOLLMS)
            foreign = [e for e in self.classify() if e[1] == 'foreign']
        if foreign:
            self.log('PY-FOREIGN %s waited_ms=%d %s' % (reason, self.os.now_ms() - start, describe_pythons(foreign)))
            raise Refusal('FOREIGN-PYTHON (%s): %s' % (reason, describe_pythons(foreign)))

    def wait_own_python(self, reason):
        # G2 race fix, slot-aware: wait only for this slot's own pythons; the siblings' are never counted, waited for or
        # killed.
        start = self.os.now_ms()
        self.assert_no_foreign_python(reason)
        own = [e for e in self.classify() if e[1] == 'own']
        while own and self.os.now_ms() - start < PYEXITWAITMS:
            self.os.sleep_ms(PYPOLLMS)
            own = [e for e in self.classify() if e[1] == 'own']
        if own:
            self.log('PY-REMAINING %s waited_ms=%d %s' % (reason, self.os.now_ms() - start, describe_pythons(own)))
            raise Refusal('python still running after PYEXITWAITMS=%d (%s): %s' % (PYEXITWAITMS, reason, describe_pythons(own)))
        self.assert_no_foreign_python(reason)
        siblings = len([e for e in self.classify() if e[1] == 'sibling'])
        self.log('PY-CLEAR %s own=0 sibling=%d waited_ms=%d' % (reason, siblings, self.os.now_ms() - start))

    def slot_lock_paths(self):
        if self.args.slots > 1:
            return [self.root + '/locks/slot%d.lock' % self.args.slot]
        return [self.root + '/locks/slot%d.lock' % s for s in range(4)] + [self.gpu_lock_path]

    def probe_gpu_lock(self, reason):
        # K >= 2: the whole-GPU lock is held by a K = 1 executor, the R2 gpucheck or the R2 bench: refuse while held.
        token = self.os.lock_try(self.gpu_lock_path)
        if token is None:
            self.log('LEGACY-GPU-MUTEX-BUSY %s lock=%s' % (reason, self.gpu_lock_path))
            raise Refusal('LEGACY-GPU-MUTEX-BUSY (%s): %s is held (a K = 1 executor, the R2 gpucheck or the R2 bench owns '
                          'the GPU of this host)' % (reason, self.gpu_lock_path))
        self.os.lock_release(token)

    # ---------------------------------------------------------------- GPU memory guard
    def nvidia_smi_path(self):
        for candidate in NVIDIA_SMI_CANDIDATES:
            if self.os.is_exec_file(candidate):
                return candidate
        raise Refusal('nvidia-smi not found (looked in: %s); the GPU memory guard cannot run' % '; '.join(NVIDIA_SMI_CANDIDATES))

    def gpu_memory(self):
        code, text = self.os.run_capture([self.nvidia_smi, '--query-gpu=memory.free,memory.total',
                                          '--format=csv,noheader,nounits', '-i', GPU_INDEX], NVSMIMS)
        lines = [line for line in split_lines(text or '') if line != '']
        memory = [line for line in lines if re.fullmatch(r'\s*[0-9]+\s*,\s*[0-9]+\s*', line)]
        if code != 0 or len(memory) != 1:
            raise Refusal('GPU-MEMORY-QUERY-FAILED exit=%s output=%s' % (code, ' | '.join(lines)))
        free, total = memory[0].split(',')
        return int(free.strip()), int(total.strip())

    def wait_gpu_memory(self, reason, cell):
        klass, constant, threshold = gpu_mem_class(cell.ctrl, cell.vessel)
        start = self.os.now_ms()
        while True:
            free, total = self.gpu_memory()
            waited = self.os.now_ms() - start
            if free >= threshold:
                self.log('GPU-MEM-OK %s class=%s threshold_mib=%d free_mib=%d total_mib=%d waited_ms=%d' % (reason, klass, threshold, free, total, waited))
                return
            self.log('GPU-MEM-LOW %s class=%s threshold_mib=%d free_mib=%d total_mib=%d waited_ms=%d' % (reason, klass, threshold, free, total, waited))
            if waited + GPUMEMPOLLMS > self.gpumemwaitms:
                raise Refusal('GPU-MEMORY-LOW (%s): free %d MiB < %s=%d (class %s) after %d ms' % (reason, free, constant, threshold, klass, waited))
            self.os.sleep_ms(GPUMEMPOLLMS)

    # ---------------------------------------------------------------- host, environment, inputs, release root
    def assert_host(self):
        """Computer identity: the record's host entry when present, otherwise the R2 qualification file (header)."""
        key = self.args.host_key
        name = self.os.hostname()
        entry = (self.rec.get('hosts') or {}).get(key)
        if entry is not None:
            if name.lower() != str(entry.get('computer_name') or '').lower():
                raise Refusal('Wrong host: %s; host key %s expects %s' % (name, key, entry.get('computer_name')))
            self.computer_name = name
            self.identity = 'record'
            return
        folder = self.root + '/qualification'
        prefix = 'qualification_%s_%s_' % (self.rec['release_id'], key)
        names = []
        if self.os.isdir(folder):
            names = sorted(n for n, is_dir in self.os.listdir_entries(folder)
                           if not is_dir and n.startswith(prefix) and n.endswith('.json')
                           and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z', n[len(prefix):-len('.json')]))
        if not names:
            raise Refusal('Host key %s is not in the record and no %s/%s<stamp>.json exists (run the R2 qualify.sh with '
                          '--host-key %s first)' % (key, folder, prefix, key))
        newest = names[-1]
        try:
            doc = json.loads(self.os.read_bytes(folder + '/' + newest).decode('utf-8'))
        except (ValueError, OSError) as exc:
            raise Refusal('Unreadable qualification file %s: %s' % (newest, exc))
        wrong = []
        for field, want in (('format', QUALIFICATION_FORMAT), ('verdict', 'QUALIFIED'), ('release_id', self.rec['release_id']),
                            ('record_sha256', self.args.record_sha256.upper()), ('host_key', key), ('data_root', self.data_root)):
            have = doc.get(field) if isinstance(doc, dict) else None
            if (str(have).upper() if field == 'record_sha256' else have) != want:
                wrong.append('%s=%r (want %r)' % (field, have, want))
        recorded = str(doc.get('hostname') or '') if isinstance(doc, dict) else ''
        if recorded.lower() != name.lower():
            wrong.append('hostname=%r (this machine %r)' % (recorded, name))
        if wrong:
            raise Refusal('Wrong host or qualification: %s/%s: %s' % (folder, newest, '; '.join(wrong)))
        self.computer_name = name
        self.identity = 'qualification:%s' % newest

    def assert_environment(self):
        # Reference environment (CONTEXT_R4 D28): every version printed; any mismatch refuses. The probe python carries
        # the slot tag, so a sibling slot never takes it for a foreign process.
        code, text = self.os.run_capture([self.python_path, '-X', self.own_tag(), '-B', '-c', ENV_PROBE_CODE], PROBEMS)
        lines = [line for line in split_lines(text or '') if line != '']
        json_lines = [line for line in lines if line.startswith('ENVJSON ')]
        if code != 0 or len(json_lines) != 1:
            raise Refusal('Version probe failed (exit %s): %s' % (code, ' | '.join(lines)))
        found = json.loads(json_lines[0][8:])
        if self.os.isfile(self.version_file):
            first = split_lines(self.os.read_bytes(self.version_file).decode('ascii', 'replace'))
            found['isaaclab'] = first[0].strip() if first else ''
        else:
            found['isaaclab'] = 'MISSING'
        wrong, report = [], []
        for name, value in (self.rec.get('reference_environment') or {}).items():
            have = ps_string(found.get(name))
            report.append('%s=%s' % (name, have))
            if have != ps_string(value):
                wrong.append('%s=%s (required %s)' % (name, have, value))
        if wrong:
            raise Refusal('REFERENCE-ENVIRONMENT-MISMATCH: %s' % '; '.join(wrong))
        return 'REFERENCE-ENVIRONMENT-OK %s' % ' '.join(report)

    def assert_inputs(self):
        for path, want in self.hash_checks:
            have = self.os.sha256_upper(path)
            if have != want:
                raise Refusal('INPUT-HASH-MISMATCH path=%s expected=%s actual=%s' % (path, want, have))

    def assert_bundle(self):
        listed = {}
        for entry in self.rec['bundle'].get('files') or []:
            path = self.bundle_path + '/' + entry['name']
            if not self.os.isfile(path):
                raise Refusal('BUNDLE-FILE-MISSING %s' % entry['name'])
            if self.os.getsize(path) != int(entry['size']):
                raise Refusal('BUNDLE-SIZE-MISMATCH %s' % entry['name'])
            if self.os.sha256_upper(path) != entry['sha256'].upper():
                raise Refusal('BUNDLE-HASH-MISMATCH %s' % entry['name'])
            listed[entry['name']] = True
        listed[self.rec['bundle']['manifest_name']] = True
        present = self.os.listdir_entries(self.bundle_path)
        for name, is_dir in present:
            if is_dir or name not in listed:
                raise Refusal('Unlisted entry in the bundle: %s' % name)
        if len(present) != len(listed):
            raise Refusal('Bundle folder file count differs from the record')

    def assert_release_root(self, reason):
        # Cheap check (one directory listing): the release root holds only the record's release_top_level entries.
        top = self.rec.get('release_top_level') or {}
        foreign = []
        for name, is_dir in sorted(self.os.listdir_entries(self.release_path)):
            kind = top.get(name)
            if kind is None:
                foreign.append(name)
                continue
            if is_dir != (kind == 'dir'):
                foreign.append('%s(type)' % name)
        if foreign:
            self.log('RELEASE-ROOT-DIRTY %s entries=%s' % (reason, ','.join(foreign)))
            raise Refusal('RELEASE-ROOT-DIRTY (%s): %s in %s' % (reason, ','.join(foreign), self.release_path))
        self.log('RELEASE-ROOT-CLEAN %s' % reason)

    # ---------------------------------------------------------------- owned process group (the G2 native helper's role)
    def write_script(self, path, inner_command, work_dir):
        # WriteBat equivalent; LF line ends (bash). The RE-1 environment script, then the same variables as the Windows
        # .bat: pool directory set to the deployed bundle, the other three cleared.
        self.write_new_ascii(path, [
            '#!/bin/bash',
            '# R3 cell script (written by usvb_dose_executor.py; run as /bin/bash --noprofile --norc <this file>)',
            '. "%s" || exit 96' % self.env_file,
            'export OMNI_KIT_ACCEPT_EULA=YES',
            'export PYTHONDONTWRITEBYTECODE=1',
            'export USVBENCH_D7_POOL_DIR="%s"' % self.bundle_path,
            'unset USVBENCH_HARBOUR_D7_POOL USVBENCH_ASSETS_V2 USVBENCH_RELEASE_DESCRIPTOR',
            'cd "%s" || exit 97' % work_dir,
            'exec %s' % inner_command], newline='\n')

    def cell_env(self):
        return {'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONDONTWRITEBYTECODE': '1', 'USVBENCH_D7_POOL_DIR': self.bundle_path,
                'USVBENCH_HARBOUR_D7_POOL': None, 'USVBENCH_ASSETS_V2': None, 'USVBENCH_RELEASE_DESCRIPTOR': None}

    @staticmethod
    def helper_files(alive_file):
        if not alive_file.endswith('.alive.txt'):
            raise Refusal('Unexpected alive path: %s' % alive_file)
        stem = alive_file[:-len('.alive.txt')]
        return stem + '.launched.txt', stem + '.heartbeat.txt'

    def report_helper_files(self, label, alive_file):
        # Copy the LAUNCHED record, every JOB-STATE line and the last heartbeat into the progress log (diagnostics only).
        try:
            launched, heartbeat = self.helper_files(alive_file)
            if self.os.isfile(launched):
                first = split_lines(self.os.read_bytes(launched).decode('ascii', 'replace'))[0]
                self.log('HELPER-LAUNCHED %s %s' % (label, first))
            else:
                self.log('HELPER-LAUNCHED-MISSING %s (the group was never started)' % label)
            if self.os.isfile(heartbeat):
                lines = [line for line in split_lines(self.os.read_bytes(heartbeat).decode('ascii', 'replace')) if line]
                for line in lines:
                    if line.startswith('JOB-STATE-'):
                        self.log('%s %s %s' % (line.split(' ')[0], label, line[line.index(' ') + 1:]))
                self.log('HELPER-HEARTBEAT-LAST %s lines=%d %s' % (label, len(lines), lines[-1] if lines else 'empty'))
            else:
                self.log('HELPER-HEARTBEAT-MISSING %s' % label)
        except Exception as exc:  # noqa: BLE001 - diagnostics never throw
            self.log('HELPER-REPORT-FAILED %s %s' % (label, exc))

    def invoke_owned(self, script, limit_ms, alive_file, watched_log, label, cwd, info):
        launched, heartbeat = self.helper_files(alive_file)
        for path in (launched, heartbeat):
            if self.os.exists(path):
                raise Refusal('Attempt path already exists: %s' % path)
        try:
            return self.run_group(script, limit_ms, alive_file, launched, heartbeat, watched_log, cwd, info)
        finally:
            self.report_helper_files(label, alive_file)

    def run_group(self, script, limit_ms, alive_file, launched, heartbeat, watched_log, cwd, info):
        """One owned run: own session and process group, named timeout, TERM / KILL of the whole group, confirmed empty.
        Returns (exit code, timed out); exit code 99 on a timeout, as the Windows helper's TerminateJobObject(job, 99)."""
        if limit_ms < 60000 or EXITWAITMS < 1 or ALIVEPROBEMS < 1000 or ALIVEPROBEMS >= limit_ms or HEARTBEATMS < 1000:
            raise Refusal('Invalid named timeout')
        start = self.os.now_ms()
        state = {'stage': 'init', 'anomaly': 'none', 'closed': False}

        def hb(line):
            if state['closed']:
                return
            try:
                self.os.append_bytes(heartbeat, (line + '\r\n').encode('ascii', 'replace'))
            except Exception:  # noqa: BLE001 - diagnostics never throw
                pass

        def snapshot():
            try:
                members = self.os.group_members(handle) if handle is not None else []
                text = 'active=%d pids=%s' % (len(members), ','.join(str(m) for m in members))
                if handle is not None:
                    rc = self.os.leader_poll(handle)
                    text += ' cmd_pid=%d %s' % (handle.r3_pid, 'cmd=running' if rc is None else 'cmd=exited:%d' % rc)
                else:
                    text += ' cmd=none'
                text += ' log_bytes=%s' % (self.os.getsize(watched_log) if self.os.isfile(watched_log) else 'missing')
                return members, text
            except Exception as exc:  # noqa: BLE001
                return [], 'snapshot_error=%s' % type(exc).__name__

        def check_anomaly(members, leader_rc):
            if handle is None or state['stage'] == 'timeout-stopping':
                return
            kind = 'cmd-exited-job-active' if (members and leader_rc is not None) else 'none'
            if kind == state['anomaly']:
                return
            previous, state['anomaly'] = state['anomaly'], kind
            _m, text = snapshot()
            if kind != 'none':
                hb('JOB-STATE-ANOMALY kind=%s utc=%s elapsed_ms=%d stage=%s %s' % (kind, self.os.utc_stamp(), self.os.now_ms() - start, state['stage'], text))
            else:
                hb('JOB-STATE-NORMAL after=%s utc=%s elapsed_ms=%d stage=%s %s' % (previous, self.os.utc_stamp(), self.os.now_ms() - start, state['stage'], text))

        handle = None
        confirmed_empty = False
        alive_written = False
        outcome = 'exception'
        hb('HB-START utc=%s queue_pid=%d batch=%s limit_ms=%d heartbeat_ms=%d watched_log=%s'
           % (self.os.utc_stamp(), self.os.self_pid(), script, limit_ms, HEARTBEATMS, watched_log))
        try:
            state['stage'] = 'creating-process'
            handle = self.os.spawn_group(script, watched_log, cwd, info)
            state['stage'] = 'running'
            _m, text = snapshot()
            try:
                self.os.write_new(launched, ('LAUNCHED utc=%s elapsed_ms=%d %s\r\n' % (self.os.utc_stamp(), self.os.now_ms() - start, text)).encode('ascii', 'replace'))
            except Exception as exc:  # noqa: BLE001
                hb('LAUNCHED-RECORD-FAILED %s' % type(exc).__name__)
            next_beat = HEARTBEATMS
            while True:
                leader_rc = self.os.leader_poll(handle)
                members = self.os.group_members(handle)
                if not members and leader_rc is not None:
                    break
                check_anomaly(members, leader_rc)
                elapsed = self.os.now_ms() - start
                if not alive_written and elapsed >= ALIVEPROBEMS:
                    self.os.write_new(alive_file, ('alive=true active=%d pids=%s elapsed_ms=%d\r\n'
                                                   % (len(members), ','.join(str(m) for m in members), elapsed)).encode('ascii'))
                    alive_written = True
                if elapsed >= next_beat:
                    _m, text = snapshot()
                    hb('HB utc=%s elapsed_ms=%d stage=%s %s' % (self.os.utc_stamp(), elapsed, state['stage'], text))
                    next_beat += HEARTBEATMS
                if elapsed >= limit_ms:
                    _m, text = snapshot()
                    hb('HB-TIMEOUT utc=%s elapsed_ms=%d stage=%s %s' % (self.os.utc_stamp(), elapsed, state['stage'], text))
                    state['stage'] = 'timeout-stopping'
                    self.stop_group(handle, hb, start)
                    confirmed_empty = True
                    outcome = 'timeout'
                    return 99, True
                self.os.sleep_ms(CELLPOLLMS)
            state['stage'] = 'job-empty'
            check_anomaly([], leader_rc)
            confirmed_empty = True
            if not alive_written:
                self.os.write_new(alive_file, ('alive=ended_before_probe elapsed_ms=%d\r\n' % (self.os.now_ms() - start)).encode('ascii'))
                alive_written = True
            outcome = 'exited:%d' % leader_rc
            return leader_rc, False
        finally:
            _m, text = snapshot()
            hb('HB-END outcome=%s utc=%s elapsed_ms=%d stage=%s %s' % (outcome, self.os.utc_stamp(), self.os.now_ms() - start, state['stage'], text))
            state['closed'] = True
            if handle is not None and not confirmed_empty:
                # an exception or SIGTERM to the executor: stop this run's own group only (the Windows job closes)
                self.stop_group(handle, None, start)

    def stop_group(self, handle, hb, start):
        """SIGTERM to the group and session, SIGKILL after KILLGRACEMS, then confirm no member within EXITWAITMS."""
        stop_start = self.os.now_ms()
        self.os.signal_group(handle, 'TERM')
        if hb:
            hb('GROUP-TERM utc=%s elapsed_ms=%d pids=%s' % (self.os.utc_stamp(), self.os.now_ms() - start, ','.join(str(p) for p in self.os.group_members(handle))))
        while self.os.now_ms() - stop_start < KILLGRACEMS:
            self.os.leader_poll(handle)
            if not self.os.group_members(handle):
                return
            self.os.sleep_ms(CELLPOLLMS)
        self.os.signal_group(handle, 'KILL')
        if hb:
            hb('GROUP-KILL utc=%s elapsed_ms=%d pids=%s' % (self.os.utc_stamp(), self.os.now_ms() - start, ','.join(str(p) for p in self.os.group_members(handle))))
        while True:
            self.os.leader_poll(handle)
            members = self.os.group_members(handle)
            if not members:
                return
            if self.os.now_ms() - stop_start >= EXITWAITMS:
                raise Refusal('GROUP-SURVIVOR pgid=%d pids=%s: the owned process group did not become empty after termination; '
                              'no further cells may run' % (handle.r3_pid, ','.join(str(m) for m in members)))
            self.os.sleep_ms(CELLPOLLMS)

    def verify_release(self, label):
        # kit verify-tree from the incoming folder: exact comparison of the deployed release with its manifests.
        base = '%s/verify_tree_%s.%s' % (self.out_root, label, self.attempt_id)
        verify_log, verify_script, verify_alive = base + '.log', base + '.sh', base + '.alive.txt'
        command = ('"%s" -X %s -B "%s" verify-tree --root "%s" --manifest "%s" --assets-manifest "%s" --expect-code-tree %s >> "%s" 2>&1'
                   % (self.python_path, self.own_tag(), self.kit_path, self.release_path, self.manifest_file_path,
                      self.assets_manifest_path, self.rec['code_tree_sha256'], verify_log))
        self.write_new_ascii(verify_log, ['verify-tree %s' % label, 'command=%s' % command])
        self.write_script(verify_script, command, self.incoming_path)
        self.wait_own_python('before verify-tree %s' % label)
        self.assert_release_root('before verify-tree %s' % label)
        self.log('VERIFY-TREE %s %s' % (label, command))
        info = {'kind': 'verify', 'argv': [self.python_path, '-X', self.own_tag(), '-B', self.kit_path, 'verify-tree', '--root', self.release_path,
                                           '--manifest', self.manifest_file_path, '--assets-manifest', self.assets_manifest_path,
                                           '--expect-code-tree', self.rec['code_tree_sha256']],
                'log': verify_log, 'env': self.cell_env()}
        rc, timed_out = self.invoke_owned(verify_script, VERIFYMS, verify_alive, verify_log, 'verify-tree-%s' % label, self.incoming_path, info)
        self.wait_own_python('after verify-tree %s' % label)
        self.assert_release_root('after verify-tree %s' % label)
        if timed_out or rc != 0:
            raise Refusal('RELEASE-VERIFY-FAILED %s rc=%s timedout=%s log=%s' % (label, rc, ps_bool(timed_out), verify_log))
        text = self.os.read_bytes(verify_log).decode('ascii', 'replace')
        if ('TREE-OK files=%s code_tree=%s' % (self.rec['verify_tree_file_count'], self.rec['code_tree_sha256'])) not in text:
            raise Refusal('RELEASE-VERIFY-NO-TREE-OK %s log=%s' % (label, verify_log))
        self.log('VERIFY-TREE-OK %s files=%s' % (label, self.rec['verify_tree_file_count']))

    # ---------------------------------------------------------------- plan input
    def read_batch(self, path, sha, attempt, from_order):
        raw = self.os.read_bytes(path)       # one read: hashed, byte-checked and parsed (G12 update 4 ReadInputLines)
        if hashlib.sha256(raw).hexdigest().upper() != sha.upper():
            raise Refusal('PLAN-HASH-MISMATCH %s' % path)
        lines = check_input_bytes(raw, path, 'PLAN')
        if any(re.search(r'[^\x20-\x7E]', line) for line in lines):
            raise Refusal('Non-ASCII or control text in %s' % path)
        meta, timeouts, reasons, ceilings = {}, {}, {}, {}
        predictions, references, rows = [], [], []
        for line in lines:
            if not line:
                continue
            if line.startswith('PREDICTION '):
                predictions.append(line[11:])
                continue
            if line.startswith('REFERENCE '):
                references.append(line)
                continue
            parts = line.split('|')
            head = parts[0].lower()          # PowerShell switch: case-insensitive
            if head == 'ceiling':
                if len(parts) != 3 or not re.fullmatch(r'-?[0-9]+', parts[2].strip()):
                    raise Refusal('Bad ceiling line in %s' % path)
                ceilings[parts[1]] = int(parts[2])
                continue
            if head == 'timeout':
                if (len(parts) != 4 or not re.fullmatch(r'CELLMS_[A-Z0-9_]+_[A-Z0-9]+', parts[1])
                        or not re.fullmatch(r'[1-9][0-9]{4,8}', parts[2]) or not parts[3]):
                    raise Refusal('Bad timeout line in %s: %s' % (path, line))
                if parts[1].upper() in timeouts:
                    raise Refusal('Timeout %s declared twice in %s' % (parts[1], path))
                timeouts[parts[1].upper()] = int(parts[2])
                reasons[parts[1].upper()] = parts[3]
                continue
            if head == 'cell':
                if len(parts) not in (13, 14) or not re.fullmatch(r'\s*[+-]?[0-9]+\s*', parts[1]):
                    raise Refusal('Bad cell line in %s: %s' % (path, line))
                host_timeouts = {}
                for pair in parts[11].split(','):
                    match = re.fullmatch(r'([A-Z0-9]+)=(CELLMS_[A-Z0-9_]+)', pair)
                    if not match or match.group(1).upper() in host_timeouts:
                        raise Refusal('Bad timeout field in %s: %s' % (path, line))
                    host_timeouts[match.group(1).upper()] = match.group(2)
                row = Cell()
                row.order, row.name, row.block, row.journal_id, row.vessel = int(parts[1]), parts[2], parts[3], parts[4], parts[5]
                row.carrier, row.ctrl, row.kind, row.axis, row.dose = parts[6], parts[7], parts[8], parts[9], parts[10]
                row.host_timeouts = host_timeouts
                row.part_count = len(parts)
                row.expected_controller = parts[12] if len(parts) == 14 else ''     # update 5: batch-v3 column
                row.suite = parts[-1].split(' ') if parts[-1] else []
                rows.append(row)
                continue
            match = re.fullmatch(r'([a-z_0-9]+)=(.*)', line)
            if match:
                meta[match.group(1)] = match.group(2)
                continue
            raise Refusal('Unknown line in %s: %s' % (path, line))
        if meta.get('format') not in ('usvbench-g12-batch-v2', 'usvbench-g12-batch-v3'):
            raise Refusal('Unexpected batch format in %s: %s' % (path, meta.get('format')))
        named = meta['format'] == 'usvbench-g12-batch-v3'
        if not named:
            raise Refusal('V2-BATCH-REFUSED %s: the G32 v2.6 executor runs only usvbench-g12-batch-v3 batches (release-aware names; '
                          'make them with work_v26 tools/plan_rerun_set.py); run v2 batches with the original executor' % path)
        if [row for row in rows if row.part_count != (14 if named else 13)]:
            raise Refusal('Cell rows of %s do not match its format %s' % (path, meta['format']))
        if named:
            if meta.get('release_id') != self.rec['release_id']:
                raise Refusal('BATCH-RELEASE-MISMATCH batch %s was generated for release %s, the record is %s'
                              % (meta.get('batch_id', ''), meta.get('release_id', ''), self.rec['release_id']))
            if not re.fullmatch(r'[0-9A-F]{64}', meta.get('controller_names_sha256', '')):
                raise Refusal('Bad or missing controller_names_sha256 in %s' % path)
        elif 'release_id' in meta or 'controller_names_sha256' in meta:
            raise Refusal('A usvbench-g12-batch-v2 batch carries no release names (%s)' % path)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', meta.get('batch_id', '')):
            raise Refusal('Bad batch_id in %s' % path)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', meta.get('affinity', '')):
            raise Refusal('Bad or missing affinity id in %s' % path)
        if meta.get('mode') not in ('DIAG42', 'FINAL'):
            raise Refusal('Only modes DIAG42 and FINAL are supported (batch %s)' % meta.get('batch_id'))
        for key in ('num_envs', 'episodes_per_env', 'eval_seed', 'level'):
            if not re.fullmatch(r'[0-9]{1,6}', meta.get(key, '')):
                raise Refusal('Bad %s in %s' % (key, path))
        if int(meta['num_envs']) < 1 or int(meta['num_envs']) > 99 or int(meta['episodes_per_env']) < 1 or int(meta['episodes_per_env']) > 2:
            raise Refusal('Grid %s x %s not supported by the gate' % (meta['num_envs'], meta['episodes_per_env']))
        if not predictions:
            raise Refusal('Batch %s registers no prediction' % meta['batch_id'])
        final = self.final_meta(meta, rows, path)            # G51: None for DIAG42
        if HOST_CLASS not in ceilings:
            raise Refusal('Batch %s has no ceiling for host %s' % (meta['batch_id'], HOST_CLASS))
        if not rows:
            raise Refusal('Batch %s has no cell' % meta['batch_id'])
        folded, last_order = set(), 0
        for row in rows:
            if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', row.name):
                raise Refusal('Bad cell name %s' % row.name)
            if row.name.lower() in folded:
                raise Refusal('Case-insensitive cell name collision: %s' % row.name)
            folded.add(row.name.lower())
            if row.order <= last_order:
                raise Refusal('Cell orders must ascend in %s (%s)' % (path, row.name))
            last_order = row.order
            if row.ctrl == 'cl' and row.carrier not in CLASSICAL_BY_CARRIER:
                raise Refusal('Unknown carrier %s' % row.carrier)
            if not re.fullmatch(r'cl|mp|ppo', row.ctrl) or (row.ctrl == 'ppo' and final is None):
                raise Refusal('Unknown controller %s (ppo only in FINAL batches)' % row.ctrl)
            if row.ctrl == 'ppo' and row.expected_controller != 'learned':
                raise Refusal("Cell %s: a PPO cell must expect controller 'learned', not '%s'" % (row.name, row.expected_controller))
            if named:
                if not re.fullmatch(r'[a-z][a-z0-9_]{0,59}', row.expected_controller):
                    raise Refusal("Cell %s: bad expected controller name '%s'" % (row.name, row.expected_controller))
                if row.ctrl == 'mp' and row.expected_controller != 'mppi':
                    raise Refusal("Cell %s: an MPPI cell must expect controller 'mppi', not '%s'" % (row.name, row.expected_controller))
            gpu_mem_class(row.ctrl, row.vessel)
            host_name = row.host_timeouts.get(HOST_CLASS, '')
            if (not host_name or not host_name.endswith('_' + HOST_CLASS) or host_name.upper() not in timeouts
                    or not reasons.get(host_name.upper())):
                raise Refusal('Cell %s: timeout %s is not a declared, reasoned timeout of host %s' % (row.name, host_name, HOST_CLASS))
            row.timeout_name = host_name
            row.timeout_ms = timeouts[host_name.upper()]
            row.reason = reasons[host_name.upper()]
        selected = [row for row in rows if row.order >= from_order]
        if not selected:
            raise Refusal('Batch %s: no cell at or after order %d' % (meta['batch_id'], from_order))
        batch = Batch()
        batch.path, batch.sha, batch.meta, batch.id, batch.attempt, batch.from_order = path, sha.upper(), meta, meta['batch_id'], attempt, from_order
        batch.num_envs, batch.seed, batch.level = int(meta['num_envs']), int(meta['eval_seed']), int(meta['level'])
        batch.episodes_per_env = int(meta['episodes_per_env'])
        batch.final = final
        batch.ceiling = ceilings[HOST_CLASS]
        batch.predictions, batch.references, batch.reasons = predictions, references, reasons
        batch.cells, batch.all_cells = selected, len(rows)
        batch.out_root = '%s/dose_%s_a%d' % (self.root, meta['batch_id'], attempt)
        return batch

    def final_meta(self, meta, rows, path):
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
        plan_list = self.args.plan_list
        raw = self.os.read_bytes(plan_list)
        if hashlib.sha256(raw).hexdigest().upper() != self.args.plan_list_sha256.upper():
            raise Refusal('PLANLIST-HASH-MISMATCH %s' % plan_list)
        lines = [line for line in check_input_bytes(raw, plan_list, 'PLANLIST') if line]
        if len(lines) < 2 or lines[0] != 'format=usvbench-g12-planlist-v1':
            raise Refusal('Unexpected plan list format: %s' % plan_list)
        list_dir = posixpath.dirname(self.os.abspath(plan_list))
        batches = []
        for line in lines[1:]:
            match = re.fullmatch(r'batch\|([A-Za-z0-9_.-]+\.plan)\|([0-9A-Fa-f]{64})\|([1-9])\|([1-9][0-9]{0,5})', line)
            if not match:
                raise Refusal('Bad plan list line: %s' % line)
            batches.append(self.read_batch(list_dir + '/' + match.group(1), match.group(2), int(match.group(3)), int(match.group(4))))
        if len(batches) > MAX_BATCHES:
            raise Refusal('More than %d batches in %s' % (MAX_BATCHES, plan_list))
        roots = set()
        for batch in batches:
            if batch.out_root in roots:
                raise Refusal('Batch directory listed twice: %s' % batch.out_root)
            roots.add(batch.out_root)
        return batches

    def batch_cells(self, batch, token):
        root = batch.out_root
        out = []
        for row in batch.cells:
            cell = Cell()
            for key in ('order', 'name', 'block', 'journal_id', 'vessel', 'carrier', 'ctrl', 'kind', 'axis', 'dose', 'suite',
                        'timeout_name', 'timeout_ms', 'reason', 'expected_controller'):
                setattr(cell, key, getattr(row, key))
            cell.num_envs, cell.seed, cell.level = batch.num_envs, batch.seed, batch.level
            cell.episodes_per_env = batch.episodes_per_env
            cell.final = getattr(batch, 'final', None)
            cells = root + '/cells/' + row.name
            cell.output = cells + '.json'
            cell.stage = '%s.%s.stage.json' % (cells, token)
            cell.log = '%s.%s.log' % (cells, token)
            cell.script = '%s.%s.sh' % (cells, token)
            cell.alive = '%s.%s.alive.txt' % (cells, token)
            cell.inflight = cells + '.json.INFLIGHT'
            cell.receipt = '%s.%s.status.txt' % (cells, token)
            cell.validated = '%s.json.VALIDATED.%s' % (cells, token)
            cell.cwd = root + '/cwd/' + row.name
            out.append(cell)
        return out

    def build_arguments(self, cell):
        if getattr(cell, 'final', None):
            return self.build_final_arguments(cell)
        evaluator = MPPI_EVALUATOR if cell.ctrl == 'mp' else CLASSICAL_BY_CARRIER[cell.carrier]
        tokens = ['-X', self.own_tag(), '-B', '%s/scripts/%s' % (self.release_path, evaluator['script'])]
        tokens.extend(evaluator['extra'])
        tokens.extend(['--task', cell.journal_id, '--num_envs', str(cell.num_envs), '--episodes', str(cell.num_envs * cell.episodes_per_env),
                       '--level', str(cell.level), evaluator['seed_flag'], str(cell.seed), '--vessel', cell.vessel,
                       '--device', 'cuda:0', '--headless', '--out', cell.stage, '--release-descriptor', self.descriptor_path])
        if cell.ctrl == 'mp':
            tokens.extend(['--mppi-seed', '0'])
        if cell.carrier == 'docking':
            tokens.extend(['--set', 'd7_pool_dir=%s' % self.bundle_path])
        if cell.suite:
            tokens.append('--suite')
            tokens.extend(cell.suite)
        for token in tokens:
            if UNSAFE_TOKEN_RE.search(token) or NONPRINT_TOKEN_RE.search(token):
                raise Refusal('Unsafe command token: %s' % token)
        return tokens

    def build_final_arguments(self, cell):
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
        return '"%s" %s >> "%s" 2>&1' % (self.python_path, ' '.join(self.build_arguments(cell)), cell.log)

    @staticmethod
    def worst_case_ms(cells):
        total = 0
        for cell in cells:
            total += cell.timeout_ms + EXITWAITMS + 2 * PYEXITWAITMS + CEILING_GPUMEM_ALLOWANCE_MS
        return total + 2 * VERIFYMS

    def build_manifest(self, batch, cells):
        meta = batch.meta
        used = sorted_unique([c.timeout_name for c in cells])
        lines = ['G12 executor manifest v4']
        lines.append(('host_key=%s; slot=%d; slots=%d; batch_id=%s; attempt=%d; from_order=%d; batch_cells=%d; selected_cells=%d; '
                      'affinity=%s; host_name=%s; mode=%s; release_id=%s; code_tree=%s; bundle_id=%s; record_sha256=%s; plan_sha256=%s; '
                      'planlist_sha256=%s; plan_json_sha256=%s; predictions_sha256=%s; executor_predictions_sha256=%s; '
                      'rental_host_key=%s; identity=%s; data_root=%s; pool_dir=%s; executor=usvb_dose_executor.py')
                     % (HOST_CLASS, self.args.slot, self.args.slots, batch.id, batch.attempt, batch.from_order, batch.all_cells,
                        len(cells), meta.get('affinity', ''), self.computer_name, meta.get('mode', ''), self.rec['release_id'],
                        self.rec['code_tree_sha256'], self.rec['bundle']['bundle_id'], self.args.record_sha256.upper(), batch.sha,
                        self.args.plan_list_sha256.upper(), meta.get('plan_json_sha256', ''), predictions_sha(batch.predictions),
                        PREDICTIONS_SHA256, self.args.host_key, self.identity, self.data_root, self.bundle_path))
        lines.append('grid: num_envs=%d episodes=%d episodes_per_env=%d level=%d eval_seed=%d' % (batch.num_envs, batch.num_envs * batch.episodes_per_env, batch.episodes_per_env, batch.level, batch.seed))
        final = getattr(batch, 'final', None)
        if final:
            ppo = final['ppo'] or {}
            lines.append('final: route=run_cell.py seed_namespace=%s seed_block=%s run_purpose=%s ppo_training_seed=%s ppo_checkpoint=%s '
                         'ppo_checkpoint_sha256=%s gate_ledger=%s gate_ledger_sha256=%s'
                         % (final['seed_namespace'], final['seed_block'], final['run_purpose'], ppo.get('ppo_training_seed', '-'),
                            ppo.get('ppo_checkpoint', '-'), ppo.get('ppo_checkpoint_sha256', '-'), ppo.get('gate_ledger', '-'),
                            ppo.get('gate_ledger_sha256', '-')))
        lines.append(('slots: own_tag=%s other_tags=%s slot_mutexes=%s legacy_probe=%s gpu_mem_min_free_mib=classical:%d,ppo:%d,'
                      'mppi_blueboat:%d,mppi_wamv:%d gpumemwaitms=%d gpumemwait_source=%s gpumempollms=%d ceiling_gpumem_allowance_ms=%d '
                      'gpu_lock=%s python_scope=re1-interpreter-or-kit-root')
                     % (self.own_tag(), ','.join(self.other_tags()), ','.join(self.slot_lock_paths()), ps_bool(self.args.slots > 1),
                        GPU_MEM_MIN_FREE_MIB_CLASSICAL, GPU_MEM_MIN_FREE_MIB_PPO, GPU_MEM_MIN_FREE_MIB_MPPI_BLUEBOAT,
                        GPU_MEM_MIN_FREE_MIB_MPPI_WAMV, self.gpumemwaitms, self.gpumemwait_source, GPUMEMPOLLMS,
                        CEILING_GPUMEM_ALLOWANCE_MS, self.gpu_lock_path))
        lines.append('timeouts_ms: %s VERIFYMS=%d EXITWAITMS=%d ALIVEPROBEMS=%d PYEXITWAITMS=%d PYPOLLMS=%d HEARTBEATMS=%d KILLGRACEMS=%d CELLPOLLMS=%d PROBEMS=%d NVSMIMS=%d'
                     % (' '.join('%s=%d' % (name, [c.timeout_ms for c in cells if c.timeout_name == name][0]) for name in used),
                        VERIFYMS, EXITWAITMS, ALIVEPROBEMS, PYEXITWAITMS, PYPOLLMS, HEARTBEATMS, KILLGRACEMS, CELLPOLLMS, PROBEMS, NVSMIMS))
        lines.append('ceiling: batch_ceiling_ms_%s=%d worst_case_ms_selected=%d' % (HOST_CLASS, batch.ceiling, self.worst_case_ms(cells)))
        lines.append('reference_environment: %s' % ' '.join('%s=%s' % (k, ps_string(v)) for k, v in (self.rec.get('reference_environment') or {}).items()))
        named = meta.get('format') == 'usvbench-g12-batch-v3'
        if named:
            lines.append('release_names: batch_format=usvbench-g12-batch-v3 release_id=%s controller_names_sha256=%s'
                         % (meta.get('release_id', ''), meta.get('controller_names_sha256', '')))
        lines.extend('PREDICTION %s' % text for text in batch.predictions)
        lines.extend(batch.references)
        lines.extend('timeout_reason|%s|%s' % (name, batch.reasons[name.upper()]) for name in used)
        lines.append('cell|order|affinity|name|block|journal_id|vessel|carrier|ctrl|kind|axis|dose|timeout_name|timeout_ms|suite')
        for c in cells:
            lines.append('cell|%d|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%d|%s' % (c.order, meta.get('affinity', ''), c.name, c.block, c.journal_id, c.vessel,
                                                                         c.carrier, c.ctrl, c.kind, c.axis, c.dose, c.timeout_name, c.timeout_ms, ' '.join(c.suite)))
            lines.append('command|%s|%s' % (c.name, self.build_command(c)))
            klass, _constant, threshold = gpu_mem_class(c.ctrl, c.vessel)
            lines.append('gpumem|%s|class=%s|threshold_mib=%d' % (c.name, klass, threshold))
            if named:
                lines.append('controller|%s|%s' % (c.name, expected_controller_name(c)))
            lines.append('paths|%s|final=%s|stage=%s|log=%s|cwd=%s|status=%s' % (c.name, c.output, c.stage, c.log, c.cwd, c.receipt))
        return lines

    # ---------------------------------------------------------------- certificate gate (dose_judge.py does the real check)
    def report(self, json_path, cell):
        if not self.os.isfile(json_path):
            partial = json_path + '.partial'
            note = '; %s exists (%d bytes, kept)' % (partial, self.os.getsize(partial)) if self.os.isfile(partial) else ''
            raise Refusal('NO-CERTIFICATE: the evaluator exited 0 without writing --out %s%s' % (json_path, note))
        grid_envs = cell.num_envs
        grid_epe = cell.episodes_per_env
        grid = grid_envs * grid_epe
        try:
            doc = json.loads(self.os.read_bytes(json_path).decode('utf-8-sig'))
        except ValueError as exc:
            raise Refusal('CERT-NOT-JSON: %s (%s)' % (json_path, exc))
        header = doc.get('header') if isinstance(doc, dict) else None
        if not isinstance(doc, dict) or not isinstance(header, dict):
            raise Refusal('CERT-NOT-OBJECT: %s' % json_path)
        if header.get('task_id') != cell.journal_id:
            raise Refusal('TASK-MISMATCH: %s' % ps_string(header.get('task_id')))
        controller = header.get('controller') if isinstance(header.get('controller'), dict) else {}
        want_controller = expected_controller_name(cell)
        if controller.get('name') != want_controller:
            raise Refusal('CONTROLLER-MISMATCH: %s' % ps_string(controller.get('name')))
        if not is_finite_number(header.get('eval_seed')) or header.get('eval_seed') != cell.seed:
            raise Refusal('SEED-MISMATCH: %s' % ps_string(header.get('eval_seed')))
        for key, want in (('num_envs', grid_envs), ('episodes_per_env', grid_epe), ('episodes_requested', grid)):
            if not is_finite_number(header.get(key)) or header.get(key) != want:
                raise Refusal('GRID-MISMATCH: %s' % json_path)
        if header.get('complete') is not True:
            raise Refusal('INCOMPLETE: %s' % json_path)
        final = getattr(cell, 'final', None)
        if final:
            reg = header.get('seed_namespace') if isinstance(header.get('seed_namespace'), dict) else {}
            if (reg.get('namespace') != final['seed_namespace'] or reg.get('purpose') != final['run_purpose']
                    or reg.get('block_id') != final['seed_block'] or reg.get('seed') != cell.seed):
                raise Refusal('SEED-REGISTRATION-MISMATCH: %s' % ps_string(json.dumps(reg, sort_keys=True)))
        code = header.get('code') if isinstance(header.get('code'), dict) else {}
        if code.get('release_id') != self.rec['release_id'] or code.get('code_tree_sha256') != self.rec['code_tree_sha256']:
            raise Refusal('RELEASE-MISMATCH: %s' % ps_string(code.get('release_id')))
        records = doc.get('records')
        if not isinstance(records, list) or len(records) != grid:
            raise Refusal('RECORD-COUNT: %d' % (len(records) if isinstance(records, list) else (0 if records is None else 1)))
        seen = set()
        for record in records:
            env_text = ps_string(record.get('env')) if isinstance(record, dict) else ''
            if (not isinstance(record, dict) or not isinstance(record.get('success'), bool)
                    or not re.fullmatch(r'0|[1-9][0-9]?', env_text) or int(env_text) >= grid_envs
                    or not re.fullmatch(r'[0-9]', ps_string(record.get('ep'))) or int(ps_string(record.get('ep'))) >= grid_epe):
                raise Refusal('INVALID record: %s' % json_path)
            key = '%s/%s' % (env_text, ps_string(record.get('ep')))
            if key in seen:
                raise Refusal('DUPLICATE env/ep: %s' % key)
            seen.add(key)
        if len(seen) != grid:
            raise Refusal('INCOMPLETE env/ep grid')
        success = len([r for r in records if r.get('success')])
        return 'success=%d/%d complete=true grid=%dx%d release=%s controller=%s' % (success, grid, grid_envs, grid_epe, self.rec['release_id'], controller.get('name'))

    # ---------------------------------------------------------------- one cell, one batch
    def record_outcome(self, cell, state, detail):
        detail = one_line(detail)
        self.write_new_ascii(cell.receipt, ['cell=%s' % cell.name, 'state=%s' % state, 'detail=%s' % detail, 'command=%s' % self.build_command(cell)])
        self.log('%s %s %s' % (state, cell.name, detail))
        self.outcomes.append((cell.name, state, detail))

    def run_cell_script(self, cell):
        # Named timeout: declared for RENTAL with a reason (checked in read_batch); value re-checked here.
        if not re.fullmatch(r'CELLMS_[A-Z0-9_]+_' + HOST_CLASS, cell.timeout_name) or not cell.reason:
            raise Refusal('Not a named, reasoned timeout of this host: %s' % cell.timeout_name)
        if cell.timeout_ms < 60000:
            raise Refusal("Timeout below the helper's 60 s minimum: %s" % cell.timeout_name)
        if self.os.exists(cell.cwd):
            raise Refusal('Cell working directory already exists: %s' % cell.cwd)
        self.os.mkdir(cell.cwd)
        inner = self.build_command(cell)
        self.write_script(cell.script, inner, cell.cwd)
        self.assert_inputs()
        if getattr(cell, 'final', None) and cell.ctrl == 'ppo':
            self.assert_ppo_inputs(cell)
        self.assert_release_root('before %s' % cell.name)
        self.wait_own_python('before %s' % cell.name)
        if self.args.slots > 1:
            self.probe_gpu_lock('before %s' % cell.name)
        self.wait_gpu_memory('before %s' % cell.name, cell)
        self.log('COMMAND %s' % inner)
        self.log('LAUNCH /bin/bash --noprofile --norc "%s" timeout=%s=%d reason=%s' % (cell.script, cell.timeout_name, cell.timeout_ms, cell.reason))
        info = {'kind': 'cell', 'argv': [self.python_path] + self.build_arguments(cell), 'log': cell.log, 'env': self.cell_env(),
                'cell': cell.name}
        return self.invoke_owned(cell.script, cell.timeout_ms, cell.alive, cell.log, cell.name, cell.cwd, info)

    def run_batch(self, batch):
        # One queue directory: folder, progress file, manifest, verify-tree, cells, verify-tree, FINISHED or ABORTED.
        # Returns 0 (all cells SUCCESS), 1 (finished with failed cells) or 2 (aborted).
        self.out_root = batch.out_root
        self.outcomes = []
        self.attempt_id = self.os.new_attempt_id()
        cells = []
        terminal_owned = finished_committed = all_cells_ended = False
        abort_reason, batch_exit = '', 2
        progress_path = self.out_root + '/progress_dose.txt'
        try:
            self.os.mkdir(self.out_root)
            terminal_owned = True
            for sub in ('cells', 'cwd'):
                self.os.mkdir(self.out_root + '/' + sub)
            self.progress = self.os.open_new_writer(progress_path)
            self.progress_owned = True
            batch_start = self.os.now_ms()
            self.log('START G12 executor batch=%s attempt=%d from_order=%d host=%s slot=%d/%d release=%s bundle=%s record_sha256=%s plan_sha256=%s'
                     % (batch.id, batch.attempt, batch.from_order, HOST_CLASS, self.args.slot, self.args.slots, self.rec['release_id'],
                        self.rec['bundle']['bundle_id'], self.args.record_sha256.upper(), batch.sha))
            self.log(self.environment_line)
            self.log('GPU-MEM-START free_mib=%d total_mib=%d nvidia_smi=%s' % (self.start_memory[0], self.start_memory[1], self.nvidia_smi))
            self.log('SLOT-MUTEXES %s own_tag=%s other_tags=%s' % (','.join(self.slot_lock_paths()), self.own_tag(), ','.join(self.other_tags())))
            self.assert_release_root('batch %s start' % batch.id)
            cells = self.batch_cells(batch, self.attempt_id)
            manifest = self.build_manifest(batch, cells)
            self.write_new_ascii(self.out_root + '/dose_manifest.txt', manifest)
            for entry in manifest:
                if not entry.startswith('paths|'):
                    self.log('MANIFEST %s' % entry)
            self.verify_release('start')
            for cell in cells:
                if self.os.now_ms() - batch_start + cell.timeout_ms + EXITWAITMS + PYEXITWAITMS + CEILING_GPUMEM_ALLOWANCE_MS > batch.ceiling:
                    self.record_outcome(cell, 'NOT-STARTED-CEILING', 'elapsed_ms=%d timeout_ms=%d ceiling_ms=%d'
                                        % (self.os.now_ms() - batch_start, cell.timeout_ms, batch.ceiling))
                    continue
                self.assert_inputs()
                for path in (cell.output, cell.inflight, cell.stage, cell.log, cell.script, cell.alive, cell.receipt, cell.validated,
                             cell.stage + '.cell.json', cell.cwd):
                    if self.os.exists(path):
                        raise Refusal('Attempt path already exists: %s' % path)
                self.write_new_ascii(cell.inflight, ['cell=%s' % cell.name, 'state=INFLIGHT', 'stage=%s' % cell.stage, 'output=%s' % cell.output,
                                                     'command=%s' % self.build_command(cell), 'manifest=%s/dose_manifest.txt' % self.out_root,
                                                     'attempt=%s' % self.attempt_id])
                self.write_new_ascii(cell.log, ['cell=%s' % cell.name, 'command=%s' % self.build_command(cell)])
                self.log('START-CELL %s order=%d %s=%d log=%s' % (cell.name, cell.order, cell.timeout_name, cell.timeout_ms, cell.log))
                cell_start = self.os.now_ms()
                rc, timed_out = self.run_cell_script(cell)
                wall = self.os.now_ms() - cell_start
                self.log('WALL %s wall_ms=%d timedout=%s rc=%d' % (cell.name, wall, ps_bool(timed_out), rc))
                self.wait_own_python('after %s' % cell.name)
                self.assert_release_root('after %s' % cell.name)
                self.assert_inputs()
                if self.os.isfile(cell.alive):
                    self.log('ALIVE %s %s' % (cell.name, self.os.read_bytes(cell.alive).decode('ascii', 'replace').strip()))
                else:
                    self.log('ALIVE-MISSING %s' % cell.name)
                if timed_out:
                    self.os.move_new(cell.inflight, '%s.TIMEOUT.%s' % (cell.inflight, self.attempt_id))
                    self.record_outcome(cell, 'TIMEOUT', 'rc=%d; wall_ms=%d; owned group empty; stage retained=%s' % (rc, wall, cell.stage))
                    continue
                failure, validated = '', ''
                if rc != 0:
                    failure = 'evaluator rc=%d' % rc
                if not failure:
                    try:
                        validated = self.report(cell.stage, cell)
                    except Exception as exc:  # noqa: BLE001 - any gate failure is the cell's ERROR
                        failure = str(exc)
                if not failure:
                    try:
                        self.os.move_new(cell.stage, cell.output)
                    except Exception as exc:  # noqa: BLE001
                        failure = 'Publish failed; destination preserved: %s' % exc
                if failure:
                    self.os.move_new(cell.inflight, '%s.ERROR.%s' % (cell.inflight, self.attempt_id))
                    self.record_outcome(cell, 'ERROR', '%s; wall_ms=%d; stage retained=%s' % (failure, wall, cell.stage))
                    continue
                self.os.move_new(cell.inflight, cell.validated)
                self.record_outcome(cell, 'SUCCESS', '%s wall_ms=%d' % (validated, wall))
            all_cells_ended = len(self.outcomes) == len(cells)
            if not all_cells_ended:
                raise Refusal('Missing terminal cell outcome')
            self.verify_release('end')
            bad = [o for o in self.outcomes if o[1] != 'SUCCESS']
            batch_exit = 0 if not bad else 1
            self.log('ALL-CELLS-ENDED batch=%s count=%d not_success=%d; run dose_judge.py' % (batch.id, len(self.outcomes), len(bad)))
        except (Exception, Interrupted) as exc:  # noqa: BLE001 - the Windows catch: every throw aborts the batch
            abort_reason = str(exc) or type(exc).__name__
            batch_exit = 2
            try:
                self.log('ABORT %s' % abort_reason)
            except Exception:  # noqa: BLE001
                self.console_line('dose progress write failed')
        finally:
            if terminal_owned:
                total = len(cells)
                if all_cells_ended and batch_exit < 2:
                    try:
                        lines = (['state=FINISHED', 'exit_code=%d' % batch_exit, 'batch=%s' % batch.id, 'attempt=%d' % batch.attempt,
                                  'slot=%d/%d' % (self.args.slot, self.args.slots), 'ended=%d/%d' % (len(self.outcomes), total),
                                  'FINISHED means all selected cells of this batch ended; read per-cell status and run dose_judge.py']
                                 + ['%s|%s|%s' % o for o in self.outcomes])
                        stage = '%s/dose_FINISHED.%s.pending.txt' % (self.out_root, uuid.uuid4().hex)
                        self.write_new_ascii(stage, lines)
                        self.os.move_new(stage, self.out_root + '/dose_FINISHED.txt')
                        finished_committed = True
                    except Exception as exc:  # noqa: BLE001
                        abort_reason, batch_exit = str(exc), 2
                if not finished_committed:
                    try:
                        ended = set(o[0] for o in self.outcomes)
                        nxt = [c.order for c in cells if c.name not in ended]
                        stage = '%s/dose_ABORTED.%s.pending.txt' % (self.out_root, uuid.uuid4().hex)
                        self.write_new_ascii(stage, ['state=ABORTED', 'batch=%s' % batch.id, 'attempt=%d' % batch.attempt,
                                                     'slot=%d/%d' % (self.args.slot, self.args.slots), 'ended=%d/%d' % (len(self.outcomes), total),
                                                     'resume_from_order=%s' % (nxt[0] if nxt else 'none'), 'reason=%s' % one_line(abort_reason)])
                        self.os.move_new(stage, self.out_root + '/dose_ABORTED.txt')
                    except Exception as exc:  # noqa: BLE001
                        self.console_line('dose ABORTED marker unavailable: %s' % exc)
                try:
                    self.log('DOSE %s batch=%s' % ('FINISHED' if finished_committed else 'ABORTED', batch.id))
                except Exception:  # noqa: BLE001
                    self.console_line('dose terminal diagnostic failed')
            if self.progress is not None:
                try:
                    self.progress.close()
                except Exception:  # noqa: BLE001
                    self.console_line('dose progress cleanup diagnostic')
            self.progress = None
            self.progress_owned = False
        return batch_exit

    # ---------------------------------------------------------------- main
    def header_predictions(self):
        with open(self.header_path, 'rb') as handle:
            lines = split_lines(handle.read().decode('ascii'))
        predictions = [line[2:] for line in lines if re.match(r'# S[1-8] \(', line)]
        references = [line[2:] for line in lines if re.match(r'# REFERENCE G2 P[45] \(', line)]
        return predictions, references

    def run(self):
        a = self.args
        if a.slot >= a.slots:
            raise Refusal('Slot %d must be below Slots %d' % (a.slot, a.slots))
        if 0 < a.gpu_memory_wait_ms < 60000:
            raise Refusal('GpuMemoryWaitMs %d is below 60000 ms' % a.gpu_memory_wait_ms)
        if a.gpu_memory_wait_ms > 0:
            self.gpumemwaitms, self.gpumemwait_source = a.gpu_memory_wait_ms, 'parameter'
        elif a.slots >= 3:
            self.gpumemwaitms, self.gpumemwait_source = GPUMEMWAITMS_DEFAULT_K3, 'GPUMEMWAITMS_DEFAULT_K3'
        else:
            self.gpumemwaitms, self.gpumemwait_source = GPUMEMWAITMS_DEFAULT_K2, 'GPUMEMWAITMS_DEFAULT_K2'
        predictions, references = self.header_predictions()
        if len(predictions) != 7 or predictions_sha(predictions) != PREDICTIONS_SHA256:
            raise Refusal('PREDICTIONS-ALTERED: %d lines, sha256 %s (pinned %s)' % (len(predictions), predictions_sha(predictions), PREDICTIONS_SHA256))
        if len(references) != 2:
            raise Refusal('Header must hold the two REFERENCE G2 lines')
        if not HOST_KEY_RE.match(a.host_key):
            raise Refusal('Bad host key %r (dispatch.py host names: [A-Za-z0-9_-]{1,32})' % a.host_key)
        self.data_root = a.data_root.rstrip('/') if a.data_root != '/' else a.data_root
        if not SAFE_ROOT_RE.match(self.data_root):
            raise Refusal('Data root %r must be an absolute path of [A-Za-z0-9_./:+-] (no space or shell character)' % a.data_root)
        self.root = self.data_root + '/usvb'
        self.gpu_lock_path = self.root + '/state/gpu.lock'
        self.rec = self.read_record()
        self.resolve_paths()
        self.batches = self.read_plan_list()
        self.computer_name = self.os.hostname()
        if a.describe_only:
            self.identity = 'not-checked(describe-only)'
            for batch in self.batches:
                self.out_root = batch.out_root
                for line in self.build_manifest(batch, self.batch_cells(batch, '{ATTEMPT}')):
                    self.console_line(line)
            return 0
        list_exit = 2
        self.nvidia_smi = ''
        try:
            self.assert_host()
            want_tag = 'ds-%s-k%ds%d' % (self.rec['release_id'], a.slots, a.slot)
            if self.os.executor_tag() != want_tag:
                raise Refusal('EXECUTOR-TAG-MISSING: start the executor with -X usvbexec=%s (found %r); untagged it would '
                              'look like a foreign python to the other slots' % (want_tag, self.os.executor_tag()))
            for batch in self.batches:
                if self.os.exists(batch.out_root):
                    raise Refusal('Prior queue directory exists; preserve and review before any new invocation: %s' % batch.out_root)
            self.os.makedirs(self.root + '/locks')
            self.os.makedirs(self.root + '/state')
            for path in self.slot_lock_paths():
                token = self.os.lock_try(path)
                if token is None:
                    raise Refusal('GPU slot lock busy: %s (slot %d of %d); refusing overlap' % (path, a.slot, a.slots))
                self.locks.append(token)
            if a.slots > 1:
                self.probe_gpu_lock('queue start')
            self.python_real = self.os.realpath(self.python_path)
            self.wait_own_python('queue start')
            for path in (self.python_path, self.kit_path, self.descriptor_path, self.manifest_file_path, self.assets_manifest_path, self.env_file):
                if not self.os.isfile(path):
                    raise Refusal('Required file missing: %s' % path)
            first = split_lines(self.os.read_bytes(self.env_file).decode('ascii', 'replace'))[0]
            if first != ENV_FILE_MARKER:
                raise Refusal('%s was not written by the R2 install.sh (first line %r)' % (self.env_file, first))
            for path in (self.release_path, self.bundle_path):
                if not self.os.isdir(path):
                    raise Refusal('Required folder missing: %s' % path)
            self.nvidia_smi = self.nvidia_smi_path()
            self.start_memory = self.gpu_memory()
            self.environment_line = self.assert_environment()
            self.assert_inputs()
            self.assert_bundle()
            self.assert_release_root('queue start')
            for batch in self.batches:
                if self.worst_case_ms(batch.cells) > batch.ceiling:
                    raise Refusal('Batch %s: ceiling %d ms is below its worst case %d ms' % (batch.id, batch.ceiling, self.worst_case_ms(batch.cells)))
            self.console_line('PLAN-LIST %s batches=%d slot=%d/%d (RELEASE-ROOT-CLEAN queue start, checked before any queue directory was created)'
                              % (a.plan_list, len(self.batches), a.slot, a.slots))
            list_exit = 0
            for batch in self.batches:
                result = self.run_batch(batch)
                self.console_line('BATCH-END %s exit=%d' % (batch.id, result))
                if result == 2:
                    list_exit = 2
                    break
                if result == 1:
                    list_exit = 1
                if self.interrupted:
                    list_exit = 2
                    break
        except (Exception, Interrupted) as exc:  # noqa: BLE001
            list_exit = 2
            self.console_line('ABORT (before any batch started or between batches): %s' % (str(exc) or type(exc).__name__))
        finally:
            for token in self.locks:
                try:
                    self.os.lock_release(token)
                except Exception:  # noqa: BLE001
                    list_exit = 2
                    self.console_line('slot lock cleanup diagnostic')
            self.locks = []
        return list_exit


def build_parser():
    parser = argparse.ArgumentParser(description='R3 Linux executor of the G12 dose smoke (host class RENTAL)')
    parser.add_argument('--host-class', required=True, choices=[HOST_CLASS])
    parser.add_argument('--host-key', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--record-sha256', required=True)
    parser.add_argument('--slots', required=True, type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--slot', required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--plan-list', required=True)
    parser.add_argument('--plan-list-sha256', required=True)
    parser.add_argument('--gpu-memory-wait-ms', type=int, default=0)
    parser.add_argument('--describe-only', action='store_true')
    return parser


def main(argv=None, layer=None, console=None):
    args = build_parser().parse_args(argv)
    if not 0 <= args.gpu_memory_wait_ms <= 14400000:
        sys.stderr.write('--gpu-memory-wait-ms must be 0..14400000\n')
        return 2
    layer = layer or LinuxOs()
    executor = Executor(args, layer, console)
    installed = []
    if getattr(layer, 'is_real', False) and not args.describe_only:
        def on_signal(signum, _frame):
            if executor.interrupted:
                return
            executor.interrupted = True
            raise Interrupted('INTERRUPTED by signal %d (abort command): the current cell group was stopped' % signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            installed.append((sig, signal.signal(sig, on_signal)))
    try:
        return executor.run()
    except Refusal as exc:
        executor.console_line('REFUSED %s' % exc)
        return 1
    finally:
        for sig, old in installed:
            signal.signal(sig, old)


if __name__ == '__main__':
    sys.exit(main())
