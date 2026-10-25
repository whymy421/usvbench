#!/usr/bin/env python3
# R3 remote tool (runs ON the rental, Python 3.11 stdlib; imports usvb_dose_executor.py from its own folder for the
# process table, the python ownership rules and the record checks). It is always started as
#   <data root>/usvb/env/bin/python -X usvbexec=r3tool -B -I <folder>/r3_remote.py <command> ...
# so that it is an R3 tooling process that no slot counts as a python (header of usvb_dose_executor.py).
# Commands (the remote steps of launch_rental.ps1, status_rental.ps1, abort_rental.ps1 and pull_rental.ps1; adapted
# from G12 launcher/lib/remote/step_launch_dose.ps1 and step_listing.ps1):
#   verify-toolkit --dir D --manifest-sha256 H   every file of a freshly copied toolkit folder against TOOLKIT_MANIFEST.txt
#   launch ...                                    install the executor, the plan list and its batch files (hash-checked)
#                                                 into <usvb>/queues_v2/<task>*, refuse a foreign python persisting
#                                                 FOREIGN_WAIT_S, start the executor detached (setsid + nohup: own session,
#                                                 SIGHUP ignored, stdin/stdout/stderr on /dev/null, the console in
#                                                 <task>_console.log) and prove it alive (executor process + first progress
#                                                 file within ALIVE_WAIT_S); nothing is ever killed
#   status --data-root DR --task T               one R3-STATUS line for that slot's task
#   abort --data-root DR --task T                SIGTERM to the task's executor (it stops the current cell group and
#                                                 writes dose_ABORTED.txt); waits ABORT_WAIT_S; never SIGKILL
#   listing --data-root DR --name N              FILE <SHA-256> <size> <relative path> for every file of <usvb>/<N>
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import usvb_dose_executor as X  # noqa: E402

FOREIGN_WAIT_S = 120
ALIVE_WAIT_S = 240
ALIVE_POLL_S = 5
ABORT_WAIT_S = 180
TASK_RE = re.compile(r'^dose_([A-Za-z0-9_-]{1,32})_k([1-4])s([0-3])_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$')
LIST_LINE_RE = re.compile(r'^batch\|([A-Za-z0-9_.-]+\.plan)\|([0-9A-Fa-f]{64})\|([1-9])\|([1-9][0-9]{0,5})$')


class StepError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_file(path):
    with open(path, 'rb') as handle:
        return handle.read()


def copy_new(src, dst):
    """Copy into a file that must not exist (CreateNew), then verify the copy's SHA-256 against the source."""
    with open(src, 'rb') as handle:
        data = handle.read()
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
    if sha256_file(dst) != hashlib.sha256(data).hexdigest().upper():
        raise StepError('Installed file differs: %s' % dst)


def write_new_text(path, text, mode=0o644):
    if re.search(r'[^\x00-\x7F]', text):
        raise StepError('Non-ASCII text: %s' % path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(text.encode('ascii'))


def parse_list(path):
    lines = [line for line in X.split_lines(read_file(path).decode('ascii')) if line]
    if len(lines) < 2 or lines[0] != 'format=usvbench-g12-planlist-v1':
        raise StepError('Unexpected plan list format: %s' % path)
    out = []
    for line in lines[1:]:
        match = LIST_LINE_RE.match(line)
        if not match:
            raise StepError('Bad plan list line: %s' % line)
        out.append((match.group(1), match.group(2).upper(), int(match.group(3)), int(match.group(4))))
    return out


def batch_info(path, from_order):
    """(batch_id, selected cell count) of one batch file."""
    batch_id, selected = None, 0
    for line in X.split_lines(read_file(path).decode('ascii', 'replace')):
        if line.startswith('batch_id='):
            batch_id = line[len('batch_id='):]
        elif line.startswith('cell|'):
            try:
                if int(line.split('|')[1]) >= from_order:
                    selected += 1
            except ValueError:
                pass
    if not batch_id or not re.match(r'^[A-Za-z0-9_-]{1,60}$', batch_id):
        raise StepError('No valid batch_id in %s' % path)
    return batch_id, selected


def executor_processes(layer, task_py, exec_tag):
    out = []
    for proc in layer.processes():
        argv = proc.argv or []
        if task_py in argv and X.carries_tag(argv, 'usvbexec=' + exec_tag):
            out.append(proc)
    return out


# ------------------------------------------------------------------ verify-toolkit
def cmd_verify_toolkit(args):
    folder = args.dir
    manifest = os.path.join(folder, 'TOOLKIT_MANIFEST.txt')
    if sha256_file(manifest) != args.manifest_sha256.upper():
        raise StepError('TOOLKIT-MANIFEST-HASH-MISMATCH %s' % manifest)
    listed = {}
    for line in X.split_lines(read_file(manifest).decode('ascii')):
        if not line:
            continue
        match = re.match(r'^([A-Za-z0-9_.-]+)=([0-9A-F]{64})$', line)
        if not match:
            raise StepError('Bad toolkit manifest line: %s' % line)
        listed[match.group(1)] = match.group(2)
    present = sorted(os.listdir(folder))
    for name in present:
        if name != 'TOOLKIT_MANIFEST.txt' and name not in listed:
            raise StepError('Unlisted toolkit file: %s' % name)
    for name, want in sorted(listed.items()):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise StepError('Toolkit file missing: %s' % name)
        have = sha256_file(path)
        if have != want:
            raise StepError('TOOLKIT-HASH-MISMATCH %s expected=%s actual=%s' % (name, want, have))
        print('TOOLKIT-FILE-OK %s sha256=%s' % (name, have))
    print('TOOLKIT-OK dir=%s files=%d' % (folder, len(listed)))
    return 0


# ------------------------------------------------------------------ launch
def wrapper_text(root, data_root, task, exec_tag, host_key, record_final, record_sha, slots, slot, plan_list, list_sha, wait_ms):
    queues = root + '/queues_v2'
    return '\n'.join([
        '#!/bin/bash',
        '# R3 wrapper written by r3_remote.py launch (CreateNew); started detached with setsid + nohup; the executor',
        '# console (refusals included) goes to %s/%s_console.log' % (queues, task),
        '. "%s/usvb_env.sh" || exit 96' % root,
        ('exec "%s/env/bin/python" -X usvbexec=%s -B -I "%s/%s.py" --host-class RENTAL --host-key %s --data-root "%s" '
         '--record "%s" --record-sha256 %s --slots %d --slot %d --plan-list "%s" --plan-list-sha256 %s --gpu-memory-wait-ms %d '
         '> "%s/%s_console.log" 2>&1') % (root, exec_tag, queues, task, host_key, data_root, record_final, record_sha, slots, slot,
                                         plan_list, list_sha, wait_ms, queues, task),
        ''])


def foreign_pythons(layer, rid, slots, slot, python_real, root):
    allowed = ['usvbslot=ds-%s-k%ds%d' % (rid, slots, s) for s in range(slots)]
    own = allowed[slot]
    others = [tag for tag in allowed if tag != own]
    return [e for e in X.classify_pythons(layer.processes(), own, others, python_real, root + '/', layer.self_pid()) if e[1] == 'foreign']


def cmd_launch(args, layer=None):
    layer = layer or X.LinuxOs()
    data_root = args.data_root.rstrip('/')
    if not X.SAFE_ROOT_RE.match(data_root):
        raise StepError('Data root %r is not an absolute path of [A-Za-z0-9_./:+-]' % args.data_root)
    if not TASK_RE.match(args.task):
        raise StepError('Bad task name %s' % args.task)
    host, k_text, s_text = TASK_RE.match(args.task).groups()
    if host != args.host_key or int(k_text) != args.slots or int(s_text) != args.slot or args.slot >= args.slots:
        raise StepError('Task name %s does not match host key %s, K=%d, slot=%d' % (args.task, args.host_key, args.slots, args.slot))
    if 0 < args.gpu_memory_wait_ms < 60000 or not 0 <= args.gpu_memory_wait_ms <= 14400000:
        raise StepError('--gpu-memory-wait-ms must be 0 or 60000..14400000')
    root = data_root + '/usvb'
    kit = args.toolkit
    record_path = kit + '/release_record_v2.json'
    if sha256_file(record_path) != args.record_sha256.upper():
        raise StepError('Record hash differs: %s' % record_path)
    record = json.loads(read_file(record_path).decode('utf-8'))
    rid = record['release_id']
    bid = record['bundle']['bundle_id']
    record_final = '%s/records/release_record_v2_%s.json' % (root, rid)
    for need in (root + '/releases/' + rid, root + '/pools/' + bid, record_final, root + '/queues_v2', root + '/env/bin/python', root + '/usvb_env.sh'):
        if not os.path.exists(need):
            raise StepError('Deploy first (R2 kit); missing %s' % need)
    if sha256_file(record_final) != args.record_sha256.upper():
        raise StepError('Installed record differs: %s' % record_final)
    executor_src = kit + '/usvb_dose_executor.py'
    if sha256_file(executor_src) != args.executor_sha256.upper():
        raise StepError('Executor hash differs: %s' % executor_src)
    list_src = kit + '/' + args.plan_list_file
    if not re.match(r'^[A-Za-z0-9_.-]+\.list$', args.plan_list_file) or sha256_file(list_src) != args.plan_list_sha256.upper():
        raise StepError('Plan list hash differs: %s' % list_src)
    entries = parse_list(list_src)
    for name, sha, _attempt, _from in entries:
        if sha256_file(kit + '/' + name) != sha:
            raise StepError('Batch file hash differs: %s/%s' % (kit, name))
    first_id, _n = batch_info(kit + '/' + entries[0][0], entries[0][3])
    out_root = '%s/dose_%s_a%d' % (root, first_id, entries[0][2])
    queues = root + '/queues_v2'
    task_py, wrapper, console, plans = (queues + '/' + args.task + '.py', queues + '/' + args.task + '.sh',
                                        queues + '/' + args.task + '_console.log', queues + '/' + args.task + '_plans')
    for reserved in (task_py, wrapper, console, plans, out_root):
        if os.path.lexists(reserved):
            raise StepError('Exists; preserve it and stop: %s' % reserved)
    python_real = os.path.realpath(root + '/env/bin/python')
    deadline = time.monotonic() + FOREIGN_WAIT_S
    foreign = foreign_pythons(layer, rid, args.slots, args.slot, python_real, root)
    while foreign and time.monotonic() < deadline:
        time.sleep(2)
        foreign = foreign_pythons(layer, rid, args.slots, args.slot, python_real, root)
    if foreign:
        raise StepError('FOREIGN-PYTHON running (%s) for %d s; slot %d of %d allows only the tags of K=%d; do not launch'
                        % (X.describe_pythons(foreign), FOREIGN_WAIT_S, args.slot, args.slots, args.slots))
    print('PYTHON-CHECK slot=%d/%d foreign=0 own_tag=usvbslot=ds-%s-k%ds%d' % (args.slot, args.slots, rid, args.slots, args.slot))
    copy_new(executor_src, task_py)
    os.mkdir(plans)
    for name in [e[0] for e in entries] + [args.plan_list_file, 'r3_remote.py', 'usvb_dose_executor.py']:
        copy_new(kit + '/' + name, plans + '/' + name)
    print('PLANS-INSTALLED %s files=%d executor=%s sha256=%s' % (plans, len(entries) + 3, task_py, sha256_file(task_py)))
    exec_tag = 'ds-%s-k%ds%d' % (rid, args.slots, args.slot)
    text = wrapper_text(root, data_root, args.task, exec_tag, args.host_key, record_final, args.record_sha256.upper(), args.slots,
                        args.slot, plans + '/' + args.plan_list_file, args.plan_list_sha256.upper(), args.gpu_memory_wait_ms)
    write_new_text(wrapper, text, 0o755)
    print('WRAPPER %s sha256=%s' % (wrapper, sha256_file(wrapper)))
    print('WRAPPER-LINE %s' % text.splitlines()[-1])
    # Detached start: own session (setsid via start_new_session), SIGHUP ignored (nohup), no inherited terminal fds.
    subprocess.Popen(['nohup', '/bin/bash', wrapper], cwd=queues, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    progress = out_root + '/progress_dose.txt'
    deadline = time.monotonic() + ALIVE_WAIT_S
    procs, has_progress, polls = [], False, 0
    while time.monotonic() < deadline:
        time.sleep(ALIVE_POLL_S)
        polls += 1
        procs = executor_processes(layer, task_py, exec_tag)
        has_progress = os.path.isfile(progress)
        if procs and has_progress:
            break
        if not procs and polls >= 3:
            break  # the executor has ended (a refused start): stop waiting and print its console
    if not procs or not has_progress:
        print('----- console %s -----' % console)
        if os.path.isfile(console):
            print('\n'.join(read_file(console).decode('ascii', 'replace').splitlines()[-60:]))
        else:
            print('console log %s was not created' % console)
        raise StepError('Executor not proven alive within %d s (a refused start writes nothing into %s; the reason is in the console log)'
                        % (ALIVE_WAIT_S, out_root))
    print('ALIVE task=%s executor_pids=%s progress=%s console=%s detach=setsid+nohup'
          % (args.task, ','.join(str(p.pid) for p in procs), progress, console))
    print('\n'.join(read_file(progress).decode('ascii', 'replace').splitlines()[-10:]))
    return 0


# ------------------------------------------------------------------ status
def task_paths(root, task):
    queues = root + '/queues_v2'
    plans = queues + '/' + task + '_plans'
    lists = sorted(n for n in os.listdir(plans) if n.endswith('.list'))
    if len(lists) != 1:
        raise StepError('Task %s: expected one plan list in %s, found %d' % (task, plans, len(lists)))
    return queues, plans, plans + '/' + lists[0]


def gpu_line(layer):
    for candidate in X.NVIDIA_SMI_CANDIDATES:
        if layer.is_exec_file(candidate):
            code, text = layer.run_capture([candidate, '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits',
                                            '-i', X.GPU_INDEX], X.NVSMIMS)
            match = re.search(r'^\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*$', text or '', re.M)
            if code == 0 and match:
                return 'gpu_util_pct=%s gpu_mem_mib=%s/%s' % match.groups()
            return 'gpu_util_pct=? gpu_mem_mib=? (nvidia-smi exit %s)' % code
    return 'gpu_util_pct=? gpu_mem_mib=? (no nvidia-smi)'


def status_line(data_root, task, layer=None, now=None):
    layer = layer or X.LinuxOs()
    now = time.time() if now is None else now
    match = TASK_RE.match(task)
    if not match:
        raise StepError('Bad task name %s' % task)
    host, k_text, s_text = match.groups()
    root = data_root.rstrip('/') + '/usvb'
    queues, plans, plan_list = task_paths(root, task)
    entries = parse_list(plan_list)
    alive = []
    for proc in layer.processes():
        argv = proc.argv or []
        if (queues + '/' + task + '.py') in argv and X.is_tooling(argv):
            alive.append(proc.pid)
    counts = {'SUCCESS': 0, 'ERROR': 0, 'TIMEOUT': 0, 'NOT-STARTED-CEILING': 0}
    total, done, aborted, current, current_cell, cell_age, last_write = 0, 0, False, '-', '-', '-', None
    for name, _sha, attempt, from_order in entries:
        batch_id, selected = batch_info(plans + '/' + name, from_order)
        total += selected
        out_root = '%s/dose_%s_a%d' % (root, batch_id, attempt)
        if not os.path.isdir(out_root):
            continue
        cells = out_root + '/cells'
        for fname in os.listdir(cells) if os.path.isdir(cells) else []:
            if fname.endswith('.status.txt'):
                state = re.search(r'^state=(\S+)', read_file(cells + '/' + fname).decode('ascii', 'replace'), re.M)
                if state and state.group(1) in counts:
                    counts[state.group(1)] += 1
        for fname in ('progress_dose.txt',):
            if os.path.isfile(out_root + '/' + fname):
                mtime = os.path.getmtime(out_root + '/' + fname)
                last_write = mtime if last_write is None else max(last_write, mtime)
        if os.path.isfile(out_root + '/dose_FINISHED.txt'):
            done += 1
            continue
        if os.path.isfile(out_root + '/dose_ABORTED.txt'):
            aborted = True
            current = '%s_a%d' % (batch_id, attempt)
            continue
        current = '%s_a%d' % (batch_id, attempt)
        inflight = [f for f in os.listdir(cells) if f.endswith('.json.INFLIGHT')] if os.path.isdir(cells) else []
        if inflight:
            current_cell = inflight[0][:-len('.json.INFLIGHT')]
            cell_age = '%d' % (now - os.path.getmtime(cells + '/' + inflight[0]))
            for fname in os.listdir(cells):
                if fname.startswith(current_cell + '.') and (fname.endswith('.log') or fname.endswith('.heartbeat.txt')):
                    last_write = max(last_write or 0, os.path.getmtime(cells + '/' + fname))
    ended = sum(counts.values())
    if done == len(entries):
        state = 'FINISHED'
    elif aborted:
        state = 'ABORTED'
    elif alive:
        state = 'RUNNING' if current != '-' else 'STARTING'
    elif current != '-' or done:
        state = 'DEAD'
    else:
        state = 'NOT-STARTED'
    return ('R3-STATUS host=%s task=%s slot=%s/%s state=%s batch=%s batches_done=%d/%d cells_ended=%d/%d success=%d errors=%d '
            'timeouts=%d not_started_ceiling=%d current=%s cell_age_s=%s last_write_s=%s %s executor_pids=%s'
            % (host, task, s_text, k_text, state, current, done, len(entries), ended, total, counts['SUCCESS'], counts['ERROR'],
               counts['TIMEOUT'], counts['NOT-STARTED-CEILING'], current_cell, cell_age,
               '-' if last_write is None else '%d' % (now - last_write), gpu_line(layer), ','.join(str(p) for p in alive) or '-'))


def cmd_status(args):
    print(status_line(args.data_root, args.task))
    return 0


# ------------------------------------------------------------------ abort
def cmd_abort(args, layer=None):
    layer = layer or X.LinuxOs()
    root = args.data_root.rstrip('/') + '/usvb'
    queues = root + '/queues_v2'
    task_py = queues + '/' + args.task + '.py'
    procs = [p for p in layer.processes() if task_py in (p.argv or []) and X.is_tooling(p.argv)]
    if not procs:
        print('ABORT-NOTHING task=%s (no executor of this task is running)' % args.task)
        print(status_line(args.data_root, args.task, layer))
        return 3
    for proc in procs:
        os.kill(proc.pid, signal.SIGTERM)
        print('SIGTERM pid=%d task=%s' % (proc.pid, args.task))
    deadline = time.monotonic() + args.wait_s
    while time.monotonic() < deadline:
        time.sleep(2)
        if not [p for p in layer.processes() if p.pid in set(q.pid for q in procs)]:
            break
    left = [p.pid for p in layer.processes() if p.pid in set(q.pid for q in procs)]
    print(status_line(args.data_root, args.task, layer))
    if left:
        print('ABORT-PENDING pids=%s still running after %d s (the executor stops its cell group first; check again; never kill '
              'by hand before reading its progress file)' % (','.join(str(p) for p in left), args.wait_s))
        return 1
    print('R3-ABORTED task=%s (the current batch has dose_ABORTED.txt with resume_from_order; pull it before any re-dispatch)' % args.task)
    return 0


# ------------------------------------------------------------------ listing
def listing_lines(base):
    out = []
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(folder, name)
            if os.path.islink(path):
                raise StepError('Symbolic link in the listing root: %s' % path)
            rel = os.path.relpath(path, base).replace(os.sep, '/')
            out.append('FILE %s %d %s' % (sha256_file(path), os.path.getsize(path), rel))
    return sorted(out, key=lambda line: line.split(' ', 3)[3])


def cmd_listing(args):
    if not re.match(r'^[A-Za-z0-9_-]+$', args.name):
        raise StepError('Bad listing name %s' % args.name)
    base = args.data_root.rstrip('/') + '/usvb/' + args.name
    if not os.path.isdir(base):
        raise StepError('Missing %s' % base)
    lines = listing_lines(base)
    for line in lines:
        print(line)
    print('LISTING-OK root=%s files=%d' % (base, len(lines)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='R3 remote tool (rental side)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('verify-toolkit')
    p.add_argument('--dir', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p = sub.add_parser('launch')
    for flag in ('--data-root', '--toolkit', '--host-key', '--record-sha256', '--executor-sha256', '--task', '--plan-list-file', '--plan-list-sha256'):
        p.add_argument(flag, required=True)
    p.add_argument('--slots', type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument('--slot', type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument('--gpu-memory-wait-ms', type=int, default=0)
    for name in ('status', 'abort'):
        p = sub.add_parser(name)
        p.add_argument('--data-root', required=True)
        p.add_argument('--task', required=True)
        if name == 'abort':
            p.add_argument('--wait-s', type=int, default=ABORT_WAIT_S)
    p = sub.add_parser('listing')
    p.add_argument('--data-root', required=True)
    p.add_argument('--name', required=True)
    args = parser.parse_args(argv)
    if not getattr(sys, '_xoptions', {}).get('usvbexec'):
        print('REFUSED start r3_remote.py with -X usvbexec=r3tool (otherwise the running slots count it as a foreign python)')
        return 2
    try:
        return {'verify-toolkit': cmd_verify_toolkit, 'launch': cmd_launch, 'status': cmd_status, 'abort': cmd_abort,
                'listing': cmd_listing}[args.command](args)
    except (StepError, X.Refusal, OSError, ValueError) as exc:
        print('STEP-FAILED %s: %s' % (args.command, exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())
