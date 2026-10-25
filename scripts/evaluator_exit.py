"""Fail-closed process exit and certificate publication for the Isaac evaluators (G6, G6b).

History.
  P5 (2026-10-01, Isaac Sim 5.1.0 / Isaac Lab 2.3.2): classical_planner_pid.py raised in
  parse_env_cfg, printed the traceback and exited 0 without a certificate (the app was not
  closed and the interpreter shutdown returned 0).
  P5 v3 (release v2, G6b): with the first guard, which closed the app BEFORE publishing,
  (1) every successful run wrote ``<out>.partial`` and still exited 0 without renaming it:
  Isaac Sim 5.1's ``SimulationApp.close()`` ends the process with code 0 on the success
  path, so nothing after close() runs; (2) a run that failed in cert_v3 (classical_baseline,
  after Hydra's "Error executing job") hung for 26 minutes inside close() until the queue's
  timeout killed it.

Rules (shared by classical_planner_pid, classical_baseline, classical_harbour, eval_mppi and
eval_v6_frozen; CLIs unchanged):

* SUCCESS: every certificate is written to ``<out>.partial`` (the same json.dump call, same
  bytes), flushed and fsynced, re-read and parsed, and renamed to ``<out>`` with os.replace;
  only THEN is the app closed. A named watchdog (CLOSE_WATCHDOG_S) is armed before close():
  if close() has not returned by then it ends the process with code 0, because the
  certificate is already published. A close() that itself ends the process with code 0 is
  fine for the same reason; a close() that raises is reported on stderr and the run still
  exits 0 (its certificate is complete and published).
* FAILURE (any stage: cfg, env build, episode loop, certificate build / write / verify /
  rename): the traceback is printed, every ``<out>.partial`` is deleted (never left behind),
  stdout/stderr are flushed and the process ends with ``os._exit(<nonzero>)`` WITHOUT calling
  close(). Deliberate: Kit's close() is what hung on the failure path, and process exit
  releases the GPU and every Kit resource anyway. Exit code: 1 for an exception, the Python
  exit code of a SystemExit (``raise SystemExit("message")`` prints it and exits 1).
* Nothing at interpreter shutdown can turn a failure into 0 (os._exit).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import traceback

PARTIAL_SUFFIX = ".partial"
# Seconds the success path waits for SimulationApp.close() before it ends the process with code 0
# (the certificate is already published at that point).
CLOSE_WATCHDOG_S = 120


class CertificateOutput:
    """One certificate file: written to ``<out>.partial``, published (renamed) only on success."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.partial = self.path + PARTIAL_SUFFIX
        self.written = False

    @contextlib.contextmanager
    def open(self):
        """``with output.open() as stream:`` -- the evaluators' former
        ``open(args_cli.out, "w", encoding="utf-8")``, on the partial file. The data is flushed
        and fsynced before the file is closed. An exception inside the block (e.g. cert.build
        refusing the run) deletes the partial file."""
        try:
            with open(self.partial, "w", encoding="utf-8") as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.discard()
            raise
        self.written = True

    def write_json(self, payload, **dump_kwargs):
        """json.dump(payload, <text stream>, **dump_kwargs) into the partial file."""
        with self.open() as stream:
            json.dump(payload, stream, **dump_kwargs)

    def verify(self):
        """Re-read the partial file and parse it as JSON (raises on a truncated or empty file)."""
        with open(self.partial, "r", encoding="utf-8") as stream:
            text = stream.read()
        if not text.strip():
            raise ValueError("certificate %s is empty" % self.partial)
        json.loads(text)

    def publish(self):
        """Verify, then rename ``<out>.partial`` to ``<out>`` (no-op when nothing was written)."""
        if self.written:
            self.verify()
            os.replace(self.partial, self.path)
            self.written = False

    def discard(self):
        self.written = False
        try:
            os.remove(self.partial)
        except FileNotFoundError:
            pass


def _flush():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass


def _hard_exit(code):
    _flush()
    code = int(code)
    os._exit(code if 0 <= code < 256 else 1)


class Guard:
    """Process-level exit guard of one evaluator run (see the module docstring)."""

    def __init__(self):
        self.app = None
        self.outputs = []
        self.done = False

    # ---- registration
    def attach_app(self, app):
        self.app = app
        return app

    def output(self, path):
        out = CertificateOutput(path)
        self.outputs.append(out)
        return out

    # ---- termination
    def _discard(self):
        for out in self.outputs:
            try:
                out.discard()
            except OSError as exc:
                print(f"[evaluator_exit] could not delete {out.partial}: {exc}", file=sys.stderr)

    def fail(self, code=1):
        """Failure path: delete every partial certificate and os._exit(code) WITHOUT close()."""
        self.done = True
        code = int(code) if code else 1
        self._discard()
        print(f"[evaluator_exit] run failed; exit code {code}; no certificate written; "
              "the app is NOT closed (process exit releases it)", file=sys.stderr)
        _hard_exit(code)

    def _close_with_watchdog(self):
        app, self.app = self.app, None
        if app is None:
            return
        closed = threading.Event()

        def watchdog():
            if not closed.wait(CLOSE_WATCHDOG_S):
                print(f"[evaluator_exit] app.close() did not return within CLOSE_WATCHDOG_S={CLOSE_WATCHDOG_S} s; "
                      "the certificate is published; exiting 0", file=sys.stderr)
                _hard_exit(0)

        threading.Thread(target=watchdog, name="evaluator_exit-close-watchdog", daemon=True).start()
        try:
            app.close()              # Isaac Sim 5.1 may end the process here with code 0: fine
        except BaseException:        # noqa: BLE001 - the certificate is already published
            print("[evaluator_exit] app.close() raised after the certificate was published:", file=sys.stderr)
            traceback.print_exc()
        finally:
            closed.set()

    def finish(self, code=0):
        """End of a run. code != 0: fail(code). code == 0: publish every certificate (verify +
        rename), then close the app under the watchdog, then return 0."""
        if self.done:
            return code
        if code != 0:
            self.fail(code)
        try:
            for out in self.outputs:
                out.publish()
        except BaseException:        # noqa: BLE001 - a certificate that cannot be published is a failure
            print("[evaluator_exit] certificate verification / publication failed:", file=sys.stderr)
            traceback.print_exc()
            self.fail(1)
        self.done = True
        _flush()
        self._close_with_watchdog()
        return 0

    # ---- module-level scripts (eval_v6_frozen): uncaught exception / abnormal shutdown
    def install_process_hooks(self):
        """For scripts whose body runs at module level: an uncaught exception after this call
        (sys.excepthook) and an interpreter shutdown that did not pass through finish() (atexit,
        registered after the app so it runs before Kit's handlers) both take the failure path."""
        import atexit

        previous = sys.excepthook

        def hook(kind, value, tb):
            previous(kind, value, tb)
            self.fail(1)

        def last_resort():
            if not self.done:
                print("[evaluator_exit] interpreter shutdown before the run finished", file=sys.stderr)
                self.fail(1)

        sys.excepthook = hook
        atexit.register(last_resort)


def exit_code(value):
    """The process exit code Python derives from SystemExit(value)."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    print(value, file=sys.stderr)
    return 1


def run(body, guard, app=None):
    """Run ``body()`` under ``guard``; a failure never returns (os._exit without close())."""
    if app is not None:
        guard.attach_app(app)
    try:
        body()
        code = 0
    except SystemExit as exc:
        code = exit_code(exc.code)
        if code:
            traceback.print_exc()
    except BaseException:  # noqa: BLE001 - every stage failure is a failed run
        traceback.print_exc()
        code = 1
    return guard.finish(code)
