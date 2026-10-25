#!/usr/bin/env python3
"""G51: assemble work/r3copy_g51 = the G43 R3 launcher (launch / status / pull / abort + lib) and the G51 executor.

Only change to the launcher: lib/r3_local_common.ps1 Get-R3ScpOptions appends '-l <kbit/s>' when the environment variable
G51_SCP_LIMIT_KBIT is a positive integer (the owner's upload rule: this PC's uploads <= 500 KB/s from 09:00 to 01:00; the
orchestrator sets 400 kbit/s = 50 KB/s in that window). Without the variable the scp options are byte-for-byte the G43
ones. The executor folder holds executor_g51/usvb_dose_executor.py and r3_remote.py. Never overwrites."""
import hashlib
from pathlib import Path
import sys

TASKS = Path(r"D:\claude code\codex_tasks")
SRC = TASKS / "20261013_g43_c3_sweep" / "work" / "r3copy_g43" / "launcher"
WORK = Path(__file__).resolve().parent.parent
DST = WORK / "r3copy_g51"
OLD = """function Get-R3ScpOptions($endpoint) {
    return @('-i', $endpoint.Key, '-P', $endpoint.Port, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', "UserKnownHostsFile=$G40_KNOWN_HOSTS",
        '-o', "ServerAliveInterval=$SSH_ALIVE_INTERVAL_S", '-o', "ServerAliveCountMax=$SSH_ALIVE_COUNT_MAX",
        '-o', "ConnectTimeout=$SSH_CONNECT_TIMEOUT_S")
}"""
NEW = """# G51 copy: scp gets '-l <kbit/s>' when G51_SCP_LIMIT_KBIT is a positive integer (owner upload rule, daytime cap).
function Get-R3ScpOptions($endpoint) {
    $limit = @()
    if ($env:G51_SCP_LIMIT_KBIT -cmatch '^[1-9][0-9]{0,6}$') { $limit = @('-l', $env:G51_SCP_LIMIT_KBIT) }
    return @($limit) + @('-i', $endpoint.Key, '-P', $endpoint.Port, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', "UserKnownHostsFile=$G40_KNOWN_HOSTS",
        '-o', "ServerAliveInterval=$SSH_ALIVE_INTERVAL_S", '-o', "ServerAliveCountMax=$SSH_ALIVE_COUNT_MAX",
        '-o', "ConnectTimeout=$SSH_CONNECT_TIMEOUT_S")
}"""


def write_new(path, data):
    if path.exists():
        if path.read_bytes() == data:
            return "SAME"
        raise SystemExit("refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)
    return "WROTE"


def main():
    for name in ("launch_rental.ps1", "status_rental.ps1", "pull_rental.ps1", "abort_rental.ps1", "lib/r3_local_common.ps1"):
        data = (SRC / name).read_bytes()
        if name.endswith("r3_local_common.ps1"):
            text = data.decode("ascii")
            text = text.replace("\r\n", "\n")
            if text.count(OLD) != 1:
                raise SystemExit("anchor not found once in %s" % name)
            text = text.replace(OLD, NEW)
            crlf = b"\r\n" in data
            data = (text.replace("\n", "\r\n") if crlf else text).encode("ascii")
        state = write_new(DST / "launcher" / name, data)
        print("%s %s sha256=%s" % (state, DST / "launcher" / name, hashlib.sha256(data).hexdigest().upper()))
    for name in ("usvb_dose_executor.py", "r3_remote.py"):
        data = (WORK / "executor_g51" / name).read_bytes()
        state = write_new(DST / "executor" / name, data)
        print("%s %s sha256=%s" % (state, DST / "executor" / name, hashlib.sha256(data).hexdigest().upper()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
