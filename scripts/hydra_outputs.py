"""Keep Hydra's run directory and log files out of the working directory (G6).

Isaac Lab 2.3.2's hydra_task_config (isaaclab_tasks/utils/hydra.py:83) wraps the task
function in ``hydra.main(config_path=None, config_name=<task>, version_base="1.3")`` and sets
nothing about Hydra's own outputs; hydra.main then parses ``sys.argv[1:]`` as overrides and
uses Hydra's defaults: run directory ``outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}`` RELATIVE TO
THE WORKING DIRECTORY, a ``.hydra/`` config snapshot inside it and a ``<job>.log`` file from
the default job logging. In the P5 run 1 the working directory was the release root, so
classical_baseline.py wrote 24 files into the deployed tree (kit verify-tree: EXTRA-DIR).

Every Hydra-decorated script appends ``run_dir_overrides(<its output path>)`` to the
``sys.argv`` it already rebuilds for hydra.main (``[sys.argv[0]] + hydra_args``), so no CLI
changes:
    hydra.run.dir=<folder of the output path>   (an existing folder: Hydra's mkdir is a no-op;
                                                 the system temporary folder when there is no
                                                 output path)
    hydra.output_subdir=null                    (no .hydra/ snapshot)
    hydra/job_logging=disabled                  (no <job>.log file)
    hydra/hydra_logging=disabled
Keys the caller already passed as Hydra overrides are left to the caller (Hydra refuses two
values for one key).
"""
from __future__ import annotations

import os
import tempfile

FIXED = ("hydra.output_subdir=null", "hydra/job_logging=disabled", "hydra/hydra_logging=disabled")


def _quoted(path):
    """A Hydra override value for a filesystem path: forward slashes, single-quoted."""
    text = os.path.abspath(path).replace("\\", "/")
    if "'" in text:
        raise ValueError("output folders containing a single quote are not supported by the Hydra override: "
                         + text)
    return "'" + text + "'"


def run_dir_overrides(output_path=None, hydra_args=()):
    """Hydra overrides that keep the run directory and logs out of the working directory."""
    if output_path:
        run_dir = os.path.dirname(os.path.abspath(output_path))
    else:
        run_dir = tempfile.gettempdir()
    given = {str(arg).lstrip("+~").split("=", 1)[0] for arg in hydra_args}
    overrides = ["hydra.run.dir=" + _quoted(run_dir)] + list(FIXED)
    return [item for item in overrides if item.split("=", 1)[0] not in given]
