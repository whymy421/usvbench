"""Minimal headless-capture probe: no usvbench code, no policy, no task.

Spawns a lit ground plane and a red cube, points the viewport camera at it,
and writes a PNG. If THIS is black, headless capture itself is broken on this
machine; if it is not, the black demos come from our scene or our recorder.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=r"C:\Users\BRADY\usvbench\demos\probe.png")
parser.add_argument("--warmup", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402

sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
    "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
)
cube = sim_utils.CuboidCfg(
    size=(2.0, 2.0, 2.0),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
)
cube.func("/World/cube", cube, translation=(0.0, 0.0, 1.0))

sim.set_camera_view(eye=(8.0, -8.0, 6.0), target=(0.0, 0.0, 1.0))
sim.reset()

# RTX needs frames to warm up; grab only after the warmup burn-in.
for _ in range(args_cli.warmup):
    sim.step(render=True)

import omni.replicator.core as rep  # noqa: E402

rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])
for _ in range(10):
    sim.step(render=True)
data = annot.get_data()
arr = np.asarray(data)
print("SHAPE", arr.shape, "DTYPE", arr.dtype)
if arr.size:
    print("MEAN", float(arr.mean()), "MAX", int(arr.max()), "MIN", int(arr.min()))
    try:
        from PIL import Image
        Image.fromarray(arr[..., :3].astype("uint8")).save(args_cli.out)
        print("SAVED", args_cli.out)
    except Exception as exc:  # noqa: BLE001
        print("PNG SAVE FAILED:", exc)
else:
    print("EMPTY ANNOTATOR DATA")

app.close()
