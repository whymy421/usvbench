"""
Patches rescue_boat.usd with RigidBodyAPI, MassAPI, and PhysX params.

Workflow:
  1. Open Isaac Sim GUI
  2. File → Import → rescue_boat_visual.obj  (Isaac Sim auto-converts to USD)
  3. File → Save As → C:/Users/arifa/usvbench/assets/rescue_boat.usd
  4. Close Isaac Sim GUI
  5. Run this script:
       conda activate env_isaaclab
       python make_rescue_boat_usd.py --headless

After running, rescue_boat.usd will have full physics APIs applied.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema, Gf

USD_PATH = r"C:\Users\arifa\usvbench\assets\rescue_boat.usd"

# ── Physics parameters for ARENA RIB 8.50m ──────────────────────────────────
# Real-world: ~300 kg, twin outboards, ~23 knots top speed (~12 m/s)
MASS_KG             = 300.0
LINEAR_DAMPING      = 0.3    # RIBs glide — low drag
ANGULAR_DAMPING     = 2.0
MAX_LINEAR_VEL      = 12.0   # m/s (~23 knots)
MAX_ANGULAR_VEL     = 120.0  # deg/s — very agile vs catamaran (90 deg/s)
SLEEP_THRESHOLD     = 0.005
# ─────────────────────────────────────────────────────────────────────────────

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise FileNotFoundError(f"Could not open USD: {USD_PATH}\n"
                            "Did you save the imported OBJ as rescue_boat.usd first?")

# Find root prim
root = stage.GetDefaultPrim()
if root is None:
    for p in stage.Traverse():
        root = p
        break

print(f"Root prim : {root.GetPath()}  type: {root.GetTypeName()}")

# RigidBodyAPI
UsdPhysics.RigidBodyAPI.Apply(root)

# MassAPI
mass_api = UsdPhysics.MassAPI.Apply(root)
mass_api.CreateMassAttr().Set(MASS_KG)

# PhysX extras
physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
physx_rb.CreateLinearDampingAttr().Set(LINEAR_DAMPING)
physx_rb.CreateAngularDampingAttr().Set(ANGULAR_DAMPING)
physx_rb.CreateMaxLinearVelocityAttr().Set(MAX_LINEAR_VEL)
physx_rb.CreateMaxAngularVelocityAttr().Set(MAX_ANGULAR_VEL)
physx_rb.CreateSleepThresholdAttr().Set(SLEEP_THRESHOLD)
physx_rb.CreateDisableGravityAttr().Set(False)

stage.GetRootLayer().Save()
print(f"[OK] RigidBodyAPI patched → {USD_PATH}")
print(f"     mass={MASS_KG} kg  | linear_damp={LINEAR_DAMPING}  | angular_damp={ANGULAR_DAMPING}")
print(f"     max_linear_vel={MAX_LINEAR_VEL} m/s | max_angular_vel={MAX_ANGULAR_VEL} deg/s")

simulation_app.close()
