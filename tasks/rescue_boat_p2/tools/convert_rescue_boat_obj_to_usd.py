"""
Converts rescue_boat_visual.obj → rescue_boat.usd using Isaac Sim's asset converter.
Run headlessly — no GUI needed.

Usage:
    conda activate env_isaaclab
    isaaclab.bat -p convert_rescue_boat_obj_to_usd.py
"""
import asyncio
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.asset_converter as asset_converter

INPUT_OBJ  = r"C:\Users\arifa\Desktop\UCL\Individual Project\Rescue Boat\rescue_boat_visual.obj"
OUTPUT_USD = r"C:\Users\arifa\usvbench\assets\rescue_boat.usd"


async def convert():
    ctx = asset_converter.AssetConverterContext()
    ctx.ignore_materials        = False
    ctx.ignore_camera           = True
    ctx.ignore_light            = True
    ctx.single_mesh             = True   # merge into one mesh prim
    ctx.smooth_normals          = True
    ctx.export_preview_surface  = False
    ctx.use_meter_as_world_unit = True   # OBJ is already in metres

    mgr = asset_converter.get_instance()
    task = mgr.create_converter_task(INPUT_OBJ, OUTPUT_USD, None, ctx)
    success = await task.wait_until_finished()

    if success:
        print(f"[OK] Converted → {OUTPUT_USD}")
    else:
        print(f"[FAIL] {task.get_error_message()}")

asyncio.get_event_loop().run_until_complete(convert())
simulation_app.close()
