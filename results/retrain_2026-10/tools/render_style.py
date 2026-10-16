"""Render-only README style, matching NavRL2026/noncausal_usv/stage2_rl/record_compare.py:
uniform blue water (the wave tasks keep their own wave-driven deforming surface, recoloured),
the official colour BlueBoat model over the gray physics mesh, amber obstacles, a white wake trail
and a follow camera. The monohull and catamaran meshes have no colour source and are left unchanged.

Nothing here is read by physics, observations, reward, termination or success.

    import render_style
    render_style.configure(task_id, env_cfg)      # before gym.make
    env = gym.make(...); env = render_style.TrailWrapper(env)   # before RecordVideo
"""
import os

import gymnasium as gym

WATER_RGB = (0.04, 0.20, 0.42)                    # stage2 record_compare.py
OBSTACLE_RGB = (0.85, 0.55, 0.15)                 # stage2 gap walls
TRAIL_RGB = (0.95, 0.97, 1.0)
_REPO_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets")
OFFICIAL_USD = os.path.join(os.environ.get("USVBENCH_ASSETS_MAIN", _REPO_ASSETS),
                            "BB120_official_visual_only.usd")
# body-frame placement of the official model, from hazard_nav_env_cfg.py (official_blueboat_visual_*)
OFFICIAL_T, OFFICIAL_Q = (-0.60396, -0.59322, -0.22357), (0.5, 0.5, 0.5, 0.5)
ROBOT = "/World/envs/env_0/Robot"


def _register_sea_colormap():
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if "usvb_sea" not in matplotlib.colormaps:
        # same blue family as WATER_RGB; trough darker, crest lighter, so a 0.1 m sea is still readable
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list(
            "usvb_sea", [(0.02, 0.13, 0.30), WATER_RGB, (0.22, 0.48, 0.72)], N=256))  # continuous, not 3 bands


def configure(task_id, env_cfg, eye=None):
    """Render-only config edits, then patch the env class so the scene gets the style after setup."""
    _register_sea_colormap()
    vis = getattr(env_cfg, "visual", None)
    if vis is not None:
        for name, value in (("water_color", WATER_RGB), ("wave_colormap", "usvb_sea"), ("wave_mesh_res", 241), ("wave_mesh_size_m", 60.0),
                            ("obstacle_color", OBSTACLE_RGB)):
            if hasattr(vis, name):
                setattr(vis, name, value)
    if eye is not None:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = tuple(eye)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    import importlib
    mod, cls_name = gym.spec(task_id).entry_point.split(":")
    cls = getattr(importlib.import_module(mod), cls_name)
    if getattr(cls, "_usvb_styled", False):
        return
    orig = cls._setup_scene

    def _setup_scene(self):
        orig(self)
        try:
            style_scene(self)
        except Exception as exc:  # render-only: never break the task
            print(f"[STYLE][WARN] {exc!r}")

    cls._setup_scene = _setup_scene
    cls._usvb_styled = True


def style_scene(env=None):
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    from pxr import Usd
    # BlueBoat body = a prim named BlueBoat that owns the gray "Visuals" scope (Robot/BlueBoat in most
    # tasks, env_0/BlueBoat/... in blueboat_calm_nav)
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    bodies = [p for p in Usd.PrimRange(env0) if p.GetName() == "BlueBoat" and p.GetChild("Visuals").IsValid()] if env0.IsValid() else []
    for bb in bodies[:1]:
        path = str(bb.GetPath())
        if not stage.GetPrimAtPath(f"{path}/OfficialVisual").IsValid():
            cfg = sim_utils.UsdFileCfg(usd_path=OFFICIAL_USD)
            cfg.func(f"{path}/OfficialVisual", cfg, translation=OFFICIAL_T, orientation=OFFICIAL_Q)
            UsdGeom.Imageable(bb.GetChild("Visuals")).MakeInvisible()
            print(f"[STYLE] official BlueBoat visual attached at {path}")
    # calm-water tasks without a configurable water layer: hide theirs, add the stage2 plane
    wave_animated = env is not None and getattr(env, "_wave_mesh", None) is not None   # hazard_nav wave surface
    for path in ("/World/Water", "/World/WaterSurface") + (() if wave_animated else ("/World/DynamicWater",)):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            UsdGeom.Imageable(p).MakeInvisible()
    if wave_animated or not stage.GetPrimAtPath("/World/StaticWater").IsValid():
        mesh = UsdGeom.Mesh.Define(stage, "/World/DisplayWater")
        s = 400.0
        z0 = -0.35 if wave_animated else 0.0   # under the wave surface, so only its far field shows
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(-s, -s, z0), Gf.Vec3f(s, -s, z0), Gf.Vec3f(s, s, z0), Gf.Vec3f(-s, s, z0)]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*WATER_RGB)]))
        mesh.GetDisplayColorPrimvar().SetInterpolation("constant")
        mesh.GetDoubleSidedAttr().Set(True)
        print("[STYLE] display water plane added")
    trail = UsdGeom.BasisCurves.Define(stage, "/World/Trail")
    trail.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    trail.GetWidthsAttr().Set(Vt.FloatArray([0.07]))
    trail.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    trail.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*TRAIL_RGB)]))


class TrailWrapper(gym.Wrapper):
    """Extends the white wake trail after every step (env 0); sits under RecordVideo."""

    def __init__(self, env, width=0.07):
        super().__init__(env)
        self._xy = []
        self._width = width

    def step(self, action):
        out = self.env.step(action)
        try:
            self._draw()
        except Exception as exc:
            print(f"[STYLE][WARN] trail {exc!r}")
        return out

    def _draw(self):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
        base = self.env.unwrapped
        p = base.robot.data.root_pos_w[0].detach().cpu().tolist()
        if self._xy and (abs(p[0] - self._xy[-1][0]) > 5 or abs(p[1] - self._xy[-1][1]) > 5):
            self._xy = []  # episode reset teleport: start a new trail
        self._xy = (self._xy + [(p[0], p[1])])[-1800:]
        if len(self._xy) < 2:
            return
        trail = UsdGeom.BasisCurves(omni.usd.get_context().get_stage().GetPrimAtPath("/World/Trail"))
        trail.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(x, y, 0.03) for x, y in self._xy]))
        trail.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(self._xy)]))
        trail.GetWidthsAttr().Set(Vt.FloatArray([self._width]))
