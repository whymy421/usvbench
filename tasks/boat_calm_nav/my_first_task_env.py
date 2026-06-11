# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from .jonswap_wave import JONSWAPWaveField

from .my_first_task_env_cfg import MyFirstTaskEnvCfg


def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),  # 青色
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色
            ),
            "current": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # 绿色
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class MyFirstTaskEnv(DirectRLEnv):
    cfg: MyFirstTaskEnvCfg

    def __init__(self, cfg: MyFirstTaskEnvCfg, render_mode: str | None = None, **kwargs):
        self.physics_cfg = cfg.underwater_physics_cfg
        self.wave_cfg = cfg.wave_cfg
        self.currents = None
        super().__init__(cfg, render_mode, **kwargs)
        # 目标点
        self.target_pos = torch.zeros(self.num_envs, 2, device=self.device)
        # 🆕 goal_radius 可被 env var 覆盖(sweep 用)
        import os as _os
        self.goal_radius = float(_os.environ.get('GOAL_RADIUS_OVERRIDE', getattr(cfg, 'goal_radius', 2.0)))
        self.max_spawn_distance = getattr(cfg, 'max_spawn_distance', 30.0)
        self.min_spawn_distance = getattr(cfg, 'min_spawn_distance', 10.0)
        # 横浪程度（用于奖励计算）
        self.lateral_exposure = torch.zeros(self.num_envs, device=self.device)
        self.wave_drag = torch.zeros(self.num_envs, device=self.device)
        # 自学习reward（只在E4用）
        self.use_learned_reward = getattr(cfg, 'use_learned_reward', False)
        if self.use_learned_reward:
            from .learned_reward import LearnedReward
            self.learned_reward = LearnedReward(
                num_envs=self.num_envs,
                state_dim=cfg.observation_space,
                action_dim=cfg.action_space,
                device=self.device
            )
        else:
            self.learned_reward = None
        # 避浪评估统计
        self.total_lateral = 0.0
        self.total_steps = 0
        # 到达率统计
        self.episode_count = 0
        self.reached_count = 0
        self._trajectory_points = []
        # V41 航行质量 metric 累积器
        self._metric_path_length = 0.0      # 实际轨迹长度
        self._metric_straight_dist = 0.0    # 直线段累计(reach 时结算)
        self._metric_backward_steps = 0     # 倒车步数
        self._metric_total_steps = 0        # 总步数
        self._metric_heading_vel_cos = 0.0  # heading-velocity cos 累计
        self._metric_energy_total = 0.0     # 能耗累计
        self._prev_pos = None               # 上一步位置(算轨迹长度)

    def _load_water_from_usd(self):
        """从ROV_TEST.usd复制水面mesh及其动画"""
        import omni.usd, os
        from pxr import Usd, UsdGeom, Sdf

        print("\n🌊 Loading water surface from ROV_TEST.usd...")

        # Optional visual water surface (calm benchmark works without it — wrapped in try/except).
        _assets = os.environ.get("USVBENCH_ASSETS", os.path.join(os.path.expanduser("~"), "usvbench", "assets"))
        source_usd_path = os.environ.get("USVBENCH_WATER_USD", os.path.join(_assets, "ROV_TEST.usd"))
        source_water_path = "/World/Water"

        try:
            source_stage = Usd.Stage.Open(source_usd_path)
            source_water = source_stage.GetPrimAtPath(source_water_path)

            if not source_water or not source_water.IsValid():
                print(f"   ❌ Could not find {source_water_path}")
                return

            print(f"   ✓ Found source water: {source_water.GetTypeName()}")

            stage = omni.usd.get_context().get_stage()
            target_water_path = "/World/WaterSurface"

            if stage.GetPrimAtPath(target_water_path):
                print(f"   ℹ️ Water already exists at {target_water_path}")
                return

            source_layer = source_stage.GetRootLayer()
            target_layer = stage.GetRootLayer()

            success = Sdf.CopySpec(
                source_layer,
                Sdf.Path(source_water_path),
                target_layer,
                Sdf.Path(target_water_path)
            )

            if success:
                water_prim = stage.GetPrimAtPath(target_water_path)
                print(f"   ✓ Copied to {target_water_path} ({water_prim.GetTypeName()})")

                for child in water_prim.GetChildren():
                    print(f"      - {child.GetName()} ({child.GetTypeName()})")

                xform = UsdGeom.Xformable(water_prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set((0.0, 0.0, self.cfg.underwater_physics_cfg.water_surface_z))

                print(f"   ✓ Positioned at Z={self.cfg.underwater_physics_cfg.water_surface_z}m")

                try:
                    import omni.graph.core as og
                    graph_path = "/World/WaterSurface/animate_water"
                    graph = og.get_graph_by_path(graph_path)
                    if graph:
                        graph.set_disabled(False)
                        print(f"   ✓ Water animation activated")
                    else:
                        print(f"   ⚠️ OmniGraph found but not accessible (may need manual activation)")
                except Exception as e:
                    print(f"   ⚠️ Could not auto-activate animation: {e}")
                    print(f"   → Manually activate: Window → Visual Scripting → Action Graph")

                print(f"✅ Water surface loaded successfully!\n")
            else:
                print(f"   ❌ Failed to copy water prim")

        except Exception as e:
            print(f"   ❌ Error loading water: {e}")

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)

        # 🔬 P1 诊断:首次 _apply_action 时打印 boat 物理参数
        self._physics_diagnostic_printed = False

        self._load_water_from_usd()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.rigid_objects["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 🆕 每个 env 一个独特颜色的箭头(HSV 色轮),显示船头朝向
        import colorsys
        env_colors = [colorsys.hsv_to_rgb(i / self.num_envs, 0.9, 1.0) for i in range(self.num_envs)]
        env_arrow_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EnvArrows",
            markers={
                f"env_{i}": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.4, 0.4, 0.8),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=env_colors[i]),
                )
                for i in range(self.num_envs)
            },
        )
        self.visualization_markers = VisualizationMarkers(cfg=env_arrow_cfg)

        # 🆕 黑色箭头:指向目标方向(所有 env 同色,但每艘船单独算朝向)
        target_arrow_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetDirArrows",
            markers={
                "target_dir": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.4, 0.4, 0.8),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
                ),
            },
        )
        self.target_dir_markers = VisualizationMarkers(cfg=target_arrow_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.yaws = torch.zeros((self.num_envs, 1), device=self.device)
        self.commands = torch.randn((self.num_envs, 3), device=self.device)
        self.commands[:, -1] = 0.0
        self.commands = self.commands / torch.linalg.norm(self.commands, dim=1, keepdim=True)

        ratio = self.commands[:, 1] / (self.commands[:, 0] + 1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:, 0] * gzero[:, 1]
        minus = lzero[:, 0] * lzero[:, 1]
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)

        self.marker_locations = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.command_marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)

        # 初始化洋流场
        self._init_current_field()

        # 计算洋流方向（用于可视化）
        self.current_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        if self.physics_cfg.enable_current:
            current_yaws = torch.atan2(self.currents[:, 1], self.currents[:, 0])
            self.current_orientations = math_utils.quat_from_angle_axis(
                current_yaws.unsqueeze(1),
                self.up_dir
            ).reshape(self.num_envs, 4)
        # boat body-(-X) = bow,forward_vec = [-1, 0, 0](ROV 用 [0, 1, 0])
        # 🆕 AXIS_FLIP=1 → 翻号到 [+1,0,0] 测试朝向假设
        import os as _os_axis
        _fwd_x = +1.0 if int(_os_axis.environ.get('AXIS_FLIP', '0')) else -1.0
        self._fwd_x = _fwd_x  # 记下来供 thrust/forward_speed 共用
        self.forward_vec = torch.tensor([_fwd_x, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarkers",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=3.0,   # 🆕 匹配 goal_radius,视觉"碰球"= 真触发 reach
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.target_markers = VisualizationMarkers(target_marker_cfg)

        # 🆕 初始化波浪场
        self._init_wave_field()
        self._init_wave_field_jonswap()
        # 🆕 创建动态波浪水面
        self._create_wave_mesh()

        # 上传代码到 wandb
        try:
            import wandb
            if wandb.run is not None:
                import os
                code_dir = os.path.dirname(os.path.abspath(__file__))
                for f in os.listdir(code_dir):
                    if f.endswith((".py", ".yaml")):
                        wandb.save(os.path.join(code_dir, f), base_path=code_dir, policy="now")
                print(f"✅ Code uploaded to wandb: {wandb.run.name}")
        except Exception as e:
            print(f"⚠️ wandb code upload skipped: {e}")

        # 训练结束时自动关闭wandb
        import atexit
        try:
            import wandb
            atexit.register(wandb.finish)
        except:
            pass



    def _visualize_markers(self):
        # 🆕 单箭头/船,按 env 上色,朝向船头(boat body-(-X) = bow → rotate 180°)
        self.marker_locations = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w

        # arrow_x.usd 默认沿 +X;boat 船头在 body-(-X) → 绕 Z 转 180°
        # fwd_x = -1 时船头在 body-(-X) → arrow 转 180°;fwd_x = +1 时无需转
        rotation_angle = torch.pi if self._fwd_x < 0 else 0.0
        rotation_quat = math_utils.quat_from_angle_axis(
            torch.ones((self.num_envs, 1), device=self.device) * rotation_angle,
            self.up_dir
        ).reshape(self.num_envs, 4)
        arrow_orientations = math_utils.quat_mul(base_quat, rotation_quat)

        # 🆕 彩色 + 黑色箭头同高度,方向一致时会重合
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = 1.2
        arrow_loc = self.marker_locations + offset

        # marker_indices = env_id → 每艘船自己的颜色(船头朝向)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.visualization_markers.visualize(arrow_loc, arrow_orientations, marker_indices=env_ids)

        # 🆕 黑色箭头:指向目标(arrow_x.usd 默认沿 +X,直接用 atan2 给 yaw)
        rpos = self.target_pos - self.robot.data.root_pos_w[:, :2]
        target_yaws = torch.atan2(rpos[:, 1], rpos[:, 0]).unsqueeze(1)
        target_quat = math_utils.quat_from_angle_axis(
            target_yaws, self.up_dir
        ).reshape(self.num_envs, 4)
        # 黑色箭头同高度(方向一致时重合,heading 错时可见夹角)
        target_offset = torch.zeros((self.num_envs, 3), device=self.device)
        target_offset[:, 2] = 1.2
        target_loc = self.marker_locations + target_offset
        self.target_dir_markers.visualize(target_loc, target_quat)

        target_pos_3d = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_3d[:, :2] = self.target_pos
        target_pos_3d[:, 2] = self.robot.data.root_pos_w[:, 2]
        self.target_markers.visualize(target_pos_3d)

        # 🆕 摄像机跟随 env 0(用 Isaac Lab 正确 API,headless + 视频都生效)
        try:
            import numpy as np
            from isaaclab.sim import SimulationContext
            pos = self.robot.data.root_pos_w[0, :3].cpu().numpy()
            eye = pos + np.array([0.0, -30.0, 25.0])
            sim = SimulationContext.instance()
            if sim is not None:
                sim.set_camera_view(eye=eye.tolist(), target=pos.tolist())
        except Exception as e:
            if not hasattr(self, '_cam_err'):
                print(f"⚠️ Camera follow error: {e}")
                self._cam_err = True

    # ============================================
    # 水下物理计算函数
    # ============================================

    def _compute_depth_dependent_damping(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_positions = self.robot.data.root_pos_w[:, 2]
        water_surface = self.physics_cfg.water_surface_z
        relative_z = z_positions - water_surface

        displacement_percentage = torch.clamp(
            -relative_z / self.physics_cfg.rov_height + 0.5,
            min=0.0,
            max=1.0
        )

        linear_damping = (
                self.physics_cfg.air_linear_damping +
                (self.physics_cfg.max_linear_damping - self.physics_cfg.air_linear_damping) *
                displacement_percentage
        )

        angular_damping = (
                self.physics_cfg.air_angular_damping +
                (self.physics_cfg.max_angular_damping - self.physics_cfg.air_angular_damping) *
                displacement_percentage
        )

        return linear_damping, angular_damping

    def _compute_buoyancy_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.robot.data.root_pos_w
        orientations = self.robot.data.root_quat_w

        z_positions = positions[:, 2]
        center_of_h = self.physics_cfg.rov_height / 2
        # 动态水面高度（统一使用 wave_field，和视觉波浪完全同步）
        if self.wave_cfg.enable_wave and self.wave_field is not None:
            t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
            wave_elevation = self.wave_field.compute_elevation(t, positions[:, 0], positions[:, 1])
            water_surface = self.physics_cfg.water_surface_z + wave_elevation
        else:
            water_surface = self.physics_cfg.water_surface_z
        depth_below_surface = water_surface - z_positions

        submerged_ratio = torch.clamp(
            (depth_below_surface + center_of_h) / self.physics_cfg.rov_height,
            min=0.0,
            max=1.0
        )

        submerged_volume = self.physics_cfg.rov_volume * submerged_ratio
        buoyancy_magnitude = (
                self.physics_cfg.water_density *
                submerged_volume *
                self.physics_cfg.gravity
        )

        buoyancy_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_force_world[:, 2] = buoyancy_magnitude

        buoyancy_center_offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_center_offset_body[:, 2] = self.physics_cfg.buoyancy_center_offset

        buoyancy_center_offset_world = math_utils.quat_apply(
            orientations,
            buoyancy_center_offset_body
        )

        buoyancy_torque = torch.cross(
            buoyancy_center_offset_world,
            buoyancy_force_world,
            dim=-1
        )

        return buoyancy_force_world, buoyancy_torque

    # ============================================
    # 洋流场模块
    # ============================================

    def _init_current_field(self):
        if not self.physics_cfg.enable_current:
            self.currents = torch.zeros((self.num_envs, 2), device=self.device)
            print("\n🌊 Ocean currents: DISABLED\n")
            return

        speeds = torch.rand(self.num_envs, device=self.device) * \
                 (self.physics_cfg.current_speed_max - self.physics_cfg.current_speed_min) + \
                 self.physics_cfg.current_speed_min

        directions = torch.rand(self.num_envs, device=self.device) * 2 * 3.14159265

        self.currents = torch.stack([
            speeds * torch.cos(directions),
            speeds * torch.sin(directions)
        ], dim=1)

        print(f"\n🌊 Ocean Current Field Initialized:")
        print(f"   Speed range: {self.physics_cfg.current_speed_min}-{self.physics_cfg.current_speed_max} m/s")
        print(f"   Mean speed: {speeds.mean():.3f} m/s")
        print(f"   Drag coefficient: {self.physics_cfg.current_drag_coeff} N/(m/s)²")
        print(f"   Per-environment randomization: ENABLED\n")

    def _compute_current_forces(self) -> torch.Tensor:
        if not self.physics_cfg.enable_current:
            return torch.zeros((self.num_envs, 3), device=self.device)

        vel_w = self.robot.data.root_com_vel_w
        if vel_w.shape[-1] == 6:
            rov_vel_xy = vel_w[:, :2]
        else:
            rov_vel_xy = vel_w[:, :2]

        v_rel = rov_vel_xy - self.currents
        v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)

        drag_force_xy = -self.physics_cfg.current_drag_coeff * v_rel * v_rel_mag

        drag_force = torch.zeros((self.num_envs, 3), device=self.device)
        drag_force[:, :2] = drag_force_xy

        return drag_force

    # ============================================
    # 🆕 波浪场模块
    # ============================================

    def _init_wave_field(self):
        """初始化波浪参数（支持JONSWAP不规则波）"""
        import os
        self.use_jonswap = os.environ.get('WAVE_MODE', 'airy') == 'jonswap'

        if not self.wave_cfg.enable_wave:
            self.wave_height = torch.zeros(self.num_envs, device=self.device)
            self.wave_period = torch.ones(self.num_envs, device=self.device) * 5.0
            self.wave_dir = torch.zeros(self.num_envs, 2, device=self.device)
            print("\n🌊 Waves: DISABLED\n")
            return

        # 波高和波周期（所有模式共用）
        self.wave_height = torch.ones(self.num_envs, device=self.device) * self.wave_cfg.wave_height
        self.wave_period = torch.ones(self.num_envs, device=self.device) * self.wave_cfg.wave_period

        # 波浪方向
        wave_dir = torch.tensor([self.wave_cfg.wave_dir_x, self.wave_cfg.wave_dir_y], device=self.device)
        wave_dir = wave_dir / torch.norm(wave_dir).clamp(min=1e-6)
        self.wave_dir = wave_dir.unsqueeze(0).repeat(self.num_envs, 1)

        if self.use_jonswap:
            # JONSWAP频谱参数
            self.n_components = 20
            Hs = self.wave_cfg.wave_height * 2.0  # 有义波高
            Tp = self.wave_cfg.wave_period  # 峰值周期
            fp = 1.0 / Tp  # 峰值频率
            gamma = 3.3  # JONSWAP峰度因子
            g = 9.81

            # 频率范围
            f_min = fp * 0.5
            f_max = fp * 3.0
            freqs = torch.linspace(f_min, f_max, self.n_components, device=self.device)
            df = (f_max - f_min) / self.n_components

            # JONSWAP频谱计算
            alpha = 0.0081  # Phillips常数
            sigma = torch.where(freqs <= fp, torch.tensor(0.07, device=self.device),
                                torch.tensor(0.09, device=self.device))
            r = torch.exp(-0.5 * ((freqs - fp) / (sigma * fp)) ** 2)
            S = (alpha * g ** 2 / ((2 * 3.14159) ** 4 * freqs ** 5)) * \
                torch.exp(-1.25 * (fp / freqs) ** 4) * gamma ** r

            # 从频谱计算各分量幅值
            amplitudes = torch.sqrt(2 * S * df)
            # 归一化使总波高匹配设定的wave_height
            scale = self.wave_cfg.wave_height / (2 * torch.sqrt(torch.sum(amplitudes ** 2)).clamp(min=1e-6))
            amplitudes = amplitudes * scale

            # 随机相位（每个env不同）
            phases = torch.rand(self.num_envs, self.n_components, device=self.device) * 2 * 3.14159

            # 角频率和波数
            omegas = 2 * 3.14159 * freqs  # (n_components,)
            wave_numbers = omegas ** 2 / g  # 深水近似 k = omega^2/g

            # 保存为实例变量
            self.jonswap_amplitudes = amplitudes  # (n_components,)
            self.jonswap_phases = phases  # (num_envs, n_components)
            self.jonswap_omegas = omegas  # (n_components,)
            self.jonswap_wave_numbers = wave_numbers  # (n_components,)

            print(f"\n🌊 JONSWAP Wave Field Initialized:")
            print(f"   Hs={Hs:.1f}m, Tp={Tp:.1f}s, gamma={gamma}")
            print(f"   {self.n_components} frequency components, f=[{f_min:.3f}, {f_max:.3f}] Hz")
            print(f"   Amplitude range: [{amplitudes.min():.4f}, {amplitudes.max():.4f}] m")
        else:
            print(f"\n🌊 Airy Wave Field Initialized:")
            print(f"   Wave height: {self.wave_cfg.wave_height} m")
            print(f"   Wave period: {self.wave_cfg.wave_period} s")

        print(f"   Wave direction: ({self.wave_cfg.wave_dir_x}, {self.wave_cfg.wave_dir_y})")
        print(f"   Wave mode: {'JONSWAP' if self.use_jonswap else 'Airy'}\n")

    def _init_wave_field_jonswap(self):
        """使用JONSWAP谱初始化不规则波浪场"""
        if not self.wave_cfg.enable_wave:
            self.wave_height = torch.zeros(self.num_envs, device=self.device)
            self.wave_period = torch.ones(self.num_envs, device=self.device) * 5.0
            self.wave_dir = torch.zeros(self.num_envs, 2, device=self.device)
            self.wave_field = None
            return
        self.wave_field = JONSWAPWaveField(
            num_envs=self.num_envs,
            device=self.device,
            hs_range=(0.3, 1.0),
            tp_range=(4.0, 7.0),
            gamma_range=(1.0, 5.0),
            n_components=30,
        )
        self.wave_height = self.wave_field.hs
        self.wave_dir = self.wave_field.wave_dir
        self.wave_period = self.wave_field.tp
        print(f"\n🌊 JONSWAP Wave Field Initialized (Hs=0.3-1.0m, Tp=4-7s, N=30)")

    def _create_wave_mesh(self):
        """创建动态波浪水面mesh"""
        import omni.usd
        from pxr import UsdGeom, Gf, Vt, Sdf, UsdShade
        import numpy as np

        stage = omni.usd.get_context().get_stage()
        water_path = "/World/DynamicWater"

        if stage.GetPrimAtPath(water_path):
            print("⚠️ DynamicWater already exists, removing and recreating...")
            stage.RemovePrim(water_path)

        mesh = UsdGeom.Mesh.Define(stage, water_path)

        self._wave_mesh_size = 300.0
        self._wave_mesh_res = 120
        res = self._wave_mesh_res
        size = self._wave_mesh_size

        x = np.linspace(-size / 2, size / 2, res)
        y = np.linspace(-size / 2, size / 2, res)
        self._wave_xx, self._wave_yy = np.meshgrid(x, y)

        points = []
        for j in range(res):
            for i in range(res):
                points.append(Gf.Vec3f(float(self._wave_xx[j, i]), float(self._wave_yy[j, i]), 0.0))
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        face_counts = []
        face_indices = []
        for j in range(res - 1):
            for i in range(res - 1):
                v0 = j * res + i
                v1 = v0 + 1
                v2 = (j + 1) * res + i + 1
                v3 = (j + 1) * res + i
                face_counts.append(4)
                face_indices.extend([v0, v1, v2, v3])

        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))

        # 顶点颜色（GUI 和 headless 都用）
        from pxr import Gf, Vt
        colors = Vt.Vec3fArray([Gf.Vec3f(0.1, 0.3, 0.8)] * (res * res))
        mesh.GetDisplayColorAttr().Set(colors)
        mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
        mesh.GetDisplayOpacityAttr().Set([1.0] * (res * res))
        mesh.GetDoubleSidedAttr().Set(True)

        self._wave_mesh = mesh
        self._wave_k = 2 * 3.14159 / 20.0
        print(f"✅ Dynamic wave mesh created ({size}m x {size}m, {res}x{res}) with USD Material")

    def _update_wave_mesh(self):
        if not hasattr(self, '_wave_mesh') or not hasattr(self, 'wave_field') or self.wave_field is None:
            if not hasattr(self, '_wave_mesh_warn'):
                print(f"⚠️ Wave mesh update skipped: _wave_mesh={hasattr(self, '_wave_mesh')}, "
                      f"wave_field={hasattr(self, 'wave_field') and self.wave_field is not None}")
                self._wave_mesh_warn = True
            return
        if self.common_step_counter % 10 != 0:
            return

        from pxr import Gf, Vt
        import numpy as np

        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
        res = self._wave_mesh_res

        # 波浪mesh跟随env 0的船移动
        ship_pos = self.robot.data.root_pos_w[0].cpu().numpy()
        cx, cy = float(ship_pos[0]), float(ship_pos[1])

        px = self._wave_xx.flatten() + cx
        py = self._wave_yy.flatten() + cy

        amps = self.wave_field.amplitudes[0].cpu().numpy()
        ks = self.wave_field.wave_numbers.cpu().numpy()
        omegas = self.wave_field.omegas.cpu().numpy()
        phi = self.wave_field.phases[0].cpu().numpy()
        comp_dx = self.wave_field.comp_dir_x[0].cpu().numpy()
        comp_dy = self.wave_field.comp_dir_y[0].cpu().numpy()

        z = np.zeros_like(px)
        for c in range(self.wave_field.n_components):
            phase = ks[c] * (comp_dx[c] * px + comp_dy[c] * py) - omegas[c] * t + phi[c]
            z += amps[c] * np.cos(phase)

        new_points = [Gf.Vec3f(float(px[i]), float(py[i]), float(z[i])) for i in range(len(px))]
        self._wave_mesh.GetPointsAttr().Set(Vt.Vec3fArray(new_points))

        import matplotlib.cm as cm

        # 动态归一化（颜色均匀好看），加最小范围兜底防止变白
        z_min = z.min()
        z_max = z.max()
        z_range = max(z_max - z_min, 0.2)
        z_norm = (z - z_min) / z_range

        cmap = cm.get_cmap('turbo')
        colors = []
        for val in z_norm:
            c = cmap(float(val))
            colors.append(Gf.Vec3f(c[0], c[1], c[2]))
        self._wave_mesh.GetDisplayColorAttr().Set(colors)


    def _compute_wave_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算波浪对船的作用力（统一使用 wave_field）"""
        if not self.wave_cfg.enable_wave or self.wave_field is None:
            self.lateral_exposure = torch.zeros(self.num_envs, device=self.device)
            self.wave_drag = torch.zeros(self.num_envs, device=self.device)
            return (
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )

        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation

        # 船头方向 (2D)
        forwards_2d = self.forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(forwards_2d, dim=-1, keepdim=True).clamp(min=1e-6)

        # 统一调用 wave_field 计算
        result = self.wave_field.compute_forces(
            t,
            self.robot.data.root_pos_w[:, 0],
            self.robot.data.root_pos_w[:, 1],
            forwards_2d,
        )

        self.lateral_exposure = result["lateral_exposure"]
        self.wave_drag = result["wave_drag"]

        return result["heave_force"], result["roll_torque"]

    # ============================================
    # 施加动作
    # ============================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        import os  # 函数级 import 防 UnboundLocalError(line 732 的 conditional import 让 Python 把 os 当 local)
        if not hasattr(self, '_printed_body_info'):
            print(f"\n{'=' * 60}")
            print(f"🔍 ROV Configuration:")
            print(f"   Bodies: {self.robot.num_bodies}")
            print(f"   Body names: {self.robot.body_names}")
            print(f"\n🌊 Underwater Physics Enabled:")
            print(f"   Volume: {self.physics_cfg.rov_volume} m³")
            print(f"   Mass: {self.cfg.robot_cfg.spawn.mass_props.mass} kg")
            print(f"   Water density: {self.physics_cfg.water_density} kg/m³")
            print(f"   Expected buoyancy (fully submerged): "
                  f"{self.physics_cfg.water_density * self.physics_cfg.rov_volume * self.physics_cfg.gravity:.1f} N")
            print(f"   Weight: {self.cfg.robot_cfg.spawn.mass_props.mass * self.physics_cfg.gravity:.1f} N")
            print(f"   Max linear damping: {self.physics_cfg.max_linear_damping} N/(m/s)")
            print(f"   Max angular damping: {self.physics_cfg.max_angular_damping} Nm/(rad/s)")
            print(f"{'=' * 60}\n")
            self._printed_body_info = True

            try:
                import wandb, sys
                if wandb.run is not None:
                    wandb.config.update({"command": " ".join(sys.argv)})
            except:
                pass


            # 上传代码到 wandb
            try:
                import wandb
                if wandb.run is not None:
                    import os
                    code_dir = os.path.dirname(os.path.abspath(__file__))
                    for f in os.listdir(code_dir):
                        if f.endswith((".py", ".yaml")):
                            wandb.save(os.path.join(code_dir, f), base_path=code_dir, policy="now")
                    print(f"✅ Code uploaded to wandb: {wandb.run.name}")
            except Exception as e:
                print(f"⚠️ wandb code upload skipped: {e}")


        num_bodies = self.robot.num_bodies
        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        # ========================================
        # 1. 推进器控制（来自RL策略）— 加延迟
        # ========================================
        if not hasattr(self, '_action_buffer'):
            self._action_buffer = torch.zeros(self.num_envs, 10, 2, device=self.device)
            self._buf_idx = 0

        # 🆕 真 action delay(原代码 buffer 定义了但没用)
        delay_len = int(os.environ.get('ACTION_DELAY', '0'))  # 0=不延迟(原 behavior),10=硬规则
        self._action_buffer[:, self._buf_idx % 10] = self.actions
        if delay_len > 0:
            # 取 delay_len 步前的 action(若 buffer 还没填满,默认 0)
            delayed_idx = (self._buf_idx - delay_len + 1) % 10
            delayed_actions = self._action_buffer[:, delayed_idx]
        else:
            delayed_actions = self.actions
        self._buf_idx += 1

        # 🔬 P1 诊断:第一次 apply 时打印 boat 物理参数(mass, inertia, com)
        if not self._physics_diagnostic_printed:
            try:
                mass = self.robot.data.default_mass[0].cpu().numpy() if hasattr(self.robot.data, 'default_mass') else "N/A"
                inertia = self.robot.data.default_inertia[0].cpu().numpy() if hasattr(self.robot.data, 'default_inertia') else "N/A"
                com = self.robot.data.default_com[0].cpu().numpy() if hasattr(self.robot.data, 'default_com') else "N/A"
                print("=" * 70)
                print("🔬 BOAT PHYSICS DIAGNOSTIC (env 0)")
                print(f"   mass:    {mass}")
                print(f"   inertia: {inertia}     (Ixx, Iyy, Izz - Z is yaw inertia)")
                print(f"   com:     {com}")
                print(f"   USD path: boat_physics.usdc")
                print(f"   thrust scale: 100 N    (need to scale by Izz_boat / Izz_ROV)")
                print(f"   torque scale: 120 N·m  (need to scale by Izz_boat / Izz_ROV)")
                print("=" * 70)
            except Exception as e:
                print(f"⚠️ Physics diagnostic failed: {e}")
            self._physics_diagnostic_printed = True

        # 🆕 thrust/torque scale env var(sweep 用 scale physics)
        thrust_scale = float(os.environ.get('THRUST_SCALE', '100'))
        torque_scale = float(os.environ.get('TORQUE_SCALE', '120'))
        # forces 沿 boat 船头方向:fwd_x = -1 时 force[X] = -thrust(body-X 反方向),fwd_x=+1 时 +
        forces[:, 0, 0] = delayed_actions[:, 0] * self._fwd_x * thrust_scale
        forces[:, 0, 1] = 0
        torques[:, 0, 2] = delayed_actions[:, 1] * torque_scale

        # 🆕 V6 obs:存 prev_action 给下次 obs 用
        self._prev_action_obs = delayed_actions[:, :2].detach().clone()

        # ========================================
        # 2. 浮力和浮力力矩
        # ========================================
        buoyancy_force, buoyancy_torque = self._compute_buoyancy_forces()
        forces[:, 0, :] = forces[:, 0, :] + buoyancy_force
        torques[:, 0, :] = torques[:, 0, :] + buoyancy_torque

        # ========================================
        # 3. 深度相关的阻尼力
        # ========================================
        linear_damping, angular_damping = self._compute_depth_dependent_damping()

        vel_w = self.robot.data.root_com_vel_w

        if vel_w.shape[-1] == 6:
            linear_velocity = vel_w[:, :3]
            angular_velocity = vel_w[:, 3:]
        else:
            linear_velocity = vel_w
            angular_velocity = self.robot.data.root_ang_vel_w

        # 🆕 非对称阻尼:倒车(stern-first)阻力 BACKWARD_DRAG_FACTOR 倍(默认 3x)
        # 物理上 boat hull 形状决定倒车水阻大,RL 会感知到 → 自然减少倒车
        backward_drag = float(os.environ.get('BACKWARD_DRAG_FACTOR', '3.0'))
        # body-X velocity:fwd_x=-1 时,vel_b[X] > 0 = 倒车,< 0 = 前进
        quat_inv_d = math_utils.quat_conjugate(self.robot.data.root_quat_w)
        vel_b_3d = math_utils.quat_apply(quat_inv_d, linear_velocity)
        body_fwd_speed = vel_b_3d[:, 0] * self._fwd_x   # > 0 = bow-first
        # asymmetric multiplier(body-X only):前进=1,倒车=backward_drag
        asym_x = torch.where(body_fwd_speed >= 0,
                             torch.ones_like(body_fwd_speed),
                             torch.full_like(body_fwd_speed, backward_drag))
        # 把 asym 应用到 world-frame drag 的"沿船体 X 轴投影" 上(简化:整体 drag × asym 系数,主向运动是船体 X)
        drag_force = -linear_damping.reshape(-1, 1) * linear_velocity * asym_x.unsqueeze(-1)
        forces[:, 0, :] = forces[:, 0, :] + drag_force

        # 🆕 ANG_DAMP_SCALE:角阻尼 scale env var(默认 1.0,< 1 让 boat 转得快)
        ang_damp_scale = float(os.environ.get('ANG_DAMP_SCALE', '1.0'))
        angular_drag = -angular_damping.reshape(-1, 1) * angular_velocity * ang_damp_scale
        torques[:, 0, :] = torques[:, 0, :] + angular_drag

        # ========================================
        # 4. 姿态稳定
        # ========================================
        quat = self.robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        pitch_angle = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        roll_angle = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))

        torques[:, 0, 0] += -pitch_angle * 200.0
        torques[:, 0, 1] += -roll_angle * 200.0
        torques[:, 0, 1] += -angular_velocity[:, 1] * 25.0

        # ========================================
        # 5. 洋流力
        # ========================================
        current_force = self._compute_current_forces()
        forces[:, 0, :] = forces[:, 0, :] + current_force

        # ========================================
        # 🆕 6. 波浪力
        # ========================================
        heave_force, roll_torque = self._compute_wave_forces()
        forces[:, 0, 2] += heave_force
        torques[:, 0, 1] += roll_torque

        # 波浪阻力
        if self.wave_cfg.enable_wave:
            wave_drag_force = -self.wave_drag.unsqueeze(-1) * self.forwards[:, :2]
            forces[:, 0, 0] += wave_drag_force[:, 0]
            forces[:, 0, 1] += wave_drag_force[:, 1]

        # ========================================
        # 7. 施加所有力和力矩
        # ========================================
        self.robot.set_external_force_and_torque(forces, torques)

        # 轨迹线绘制
        if self.common_step_counter % 10 == 0:
            pos = self.robot.data.root_pos_w[0].cpu().tolist()
            self._trajectory_points.append(pos)
            if len(self._trajectory_points) > 2:
                try:
                    from omni.isaac.debug_draw import _debug_draw
                    draw = _debug_draw.acquire_debug_draw_interface()
                    p1 = self._trajectory_points[-2]
                    p2 = self._trajectory_points[-1]
                    draw.draw_lines([p1], [p2], [(1.0, 0.2, 0.2, 1.0)], [3.0])
                except:
                    pass
        # ========================================
        # 8. 调试输出
        # ========================================
        if self.common_step_counter % 500 == 0 and self.num_envs > 0:
            env_idx = 0

            pos_2d = self.robot.data.root_pos_w[env_idx, :2]
            target = self.target_pos[env_idx]
            dist = torch.norm(target - pos_2d).item()

            quat = self.robot.data.root_quat_w[env_idx]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            pitch_angle = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            roll_angle = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
            pitch_deg = pitch_angle.item() * 57.2958
            roll_deg = roll_angle.item() * 57.2958

            yaw_deg = torch.atan2(self.forwards[env_idx, 0], self.forwards[env_idx, 1]).item() * 57.2958

            rpos = target - pos_2d
            target_dir_deg = torch.atan2(rpos[0], rpos[1]).item() * 57.2958

            yaw_error = abs(yaw_deg - target_dir_deg)
            if yaw_error > 180:
                yaw_error = 360 - yaw_error

            boat_speed = torch.norm(linear_velocity[env_idx, :2]).item()

            current_speed = torch.norm(self.currents[env_idx]).item()
            current_dir_deg = torch.atan2(self.currents[env_idx, 1], self.currents[env_idx, 0]).item() * 57.2958

            print(f"\n📊 Step {self.common_step_counter} | Env {env_idx}:")
            print(
                f"   Position: ({pos_2d[0].item():.1f}, {pos_2d[1].item():.1f}) | Z={self.robot.data.root_pos_w[env_idx, 2]:.2f}m")
            print(f"   Target:   ({target[0].item():.1f}, {target[1].item():.1f}) | Distance: {dist:.1f}m")
            print(f"   Attitude: Roll={roll_deg:+.1f}° Pitch={pitch_deg:+.1f}°")
            print(f"   Heading:  Yaw={yaw_deg:+6.1f}° | ToTarget={target_dir_deg:+6.1f}° | Error={yaw_error:5.1f}°")
            print(f"   Speed:    {boat_speed:.2f} m/s")
            print(f"   Buoyancy: {buoyancy_force[env_idx, 2]:.0f}N")

            if self.physics_cfg.enable_current:
                print(f"   Current:  Dir={current_dir_deg:+6.1f}° | Speed={current_speed:.2f} m/s")
                print(f"   Current Force: [{current_force[env_idx, 0]:.1f}, {current_force[env_idx, 1]:.1f}] N")

            # 🆕 波浪信息
            if self.wave_cfg.enable_wave:
                wave_dir_deg = torch.atan2(self.wave_dir[env_idx, 1], self.wave_dir[env_idx, 0]).item() * 57.2958
                roll_rate = self.robot.data.root_ang_vel_w[env_idx, 0].item()
                print(
                    f"   Wave:     Dir={wave_dir_deg:+6.1f}° | Height={self.wave_height[env_idx]:.1f}m | Lateral={self.lateral_exposure[env_idx]:.2f}")
                print(f"   Roll Rate: {roll_rate:.2f} rad/s")
                # 🆕 更新波浪动画

            # 避浪统计（仅波浪模式打印）
            self.total_lateral += self.lateral_exposure.mean().item()
            self.total_steps += 1
            avg_lateral = self.total_lateral / self.total_steps
            if self.wave_cfg.enable_wave:
                print(f"   📊 Avg Lateral Exposure: {avg_lateral:.3f}")

            try:
                import wandb
                if wandb.run is not None:
                    # 计算最近的成功率
                    recent_success = 0.0
                    if self.learned_reward and len(self.learned_reward.episode_successes) >= 50:
                        recent_success = sum(self.learned_reward.episode_successes[-50:]) / 50
                    # 船的Z坐标和波浪高度
                    t_now = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
                    rov_z = self.robot.data.root_pos_w[env_idx, 2].item()
                    wave_eta = 0.0
                    if self.wave_cfg.enable_wave and self.wave_field is not None:
                        # compute_elevation 返回 (num_envs,)，取 env_idx
                        all_eta = self.wave_field.compute_elevation(
                            t_now,
                            self.robot.data.root_pos_w[:, 0],
                            self.robot.data.root_pos_w[:, 1],
                        )
                        wave_eta = all_eta[env_idx].item()

                    # 🆕 能耗指标(批量平均):
                    # power_total = |thrust × forward_speed| + |torque × yaw_rate|
                    # forward_speed_all: body-frame forward velocity for all envs
                    quat_inv_log = math_utils.quat_conjugate(self.robot.data.root_quat_w)
                    vel_b_log = math_utils.quat_apply(quat_inv_log, self.robot.data.root_com_vel_w[:, :3])
                    fwd_speed_all = vel_b_log[:, 0] * self._fwd_x   # body forward (positive = bow-first)
                    thrust_act = self.actions[:, 0]                    # [-1, 1]
                    yaw_act = self.actions[:, 1]
                    yaw_rate_all = self.robot.data.root_ang_vel_w[:, 2]
                    # power(单位:|action × speed|,无量纲)
                    power_linear = (thrust_act.abs() * fwd_speed_all.abs()).mean().item()
                    power_angular = (yaw_act.abs() * yaw_rate_all.abs()).mean().item()
                    energy_total = power_linear + power_angular
                    # 倒车浪费能量(thrust 推 bow 但 boat 反向滑动):
                    backward_waste = (thrust_act.abs() * fwd_speed_all.clamp(max=0).abs()).mean().item()

                    # V41 航行质量 metric 累积
                    cur_pos = self.robot.data.root_pos_w[:, :2].mean(dim=0)  # 批量平均位置
                    if self._prev_pos is not None:
                        step_dist = torch.norm(cur_pos - self._prev_pos).item()
                        self._metric_path_length += step_dist
                    self._prev_pos = cur_pos.clone()
                    self._metric_total_steps += 1
                    # backward_ratio: 倒车的 env 占比
                    self._metric_backward_steps += (fwd_speed_all < 0).float().mean().item()
                    # heading-velocity consistency: cos(heading, velocity_dir)
                    vel_2d_log = self.robot.data.root_com_vel_w[:, :2]
                    speed_2d_log = torch.norm(vel_2d_log, dim=-1, keepdim=True).clamp(min=0.3)
                    vel_dir_log = vel_2d_log / speed_2d_log
                    fwd_2d_log = self.forwards[:, :2]
                    fwd_2d_log = fwd_2d_log / torch.norm(fwd_2d_log, dim=-1, keepdim=True).clamp(min=1e-6)
                    hv_cos = (fwd_2d_log * vel_dir_log).sum(dim=-1).mean().item()
                    self._metric_heading_vel_cos += hv_cos
                    self._metric_energy_total += energy_total

                    # 计算瞬时航行质量指标
                    n = max(self._metric_total_steps, 1)
                    backward_ratio = self._metric_backward_steps / n
                    heading_consistency = self._metric_heading_vel_cos / n
                    energy_per_target = (self._metric_energy_total / max(self.reached_count, 1))

                    wandb.log({
                        # 波浪相关
                        "Wave/lateral_exposure_mean": self.lateral_exposure.mean().item(),
                        "Wave/avg_lateral": avg_lateral,
                        "Wave/rov_z": rov_z,
                        "Wave/wave_elevation": wave_eta,
                        "Wave/rov_z_vs_wave": rov_z - wave_eta,
                        "Wave/wave_height_hs": self.wave_field.hs[env_idx].item() if self.wave_field is not None else 0.0,
                        # 导航相关
                        "Nav/heading_error": yaw_error,
                        "Nav/speed": boat_speed,
                        "Nav/distance_to_target": dist,
                        "Nav/targets_reached": self.reached_count,
                        # 能耗
                        "Energy/power_linear": power_linear,
                        "Energy/power_angular": power_angular,
                        "Energy/total": energy_total,
                        "Energy/backward_waste": backward_waste,
                        # 航行质量(V41+)
                        "Quality/backward_ratio": backward_ratio,
                        "Quality/heading_consistency": heading_consistency,
                        "Quality/energy_per_target": energy_per_target,
                        "Quality/path_length": self._metric_path_length,
                    }, step=self.common_step_counter)
            except Exception as e:
                if not hasattr(self, '_wandb_err_printed'):
                    print(f"⚠️ wandb log error: {e}")
                    import traceback; traceback.print_exc()
                    self._wandb_err_printed = True


        self._update_wave_mesh()

    # ============================================
    # RL接口函数
    # ============================================

    def _get_observations(self) -> dict:
        import os  # 防 UnboundLocalError(后面 conditional import)
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_quat_w,
            self.forward_vec
        )

        pos_2d = self.robot.data.root_pos_w[:, :2]
        rpos = self.target_pos - pos_2d
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rpos / distance

        forwards_2d = self.forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(forwards_2d, dim=-1, keepdim=True).clamp(min=1e-6)

        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        cross = forwards_2d[:, 0:1] * direction[:, 1:2] - forwards_2d[:, 1:2] * direction[:, 0:1]
        distance_norm = distance / self.max_spawn_distance

        # CALM: 波浪关闭时返回 obs(支持 3D 或 9D self-state 模式)
        if not self.wave_cfg.enable_wave:
            # 🆕 V6 style:obs 加 self-state(velocity body, ang_vel, prev_action)防 information bottleneck
            obs_extended = int(os.environ.get('OBS_EXTENDED', '0'))
            if obs_extended == 1:
                # body-frame velocity
                vel_w_3d = self.robot.data.root_com_vel_w[:, :3]
                quat_inv = math_utils.quat_conjugate(self.robot.data.root_quat_w)
                vel_b_3d = math_utils.quat_apply(quat_inv, vel_w_3d)
                ang_vel_z = self.robot.data.root_ang_vel_w[:, 2:3]
                # prev action(下一步会更新;初始 0)
                if not hasattr(self, '_prev_action_obs'):
                    self._prev_action_obs = torch.zeros(self.num_envs, 2, device=self.device)
                obs = torch.hstack([
                    dot, cross, distance_norm,        # 3D target geometry
                    vel_b_3d,                          # 3D body velocity
                    ang_vel_z,                         # 1D yaw rate
                    self._prev_action_obs,             # 2D prev action
                ])  # 9D total
            else:
                obs = torch.hstack([dot, cross, distance_norm])
            return {"policy": obs}

        # 波浪观测
        wave_dot = torch.sum(forwards_2d * self.wave_dir, dim=-1, keepdim=True)
        wave_cross = forwards_2d[:, 0:1] * self.wave_dir[:, 1:2] - forwards_2d[:, 1:2] * self.wave_dir[:, 0:1]
        wave_height_norm = self.wave_height.unsqueeze(-1) / 1.0

        # 未来横浪力预测（非因果，含空间相位）
        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
        omega = 2 * 3.14159 / self.wave_period
        k = 2 * 3.14159 / 20.0
        rov_x = self.robot.data.root_pos_w[:, 0]
        rov_y = self.robot.data.root_pos_w[:, 1]
        spatial_phase = k * (self.wave_dir[:, 0] * rov_x + self.wave_dir[:, 1] * rov_y)
        future_horizons = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        future_intensities = []
        if self.use_jonswap:
            rov_x = self.robot.data.root_pos_w[:, 0]
            rov_y = self.robot.data.root_pos_w[:, 1]
            spatial = self.wave_dir[:, 0:1] * rov_x.unsqueeze(-1) + \
                      self.wave_dir[:, 1:2] * rov_y.unsqueeze(-1)
            for dt_future in future_horizons:
                phase = self.jonswap_omegas.unsqueeze(0) * (t + dt_future) - \
                        self.jonswap_wave_numbers.unsqueeze(0) * spatial + \
                        self.jonswap_phases
                fi = torch.abs(torch.sum(self.jonswap_amplitudes.unsqueeze(0) * torch.sin(phase), dim=-1))
                future_intensities.append(fi.unsqueeze(-1))
        else:
            for dt_future in future_horizons:
                fi = torch.abs(torch.sin(omega * (t + dt_future) - spatial_phase)).unsqueeze(-1)
                future_intensities.append(fi)
        future_wave_profile = torch.cat(future_intensities, dim=-1)

        import os
        obs_mode = os.environ.get('OBS_MODE', 'full')

        if obs_mode == 'no_wave':
            obs = torch.hstack([dot, cross, distance_norm])
        elif obs_mode == 'current_only':
            obs = torch.hstack([dot, cross, distance_norm,
                                wave_dot, wave_cross, wave_height_norm])


        else:
            obs_dim = int(os.environ.get('OBS_DIM', '12'))
            if obs_dim == 7:
                future_single = future_wave_profile[:, -1:]
                obs = torch.hstack([dot, cross, distance_norm,
                                    wave_dot, wave_cross, wave_height_norm,
                                    future_single])
            else:
                obs = torch.hstack([dot, cross, distance_norm,
                                    wave_dot, wave_cross, wave_height_norm,
                                    future_wave_profile])

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        import os
        wave_coef = float(os.environ.get('WAVE_COEF', '0.3'))

        pos_2d = self.robot.data.root_pos_w[:, :2]
        rpos = self.target_pos - pos_2d
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rpos / distance

        forwards_2d = self.forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(forwards_2d, dim=-1, keepdim=True).clamp(min=1e-6)

        # 1. 导航reward：体坐标系前进速度 × 目标对齐（保留原始base设计）
        vel_w = self.robot.data.root_com_vel_w[:, :3]
        quat_inv = math_utils.quat_conjugate(self.robot.data.root_quat_w)
        vel_b = math_utils.quat_apply(quat_inv, vel_w)
        # boat: forward_speed = vel_b · forward_vec(只在 X 轴) → vel_b[X] × fwd_x
        forward_speed = vel_b[:, 0:1] * self._fwd_x
        alignment = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)  # [-1, 1]

        # 🆕 reward 变体(env var REWARD_VARIANT 控制),默认 V12
        variant = os.environ.get('REWARD_VARIANT', 'V12')
        if variant == 'V11':
            # 原(失败):反方向也获 +0.37 奖
            reward_nav = forward_speed * torch.exp(alignment)
        elif variant == 'V12':
            # smooth gate: (align+1)/2 → 反方向 = 0,smooth grad
            gate = (alignment + 1.0) * 0.5
            reward_nav = forward_speed * torch.exp(alignment) * gate
        elif variant == 'V13':
            # 硬 ReLU gate,alignment 必须 > 0 才有奖
            reward_nav = forward_speed * torch.relu(alignment)
        elif variant == 'V14':
            # 只算前进的速度,后退 = 0 奖
            reward_nav = torch.relu(forward_speed) * torch.exp(alignment)
        elif variant == 'V15':
            # V12 + 显式 heading reward(V2 风格)
            gate = (alignment + 1.0) * 0.5
            reward_nav = forward_speed * torch.exp(alignment) * gate + 0.5 * alignment.clamp(min=-1)
        elif variant == 'V16':
            # 更陡的对齐惩罚 exp(2×align)
            gate = (alignment + 1.0) * 0.5
            reward_nav = forward_speed * torch.exp(2 * alignment) * gate
        elif variant == 'V23':
            # V2/V8 风格 + E7 speed-coupling(防"驻足凝视"局部最优)
            # 硬规则 line 432:任何速度无关的奖励 → "不动=最优"
            heading_reward_raw = alignment.clamp(min=-1.0, max=1.0) * 1.0
            vel_toward = torch.sum(
                self.robot.data.root_com_vel_w[:, :2] * direction, dim=-1, keepdim=True
            )
            velocity_reward = vel_toward.clamp(min=0) * 0.3   # 本身就是速度
            dist_penalty_v23 = -(distance / self.max_spawn_distance) * 0.3
            # 🆕 V40:SIDE_APPROACH=1 → 不要求船头对准目标,侧面碰也算 reach
            #   把 heading_reward 用 vel_toward 的方向(速度对目标的投影)替代
            #   alive_bonus 也 × |velocity|(不限制 forward_speed)
            side_approach = int(os.environ.get('SIDE_APPROACH', '0'))
            forward_transit = int(os.environ.get('FORWARD_TRANSIT', '0'))
            speed_coupling = int(os.environ.get('SPEED_COUPLE', '1'))
            if side_approach and forward_transit:
                # V42: SIDE_APPROACH + FORWARD_TRANSIT (修复 V41 逃跑 bug)
                # V41 失败根因: 没有朝目标方向的引导, 船高速背对目标逃跑
                # 修复: vel_toward 系数 ↑ 到主导项, alive_bonus 不再奖励乱跑
                vel_2d = self.robot.data.root_com_vel_w[:, :2]
                speed_2d = torch.norm(vel_2d, dim=-1, keepdim=True).clamp(min=0.1)
                vel_dir = vel_2d / speed_2d
                heading_dir = forwards_2d  # 船头方向(已归一化)
                forward_align = torch.sum(heading_dir * vel_dir, dim=-1, keepdim=True)
                fwd_transit_coef = float(os.environ.get('FWD_TRANSIT_COEF', '0.2'))
                # 正着走奖励: 只有 vel_toward > 0 (朝目标走) 时才给
                heading_reward = forward_align.clamp(min=0) * vel_toward.clamp(min=0) * fwd_transit_coef
                backward_cost_coef = float(os.environ.get('BACKWARD_COST', '0.3'))
                backward_cost = forward_speed.clamp(max=0) * backward_cost_coef
                # vel_toward 是主导导航信号 (系数 1.0, V41 是 0.8)
                velocity_reward = vel_toward.clamp(min=0) * 1.0
                # alive_bonus × vel_toward 方向: 只有朝目标运动才奖
                alive_bonus = vel_toward.clamp(min=0) * 0.05
            elif side_approach:
                # V40: 纯侧面 approach,不管行进朝向
                heading_reward = torch.zeros_like(distance)
                vel_2d_mag = torch.norm(self.robot.data.root_com_vel_w[:, :2], dim=-1, keepdim=True)
                alive_bonus = vel_2d_mag * 0.05
                backward_cost = torch.zeros_like(distance)
                velocity_reward = vel_toward.clamp(min=0) * 0.8
            elif speed_coupling:
                fs = forward_speed.clamp(min=0)
                heading_reward = heading_reward_raw * fs
                alive_bonus = fs * 0.05    # 必须动才有 alive 奖(变相鼓励持续运动)
                backward_cost_coef = float(os.environ.get('BACKWARD_COST', '0.5'))
                backward_cost = forward_speed.clamp(max=0) * backward_cost_coef
            else:
                heading_reward = heading_reward_raw
                alive_bonus = torch.full_like(distance, 0.05)
                backward_cost = torch.zeros_like(distance)
            reward_nav = heading_reward + velocity_reward + dist_penalty_v23 + alive_bonus + backward_cost
        else:
            raise ValueError(f"Unknown REWARD_VARIANT: {variant}")

        # 2. 波浪安全reward：log形，纯几何（模仿NavRL）
        # 2. 波浪安全reward：NavRL-style log(safety_distance)
        #    将 lateral_exposure 映射为 safety_distance ∈ (0, RANGE]
        #    与 NavRL 的 log(lidar距离) 结构完全对应
        wave_safety = torch.zeros_like(distance)
        if self.wave_cfg.enable_wave:
            WAVE_SAFETY_RANGE = 10.0  # 对应 NavRL 的 lidar_range
            t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
            if self.use_jonswap:
                # V3b: 纯几何 + 去掉Hs缩放
                # V3失败原因：hs_range=(0.3,1.0)，乘Hs后惩罚只有nav的3-4%
                # 直接用lateral_exposure × speed，不乘Hs
                future_danger = self.lateral_exposure
            else:
                future_danger = self.lateral_exposure  # Airy模式

            if self.use_jonswap:
                # JONSWAP: 不乘Hs（Hs=0.3-1.0太小，会稀释信号）
                wave_safety = -future_danger.unsqueeze(-1) * forward_speed.clamp(min=0)
            else:
                # Airy: 保留乘Hs（Airy的wave_height设置通常较大）
                wave_safety = -future_danger.unsqueeze(-1) * self.wave_height.unsqueeze(-1) * forward_speed.clamp(min=0)

        # 🆕 距离 shaping(让 agent 学会"近了就慢")
        dist_penalty_coef = float(os.environ.get('DIST_PENALTY_COEF', '0.0'))
        if dist_penalty_coef > 0:
            # distance normalized to [0, 1], penalty = -coef × dist_norm
            dist_penalty = -(distance / self.max_spawn_distance) * dist_penalty_coef
            reward_nav = reward_nav + dist_penalty

        # 3. 到达reward + 目标重置
        # 🆕 REACH_BONUS env var:boat 慢,默认 +10 太小被 respawn 旅行成本吃掉
        # boat 至少要 +50~+100,见硬规则 line 438 (连续导航需要 reach_reward 足够大)
        reach_bonus = float(os.environ.get('REACH_BONUS', '10.0'))
        reached = (distance < self.goal_radius).float()
        reach_reward = reached * reach_bonus

        reached_mask = reached.squeeze(-1).bool()
        if reached_mask.any():
            env_ids = torch.where(reached_mask)[0]
            distances = self.min_spawn_distance + \
                        torch.rand(len(env_ids), device=self.device) * \
                        (self.max_spawn_distance - self.min_spawn_distance)
            angles = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
            origin_xy = self.scene.env_origins[env_ids, :2]
            self.target_pos[env_ids, 0] = origin_xy[:, 0] + distances * torch.cos(angles)
            self.target_pos[env_ids, 1] = origin_xy[:, 1] + distances * torch.sin(angles)
            self.reached_count += len(env_ids)
            # 🔬 DEBUG:reach 触发时打印,确认 respawn 真的发生
            if 0 in env_ids.cpu().numpy().tolist():
                new_x = self.target_pos[0, 0].item()
                new_y = self.target_pos[0, 1].item()
                boat_x = self.robot.data.root_pos_w[0, 0].item()
                boat_y = self.robot.data.root_pos_w[0, 1].item()
                print(f"🎯 env 0 REACH! boat=({boat_x:.1f},{boat_y:.1f}) → new target=({new_x:.1f},{new_y:.1f})")

        reward = (reward_nav
                  + wave_safety * wave_coef
                  + reach_reward)

        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 越界检测：船离 env_origin 超过边界则终止
        pos_2d = self.robot.data.root_pos_w[:, :2]
        origin_2d = self.scene.env_origins[:, :2]
        dist_from_origin = torch.norm(pos_2d - origin_2d, dim=-1)
        out_of_bounds = dist_from_origin > (self.max_spawn_distance + 20.0)  # 留 20m 余量

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 连续导航模式：到达统计在 _get_rewards 里已经做了
        # 这里只负责 log
        self.episode_count += len(env_ids)
        if self.episode_count >= 50:
            targets_per_ep = self.reached_count / self.episode_count
            n = max(self._metric_total_steps, 1)
            bw_ratio = self._metric_backward_steps / n
            hd_cons = self._metric_heading_vel_cos / n
            e_per_t = self._metric_energy_total / max(self.reached_count, 1)
            print(f"📊 Targets/ep: {targets_per_ep:.2f} ({self.reached_count}/{self.episode_count})"
                  f" | bw_ratio={bw_ratio:.2f} | heading_cons={hd_cons:.2f} | energy/target={e_per_t:.1f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "Metrics/targets_per_episode": targets_per_ep,
                        "Metrics/backward_ratio": bw_ratio,
                        "Metrics/heading_consistency": hd_cons,
                        "Metrics/energy_per_target": e_per_t,
                        "Metrics/path_length": self._metric_path_length,
                    })
            except:
                pass
            self.episode_count = 0
            self.reached_count = 0
            self._metric_backward_steps = 0
            self._metric_total_steps = 0
            self._metric_heading_vel_cos = 0.0
            self._metric_energy_total = 0.0
            self._metric_path_length = 0.0

        super()._reset_idx(env_ids)

        num = len(env_ids)

        # 重置 ROV 状态
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # 生成目标点（相对 env_origin，确保船不会漂移出边界）
        distances = self.min_spawn_distance + \
                    torch.rand(num, device=self.device) * \
                    (self.max_spawn_distance - self.min_spawn_distance)
        angles = torch.rand(num, device=self.device) * 2 * torch.pi

        origin_xy = self.scene.env_origins[env_ids, :2]
        self.target_pos[env_ids, 0] = origin_xy[:, 0] + distances * torch.cos(angles)
        self.target_pos[env_ids, 1] = origin_xy[:, 1] + distances * torch.sin(angles)

        # 🆕 随机化波浪方向
        if self.wave_cfg.enable_wave:
            random_angles = torch.rand(num, device=self.device) * 2 * 3.14159
            self.wave_dir[env_ids, 0] = torch.cos(random_angles)
            self.wave_dir[env_ids, 1] = torch.sin(random_angles)

        # 随机化JONSWAP波浪场
        if self.wave_field is not None:
            self.wave_field.randomize(env_ids)
            # 记录env 0的波浪参数（用于复现好看的波浪）
            if 0 in env_ids:
                hs = self.wave_field.hs[0].item()
                tp = self.wave_field.tp[0].item()
                gm = self.wave_field.gamma[0].item()
                wd = torch.atan2(self.wave_field.wave_dir[0, 1], self.wave_field.wave_dir[0, 0]).item()
                print(f"🌊 Env0 wave: Hs={hs:.2f}m Tp={tp:.2f}s gamma={gm:.2f} dir={wd:.2f}rad")

        self._visualize_markers()