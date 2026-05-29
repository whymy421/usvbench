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
        self.goal_radius = getattr(cfg, 'goal_radius', 2.0)
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


    def _load_water_from_usd(self):
        """从ROV_TEST.usd复制水面mesh及其动画"""
        import omni.usd
        from pxr import Usd, UsdGeom, Sdf

        print("\n🌊 Loading water surface from ROV_TEST.usd...")

        source_usd_path = "C:/Users/Yutong/NavRL/NavRL2026/isaac_underwater/ROV_TEST.usd"
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

        self._load_water_from_usd()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

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
            ).squeeze()
        # FIX: boat USD body-(-X) = 船头方向(SolidWorks约定,body-X 指船尾)
        self.forward_vec = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarkers",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.target_markers = VisualizationMarkers(target_marker_cfg)

        # 🆕 初始化波浪场
        self._init_wave_field()
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
        self.marker_locations = self.robot.data.root_pos_w

        base_quat = self.robot.data.root_quat_w

        rotation_angle = +torch.pi / 2
        rotation_quat = math_utils.quat_from_angle_axis(
            torch.ones((self.num_envs, 1), device=self.device) * rotation_angle,
            self.up_dir
        ).squeeze()

        self.forward_marker_orientations = math_utils.quat_mul(base_quat, rotation_quat)

        self.command_marker_orientations = math_utils.quat_from_angle_axis(
            self.yaws, self.up_dir
        ).squeeze()

        forward_command_offset = torch.zeros((self.num_envs, 3), device=self.device)
        forward_command_offset[:, 2] = 0.5
        forward_loc = self.marker_locations + forward_command_offset
        command_loc = self.marker_locations + forward_command_offset

        current_offset = torch.zeros((self.num_envs, 3), device=self.device)
        current_offset[:, 2] = 2.0
        current_loc = self.marker_locations + current_offset

        loc = torch.vstack((forward_loc, command_loc, current_loc))
        rots = torch.vstack((
            self.forward_marker_orientations,
            self.command_marker_orientations,
            self.current_orientations
        ))

        all_envs = torch.arange(self.num_envs, device=self.device)
        indices = torch.hstack((
            torch.zeros_like(all_envs),
            torch.ones_like(all_envs),
            torch.ones_like(all_envs) * 2
        ))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

        target_pos_3d = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_3d[:, :2] = self.target_pos
        target_pos_3d[:, 2] = self.robot.data.root_pos_w[:, 2]
        self.target_markers.visualize(target_pos_3d)

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
        # 动态水面高度（和视觉波浪同步）
        if self.wave_cfg.enable_wave:
            t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
            omega = 2 * 3.14159 / self.wave_cfg.wave_period
            k = 2 * 3.14159 / 20.0  # 波长20m，和视觉一致
            rov_x = positions[:, 0]
            rov_y = positions[:, 1]
            wave_phase = omega * t - k * (self.wave_dir[:, 0] * rov_x + self.wave_dir[:, 1] * rov_y)
            water_surface = self.physics_cfg.water_surface_z + self.wave_height * torch.sin(wave_phase)
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
        """初始化波浪参数"""
        if not self.wave_cfg.enable_wave:
            self.wave_height = torch.zeros(self.num_envs, device=self.device)
            self.wave_period = torch.ones(self.num_envs, device=self.device) * 5.0
            self.wave_dir = torch.zeros(self.num_envs, 2, device=self.device)
            print("\n🌊 Waves: DISABLED\n")
            return

        # 波高（可以每个环境随机）
        self.wave_height = torch.ones(self.num_envs, device=self.device) * self.wave_cfg.wave_height

        # 波周期
        self.wave_period = torch.ones(self.num_envs, device=self.device) * self.wave_cfg.wave_period

        # 波浪方向（归一化）
        wave_dir = torch.tensor([self.wave_cfg.wave_dir_x, self.wave_cfg.wave_dir_y], device=self.device)
        wave_dir = wave_dir / torch.norm(wave_dir).clamp(min=1e-6)
        self.wave_dir = wave_dir.unsqueeze(0).repeat(self.num_envs, 1)

        print(f"\n🌊 Wave Field Initialized:")
        print(f"   Wave height: {self.wave_cfg.wave_height} m")
        print(f"   Wave period: {self.wave_cfg.wave_period} s")
        print(f"   Wave direction: ({self.wave_cfg.wave_dir_x}, {self.wave_cfg.wave_dir_y})")
        print(f"   Wave physics: ENABLED\n")

    def _create_wave_mesh(self):
        """创建动态波浪水面mesh"""
        import omni.usd
        from pxr import UsdGeom, Gf, Vt, Sdf, UsdShade
        import numpy as np

        stage = omni.usd.get_context().get_stage()
        water_path = "/World/DynamicWater"

        if stage.GetPrimAtPath(water_path):
            return

        mesh = UsdGeom.Mesh.Define(stage, water_path)

        self._wave_mesh_size = 100.0
        self._wave_mesh_res = 60
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

        material_path = "/World/DynamicWaterMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.3, 0.8))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        self._wave_mesh = mesh
        self._wave_k = 2 * 3.14159 / 20.0
        print("✅ Dynamic wave mesh created (100m x 100m, 60x60)")

    def _update_wave_mesh(self):
        """每步更新波浪水面顶点"""
        if not hasattr(self, '_wave_mesh'):
            return
        if not self.wave_cfg.enable_wave:
            return  # FIX: 静水时不要动画水面,否则视觉上像船在上下振荡
        if self.common_step_counter % 5 != 0:
            return

        from pxr import Gf, Vt
        import math

        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
        omega = 2 * 3.14159 / self.wave_cfg.wave_period
        k = self._wave_k
        res = self._wave_mesh_res
        h = self.wave_cfg.wave_height

        wave_dx = self.wave_dir[0, 0].item()
        wave_dy = self.wave_dir[0, 1].item()

        cx = self.robot.data.root_pos_w[0, 0].item()
        cy = self.robot.data.root_pos_w[0, 1].item()

        new_points = []
        for j in range(res):
            for i in range(res):
                px = self._wave_xx[j, i] + cx
                py = self._wave_yy[j, i] + cy
                phase = omega * t - k * (wave_dx * px + wave_dy * py)
                z = h * math.sin(phase)
                new_points.append(Gf.Vec3f(float(px), float(py), float(z)))

        self._wave_mesh.GetPointsAttr().Set(Vt.Vec3fArray(new_points))


    def _compute_wave_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算波浪对船的作用力"""
        if not self.wave_cfg.enable_wave:
            return (
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )

        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation

        # 波浪相位
        phase = 2 * 3.14159 * t / self.wave_period

        # 船头方向 (2D)
        forwards_2d = self.forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(forwards_2d, dim=-1, keepdim=True).clamp(min=1e-6)

        # 船侧方向
        sideways = torch.stack([-forwards_2d[:, 1], forwards_2d[:, 0]], dim=-1)

        # 横浪程度
        self.lateral_exposure = torch.abs(torch.sum(self.wave_dir * sideways, dim=-1))

        # 迎浪程度
        frontal_exposure = torch.sum(self.wave_dir * forwards_2d, dim=-1)

        roll_torque = self.lateral_exposure * self.wave_height * 150.0 * torch.sin(phase)
        heave_force = self.wave_height * 50.0 * torch.sin(phase)
        self.wave_drag = torch.clamp(-frontal_exposure, min=0) * self.wave_height * 10.0

        return heave_force, roll_torque

    # ============================================
    # 施加动作
    # ============================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
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
        # 1. 推进器控制（来自RL策略）
        # ========================================
        import os
        if os.environ.get('NO_ACTION') == '1':
            # DEMO模式:不施加策略动作,纯看浮力效果
            forces[:, 0, 0] = 0
            forces[:, 0, 1] = 0
            torques[:, 0, 2] = 0
        else:
            # DEBUG 模式:DEBUG_THRUST=X 强行往哪个轴推 200N,验证 USD 朝向
            #   DEBUG_THRUST=X → 往 body-X 推
            #   DEBUG_THRUST=Y → 往 body-Y 推(原"前进推力")
            #   DEBUG_THRUST=Z → 往 body-Z 推
            #   不设 → 正常 RL 模式
            debug_thrust = os.environ.get('DEBUG_THRUST', '')
            if debug_thrust == 'X':
                forces[:, 0, 0] = 200.0
            elif debug_thrust == 'Y':
                forces[:, 0, 1] = 200.0
            elif debug_thrust == 'Z':
                forces[:, 0, 2] = 200.0
            else:
                # FIX: boat USD body-X = 船尾(stern),-X = 船头(bow);body-Y = 右舷
                forces[:, 0, 0] = self.actions[:, 0] * -200.0  # 负号:正 action → 船头方向推力
                forces[:, 0, 1] = 0  # 右舷方向无推力
                torques[:, 0, 2] = self.actions[:, 1] * 120    # body-Z 是上,绕它转就是 yaw
                if self.common_step_counter % 2000 == 0:
                    act_mean = self.actions.mean(dim=0).tolist()
                    act_std = self.actions.std(dim=0).tolist()
                    print(f"[step {self.common_step_counter}] action mean={act_mean}, std={act_std}")

        # ========================================
        # 2. 浮力和浮力力矩
        # ========================================
        buoyancy_force, buoyancy_torque = self._compute_buoyancy_forces()
        forces[:, 0, :] = forces[:, 0, :] + buoyancy_force
        torques[:, 0, :] = torques[:, 0, :] + buoyancy_torque

        # ========================================
        # 3. 阻尼:改用PhysX原生阻尼(在rigid_props里设),不在Python里算
        # ========================================
        # (linear_damping/angular_damping在rigid_props里已经设了)

        # ========================================
        # 4. 姿态稳定:关掉。ROV留下的hack,对boat会持续oscillate
        # ========================================
        # (pitch/roll restoring torques removed - boat uses native PhysX dynamics)

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

        # ========================================
        # DEBUG: 每30步打印 env_0 的 z 状态 (设 DEBUG_Z=1 开启)
        # ========================================
        import os
        if os.environ.get('DEBUG_Z') == '1' and self.common_step_counter % 30 == 0:
            z_cm = self.robot.data.root_pos_w[0, 2].item() * 100
            vz_cms = self.robot.data.root_com_vel_w[0, 2].item() * 100
            buoy_z = buoyancy_force[0, 2].item()
            print(f"[step {self.common_step_counter:5d}] z={z_cm:+7.2f}cm  vz={vz_cms:+7.2f}cm/s  buoyancy_z={buoy_z:6.1f}N  (weight=980N)")

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

            boat_speed = torch.norm(self.robot.data.root_com_vel_w[env_idx, :2]).item()

            current_speed = torch.norm(self.currents[env_idx]).item()
            current_dir_deg = torch.atan2(self.currents[env_idx, 1], self.currents[env_idx, 0]).item() * 57.2958

            print(f"\n📊 Step {self.common_step_counter} | Env {env_idx}:")
            print(
                f"   Position: ({pos_2d[0].item():.1f}, {pos_2d[1].item():.1f}) | Z={self.robot.data.root_pos_w[env_idx, 2]:.2f}m")
            print(f"   Target:   ({target[0].item():.1f}, {target[1].item():.1f}) | Distance: {dist:.1f}m")
            print(f"   Attitude: Roll={roll_deg:+.1f}° Pitch={pitch_deg:+.1f}°")
            print(f"   Heading:  Yaw={yaw_deg:+6.1f}° | ToTarget={target_dir_deg:+6.1f}° | Error={yaw_error:5.1f}°")
            print(f"   Speed:    {boat_speed:.2f} m/s")
            print(f"   Current:  Dir={current_dir_deg:+6.1f}° | Speed={current_speed:.2f} m/s")
            print(f"   Buoyancy: {buoyancy_force[env_idx, 2]:.0f}N")

            if self.physics_cfg.enable_current:
                print(f"   Current Force: [{current_force[env_idx, 0]:.1f}, {current_force[env_idx, 1]:.1f}] N")

            # 🆕 波浪信息
            if self.wave_cfg.enable_wave:
                wave_dir_deg = torch.atan2(self.wave_dir[env_idx, 1], self.wave_dir[env_idx, 0]).item() * 57.2958
                roll_rate = self.robot.data.root_ang_vel_w[env_idx, 0].item()
                print(
                    f"   Wave:     Dir={wave_dir_deg:+6.1f}° | Height={self.wave_height[env_idx]:.1f}m | Lateral={self.lateral_exposure[env_idx]:.2f}")
                print(f"   Roll Rate: {roll_rate:.2f} rad/s")
                # 🆕 更新波浪动画

            # 避浪统计
            self.total_lateral += self.lateral_exposure.mean().item()
            self.total_steps += 1
            avg_lateral = self.total_lateral / self.total_steps
            print(f"   📊 Avg Lateral Exposure: {avg_lateral:.3f}")

            try:
                import wandb
                if wandb.run is not None:
                    # 计算最近的成功率
                    recent_success = 0.0
                    if self.learned_reward and len(self.learned_reward.episode_successes) >= 50:
                        recent_success = sum(self.learned_reward.episode_successes[-50:]) / 50
                    wandb.log({
                        "Wave/lateral_exposure_mean": self.lateral_exposure.mean().item(),
                        "Wave/avg_lateral": avg_lateral,
                        "Wave/heading_error": yaw_error,
                        "Wave/speed": boat_speed,
                        "Wave/distance": dist,
                        "Nav/success_rate": recent_success,
                    }, step=self.common_step_counter)
            except:
                pass


        self._update_wave_mesh()

    # ============================================
    # RL接口函数
    # ============================================

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w,
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

        # 波浪观测
        wave_dot = torch.sum(forwards_2d * self.wave_dir, dim=-1, keepdim=True)
        wave_cross = forwards_2d[:, 0:1] * self.wave_dir[:, 1:2] - forwards_2d[:, 1:2] * self.wave_dir[:, 0:1]
        wave_height_norm = self.wave_height.unsqueeze(-1) / 1.0

        # 未来横浪力预测（非因果）
        t = self.common_step_counter * self.cfg.sim.dt * self.cfg.decimation
        omega = 2 * 3.14159 / self.wave_period
        future_phase = torch.sin(omega * (t + 3.0)).unsqueeze(-1)
        future_lateral_force = wave_cross * future_phase

        obs = torch.hstack([
            dot, cross, distance_norm,
        ])

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_2d = self.robot.data.root_pos_w[:, :2]
        rpos = self.target_pos - pos_2d
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rpos / distance

        if self.cfg.use_learned_reward:
            # E4: 自学习reward
            obs = self._get_observations()["policy"]
            reward = self.learned_reward.compute_reward(
                obs, self.actions, distance,
                self.goal_radius, self.max_spawn_distance
            )
            return reward

        # E2/E3: 手动reward
        forwards_2d = self.forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(forwards_2d, dim=-1, keepdim=True).clamp(min=1e-6)

        # 指纹:确认本次运行加载的是新版代码 (用class属性,只打一次)
        if not getattr(self, '_reward_v2_announced', False):
            print(">>> REWARD V2 LOADED: velocity*2.0, reach*30, speed_gate, dist*1.0 <<<")
            self._reward_v2_announced = True

        # 1. 朝向奖励——只有动起来才算!避免静止刷分
        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        # 2. 速度奖励（向目标方向的速度）—— 主奖励信号,权重大幅提升
        vel_2d = self.robot.data.root_com_vel_w[:, :2]
        vel_toward = torch.sum(vel_2d * direction, dim=-1, keepdim=True)
        velocity_reward = torch.clamp(vel_toward, min=0) * 2.0   # 0.3 → 2.0,真正激励"动得快"
        # heading 一半无条件给信号(让 PPO 静止时也有梯度),一半 speed-gated 鼓励"动起来对准"
        speed_mag = torch.norm(vel_2d, dim=-1, keepdim=True)
        speed_gate = torch.clamp(speed_mag / 0.5, max=1.0)
        heading_reward = torch.clamp(dot, min=-1.0) * (0.3 + 0.5 * speed_gate)

        # 3. 到达奖励
        reached = (distance < self.goal_radius).float()
        reach_reward = reached * 30.0   # 10 → 30, 终极目标更显眼

        # 4. 距离惩罚——加大让"停在远处"代价更高
        dist_penalty = -distance / self.max_spawn_distance * 1.0   # 0.3 → 1.0

        # 5. 避浪奖励（只在波浪开启时生效）
        wave_penalty = torch.zeros_like(distance)
        roll_penalty = torch.zeros_like(distance)
        head_wave_bonus = torch.zeros_like(distance)

        if self.wave_cfg.enable_wave:
            # 横浪惩罚: lateral越大惩罚越重
            wave_penalty = -self.lateral_exposure.unsqueeze(-1) * self.wave_height.unsqueeze(-1) * 2.0

            # 横摇惩罚: roll rate越大惩罚越重
            roll_rate = torch.abs(self.robot.data.root_ang_vel_w[:, 0:1])
            roll_penalty = -roll_rate * 1.5

            # 迎浪/顺浪奖励: 船头或船尾对着波浪方向时给奖励
            wave_dot = torch.sum(forwards_2d * self.wave_dir, dim=-1, keepdim=True)
            head_wave_bonus = torch.abs(wave_dot) * 0.5  # |cos|大=迎浪或顺浪

        # 总reward
        reward = (heading_reward
                  + velocity_reward
                  + reach_reward
                  + dist_penalty
                  + wave_penalty
                  + roll_penalty
                  + head_wave_bonus
                  + 0.05)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_2d = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(self.target_pos - pos_2d, dim=-1)

        reached = distance < self.goal_radius

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return reached, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # ⚠️ 先计算是否到达（在重置目标之前！）
        pos_2d = self.robot.data.root_pos_w[env_ids, :2]
        dist = torch.norm(self.target_pos[env_ids] - pos_2d, dim=-1)
        reached = dist < self.goal_radius

        # 通知自学习reward：episode结束
        if self.learned_reward is not None:
            self.learned_reward.on_episode_end(env_ids, reached)

        super()._reset_idx(env_ids)

        num = len(env_ids)

        # 重置 ROV 状态
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # 生成目标点
        distances = self.min_spawn_distance + \
                    torch.rand(num, device=self.device) * \
                    (self.max_spawn_distance - self.min_spawn_distance)
        angles = torch.rand(num, device=self.device) * 2 * torch.pi

        rov_pos = self.robot.data.root_pos_w[env_ids, :2]
        self.target_pos[env_ids, 0] = rov_pos[:, 0] + distances * torch.cos(angles)
        self.target_pos[env_ids, 1] = rov_pos[:, 1] + distances * torch.sin(angles)

        # 🆕 随机化波浪方向
        if self.wave_cfg.enable_wave:
            random_angles = torch.rand(num, device=self.device) * 2 * 3.14159
            self.wave_dir[env_ids, 0] = torch.cos(random_angles)
            self.wave_dir[env_ids, 1] = torch.sin(random_angles)

        self._visualize_markers()