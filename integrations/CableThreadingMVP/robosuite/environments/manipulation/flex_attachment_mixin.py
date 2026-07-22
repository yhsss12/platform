"""
FlexAttachmentMixin -- Flex 线缆夹取能力（延迟附着 + 顶点直接操控）

提供 flex 线缆的统一夹取机制，供 CableRouting、CableThreading 等非 CableBaseEnv 环境使用。

核心能力：
- 延迟附着：set_attachment_enabled 只设置 pending，等夹爪闭合后才真正激活
- 顶点直接操控：中点抓取时直接写入 flex vertex qpos，让弹性自然处理形变
- 距离检查：夹爪与目标点距离超阈值时不激活

使用方式：
    class MyEnv(FlexAttachmentMixin, ManipulationEnv):
        def _load_model(self, ...):
            ...
            if self._is_flex_cable:
                self._init_flex_attachment_state()

        def _pre_action(self, action, policy_step=False):
            super()._pre_action(action, policy_step=policy_step)
            self._flex_pre_action_check(action)
            if self.attachment_enabled:
                self._flex_update_attachment()
"""

import numpy as np


class FlexAttachmentMixin:
    """Flex 线缆夹取能力 mixin。

    宿主类需要提供：
    - self.robots[0]（robosuite Robot）
    - self._get_gripper_site_position() -> np.ndarray(3,)
    - self.sim（MuJoCo Sim）
    - self._is_flex_cable (bool)
    - self._flex_vertadr (int)
    - self._flex_vertnum (int)
    - self.attach_offset (np.ndarray(3,))
    - self.attachment_follow_gain (float)
    - self._cable_in_task.flex_vertex_spacing (float, optional) -- flex 顶点间距
    """

    def _init_flex_attachment_state(self):
        """初始化 flex 夹取状态变量。在 _load_model 的 flex 分支中调用。"""
        self._flex_grasp_active = False
        self._flex_grasp_vtx_idx = -1
        self._flex_grasp_offset = np.zeros(3, dtype=float)
        self._flex_grasp_window = 1
        self._attach_pending = False
        self._attach_pending_params = {}
        self._attach_grip_threshold = 0.3
        self._attach_max_distance = 0.03

    def _is_gripper_closed(self, action):
        """检查夹爪是否已闭合。通过 action 的 gripper 分量判断。"""
        robot = self.robots[0]
        arm = robot.arms[0]
        gripper_dof = robot.gripper[arm].dof
        if gripper_dof == 0:
            return True
        gripper_val = float(action[-gripper_dof])
        return gripper_val > self._attach_grip_threshold

    def _is_flex_target_close_enough(self, target_pos):
        """检查夹爪与目标点的距离是否在阈值内。"""
        grip_pos = self._get_gripper_site_position()
        dist = float(np.linalg.norm(grip_pos - target_pos))
        return dist < self._attach_max_distance

    def _flex_pre_action_check(self, action):
        """在 _pre_action 中调用：检查待激活附着（夹爪闭合 + 距离检查）。

        子类调用此方法后，如果条件满足会调用 _activate_flex_attachment()。
        """
        if not self._attach_pending:
            return
        grip_closed = self._is_gripper_closed(action)
        # 获取目标位置进行距离检查
        target_pos = self._get_flex_attach_target_pos()
        close_enough = self._is_flex_target_close_enough(target_pos) if target_pos is not None else True
        if grip_closed and close_enough:
            self._activate_flex_attachment()

    def _get_flex_attach_target_pos(self):
        """获取 flex 附着目标位置（用于距离检查）。

        子类应覆写此方法以提供正确的目标位置。
        默认返回 None（跳过距离检查）。
        """
        return None

    def _activate_flex_attachment(self):
        """正式激活附着（延迟到夹爪闭合后）。

        默认实现：对于 flex 中点抓取，激活顶点直接操控。
        子类可覆写以处理端点 weld 约束等特定逻辑。
        """
        params = self._attach_pending_params
        self._attach_pending = False
        self._attach_pending_params = {}

        point_idx = params.get("point_idx")

        # flex 中点抓取：直接操控顶点
        if self._is_flex_cable and point_idx is not None:
            nvert = self._flex_vertnum
            if 0 < point_idx < nvert - 1:
                self._flex_grasp_active = True
                self._flex_grasp_vtx_idx = int(point_idx)
                grip_pos = self._get_gripper_site_position()
                vtx_pos = self.sim.data.flexvert_xpos[point_idx].copy()
                self._flex_grasp_offset = grip_pos - vtx_pos
                self.attachment_enabled = True
                self.sim.forward()
                return
            self._flex_grasp_active = False

        # 端点：设置 attachment_enabled，由子类处理具体 weld/mocap 逻辑
        self.attachment_enabled = True
        self.sim.forward()

    def _flex_vertex_follow_gripper(self):
        """Flex 顶点直接操控：每步将 gripper 世界坐标转换为 body-local qpos 并写入。

        在 _attach_cable_end_to_gripper 或类似方法中，当 _flex_grasp_active 时调用。
        """
        grip_pos = self._get_gripper_site_position()
        clamp_offset = (
            self._gripper_clamp_center_offset()
            if hasattr(self, "_gripper_clamp_center_offset")
            else np.zeros(3, dtype=float)
        )
        target_world = grip_pos + clamp_offset + self.attach_offset
        vtx_idx = self._flex_grasp_vtx_idx
        adr = self._flex_vertadr

        # 获取 "object" body 的世界变换（flexcomp 所在的 body）
        obj_body_id = self.sim.model.body_name2id("object")
        body_pos = self.sim.data.body_xpos[obj_body_id]
        body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)
        spacing = getattr(getattr(self, "_cable_in_task", None), "flex_vertex_spacing", None) or 0.01

        # 将目标世界坐标转换为 body-local qpos 偏移
        target_local = body_rot.T @ (target_world - body_pos)
        grid_rest = np.array([vtx_idx * spacing, 0.0, 0.0], dtype=float)
        local_offset = target_local - grid_rest
        self.sim.data.qpos[adr + vtx_idx * 3: adr + vtx_idx * 3 + 3] = local_offset

    def _flex_tail_follow_gripper(self, window_size=3):
        """让 flex 端点附近的一小段尾部跟随 gripper。

        与只控制最后一个顶点相比，这种方式能减少末端奇异卷起，
        更接近“夹住线缆末端一小段”的视觉效果。
        """
        nvert = self._flex_vertnum
        adr = self._flex_vertadr
        spacing = getattr(getattr(self, "_cable_in_task", None), "flex_vertex_spacing", None) or 0.01
        tail_end = int(np.clip(self._flex_grasp_vtx_idx, 0, nvert - 1))
        tail_start = max(0, tail_end - max(int(window_size), 1) + 1)

        obj_body_id = self.sim.model.body_name2id("object")
        body_pos = self.sim.data.body_xpos[obj_body_id]
        body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)

        grip_pos = self._get_gripper_site_position()
        clamp_offset = (
            self._gripper_clamp_center_offset()
            if hasattr(self, "_gripper_clamp_center_offset")
            else np.zeros(3, dtype=float)
        )
        target_world = grip_pos + clamp_offset + self.attach_offset
        target_local = body_rot.T @ (target_world - body_pos)

        verts_world = self.sim.data.flexvert_xpos[adr: adr + nvert].copy().reshape(nvert, 3)
        verts_local = (body_rot.T @ (verts_world - body_pos).T).T

        if tail_end > 0:
            tail_dir = verts_local[tail_end] - verts_local[tail_end - 1]
            tail_dir_norm = float(np.linalg.norm(tail_dir))
        else:
            tail_dir = np.array([1.0, 0.0, 0.0], dtype=float)
            tail_dir_norm = 1.0
        if tail_dir_norm < 1e-8:
            tail_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            tail_dir = tail_dir / tail_dir_norm

        for rev_idx, vtx_idx in enumerate(range(tail_end, tail_start - 1, -1)):
            desired_local = target_local - tail_dir * spacing * rev_idx
            current_local = verts_local[vtx_idx]
            blend = 1.0 if rev_idx <= 1 else (0.9 if rev_idx == 2 else max(0.55, 0.9 - 0.1 * rev_idx))
            local_target = current_local * (1.0 - blend) + desired_local * blend
            grid_rest = np.array([vtx_idx * spacing, 0.0, 0.0], dtype=float)
            local_offset = local_target - grid_rest
            self.sim.data.qpos[adr + vtx_idx * 3: adr + vtx_idx * 3 + 3] = local_offset

    def set_flex_attachment_enabled(self, enabled, endpoint_name=None, point_idx=None, body_name=None):
        """设置 flex 附着状态（延迟激活版本）。

        启用时只存储参数，等 _flex_pre_action_check 检测到夹爪闭合后才真正激活。
        禁用时立即清除所有状态。
        """
        if not bool(enabled):
            self._flex_grasp_active = False
            self._attach_pending = False
            self._attach_pending_params = {}
            self.attachment_enabled = False
        else:
            self._attach_pending = True
            self._attach_pending_params = {
                "endpoint_name": endpoint_name,
                "point_idx": point_idx,
                "body_name": body_name,
            }

    def _enforce_flex_inextensible(self):
        """Post-step correction: clamp adjacent vertex distances to rest length.

        MuJoCo flexcomp does not natively enforce edge lengths, so we correct
        stretching by projecting oversized edges back to rest length. The
        correction operates in body-local space and writes offsets to qpos
        (qpos = world_local_pos - grid_rest_pos).
        """
        nvert = self._flex_vertnum
        adr = self._flex_vertadr
        spacing = getattr(getattr(self, '_cable_in_task', None), 'flex_vertex_spacing', None) or 0.01
        rest_len = spacing

        # Get body transform
        obj_body_id = self.sim.model.body_name2id("object")
        body_pos = self.sim.data.body_xpos[obj_body_id]
        body_rot = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)

        # Read current vertex positions (world) and convert to body-local
        verts_world = self.sim.data.flexvert_xpos[adr:adr + nvert].copy().reshape(nvert, 3)
        verts_local = (body_rot.T @ (verts_world - body_pos).T).T  # (nvert, 3)

        # Grid rest positions: vertex i is at (i * spacing, 0, 0) in body-local
        grid_rest = np.zeros((nvert, 3), dtype=float)
        grid_rest[:, 0] = np.arange(nvert) * spacing

        # Current offsets from rest
        offsets = verts_local - grid_rest

        # Forward pass: clamp edges that are too long (toward endpoint)
        for i in range(nvert - 1):
            edge = offsets[i + 1] - offsets[i]
            edge_len = float(np.linalg.norm(edge))
            if edge_len > rest_len * 1.01 and edge_len > 1e-8:
                correction = edge * (1.0 - rest_len / edge_len) * 0.5
                offsets[i] += correction
                offsets[i + 1] -= correction

        # Backward pass: propagate corrections toward fixed end (vertex 0)
        for i in range(nvert - 2, -1, -1):
            edge = offsets[i + 1] - offsets[i]
            edge_len = float(np.linalg.norm(edge))
            if edge_len > rest_len * 1.01 and edge_len > 1e-8:
                correction = edge * (1.0 - rest_len / edge_len)
                offsets[i] += correction

        # Write corrected offsets to qpos
        for i in range(nvert):
            self.sim.data.qpos[adr + i * 3: adr + i * 3 + 3] = offsets[i]
