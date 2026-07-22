"""
轨迹质量检查与报告生成。

本模块对机器人执行轨迹（EEF 位置序列）进行质量检查，
检测可能的问题并生成结构化报告。

检查项目（分为 failure 和 warning 两个级别）：

  Failure 级别（直接判定为不合格）：
    - max_eef_step: 单步 EEF 位移过大（可能碰撞或控制异常）
    - max_eef_z: EEF 高度过高（可能脱离工作空间）
    - max_eef_accel_step: EEF 加速度过大（控制不稳定）
    - phase_target_jump: 阶段切换时目标跳变过大（规划不连续）
    - nonfinite_eef: EEF 位置包含 NaN/inf

  Warning 级别（提示但不直接失败）：
    - max_eef_step: 单步位移偏大
    - p95_eef_step: 95th percentile 位移偏大
    - target_stall_steps: 连续多步未接近目标（停滞）

数据来源支持：
  - NPZ 文件（numpy 压缩格式）
  - CSV 文件（人类可读格式）
  - 直接传入 numpy 数组
"""

import csv
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 默认阈值配置
# ---------------------------------------------------------------------------
# 每个阈值的命名格式：<指标名>_<fail|warn>
# fail = 超过则判定为不合格，warn = 超过则发出警告
DEFAULT_THRESHOLDS = {
    "max_eef_step_fail": 0.08,           # 单步最大位移 fail 阈值（米）
    "max_eef_z_above_table_fail": 0.45,  # EEF 高度超过桌面的 fail 阈值（米）
    "max_eef_accel_step_fail": 0.12,     # 单步加速度 fail 阈值（米/步^2）
    "phase_target_jump_fail": 0.10,      # 阶段切换目标跳变 fail 阈值（米）
    "max_eef_step_warn": 0.05,           # 单步最大位移 warn 阈值
    "p95_eef_step_warn": 0.035,          # 95th percentile 位移 warn 阈值
    "target_distance_warn": 0.04,        # EEF 到目标距离 warn 阈值（米）
    "target_stall_steps_warn": 20,       # 连续停滞步数 warn 阈值
}


# ---------------------------------------------------------------------------
# 序列化/反序列化辅助
# ---------------------------------------------------------------------------

def _jsonable(value):
    """将 numpy 类型转换为 JSON 可序列化的 Python 原生类型。

    递归处理 dict/list/tuple 中的嵌套 numpy 对象。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _load_json_scalar(value, default=None):
    """安全地将值解析为 JSON 对象。

    处理各种 numpy 标量类型和字符串编码。
    """
    try:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, dict):
            return value
    except Exception:
        return default
    return default


def _parse_step_metrics(values):
    """解析每步的指标值（可能是 JSON 字符串或 dict）。"""
    out = []
    for value in values:
        parsed = _load_json_scalar(value, default={})
        out.append(parsed if isinstance(parsed, dict) else {})
    return out


def infer_table_z(metadata=None, step_metrics=None, default=0.825):
    """从元数据或步指标中推断桌面高度。

    按优先级在 metadata 和 step_metrics 中查找以下 key：
      table_top_z > table_z > table_height_reference

    如果都找不到，返回 default 值。
    """
    metadata = metadata or {}
    for key in ("table_top_z", "table_z", "table_height_reference"):
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                pass
    for row in step_metrics or []:
        if not isinstance(row, dict):
            continue
        for key in ("table_top_z", "table_z", "table_height_reference"):
            if key in row:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
    return float(default)


# ---------------------------------------------------------------------------
# 内部分析辅助
# ---------------------------------------------------------------------------

def _episode_ranges(episode_lengths, total_steps):
    """将 episode 长度列表转换为 (episode_index, start, end) 的范围列表。

    用于将扁平化的轨迹数据切分回各个 episode。
    """
    lengths = np.asarray(episode_lengths, dtype=int).reshape(-1)
    if lengths.size == 0:
        lengths = np.asarray([total_steps], dtype=int)
    ranges = []
    start = 0
    for episode, length in enumerate(lengths.tolist()):
        end = start + int(length)
        ranges.append((episode, start, end))
        start = end
    if start != total_steps:
        raise ValueError(f"episode_lengths sum {start} does not match total steps {total_steps}")
    return ranges


def _max_consecutive(mask):
    """计算布尔掩码中最长连续 True 段的长度和位置。

    用于检测"停滞"（连续多步未接近目标）。

    Returns:
        (best_length, start_index, end_index)
        如果没有 True，start_index 和 end_index 为 -1。
    """
    best = 0
    current = 0
    end_index = -1
    for idx, value in enumerate(mask):
        if bool(value):
            current += 1
            if current > best:
                best = current
                end_index = idx
        else:
            current = 0
    start_index = end_index - best + 1 if best else -1
    return best, start_index, end_index


def _failure(kind, episode, step, phase, value, threshold):
    """构造一个 failure 记录。"""
    return {
        "kind": kind,
        "episode": int(episode),
        "step": int(step),
        "phase": str(phase),
        "value": float(value),
        "threshold": float(threshold),
    }


def _warning(kind, episode, step, phase, value, threshold):
    """构造一个 warning 记录（比 failure 多一个 severity 字段）。"""
    item = _failure(kind, episode, step, phase, value, threshold)
    item["severity"] = "warning"
    return item


# ---------------------------------------------------------------------------
# 核心报告生成
# ---------------------------------------------------------------------------

def trajectory_quality_report(
    *,
    eef_positions,
    eef_targets,
    phases,
    episode_lengths,
    table_z=None,
    thresholds=None,
):
    """生成轨迹质量报告（核心函数）。

    对每个 episode 逐一检查：
      1. 计算每步 EEF 位移（一阶差分）
      2. 计算每步加速度（二阶差分）
      3. 检测阶段切换时的目标跳变
      4. 检测连续停滞（未接近目标）
      5. 对所有指标应用 fail/warn 阈值

    Args:
        eef_positions: EEF 位置序列，shape (N, 3)。
        eef_targets:   EEF 目标位置序列，shape (N, 3)。
        phases:        每步的阶段标签，shape (N,)。
        episode_lengths: 各 episode 的长度列表。
        table_z:       桌面高度（用于 max_eef_z 检查）。
        thresholds:    自定义阈值（覆盖 DEFAULT_THRESHOLDS）。

    Returns:
        dict 包含：
          - passed: 是否通过（无 failure）
          - failures / warnings: 问题列表
          - episodes: 各 episode 的统计
          - 各种全局统计量
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    eef_positions = np.asarray(eef_positions, dtype=float)
    eef_targets = np.asarray(eef_targets, dtype=float)
    phases = np.asarray(phases, dtype=object).reshape(-1)
    if eef_positions.ndim != 2 or eef_positions.shape[1] != 3:
        raise ValueError(f"eef_positions must have shape (N, 3), got {eef_positions.shape}")
    if eef_targets.shape != eef_positions.shape:
        raise ValueError(f"eef_targets shape {eef_targets.shape} must match eef_positions {eef_positions.shape}")
    if phases.shape[0] != eef_positions.shape[0]:
        raise ValueError(f"phases length {phases.shape[0]} must match steps {eef_positions.shape[0]}")
    table_z = float(0.825 if table_z is None else table_z)

    failures = []
    warnings = []
    episodes = []
    # 全局统计量（跨所有 episode 取最大值）
    global_max_eef_step = 0.0
    global_p95_eef_step = 0.0
    global_max_eef_z = float(np.nanmax(eef_positions[:, 2])) if eef_positions.size else float("nan")
    global_max_eef_accel_step = 0.0
    global_max_phase_target_jump = 0.0

    for episode, start, end in _episode_ranges(episode_lengths, eef_positions.shape[0]):
        pos = eef_positions[start:end]
        targets = eef_targets[start:end]
        episode_phases = phases[start:end]
        if len(pos) == 0:
            failures.append(_failure("empty_episode", episode, start, "", 0.0, 1.0))
            continue

        # 检查是否有 NaN/inf
        finite = bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(targets)))
        if not finite:
            bad = int(np.where(~np.all(np.isfinite(pos), axis=1) | ~np.all(np.isfinite(targets), axis=1))[0][0])
            failures.append(_failure("nonfinite_eef", episode, start + bad, episode_phases[bad], 1.0, 0.0))

        # 计算一阶差分（步长）和二阶差分（加速度）
        eef_steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.asarray([], dtype=float)
        accel_steps = (
            np.linalg.norm(np.diff(pos, n=2, axis=0), axis=1) if len(pos) > 2 else np.asarray([], dtype=float)
        )
        # 检测阶段切换点
        phase_switch = np.asarray([str(episode_phases[idx]) != str(episode_phases[idx - 1]) for idx in range(1, len(pos))])
        target_jumps = np.linalg.norm(np.diff(targets, axis=0), axis=1) if len(targets) > 1 else np.asarray([], dtype=float)
        phase_target_jumps = target_jumps[phase_switch] if target_jumps.size else np.asarray([], dtype=float)

        # 统计量
        max_step = float(np.max(eef_steps)) if eef_steps.size else 0.0
        p95_step = float(np.percentile(eef_steps, 95)) if eef_steps.size else 0.0
        max_z = float(np.max(pos[:, 2]))
        max_accel = float(np.max(accel_steps)) if accel_steps.size else 0.0
        max_phase_jump = float(np.max(phase_target_jumps)) if phase_target_jumps.size else 0.0
        # 停滞检测：连续多步 EEF 距离目标超过阈值
        target_distances = np.linalg.norm(pos - targets, axis=1)
        stall_mask = target_distances > thresholds["target_distance_warn"]
        stall_count, stall_start, stall_end = _max_consecutive(stall_mask)

        # 更新全局统计
        global_max_eef_step = max(global_max_eef_step, max_step)
        global_p95_eef_step = max(global_p95_eef_step, p95_step)
        global_max_eef_accel_step = max(global_max_eef_accel_step, max_accel)
        global_max_phase_target_jump = max(global_max_phase_target_jump, max_phase_jump)

        # --- Failure 检查 ---
        if max_step > thresholds["max_eef_step_fail"]:
            idx = int(np.argmax(eef_steps)) + 1
            failures.append(_failure("max_eef_step", episode, start + idx, episode_phases[idx], max_step, thresholds["max_eef_step_fail"]))
        if max_z > table_z + thresholds["max_eef_z_above_table_fail"]:
            idx = int(np.argmax(pos[:, 2]))
            failures.append(
                _failure(
                    "max_eef_z",
                    episode,
                    start + idx,
                    episode_phases[idx],
                    max_z,
                    table_z + thresholds["max_eef_z_above_table_fail"],
                )
            )
        if max_accel > thresholds["max_eef_accel_step_fail"]:
            idx = int(np.argmax(accel_steps)) + 2
            failures.append(
                _failure("max_eef_accel_step", episode, start + idx, episode_phases[idx], max_accel, thresholds["max_eef_accel_step_fail"])
            )
        if max_phase_jump > thresholds["phase_target_jump_fail"]:
            switch_indices = np.where(phase_switch)[0] + 1
            idx = int(switch_indices[np.argmax(phase_target_jumps)])
            failures.append(
                _failure("phase_target_jump", episode, start + idx, episode_phases[idx], max_phase_jump, thresholds["phase_target_jump_fail"])
            )

        # --- Warning 检查 ---
        if max_step > thresholds["max_eef_step_warn"]:
            idx = int(np.argmax(eef_steps)) + 1 if eef_steps.size else 0
            warnings.append(_warning("max_eef_step", episode, start + idx, episode_phases[idx], max_step, thresholds["max_eef_step_warn"]))
        if p95_step > thresholds["p95_eef_step_warn"]:
            warnings.append(_warning("p95_eef_step", episode, start, episode_phases[0], p95_step, thresholds["p95_eef_step_warn"]))
        if stall_count >= int(thresholds["target_stall_steps_warn"]):
            warnings.append(
                _warning(
                    "target_not_reached_consecutive_steps",
                    episode,
                    start + stall_start,
                    episode_phases[stall_start],
                    stall_count,
                    thresholds["target_stall_steps_warn"],
                )
            )

        episodes.append(
            {
                "episode": int(episode),
                "start_step": int(start),
                "steps": int(end - start),
                "max_eef_step": max_step,
                "p95_eef_step": p95_step,
                "max_eef_z": max_z,
                "max_eef_accel_step": max_accel,
                "max_phase_target_jump": max_phase_jump,
                "max_consecutive_target_miss": int(stall_count),
            }
        )

    return {
        "passed": not failures,
        "table_z": table_z,
        "thresholds": thresholds,
        "num_episodes": len(episodes),
        "num_steps": int(eef_positions.shape[0]),
        "max_eef_step": float(global_max_eef_step),
        "p95_eef_step": float(global_p95_eef_step),
        "max_eef_z": float(global_max_eef_z),
        "max_eef_accel_step": float(global_max_eef_accel_step),
        "max_phase_target_jump": float(global_max_phase_target_jump),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# 从轨迹数据生成报告
# ---------------------------------------------------------------------------

def quality_report_from_trajectories(trajectories, *, metadata=None, table_z=None, thresholds=None):
    """从轨迹列表生成质量报告。

    每条轨迹是一个 step 字典的列表，每个 step 包含：
      - eef_pos: EEF 位置
      - eef_target: EEF 目标位置
      - phase: 阶段标签
      - metrics: 指标字典（可选）

    将多条轨迹拼接后调用 trajectory_quality_report。
    """
    eef_positions = np.concatenate(
        [np.stack([np.asarray(step.get("eef_pos", np.full(3, np.nan)), dtype=float) for step in traj], axis=0) for traj in trajectories],
        axis=0,
    )
    eef_targets = np.concatenate(
        [np.stack([np.asarray(step.get("eef_target", np.full(3, np.nan)), dtype=float) for step in traj], axis=0) for traj in trajectories],
        axis=0,
    )
    phases = np.concatenate([np.asarray([str(step.get("phase", "")) for step in traj], dtype=object) for traj in trajectories], axis=0)
    episode_lengths = np.asarray([len(traj) for traj in trajectories], dtype=np.int32)
    step_metrics = [step.get("metrics", {}) for traj in trajectories for step in traj]
    inferred_table_z = table_z if table_z is not None else infer_table_z(metadata=metadata, step_metrics=step_metrics)
    return trajectory_quality_report(
        eef_positions=eef_positions,
        eef_targets=eef_targets,
        phases=phases,
        episode_lengths=episode_lengths,
        table_z=inferred_table_z,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------

def load_npz_quality_inputs(path):
    """从 NPZ 文件加载轨迹质量检查所需的输入数据。

    NPZ 文件必须包含以下 key：
      - eef_positions: EEF 位置序列
      - eef_targets: EEF 目标位置序列
      - phases: 阶段标签序列
      - episode_lengths: 各 episode 的长度

    可选 key：
      - metadata: JSON 字符串或 dict
      - step_metrics: 每步的指标
    """
    path = Path(path).expanduser()
    data = np.load(path, allow_pickle=True)
    required = ("eef_positions", "eef_targets", "phases", "episode_lengths")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing trajectory quality fields: {', '.join(missing)}")
    metadata = _load_json_scalar(data["metadata"], default={}) if "metadata" in data else {}
    step_metrics = _parse_step_metrics(data["step_metrics"]) if "step_metrics" in data else []
    return {
        "eef_positions": np.asarray(data["eef_positions"], dtype=float),
        "eef_targets": np.asarray(data["eef_targets"], dtype=float),
        "phases": np.asarray(data["phases"], dtype=object),
        "episode_lengths": np.asarray(data["episode_lengths"], dtype=np.int32),
        "table_z": infer_table_z(metadata=metadata, step_metrics=step_metrics),
        "metadata": metadata,
    }


def _csv_value(row, names, default=np.nan):
    """从 CSV 行中按多个候选列名读取浮点值。

    支持不同的列命名约定（如 eef_pos_x / eef_x / robot0_eef_x）。
    """
    for name in names:
        if name in row and row[name] not in {"", None}:
            return float(row[name])
    return default


def load_csv_quality_inputs(path, table_z=None):
    """从 CSV 文件加载轨迹质量检查所需的输入数据。

    CSV 文件的列名支持多种约定：
      - 位置：eef_pos_x / eef_x / robot0_eef_x
      - 目标：eef_target_x / target_x
      - 阶段：phase
      - episode：episode（用于切分多条轨迹）
    """
    rows = list(csv.DictReader(Path(path).expanduser().open(newline="")))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    positions = []
    targets = []
    phases = []
    episode_ids = []
    for idx, row in enumerate(rows):
        positions.append(
            [
                _csv_value(row, ("eef_pos_x", "eef_x", "robot0_eef_x")),
                _csv_value(row, ("eef_pos_y", "eef_y", "robot0_eef_y")),
                _csv_value(row, ("eef_pos_z", "eef_z", "robot0_eef_z")),
            ]
        )
        targets.append(
            [
                _csv_value(row, ("eef_target_x", "target_x"), positions[-1][0]),
                _csv_value(row, ("eef_target_y", "target_y"), positions[-1][1]),
                _csv_value(row, ("eef_target_z", "target_z"), positions[-1][2]),
            ]
        )
        phases.append(row.get("phase", ""))
        episode_ids.append(int(float(row.get("episode", 0) or 0)))
    episode_lengths = [episode_ids.count(episode) for episode in sorted(set(episode_ids))]
    if not np.all(np.isfinite(np.asarray(positions, dtype=float))):
        raise ValueError("CSV must contain finite eef position columns")
    return {
        "eef_positions": np.asarray(positions, dtype=float),
        "eef_targets": np.asarray(targets, dtype=float),
        "phases": np.asarray(phases, dtype=object),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int32),
        "table_z": float(0.825 if table_z is None else table_z),
        "metadata": {},
    }


def load_quality_inputs(path, table_z=None):
    """根据文件扩展名自动选择加载方式（NPZ 或 CSV）。"""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npz":
        inputs = load_npz_quality_inputs(path)
        if table_z is not None:
            inputs["table_z"] = float(table_z)
        return inputs
    if path.suffix.lower() == ".csv":
        return load_csv_quality_inputs(path, table_z=table_z)
    raise ValueError(f"Unsupported trajectory file type: {path.suffix}")


# ---------------------------------------------------------------------------
# 报告 I/O
# ---------------------------------------------------------------------------

def report_from_file(path, *, table_z=None, thresholds=None):
    """从文件加载轨迹数据并生成质量报告（一步到位的便捷函数）。"""
    inputs = load_quality_inputs(path, table_z=table_z)
    return trajectory_quality_report(
        eef_positions=inputs["eef_positions"],
        eef_targets=inputs["eef_targets"],
        phases=inputs["phases"],
        episode_lengths=inputs["episode_lengths"],
        table_z=inputs["table_z"],
        thresholds=thresholds,
    )


def write_quality_json(path, report):
    """将质量报告写入 JSON 文件。"""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")


def write_quality_csv(path, report):
    """将质量报告的 failures 和 warnings 写入 CSV 文件。

    CSV 格式便于在 Excel 或数据分析工具中查看。
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("severity", "kind", "episode", "step", "phase", "value", "threshold")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("failures", []):
            writer.writerow({"severity": "failure", **{key: row.get(key) for key in fieldnames if key != "severity"}})
        for row in report.get("warnings", []):
            writer.writerow({"severity": "warning", **{key: row.get(key) for key in fieldnames if key != "severity"}})
