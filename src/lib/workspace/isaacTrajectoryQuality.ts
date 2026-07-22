/** Isaac 物块堆叠轨迹质量展示（统一入口，避免多页面逻辑分叉）。 */

export type TrajectoryQualitySeverity = 'passed' | 'mild' | 'motion' | 'failed';

const MOTION_WARNING_MARKERS = [
  'arm_action_delta',
  'rot_cmd_saturated',
  'wrist_flip',
  'joint_jump',
];

function isMotionAnomalyWarning(tag: string): boolean {
  const lower = tag.toLowerCase();
  return MOTION_WARNING_MARKERS.some((marker) => lower.includes(marker));
}

function isEpisodeLengthOnlyWarning(warnings: string[]): boolean {
  if (warnings.length === 0) return false;
  return warnings.every((w) => w.toLowerCase().includes('episode_length'));
}

export interface TrajectoryQualityDisplay {
  label: string;
  severity: TrajectoryQualitySeverity;
  description: string | null;
  recommendation: string | null;
}

export function resolveTrajectoryQualityDisplay(input: {
  qualityStatus?: string | null;
  qualityWarnings?: string[] | null;
  generationMode?: string | null;
  suspiciousWristFlipCount?: number | null;
  suspiciousJointJumpCount?: number | null;
  qualityDisplayLabel?: string | null;
}): TrajectoryQualityDisplay {
  const status = (input.qualityStatus ?? '').trim();
  const warnings = input.qualityWarnings ?? [];
  const wristFlips = input.suspiciousWristFlipCount ?? 0;
  const jointJumps = input.suspiciousJointJumpCount ?? 0;
  const mode = input.generationMode ?? '';

  if (status === 'failed') {
    return {
      label: input.qualityDisplayLabel ?? '未通过',
      severity: 'failed',
      description: '轨迹质量检查未通过，不建议用于训练。',
      recommendation: '请重新生成数据或调整策略参数后再次导出。',
    };
  }

  if (status === 'passed') {
    return {
      label: input.qualityDisplayLabel ?? '通过',
      severity: 'passed',
      description: '轨迹动作质量正常。',
      recommendation: null,
    };
  }

  if (status === 'warning') {
    const hasMotion =
      wristFlips > 0 ||
      jointJumps > 0 ||
      warnings.some((w) => isMotionAnomalyWarning(w));

    if (!hasMotion && (isEpisodeLengthOnlyWarning(warnings) || warnings.length === 0)) {
      const scriptedHint =
        mode === 'expert_policy' || mode === 'scripted_expert'
          ? '该数据集动作质量较稳定，但执行步数偏长，后续可继续优化策略效率。'
          : '轨迹动作质量正常，但执行步数较长，后续可优化策略效率。';
      return {
        label: input.qualityDisplayLabel ?? '可用（episode 较长）',
        severity: 'mild',
        description: '轨迹动作质量正常，但执行步数较长。',
        recommendation: scriptedHint,
      };
    }

    return {
      label: input.qualityDisplayLabel ?? '存在警告',
      severity: 'motion',
      description: '检测到部分轨迹存在动作突变、旋转饱和或机械臂姿态异常。',
      recommendation:
        '建议使用脚本专家策略重新生成或更换 seed demonstrations。',
    };
  }

  return {
    label: input.qualityDisplayLabel ?? (status || '—'),
    severity: 'passed',
    description: null,
    recommendation: null,
  };
}

/** @deprecated use resolveTrajectoryQualityDisplay */
export function resolveIsaacDatasetQualityAdvice(input: {
  generationMode?: string | null;
  qualityStatus?: string | null;
  qualityWarnings?: string[] | null;
  suspiciousWristFlipCount?: number | null;
  suspiciousJointJumpCount?: number | null;
}): string | null {
  const d = resolveTrajectoryQualityDisplay(input);
  return d.recommendation ?? d.description;
}

/** @deprecated use resolveTrajectoryQualityDisplay */
export function classifyTrajectoryQualityDisplay(input: {
  qualityStatus?: string | null;
  qualityWarnings?: string[] | null;
  suspiciousWristFlipCount?: number | null;
  suspiciousJointJumpCount?: number | null;
}): { tier: string; label: string; hint: string | null } {
  const d = resolveTrajectoryQualityDisplay(input);
  const tierMap: Record<TrajectoryQualitySeverity, string> = {
    passed: 'passed',
    mild: 'usable_long_episode',
    motion: 'motion_warning',
    failed: 'failed',
  };
  return {
    tier: tierMap[d.severity],
    label: d.label,
    hint: d.recommendation ?? d.description,
  };
}
