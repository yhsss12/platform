import type { CSSProperties } from 'react';
import type { TrainingBackendRequest, TrainingCapabilities } from '@/lib/api/trainingClient';

export const DOWNSTREAM_MODEL_TYPES = [
  'Robomimic',
  'ACT',
  'DT',
  'Diffusion Policy',
  'LeRobot',
  '自定义模型',
] as const;

export type DownstreamModelType = (typeof DOWNSTREAM_MODEL_TYPES)[number];

/** 内部能力标记 — 不直接展示给用户 */
export type TrainingTrainability = 'real' | 'placeholder';

export const TRAINING_FRAMEWORK_OPTIONS: {
  value: TrainingBackendRequest;
  label: string;
}[] = [
  { value: 'act', label: 'ACT Trainer' },
  { value: 'diffusion_policy', label: 'Diffusion Policy Trainer' },
  { value: 'robomimic_bc', label: 'Robomimic BC' },
  { value: 'torch_bc', label: 'PyTorch BC (torch_bc)' },
  { value: 'dt', label: 'Decision Transformer Trainer' },
];

export function hasTorchBcBackend(capabilities: TrainingCapabilities | null | undefined): boolean {
  return Boolean(capabilities?.supportedTrainingBackends?.includes('torch_bc'));
}

export function hasRobomimicBackend(capabilities: TrainingCapabilities | null | undefined): boolean {
  return Boolean(capabilities?.supportedTrainingBackends?.includes('robomimic_bc'));
}

export function hasDiffusionPolicyBackend(
  capabilities: TrainingCapabilities | null | undefined
): boolean {
  return Boolean(capabilities?.supportedTrainingBackends?.includes('diffusion_policy'));
}

export function hasActBackend(capabilities: TrainingCapabilities | null | undefined): boolean {
  return Boolean(capabilities?.supportedTrainingBackends?.includes('act'));
}

export function hasIsaacRobomimicBackend(
  capabilities: TrainingCapabilities | null | undefined
): boolean {
  return Boolean(capabilities?.supportedTrainingBackends?.includes('isaac_robomimic_bc'));
}

export function datasetSupportsHdf5(dataFormat?: string | null): boolean {
  if (!dataFormat) return false;
  return dataFormat.includes('HDF5');
}

export function datasetSupportsDiffusionPolicyTraining(options: {
  dataFormat?: string | null;
  isDualArm?: boolean;
}): boolean {
  if (options.isDualArm) return false;
  return datasetSupportsHdf5(options.dataFormat);
}

export function recommendDownstreamModelType(
  capabilities: TrainingCapabilities | null | undefined,
  dataFormat?: string | null,
  datasetModelFormat?: string | null
): DownstreamModelType {
  if (hasRobomimicBackend(capabilities) && datasetSupportsHdf5(dataFormat)) {
    return 'Robomimic';
  }
  if (datasetModelFormat && DOWNSTREAM_MODEL_TYPES.includes(datasetModelFormat as DownstreamModelType)) {
    return datasetModelFormat as DownstreamModelType;
  }
  return 'Robomimic';
}

export function recommendDataFormat(dataFormat?: string | null): string {
  if (!dataFormat) return 'HDF5';
  if (dataFormat.includes('HDF5')) return 'HDF5';
  return dataFormat;
}

export function defaultBackendForModelType(
  modelType: string,
  capabilities?: TrainingCapabilities | null,
  isDualArm?: boolean
): TrainingBackendRequest {
  if (isDualArm && hasTorchBcBackend(capabilities)) {
    return 'torch_bc';
  }
  switch (modelType) {
    case 'ACT':
      return 'act';
    case 'Diffusion Policy':
      return 'diffusion_policy';
    case 'DT':
      return 'dt';
    case 'Robomimic':
    default:
      return 'robomimic_bc';
  }
}

export function backendOptionsForDownstream(
  downstreamModelType: string,
  _capabilities: TrainingCapabilities | null | undefined
): { value: TrainingBackendRequest; label: string; disabled?: boolean }[] {
  void downstreamModelType;
  return TRAINING_FRAMEWORK_OPTIONS.map((item) => ({ ...item }));
}

export function resolveTrainingTrainability(
  downstreamModelType: string,
  capabilities: TrainingCapabilities | null | undefined,
  isDualArm?: boolean
): TrainingTrainability {
  if (isDualArm && hasTorchBcBackend(capabilities)) {
    return 'real';
  }
  if (downstreamModelType === 'Robomimic' && hasRobomimicBackend(capabilities)) {
    return 'real';
  }
  if (downstreamModelType === 'ACT' && hasActBackend(capabilities)) {
    return 'real';
  }
  if (downstreamModelType === 'Diffusion Policy' && hasDiffusionPolicyBackend(capabilities)) {
    return 'real';
  }
  return 'placeholder';
}

/** @deprecated 仅保留内部兼容，UI 不再使用 */
export function resolveTrainingCapabilityHint(
  downstreamModelType: string,
  capabilities: TrainingCapabilities | null | undefined
) {
  const trainability = resolveTrainingTrainability(downstreamModelType, capabilities);
  return {
    trainability,
    tone: 'neutral' as const,
    title: '训练配置',
    message: '提交后将创建训练任务，可在训练任务列表查看进度与结果。',
    buttonLabel: '创建训练任务',
    defaultBackend: defaultBackendForModelType(downstreamModelType),
  };
}

export function trainabilityLabel(_trainability: TrainingTrainability): string {
  return '标准训练流程';
}

export function unavailableDetailExplanation(
  modelType: string,
  _capabilities: TrainingCapabilities | null | undefined
): string[] {
  return [
    `模型类型：${modelType}`,
    '任务未能进入训练执行阶段，请检查训练配置与数据集资产后重试。',
    '如需帮助，请联系平台管理员查看训练日志。',
  ];
}

export function resolveTrainabilityFromJobStatus(status: {
  status: string;
  downstreamModelType?: string | null;
}): TrainingTrainability {
  if (status.status === 'backend_unavailable') return 'placeholder';
  if (['queued', 'running', 'completed', 'failed'].includes(status.status)) {
    if (status.downstreamModelType === 'Robomimic') return 'real';
    if (status.downstreamModelType === 'ACT') return 'real';
    if (status.downstreamModelType === 'Diffusion Policy') return 'real';
  }
  return 'placeholder';
}

export function hintPanelStyle(_tone: 'success' | 'warning' | 'neutral'): CSSProperties {
  return {
    padding: '10px 12px',
    borderRadius: 8,
    backgroundColor: '#f8fafc',
    border: '1px solid #e2e8f0',
    color: '#475569',
    fontSize: 13,
    lineHeight: 1.55,
  };
}
