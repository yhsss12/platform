import { normalizeTrainingBackendStatus } from '@/lib/workspace/trainingStatus';

export type TrainingLossLabelKind = 'current' | 'final' | 'last' | 'none';

const RUNNING_STATUSES = new Set(['running', 'training', 'pending', 'queued']);
const COMPLETED_STATUSES = new Set(['completed', 'succeeded', 'success']);
const FAILED_STATUSES = new Set(['failed', 'canceled', 'cancelled', 'error', 'backend_unavailable']);

export function resolveTrainingLossLabelKind(status?: string | null): TrainingLossLabelKind {
  const normalized = normalizeTrainingBackendStatus(status);
  if (RUNNING_STATUSES.has(normalized)) return 'current';
  if (COMPLETED_STATUSES.has(normalized)) return 'final';
  if (FAILED_STATUSES.has(normalized)) return 'last';
  return 'none';
}

export function resolveTrainingLossFieldLabel(status?: string | null): string {
  const kind = resolveTrainingLossLabelKind(status);
  switch (kind) {
    case 'current':
      return '当前 Loss';
    case 'final':
      return '最终 Loss';
    case 'last':
      return '最后 Loss';
    default:
      return 'Loss';
  }
}

/** @deprecated 使用 resolveTrainingLossFieldLabel */
export function resolveTrainingLossLabel(status?: string | null, _loss?: number | null): string {
  return resolveTrainingLossFieldLabel(status);
}

/** 模型资产默认来自已完成 checkpoint */
export function resolveModelAssetLossLabel(status?: string | null): string {
  const kind = resolveTrainingLossLabelKind(status ?? 'completed');
  if (kind === 'current') return '当前 Loss';
  if (kind === 'last') return '最后 Loss';
  return '最终 Loss';
}
