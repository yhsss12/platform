export type ModelAssetDisplayStatus = 'waiting' | 'ready' | 'generating' | 'superseded' | 'missing';

import { normalizeTrainingJobStatus } from '@/lib/workspace/trainingStatus';

const DISPLAY_STATUS_LABELS: Record<ModelAssetDisplayStatus, string> = {
  waiting: '等待生成',
  ready: '已就绪',
  generating: '生成中',
  superseded: '已替换',
  missing: '训练已完成，但未找到最终模型文件',
};

export function modelAssetDisplayStatusLabel(
  status?: ModelAssetDisplayStatus | string | null
): string {
  const key = String(status ?? 'waiting').toLowerCase() as ModelAssetDisplayStatus;
  return DISPLAY_STATUS_LABELS[key] ?? '等待生成';
}

export function isModelAssetReady(asset: { status?: string | null }): boolean {
  const status = String(asset.status ?? '').toLowerCase();
  return status === 'ready' || status === 'available' || status === 'active';
}

export function isTrainingJobInProgressForAssets(
  status?: string | null,
  options?: { currentEpoch?: number; totalEpochs?: number; progressPercent?: number | null }
): boolean {
  return normalizeTrainingJobStatus({
    backendStatus: status,
    currentEpoch: options?.currentEpoch,
    totalEpochs: options?.totalEpochs,
    progressPercent: options?.progressPercent,
  }).inProgress;
}

export interface ModelAssetEvalFields {
  status?: string | null;
  checkpointPath?: string | null;
  checkpointKind?: string | null;
  isPlaceholder?: boolean;
  canEvaluate?: boolean;
  canEvaluateReason?: string | null;
  displayStatus?: string | null;
  modelType?: string | null;
  framework?: string | null;
  evalExecutor?: string | null;
  controllerType?: string | null;
  actionDim?: number | null;
}

export function getModelAssetEvalDisabledReason(
  asset: ModelAssetEvalFields,
  options?: { jobInProgress?: boolean }
): string | null {
  if (options?.jobInProgress) {
    return '训练进行中，暂不可发起评测';
  }
  if (asset.isPlaceholder) {
    return '最终模型尚未生成';
  }
  const backendReason = String(asset.canEvaluateReason ?? '').trim();
  if (backendReason) return backendReason;

  const displayStatus = String(asset.displayStatus ?? '').toLowerCase();
  if (displayStatus === 'waiting' || displayStatus === 'generating') {
    return '模型资产生成中';
  }
  if (displayStatus === 'superseded') {
    return '该 checkpoint 已被更新的最佳模型替换';
  }
  if (displayStatus && displayStatus !== 'ready') {
    return `模型资产未就绪（${displayStatus}）`;
  }
  if (!asset.checkpointPath?.trim()) {
    return 'checkpoint 路径缺失';
  }
  if (!isModelAssetReady(asset)) {
    return '模型资产状态不是 ready';
  }

  const modelType = String(asset.modelType ?? asset.framework ?? '').toLowerCase();
  if (modelType === 'act' || modelType === 'diffusion_policy') {
    const evalExecutor = String(asset.evalExecutor ?? '').trim();
    const controllerType = String(asset.controllerType ?? '').trim();
    const actionDim = asset.actionDim;
    if (evalExecutor === 'joint_position' || controllerType === 'JOINT_POSITION') {
      if (!evalExecutor) return 'evalExecutor 缺失';
      if (!controllerType) return 'controllerType 缺失';
      if (actionDim == null) return 'actionDim 缺失';
    }
  }
  return '模型未就绪或缺少 checkpoint，暂无法发起评测';
}

export function canEvaluateModelAsset(
  asset: ModelAssetEvalFields,
  options?: { jobInProgress?: boolean }
): boolean {
  if (options?.jobInProgress) return false;
  if (asset.isPlaceholder) return false;
  if (typeof asset.canEvaluate === 'boolean') {
    return asset.canEvaluate;
  }
  return getModelAssetEvalDisabledReason(asset, options) === null;
}

export function sortModelAssetsForDisplay<
  T extends { checkpointKind?: string | null; checkpointEpoch?: number | null }
>(assets: T[]): T[] {
  const kindOrder: Record<string, number> = { final: 0, best: 1, epoch: 2 };
  return [...assets].sort((a, b) => {
    const aKind = String(a.checkpointKind ?? '').toLowerCase();
    const bKind = String(b.checkpointKind ?? '').toLowerCase();
    const orderDiff = (kindOrder[aKind] ?? 9) - (kindOrder[bKind] ?? 9);
    if (orderDiff !== 0) return orderDiff;
    return Number(a.checkpointEpoch ?? 0) - Number(b.checkpointEpoch ?? 0);
  });
}
