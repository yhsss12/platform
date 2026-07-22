import type { ModelAsset } from '@/types/benchmark';
import type { TrainingJobModelAssetItem } from '@/lib/api/modelAssetsClient';
import { resolveModelAssetColumnLabel } from '@/lib/workspace/modelAssetDisplay';
import { resolveTaskTemplateIdFromModelAsset } from '@/lib/workspace/taskTemplateMapping';

export function formatModelEvaluationTaskName(assetLabel: string, date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const ts = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}`;
  const base = assetLabel.trim() || '模型';
  return `${base}_评测_${ts}`;
}

type ModelEvaluationAssetLike = Pick<
  ModelAsset,
  | 'id'
  | 'name'
  | 'displayName'
  | 'sourceTrainingJobId'
  | 'sourceDatasetId'
  | 'taskTemplateId'
  | 'modelType'
  | 'framework'
  | 'checkpointPath'
  | 'checkpointKind'
  | 'checkpointEpoch'
  | 'checkpointMetricName'
> & {
  datasetDisplayName?: string | null;
};

/** 从训练详情 / 模型资产详情跳转到评测中心并预填模型评测 */
export function buildModelEvaluationCreateFromAssetUrl(
  asset: ModelEvaluationAssetLike | TrainingJobModelAssetItem
): string {
  const params = new URLSearchParams({
    openCreate: '1',
    create: '1',
    type: 'model',
    modelAssetId: asset.id,
    modelAsset: asset.id,
    checkpointJobId: asset.sourceTrainingJobId,
  });

  const templateId = resolveTaskTemplateIdFromModelAsset(asset as ModelAsset);
  params.set('taskTemplateId', templateId);

  const label = resolveModelAssetColumnLabel(asset as ModelAsset);
  if (label) params.set('modelAssetName', label);
  if (asset.modelType) params.set('modelType', asset.modelType);
  if (asset.checkpointPath) params.set('checkpointPath', asset.checkpointPath);
  if (asset.sourceDatasetId) params.set('datasetId', asset.sourceDatasetId);
  if (asset.datasetDisplayName) params.set('datasetName', asset.datasetDisplayName);
  if (asset.taskTemplateId) params.set('taskConfigId', asset.taskTemplateId);

  const kind = (asset.checkpointKind ?? '').toLowerCase();
  if (kind === 'final') {
    params.set('checkpointKind', 'Final');
  } else if (kind === 'best') {
    const metric = asset.checkpointMetricName?.trim() || 'Loss';
    params.set('checkpointKind', `Best ${metric}`);
  } else if (kind === 'epoch' && asset.checkpointEpoch != null) {
    params.set('checkpointKind', `Epoch ${asset.checkpointEpoch}`);
  }

  return `/workspace/evaluation?${params.toString()}`;
}
