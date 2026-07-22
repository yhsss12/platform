export type EvaluationTypeKey = 'expert_policy' | 'model' | 'dataset';

export type EvaluationTypeLabel = '专家策略评测' | '模型评测' | '数据集评测';

const EVALUATION_TYPE_LABELS: Record<EvaluationTypeKey, EvaluationTypeLabel> = {
  expert_policy: '专家策略评测',
  model: '模型评测',
  dataset: '数据集评测',
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function pickString(...values: unknown[]): string {
  for (const value of values) {
    if (value == null) continue;
    const text = String(value).trim();
    if (text) return text;
  }
  return '';
}

function modeImpliesModel(mode: string): boolean {
  const n = mode.toLowerCase();
  return (
    n === 'trained_model_evaluation' ||
    n === 'model_evaluation' ||
    n === 'model' ||
    n === 'robomimic' ||
    n === 'robomimic_bc'
  );
}

function modeImpliesExpert(mode: string): boolean {
  const n = mode.toLowerCase();
  return (
    n === 'expert_policy_evaluation' ||
    n === 'expert_policy' ||
    n === 'expert' ||
    n === 'policy_evaluation' ||
    n === 'policy' ||
    n === 'episode_stability' ||
    n === 'scripted'
  );
}

function modeImpliesDataset(mode: string): boolean {
  const n = mode.toLowerCase();
  return (
    n === 'dataset_evaluation' ||
    n === 'dataset_offline' ||
    n === 'dataset_offline_evaluation' ||
    n === 'offline_dataset_evaluation' ||
    n === 'dataset'
  );
}

function objectImpliesType(evaluationObject: string): EvaluationTypeKey | null {
  const n = evaluationObject.toLowerCase();
  if (n === 'trained_model' || n === 'model') return 'model';
  if (n === 'dataset') return 'dataset';
  if (n === 'expert_policy' || n === 'expert') return 'expert_policy';
  return null;
}

export function normalizeEvaluationTypeKey(input: {
  evaluationType?: string | null;
  evaluationTypeLabel?: string | null;
  evaluationMode?: string | null;
  evaluationObject?: string | null;
  modelAssetId?: string | null;
  modelAssetName?: string | null;
  datasetId?: string | null;
  datasetName?: string | null;
  taskType?: string | null;
  runner?: string | null;
  taskName?: string | null;
  metadata?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
}): EvaluationTypeKey {
  const metadata = asRecord(input.metadata);
  const metrics = asRecord(input.metrics);
  const evalRequest = asRecord(metadata.evaluationRequest);

  const evaluationObject = pickString(
    input.evaluationObject,
    evalRequest.evaluationObject,
    metadata.evaluationObject,
    metrics.evaluationObject
  );
  const evaluationMode = pickString(
    input.evaluationMode,
    evalRequest.productEvaluationMode,
    evalRequest.evaluationMode,
    metadata.evaluationMode,
    metrics.evaluationMode
  );
  const modelAssetId = pickString(
    input.modelAssetId,
    evalRequest.modelAssetId,
    metadata.modelAssetId,
    metrics.modelAssetId
  );
  const modelAssetName = pickString(
    input.modelAssetName,
    evalRequest.modelAssetName,
    evalRequest.modelName,
    metadata.modelAssetName,
    metrics.modelAssetName,
    metrics.modelName
  );
  const datasetId = pickString(
    input.datasetId,
    evalRequest.datasetId,
    metadata.datasetId,
    metrics.datasetId
  );
  const datasetName = pickString(
    input.datasetName,
    evalRequest.datasetName,
    metadata.datasetName,
    metrics.datasetName
  );
  const taskType = pickString(input.taskType, evalRequest.taskType, metadata.taskType);
  const runner = pickString(input.runner, metadata.runner);
  const taskName = pickString(input.taskName, evalRequest.taskName, evalRequest.modelName);

  const explicitType = pickString(input.evaluationType, evalRequest.evaluationType, metadata.evaluationType);
  if (explicitType === 'expert_policy' || explicitType === 'model' || explicitType === 'dataset') {
    return explicitType;
  }

  const explicitLabel = pickString(
    input.evaluationTypeLabel,
    evalRequest.evaluationTypeLabel,
    metadata.evaluationTypeLabel
  );
  if (explicitLabel === '专家策略评测') return 'expert_policy';
  if (explicitLabel === '模型评测') return 'model';
  if (explicitLabel === '数据集评测') return 'dataset';

  const objectType = evaluationObject ? objectImpliesType(evaluationObject) : null;

  if (objectType === 'model' || modeImpliesModel(evaluationMode) || modelAssetId) {
    return 'model';
  }

  if (
    objectType === 'dataset' ||
    modeImpliesDataset(evaluationMode) ||
    taskType === 'dataset_offline' ||
    runner === 'dataset_offline_eval' ||
    /离线数据集评测/i.test(taskName) ||
    ((datasetId || datasetName) && !modelAssetId)
  ) {
    return 'dataset';
  }

  if (objectType === 'expert_policy' || modeImpliesExpert(evaluationMode) || /专家策略/.test(taskName)) {
    return 'expert_policy';
  }

  if (modelAssetName && !modeImpliesExpert(evaluationMode)) {
    return 'model';
  }

  return 'expert_policy';
}

export function normalizeEvaluationTypeLabel(input: {
  evaluationType?: string | null;
  evaluationTypeLabel?: string | null;
  evaluationMode?: string | null;
  evaluationObject?: string | null;
  modelAssetId?: string | null;
  modelAssetName?: string | null;
  datasetId?: string | null;
  datasetName?: string | null;
  taskType?: string | null;
  runner?: string | null;
  taskName?: string | null;
  metadata?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
}): EvaluationTypeLabel {
  return EVALUATION_TYPE_LABELS[normalizeEvaluationTypeKey(input)];
}

export function buildProductEvaluationFields(input: {
  evaluationModeApi: string;
  modelAssetId?: string | null;
  datasetId?: string | null;
  evaluationTopType?: 'model' | 'dataset';
}): {
  evaluationObject: string;
  productEvaluationMode: string;
  evaluationType: EvaluationTypeKey;
  evaluationTypeLabel: EvaluationTypeLabel;
} {
  if (input.evaluationTopType === 'dataset') {
    return {
      evaluationObject: 'dataset',
      productEvaluationMode: 'dataset_evaluation',
      evaluationType: 'dataset',
      evaluationTypeLabel: '数据集评测',
    };
  }

  const evaluationType = normalizeEvaluationTypeKey({
    evaluationMode: input.evaluationModeApi,
    modelAssetId: input.modelAssetId,
    datasetId: input.datasetId,
  });

  return {
    evaluationObject:
      evaluationType === 'model' ? 'trained_model' : evaluationType === 'dataset' ? 'dataset' : 'expert_policy',
    productEvaluationMode:
      evaluationType === 'model'
        ? 'model_evaluation'
        : evaluationType === 'dataset'
          ? 'dataset_evaluation'
          : 'expert_policy_evaluation',
    evaluationType,
    evaluationTypeLabel: EVALUATION_TYPE_LABELS[evaluationType],
  };
}

export const EVALUATION_TYPE_FILTER_OPTIONS = [
  '专家策略评测',
  '模型评测',
  '数据集评测',
] as const;
