import type {
  ActTrainingAdvancedParams,
  DiffusionPolicyAdvancedParams,
  RobomimicAdvancedParams,
  TorchBcAdvancedParams,
  TrainingModelAdvancedParams,
  TrainingSeedMode,
} from '@/lib/mock/workspaceTrainingMock';

import type { TrainingPretrainedOptions } from '@/lib/mock/workspaceTrainingMock';

export interface StoredTrainingJobConfig {
  taskName?: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  seed?: number;
  seedMode?: TrainingSeedMode;
  advancedEnabled?: boolean;
  downstreamModelType?: string;
  trainingBackend?: string;
  modelParams?: TrainingModelAdvancedParams;
  device?: string;
  deviceLabel?: string;
  pretrained?: TrainingPretrainedOptions | null;
}

export function isTorchBcAdvancedParams(
  params: TrainingModelAdvancedParams | undefined,
  trainingBackend?: string | null
): params is TorchBcAdvancedParams {
  if (!params) return false;
  if (trainingBackend && trainingBackend !== 'torch_bc') return false;
  if ('actor_hidden_dims' in params || 'actor_hidden_dim_1' in params) return false;
  return 'hidden_dims' in params;
}

export function isRobomimicAdvancedParams(
  params: TrainingModelAdvancedParams | undefined,
  modelType?: string
): params is RobomimicAdvancedParams {
  if (!params) return false;
  if (modelType && modelType !== 'Robomimic') return false;
  return 'actor_hidden_dims' in params || 'actor_hidden_dim_1' in params;
}

export function isActAdvancedParams(
  params: TrainingModelAdvancedParams | undefined,
  modelType?: string
): params is ActTrainingAdvancedParams {
  if (!params) return false;
  if (modelType && modelType !== 'ACT') return false;
  return 'chunk_size' in params;
}

export function isDiffusionPolicyAdvancedParams(
  params: TrainingModelAdvancedParams | undefined,
  modelType?: string
): params is DiffusionPolicyAdvancedParams {
  if (!params) return false;
  if (modelType && modelType !== 'Diffusion Policy') return false;
  return 'num_inference_steps' in params;
}

export function formatSeedDisplay(seed?: number, _seedMode?: TrainingSeedMode): string {
  if (seed == null || Number.isNaN(seed)) return '—';
  return String(seed);
}

export interface TrainingConfigDisplayRow {
  label: string;
  value: string;
}

export function buildTrainingConfigDisplayRows(
  config: StoredTrainingJobConfig | null | undefined
): TrainingConfigDisplayRow[] {
  if (!config) return [];

  const rows: TrainingConfigDisplayRow[] = [];
  if (config.epochs != null) rows.push({ label: 'Epochs', value: String(config.epochs) });
  if (config.batchSize != null) rows.push({ label: 'Batch Size', value: String(config.batchSize) });
  if (config.learningRate != null) rows.push({ label: 'Learning Rate', value: String(config.learningRate) });
  if (config.seed != null) {
    rows.push({
      label: 'Seed',
      value: formatSeedDisplay(config.seed, config.seedMode),
    });
  }

  if (config.pretrained?.modelAssetId) {
    rows.push({
      label: '初始化权重',
      value: config.pretrained.modelAssetName || config.pretrained.modelAssetId,
    });
  }

  if (!config.advancedEnabled || !config.modelParams) {
    return rows;
  }

  const modelType = config.downstreamModelType;
  const trainingBackend = config.trainingBackend;
  const params = config.modelParams;

  if (isTorchBcAdvancedParams(params, trainingBackend)) {
    rows.push(
      { label: 'Hidden Dims', value: params.hidden_dims },
      { label: 'Weight Decay', value: String(params.weight_decay) }
    );
    return rows;
  }

  if (isRobomimicAdvancedParams(params, modelType)) {
    const hiddenDims =
      params.actor_hidden_dims ??
      ('actor_hidden_dim_1' in params
        ? `${(params as { actor_hidden_dim_1?: number }).actor_hidden_dim_1 ?? 512},${
            (params as { actor_hidden_dim_2?: number }).actor_hidden_dim_2 ?? 512
          }`
        : '512,512');
    rows.push(
      { label: 'Hidden Dims', value: hiddenDims },
      { label: 'Weight Decay', value: String(params.l2_regularization) }
    );
    return rows;
  }

  if (isActAdvancedParams(params, modelType)) {
    rows.push(
      { label: 'Chunk Size', value: String(params.chunk_size) },
      { label: 'Action Steps', value: String(params.n_action_steps) },
      { label: 'KL Weight', value: String(params.kl_weight) },
      { label: 'Latent Dim', value: String(params.latent_dim) },
      { label: 'Hidden Dim', value: String(params.hidden_dim) }
    );
    return rows;
  }

  if (isDiffusionPolicyAdvancedParams(params, modelType)) {
    rows.push(
      { label: 'N Obs Steps', value: String(params.n_obs_steps) },
      { label: 'Horizon', value: String(params.horizon) },
      { label: 'Action Steps', value: String(params.n_action_steps) },
      { label: 'Inference Steps', value: String(params.num_inference_steps) },
      { label: 'Use EMA', value: params.use_ema ? '是' : '否' },
      { label: 'EMA Decay', value: String(params.ema_decay) },
      { label: 'Weight Decay', value: String(params.weight_decay) },
      { label: 'Save Best', value: params.save_best ? '是' : '否' }
    );
  }

  return rows;
}

export function extractStoredTrainingJobConfig(
  metadata: Record<string, unknown> | null | undefined
): StoredTrainingJobConfig | null {
  if (!metadata) return null;
  const trainConfig = metadata.trainConfig;
  if (!trainConfig || typeof trainConfig !== 'object') return null;
  return trainConfig as StoredTrainingJobConfig;
}
