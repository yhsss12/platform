// Historical snapshot retained from before the advanced torch-BC refactor.
import type {
  ActTrainingAdvancedParams,
  DiffusionPolicyAdvancedParams,
  RobomimicAdvancedParams,
  TrainingModelAdvancedParams,
  TrainingSeedMode,
} from '@/lib/mock/workspaceTrainingMock';

export interface StoredTrainingJobConfig {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  seed?: number;
  seedMode?: TrainingSeedMode;
  advancedEnabled?: boolean;
  downstreamModelType?: string;
  modelParams?: TrainingModelAdvancedParams;
  device?: string;
  deviceLabel?: string;
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

export function formatSeedDisplay(seed?: number, seedMode?: TrainingSeedMode): string {
  if (seed == null || Number.isNaN(seed)) return '—';
  if (seedMode === 'random') return `随机 (${seed})`;
  if (seedMode === 'manual') return `指定 (${seed})`;
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

  if (!config.advancedEnabled || !config.modelParams) {
    return rows;
  }

  const modelType = config.downstreamModelType;
  const params = config.modelParams;

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
      { label: 'L2 Regularization', value: String(params.l2_regularization) }
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
