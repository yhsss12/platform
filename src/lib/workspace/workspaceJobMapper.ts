import type { WorkspaceJobSummary, WorkspaceArtifactItem } from '@/lib/api/workspaceJobClient';
import type { EvaluationJobListItem, EvaluationSuccessStats } from '@/lib/api/evaluationClient';
import {
  resolveEvaluationJobId,
} from '@/lib/workspace/evaluationJobId';
import type { TrainingJobListItem } from '@/lib/api/trainingClient';
import type { WorkspaceDataItem, WorkspaceDataStatus } from '@/lib/mock/workspaceDataMock';
import type { EvaluationTaskRow, EvaluationTaskStatus } from '@/lib/mock/workspaceEvaluationRecordsMock';
import type { TrainingTaskRow, TrainingTaskStatus } from '@/lib/mock/workspaceTrainingMock';
import { CABLE_THREADING_TASK_NAME } from '@/lib/workspace/cableThreading';
import { DUAL_ARM_CABLE_TASK_NAME } from '@/lib/workspace/dualArmCable';
import { buildDualArmEvalResultSummary } from '@/lib/workspace/dualArmEvaluation';
import { ISAAC_BLOCK_STACKING_DISPLAY_NAME } from '@/lib/workspace/isaacBlockStacking';
import {
  normalizeEvaluationTypeLabel,
  normalizeEvaluationTypeKey,
  type EvaluationTypeKey,
  type EvaluationTypeLabel,
} from '@/lib/workspace/evaluationType';
import {
  formatTaskTemplateDisplayName,
  normalizeTaskDisplayName,
  resolveTemplateIdFromBackendTaskType,
} from '@/lib/workspace/taskTemplateMapping';

function pickNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}
import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
} from '@/lib/workspace/taskDisplayNames';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import {
  mapTrainingStatusToDisplay,
  normalizeTrainingJobStatus,
  trainingProgressPercent,
} from '@/lib/workspace/trainingStatus';
import { resolveTrainingTaskDisplayName } from '@/lib/workspace/trainingDisplay';
import {
  isInvalidDatasetDisplayName,
  normalizeDatasetDisplayName,
} from '@/lib/workspace/datasetNaming';
import { parseMetricsLossHistory } from '@/lib/workspace/trainingLossSeries';

function pickFirstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (value == null) continue;
    const trimmed = String(value).trim();
    if (trimmed) return trimmed;
  }
  return null;
}

const EVALUATION_RELATED_TASK_LABELS = new Set([
  CABLE_THREADING_TASK_NAME,
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_TASK_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
  '物块堆叠',
  'block_stacking',
  'isaac_block_stacking',
]);

function resolveEvaluationRelatedTaskLabel(params: {
  taskType?: string | null;
  taskTemplateId?: string | null;
  metrics?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}): string {
  const metrics = params.metrics ?? {};
  const metadata = params.metadata ?? {};
  const templateId = pickFirstString(
    params.taskTemplateId,
    metadata.taskTemplateId,
    metrics.taskTemplateId,
    resolveTemplateIdFromBackendTaskType(String(params.taskType ?? ''))
  );
  const fromTemplate = formatTaskTemplateDisplayName(templateId);
  if (fromTemplate) return fromTemplate;
  const taskType = String(params.taskType ?? '');
  if (taskType === 'cable_threading') return CABLE_THREADING_DISPLAY_NAME;
  if (taskType === 'dual_arm_cable_manipulation') return DUAL_ARM_CABLE_DISPLAY_NAME;
  if (taskType === 'block_stacking' || taskType === 'isaac_block_stacking') {
    return ISAAC_BLOCK_STACKING_DISPLAY_NAME;
  }
  return taskType || '—';
}

function resolveEvaluationModelDisplayName(params: {
  modelName?: string | null;
  taskName?: string | null;
  name?: string | null;
  title?: string | null;
  metrics?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  config?: Record<string, unknown>;
  modelEvaluationConfig?: Record<string, unknown>;
  relatedTaskLabel: string;
}): string {
  const metrics = params.metrics ?? {};
  const metadata = params.metadata ?? {};
  const config = params.config ?? {};
  const modelEvaluationConfig = params.modelEvaluationConfig ?? {};
  const evaluationRequest =
    metadata.evaluationRequest && typeof metadata.evaluationRequest === 'object'
      ? (metadata.evaluationRequest as Record<string, unknown>)
      : {};
  const userTaskName = pickFirstString(
    params.taskName,
    metadata.taskName,
    metrics.taskName,
    evaluationRequest.taskName,
    params.name,
    metadata.name
  );
  if (userTaskName) return userTaskName;

  const explicit = pickFirstString(
    params.modelName,
    metadata.modelName,
    metrics.modelName,
    config.modelName,
    modelEvaluationConfig.modelName,
    metrics.name,
    params.title,
    metadata.title,
    metrics.title
  );
  if (explicit) return explicit;

  const templateDisplayName = pickFirstString(
    metadata.templateDisplayName,
    metadata.displayName,
    metrics.displayName
  );
  if (templateDisplayName) return templateDisplayName;

  const rawTaskName = pickFirstString(params.taskName, metadata.taskName, metrics.taskName);
  if (rawTaskName) {
    if (!EVALUATION_RELATED_TASK_LABELS.has(rawTaskName) && rawTaskName !== params.relatedTaskLabel) {
      return rawTaskName;
    }
    if (/评测_|稳定性评测|rollout/i.test(rawTaskName)) {
      return rawTaskName;
    }
  }

  return rawTaskName && !EVALUATION_RELATED_TASK_LABELS.has(rawTaskName) ? rawTaskName : '—';
}

function formatIsoToLabel(iso?: string | null): string {
  return formatDateTimeMinuteYmdSlash(iso);
}

function pickFirstTimestamp(...values: (string | null | undefined)[]): string | null {
  for (const value of values) {
    if (value == null) continue;
    const trimmed = String(value).trim();
    if (trimmed) return trimmed;
  }
  return null;
}

export function formatEvaluationCreatedAt(fields: {
  createdAt?: string | null;
  created_at?: string | null;
  submittedAt?: string | null;
  submitted_at?: string | null;
  startedAt?: string | null;
  started_at?: string | null;
  updatedAt?: string | null;
  updated_at?: string | null;
  finishedAt?: string | null;
  finished_at?: string | null;
  completedAt?: string | null;
  completed_at?: string | null;
}): string {
  const iso = pickFirstTimestamp(
    fields.createdAt,
    fields.created_at,
    fields.submittedAt,
    fields.submitted_at,
    fields.startedAt,
    fields.started_at,
    fields.updatedAt,
    fields.updated_at,
    fields.finishedAt,
    fields.finished_at,
    fields.completedAt,
    fields.completed_at
  );
  return formatIsoToLabel(iso);
}

export function evaluationListItemSortIso(item: EvaluationJobListItem): string {
  const metrics = item.metrics ?? {};
  return (
    pickFirstTimestamp(
      item.createdAt,
      item.startedAt,
      item.updatedAt,
      item.finishedAt,
      metrics.completedAt as string | undefined,
      metrics.createdAt as string | undefined
    ) ?? ''
  );
}

export function workspaceEvaluationJobSortIso(job: WorkspaceJobSummary): string {
  const metrics = job.metricsSummary ?? {};
  return (
    pickFirstTimestamp(
      job.createdAt,
      job.startedAt,
      job.updatedAt,
      job.finishedAt,
      metrics.completedAt as string | undefined,
      metrics.createdAt as string | undefined
    ) ?? ''
  );
}

function mapGenerateStatus(status: string): WorkspaceDataStatus {
  switch (status) {
    case 'completed':
      return 'completed';
    case 'failed':
    case 'canceled':
      return 'failed';
    case 'running':
    case 'pending':
    case 'queued':
      return 'generating';
    default:
      return 'pending';
  }
}

function mapEvaluationStatus(status: string): EvaluationTaskStatus {
  switch (status) {
    case 'completed':
    case 'succeeded':
      return '已完成';
    case 'failed':
    case 'canceled':
    case 'cancelled':
    case 'stale':
      return '失败';
    case 'running':
    case 'evaluating':
      return '评测中';
    case 'pending':
    case 'queued':
    case 'draft':
      return '待评测';
    default:
      return '待评测';
  }
}

function mapTrainingStatus(status: string): TrainingTaskStatus {
  return mapTrainingStatusToDisplay(status);
}

function taskNameForJob(job: WorkspaceJobSummary): string {
  if (job.taskName) return normalizeTaskDisplayName(job.taskName);
  if (job.taskType === 'cable_threading') return CABLE_THREADING_TASK_NAME;
  if (job.taskType === 'dual_arm_cable_manipulation') return DUAL_ARM_CABLE_TASK_NAME;
  return job.taskType;
}

function cableThreadingGenerateArtifactPaths(runtimePath: string) {
  const root = runtimePath.replace(/\/$/, '');
  return {
    npzPath: `${root}/datasets/dataset.npz`,
    hdf5Path: `${root}/datasets/dataset.hdf5`,
    manifestPath: `${root}/datasets/dataset.manifest.json`,
    collectCsvPath: `${root}/results/collect.csv`,
    failuresPath: `${root}/results/failures.json`,
    generateVideoPath: `${root}/videos/generate.mp4`,
  };
}

export function workspaceGenerateJobToDataItem(job: WorkspaceJobSummary): WorkspaceDataItem {
  const metrics = job.metricsSummary ?? {};
  const episodes = Number(
    metrics.episodes ??
      metrics.total_episodes ??
      metrics.totalEpisodes ??
      metrics.numEpisodes ??
      metrics.max_cables ??
      metrics.maxCables ??
      0
  );
  const successful = Number(
    metrics.successfulEpisodes ??
      metrics.success_episodes ??
      metrics.num_cables_succeeded ??
      metrics.numCablesSucceeded ??
      metrics.num_successful ??
      metrics.success ??
      0
  );
  const frameCount = Number(
    metrics.frameCount ?? metrics.savedFrameCount ?? metrics.totalFrames ?? metrics.numFrames ?? 0
  );
  const sizeBytes = Number(
    metrics.sizeBytes ??
      metrics.size_bytes ??
      metrics.storageBytes ??
      metrics.totalBytes ??
      metrics.dataSizeBytes ??
      0
  );
  const finalSuccessRateRaw =
    metrics.finalSuccessRate != null ? Number(metrics.finalSuccessRate) : null;
  const successRateRaw =
    metrics.successRate != null ? Number(metrics.successRate) : null;
  const successRate =
    finalSuccessRateRaw != null && Number.isFinite(finalSuccessRateRaw)
      ? finalSuccessRateRaw * (finalSuccessRateRaw <= 1 ? 100 : 1)
      : successRateRaw != null && Number.isFinite(successRateRaw)
        ? successRateRaw * (successRateRaw <= 1 ? 100 : 1)
        : undefined;

  const trajectoryCount = successful > 0 ? successful : episodes > 0 ? episodes : undefined;
  const dataVolume =
    successful > 0 && episodes > 0 && successful !== episodes
      ? `${successful}/${episodes} 条`
      : trajectoryCount != null
        ? `${trajectoryCount} 条`
        : episodes > 0
          ? `${episodes} 条`
          : '—';

  const manifestAvailable = (job.artifactCounts?.manifest ?? 0) > 0;
  const taskName = taskNameForJob(job);
  const isCableThreadingGenerate =
    job.taskType === 'cable_threading' && Boolean(job.runtimePath?.trim());
  const cableArtifacts = isCableThreadingGenerate
    ? cableThreadingGenerateArtifactPaths(job.runtimePath)
    : undefined;

  return {
    id: job.jobId,
    name: taskName,
    taskId: job.jobId,
    taskName,
    simulationId: job.jobId,
    dataCategory: '真实数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume,
    size: sizeBytes > 0 ? formatDataScaleBytes(sizeBytes) : '—',
    status: mapGenerateStatus(job.status),
    generatedAt: formatIsoToLabel(job.createdAt),
    creator: '平台用户',
    taskType:
      job.taskType === 'cable_threading' || job.taskType === 'dual_arm_cable_manipulation'
        ? job.taskType
        : undefined,
    jobId: job.jobId,
    backendJobId: job.jobId,
    backendJobStatus: job.status,
    successRate,
    successfulEpisodes: successful || undefined,
    trajectoryCount,
    successTrajectoryCount: successful || undefined,
    episodeCount: episodes || undefined,
    totalEpisodes: episodes || undefined,
    frameCount: frameCount > 0 ? frameCount : undefined,
    sizeBytes: sizeBytes > 0 ? sizeBytes : undefined,
    generateVideoExists: job.videoAvailable,
    generateVideoPath: cableArtifacts?.generateVideoPath,
    npzPath: cableArtifacts?.npzPath,
    hdf5Path: cableArtifacts?.hdf5Path,
    manifestPath: cableArtifacts?.manifestPath,
    collectCsvPath: cableArtifacts?.collectCsvPath,
    failuresPath: cableArtifacts?.failuresPath,
    saveTrajectory: isCableThreadingGenerate ? true : undefined,
    datasetBuildSupported: job.taskType === 'cable_threading',
    isDatasetAsset: false,
    datasetBuildStatus: manifestAvailable ? 'built' : 'none',
    datasetManifestPath: cableArtifacts?.manifestPath,
  };
}

function formatDataScaleBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function resolveEvaluationTypeFields(input: {
  evaluationType?: string | null;
  evaluationTypeLabel?: string | null;
  evaluationObject?: string | null;
  evaluationModeApi?: string | null;
  modelAssetId?: string | null;
  datasetId?: string | null;
  datasetName?: string | null;
  taskType?: string | null;
  taskName?: string | null;
  runner?: string | null;
  metadata?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
}): {
  evaluationType: EvaluationTypeKey;
  evaluationTypeLabel: EvaluationTypeLabel;
  evaluationObject: string;
} {
  const typeInput = {
    evaluationType: input.evaluationType,
    evaluationTypeLabel: input.evaluationTypeLabel,
    evaluationObject: input.evaluationObject,
    evaluationMode: input.evaluationModeApi,
    modelAssetId: input.modelAssetId,
    datasetId: input.datasetId,
    datasetName: input.datasetName,
    taskType: input.taskType,
    taskName: input.taskName,
    runner: input.runner,
    metadata: input.metadata,
    metrics: input.metrics,
  };
  const evaluationType = normalizeEvaluationTypeKey(typeInput);
  const evaluationTypeLabel = normalizeEvaluationTypeLabel(typeInput);
  const evaluationObject =
    input.evaluationObject ??
    (evaluationType === 'model'
      ? 'trained_model'
      : evaluationType === 'dataset'
        ? 'dataset'
        : 'expert_policy');
  return { evaluationType, evaluationTypeLabel, evaluationObject };
}

export function workspaceEvaluationJobToRow(job: WorkspaceJobSummary): EvaluationTaskRow {
  const metrics = job.metricsSummary ?? {};
  const meta = (job as { metadata?: Record<string, unknown> }).metadata ?? {};
  const isDualArm = job.taskType === 'dual_arm_cable_manipulation';
  const isDatasetOffline =
    job.runner === 'dataset_offline_eval' || /离线数据集评测/i.test(String(job.taskName ?? ''));
  const datasetNameFromTask = (() => {
    const match = String(job.taskName ?? '').match(/离线数据集评测\s*[·•]\s*(.+)/);
    return match?.[1]?.trim() ?? String(metrics.datasetName ?? '');
  })();
  const evaluationModeRaw = String(meta.evaluationMode ?? metrics.evaluationMode ?? metrics.policy ?? '');
  const isTrainedModel =
    evaluationModeRaw === 'trained_model_evaluation' ||
    evaluationModeRaw === 'robomimic' ||
    Boolean(metrics.modelAssetId || metrics.checkpointPath);
  const evaluationModeApi = (() => {
    const raw = String(meta.evaluationMode ?? metrics.evaluationMode ?? '').trim() || evaluationModeRaw;
    if (raw === 'robomimic') return 'trained_model_evaluation';
    if (raw === 'scripted') return 'expert_policy_evaluation';
    if (raw === 'policy_evaluation') return 'expert_policy_evaluation';
    return raw || undefined;
  })();
  const evalMode: EvaluationTaskRow['evaluationMode'] = isDatasetOffline
    ? '数据过程评测'
    : isDualArm
      ? 'episode 稳定性评测'
      : '策略评测';
  const successRateRaw = metrics.successRate ?? metrics.success_rate;
  const successRate =
    successRateRaw != null && job.status === 'completed' ? Number(successRateRaw) : null;
  const relatedTask = isDatasetOffline
    ? '—'
    : resolveEvaluationRelatedTaskLabel({
        taskType: job.taskType,
        metrics,
      });
  const displayName = isDatasetOffline
    ? String(job.taskName ?? `离线数据集评测 · ${datasetNameFromTask || '—'}`)
    : resolveEvaluationModelDisplayName({
        modelName: metrics.modelName as string | undefined,
        taskName: job.taskName,
        metadata: meta,
        metrics,
        relatedTaskLabel: relatedTask,
      });
  const episodeCount = Number(
    metrics.numEpisodes ?? metrics.episodes ?? metrics.totalEpisodes ?? 1
  );
  const modelType: EvaluationTaskRow['modelType'] = isDatasetOffline
    ? '—'
    : isTrainedModel
      ? '已训练模型'
      : isDualArm && !isTrainedModel
        ? '专家策略'
        : job.taskType === 'cable_threading'
          ? isTrainedModel
            ? '已训练模型'
            : '专家策略'
          : '—';
  const checkpoint = isTrainedModel
    ? String(metrics.modelAssetId ?? metrics.checkpointPath ?? '—')
    : isDatasetOffline
      ? '—'
      : 'scripted';

  const evalJobId = resolveEvaluationJobId({
    evalJobId: job.jobId,
    jobId: job.jobId,
    runtimePath: job.runtimePath,
  });

  const typeFields = resolveEvaluationTypeFields({
    evaluationModeApi,
    modelAssetId: isTrainedModel ? String(metrics.modelAssetId ?? metrics.checkpointPath ?? '') : null,
    datasetId: isDatasetOffline ? String(metrics.datasetId ?? '') : null,
    datasetName: datasetNameFromTask,
    taskType: job.taskType,
    taskName: job.taskName,
    runner: job.runner,
    metadata: meta,
    metrics,
  });

  return {
    id: evalJobId || job.jobId || '',
    evalJobId: evalJobId || undefined,
    jobId: evalJobId || undefined,
    name: job.taskName ? String(job.taskName) : displayName,
    taskName: job.taskName ? String(job.taskName) : displayName,
    source: job.source === 'demo' ? 'demo' : 'real',
    evaluationMode: evalMode,
    evaluationModeApi,
    evaluationType: typeFields.evaluationType,
    evaluationTypeLabel: typeFields.evaluationTypeLabel,
    evaluationObject: typeFields.evaluationObject,
    relatedTask,
    checkpoint,
    modelType,
    dataVolume: isDualArm ? `${episodeCount} episode` : `${episodeCount || '—'} 条`,
    evalBackend: job.jobId.startsWith('isaac_eval_') ? 'Isaac Lab' : 'MuJoCo',
    evalRounds: episodeCount,
    status: mapEvaluationStatus(job.status),
    successRate,
    rawName: typeof job.taskName === 'string' ? job.taskName : undefined,
    createdAtIso: job.createdAt ?? undefined,
    createdAt: formatEvaluationCreatedAt({
      createdAt: job.createdAt,
      startedAt: job.startedAt,
      updatedAt: job.updatedAt,
      finishedAt: job.finishedAt,
      completedAt: metrics.completedAt as string | undefined,
    }),
    updatedAt: formatIsoToLabel(job.updatedAt),
    startedAt: formatIsoToLabel(job.startedAt),
    finishedAt: formatIsoToLabel(job.finishedAt),
    runner: job.runner ?? undefined,
    runtimePath: job.runtimePath,
    dataName: isDatasetOffline ? datasetNameFromTask || undefined : undefined,
    datasetId: isDatasetOffline ? String(metrics.datasetId ?? '') || undefined : undefined,
    resultSummary: isDualArm
      ? buildDualArmEvalResultSummary({
          jobStatus: job.status,
          evaluationMode: evaluationModeApi,
          message: typeof job.metricsSummary?.message === 'string'
            ? job.metricsSummary.message
            : typeof job.metricsSummary?.runtimeHealthReason === 'string'
              ? job.metricsSummary.runtimeHealthReason
              : undefined,
          metrics,
        })
      : job.status === 'completed'
        ? '评测已完成，可查看报告与回放。'
        : job.status === 'failed'
          ? '评测失败'
          : '评测运行中…',
    taskType:
      job.taskType === 'cable_threading' || job.taskType === 'dual_arm_cable_manipulation'
        ? job.taskType
        : undefined,
    backendJobStatus: job.status,
    evalVideoExists: job.videoAvailable,
    videoExists: job.videoAvailable,
    aggregate: metrics.aggregate as Record<string, unknown> | undefined,
    successStats: fallbackSuccessStats(
      metrics.successStats as EvaluationSuccessStats | undefined
    ),
  };
}

export function workspaceTrainingJobToRow(job: WorkspaceJobSummary): TrainingTaskRow {
  const metrics = job.metricsSummary ?? {};
  const meta = (job as { metadata?: Record<string, unknown> }).metadata ?? {};
  const checkpointExists =
    (job.artifactCounts?.checkpoint ?? 0) > 0 ||
    Boolean(metrics.checkpointPath || metrics.modelAssetId) ||
    (job.status === 'completed' && (job.artifactCounts?.metrics ?? 0) > 0);
  const hasModelManifest =
    (job.artifactCounts?.manifest ?? 0) > 0 ||
    (job.artifactCounts?.metrics ?? 0) > 0 ||
    Boolean(metrics.modelAssetId);
  const datasetName = String(meta.datasetName ?? metrics.datasetName ?? '');
  const datasetId = String(meta.datasetId ?? metrics.datasetId ?? '');
  const datasetManifestPath = String(
    meta.datasetManifestPath ?? metrics.datasetManifestPath ?? ''
  );
  const downstream = String(meta.downstreamModelType ?? metrics.downstreamModelType ?? '');
  const epoch = Number(metrics.epoch ?? 0);
  const totalEpochs = Number(metrics.totalEpochs ?? metrics.total_epochs ?? 0);
  const progressPercent = trainingProgressPercent({
    backendStatus: job.status,
    epoch,
    totalEpochs,
  });
  const normalized = normalizeTrainingJobStatus({
    backendStatus: job.status,
    currentEpoch: epoch,
    totalEpochs,
    progressPercent,
    checkpointExists,
  });
  const trainConfig =
    meta.trainConfig && typeof meta.trainConfig === 'object'
      ? (meta.trainConfig as Record<string, unknown>)
      : null;
  const taskDisplayName = resolveTrainingTaskDisplayName({
    taskName: job.taskName,
    metaTaskName: typeof meta.taskName === 'string' ? meta.taskName : null,
    trainConfigTaskName:
      typeof trainConfig?.taskName === 'string' ? trainConfig.taskName : null,
    datasetName: datasetName || null,
    trainingBackend: String(meta.trainingBackend ?? metrics.trainingBackend ?? ''),
    modelType: downstream || null,
    jobId: job.jobId,
  });

  return {
    id: job.jobId,
    trainJobId: job.jobId,
    source: job.source === 'demo' ? 'demo' : 'real',
    name: taskDisplayName,
    relatedTask: datasetName || taskDisplayName,
    modelType: downstream || 'unknown',
    dataset: datasetId || '—',
    datasetName: datasetName || undefined,
    datasetManifestPath: datasetManifestPath || undefined,
    dataVolume: '—',
    status: normalized.displayStatus,
    backendStatus: normalized.backendStatus,
    taskType: job.taskType,
    runner: job.runner ?? undefined,
    runtimePath: job.runtimePath,
    trainingBackend: String(meta.trainingBackend ?? metrics.trainingBackend ?? ''),
    dataFormat: String(meta.dataFormat ?? metrics.dataFormat ?? trainConfig?.dataFormat ?? ''),
    deviceLabel: String(
      metrics.trainingNodeDisplayName ?? metrics.deviceLabel ?? meta.trainingNodeDisplayName ?? meta.deviceLabel ?? 'L20'
    ),
    trainingNodeId: String(
      metrics.trainingNodeId ?? trainConfig?.trainingNodeId ?? meta.trainingNodeId ?? ''
    ) || undefined,
    trainingNodeDisplayName: String(
      metrics.trainingNodeDisplayName ?? trainConfig?.trainingNodeDisplayName ?? meta.trainingNodeDisplayName ?? metrics.deviceLabel ?? ''
    ) || undefined,
    currentEpoch: epoch,
    totalEpochs,
    progressPercent,
    loss: metrics.loss != null ? Number(metrics.loss) : null,
    message: String(metrics.message ?? ''),
    checkpoint: checkpointExists ? String(metrics.modelAssetId ?? job.jobId) : null,
    checkpointExists,
    hasModelManifest,
    checkpointPath: metrics.checkpointPath ? String(metrics.checkpointPath) : null,
    modelAssetId: metrics.modelAssetId ? String(metrics.modelAssetId) : null,
    createdAt: formatEvaluationCreatedAt({
      createdAt: job.createdAt,
      startedAt: job.startedAt,
      updatedAt: job.updatedAt,
      finishedAt: job.finishedAt,
      completedAt: metrics.completedAt as string | undefined,
    }),
    updatedAt: formatIsoToLabel(job.updatedAt),
    startedAt: formatIsoToLabel(job.startedAt),
    finishedAt: formatIsoToLabel(job.finishedAt),
    batchSize: Number(trainConfig?.batchSize ?? meta.batchSize ?? metrics.batchSize ?? 0),
    learningRate: Number(trainConfig?.learningRate ?? meta.learningRate ?? metrics.learningRate ?? 0),
    seed: Number(trainConfig?.seed ?? meta.seed ?? metrics.seed ?? 0),
  };
}

export function trainingListItemToRow(item: TrainingJobListItem): TrainingTaskRow {
  const statusEpoch = Number(item.epoch ?? 0);
  const history = parseMetricsLossHistory(
    item.lossHistory ? { lossHistory: item.lossHistory } : null
  );
  const seriesMax = history.length > 0 ? Math.max(...history.map((p) => p.epoch)) : 0;
  const epoch = Math.max(statusEpoch, seriesMax);
  const totalEpochs = Number(item.totalEpochs ?? 0);
  const downstream = String(item.downstreamModelType ?? '').trim();
  const trainingBackend = String(item.trainingBackend ?? '').trim();
  const rawDatasetName = String(item.datasetName ?? '').trim();
  const resolvedDatasetName =
    normalizeDatasetDisplayName({
      displayName: rawDatasetName || null,
      name: rawDatasetName || null,
      sourceJobId: item.datasetId || null,
    }) || rawDatasetName;
  const displayDataset = (() => {
    if (!isInvalidDatasetDisplayName(resolvedDatasetName)) return resolvedDatasetName;
    const id = String(item.datasetId ?? '').trim();
    if (id && !isInvalidDatasetDisplayName(id) && id.toLowerCase() !== 'unknown') return id;
    return '—';
  })();
  const datasetName =
    !isInvalidDatasetDisplayName(resolvedDatasetName) ? resolvedDatasetName : undefined;
  const taskDisplayName = resolveTrainingTaskDisplayName({
    taskName: item.taskName,
    metaTaskName: item.taskName,
    trainConfigTaskName: item.taskName,
    datasetName: datasetName ?? null,
    trainingBackend,
    modelType: downstream || trainingBackend || null,
    jobId: item.trainJobId,
  });
  const rawProgress = item.progress != null ? Number(item.progress) : undefined;
  const normalized = normalizeTrainingJobStatus({
    backendStatus: item.status,
    currentEpoch: epoch,
    totalEpochs,
    progress: rawProgress,
    checkpointExists: Boolean(item.checkpointExists),
  });
  const progressPercent = trainingProgressPercent({
    backendStatus: normalized.backendStatus,
    epoch,
    totalEpochs,
    progress: rawProgress,
  });

  return {
    id: item.trainJobId,
    trainJobId: item.trainJobId,
    source: 'real',
    name: taskDisplayName,
    relatedTask: datasetName || taskDisplayName,
    modelType: downstream || trainingBackend || '—',
    dataset: displayDataset,
    datasetName,
    dataVolume: '—',
    status: normalized.displayStatus,
    backendStatus: normalized.backendStatus,
    trainingBackend: String(item.trainingBackend ?? ''),
    dataFormat: String(item.dataFormat ?? ''),
    deviceLabel: String(item.trainingNodeDisplayName ?? item.deviceLabel ?? 'L20'),
    trainingNodeId: item.trainingNodeId ? String(item.trainingNodeId) : undefined,
    trainingNodeDisplayName: item.trainingNodeDisplayName
      ? String(item.trainingNodeDisplayName)
      : item.deviceLabel
        ? String(item.deviceLabel)
        : undefined,
    currentEpoch: epoch,
    totalEpochs,
    progressPercent,
    loss: item.loss != null ? Number(item.loss) : null,
    message: String(item.message ?? ''),
    checkpoint: item.checkpointExists ? String(item.modelAssetId ?? item.trainJobId) : null,
    checkpointExists: Boolean(item.checkpointExists),
    hasModelManifest: Boolean(item.modelAssetId),
    checkpointPath: null,
    modelAssetId: item.modelAssetId ? String(item.modelAssetId) : null,
    createdAt: formatIsoToLabel(item.createdAt),
    updatedAt: formatIsoToLabel(item.updatedAt),
    startedAt: formatIsoToLabel(item.createdAt),
    finishedAt: formatIsoToLabel(item.updatedAt),
    batchSize: 0,
    learningRate: 0,
    seed: 0,
  };
}

function normalizeSuccessStats(
  stats?: EvaluationSuccessStats | Record<string, unknown> | null
): EvaluationSuccessStats | null {
  if (!stats || typeof stats !== 'object') return null;
  const raw = stats as Record<string, unknown>;
  const display =
    typeof raw.display === 'string' && raw.display.trim() ? raw.display.trim() : null;
  if (!display) return null;
  return {
    successEpisodes:
      typeof raw.successEpisodes === 'number'
        ? raw.successEpisodes
        : raw.successEpisodes == null
          ? null
          : Number(raw.successEpisodes),
    totalEpisodes:
      typeof raw.totalEpisodes === 'number'
        ? raw.totalEpisodes
        : raw.totalEpisodes == null
          ? null
          : Number(raw.totalEpisodes),
    display,
    available: Boolean(raw.available),
    source: typeof raw.source === 'string' ? raw.source : undefined,
    reason: typeof raw.reason === 'string' ? raw.reason : undefined,
  };
}

function fallbackSuccessStats(
  stats?: EvaluationSuccessStats | Record<string, unknown> | null
): EvaluationSuccessStats {
  return (
    normalizeSuccessStats(stats) ?? {
      successEpisodes: null,
      totalEpisodes: null,
      display: '-/-',
      available: false,
    }
  );
}

function resolveListItemSuccessStats(
  item: EvaluationJobListItem
): EvaluationSuccessStats | null {
  const direct = normalizeSuccessStats(item.successStats);
  if (direct) return direct;
  const metrics = item.metrics ?? {};
  const fromMetrics = normalizeSuccessStats(
    metrics.successStats as EvaluationSuccessStats | undefined
  );
  if (fromMetrics) return fromMetrics;
  const legacy = item as EvaluationJobListItem & { success_stats?: EvaluationSuccessStats };
  return normalizeSuccessStats(legacy.success_stats);
}

export function evaluationListItemToRow(item: EvaluationJobListItem): EvaluationTaskRow {
  const metrics = item.metrics ?? {};
  const taskType = item.taskType ?? undefined;
  const isDualArm = taskType === 'dual_arm_cable_manipulation';
  const isDatasetOffline =
    item.runner === 'dataset_offline_eval' || taskType === 'dataset_offline';
  const datasetNameFromTask = (() => {
    const match = String(item.taskName ?? '').match(/离线数据集评测\s*[·•]\s*(.+)/);
    return match?.[1]?.trim() ?? String(metrics.datasetName ?? '');
  })();
  const evaluationModeRaw = String(item.evaluationMode ?? metrics.evaluationMode ?? metrics.policy ?? '');
  const isTrainedModel =
    evaluationModeRaw === 'trained_model_evaluation' ||
    evaluationModeRaw === 'robomimic' ||
    Boolean(metrics.modelAssetId || metrics.checkpointPath);
  const evaluationModeApi = (() => {
    const raw = String(item.evaluationMode ?? metrics.evaluationMode ?? '').trim() || evaluationModeRaw;
    if (raw === 'robomimic') return 'trained_model_evaluation';
    if (raw === 'scripted') return 'expert_policy_evaluation';
    if (raw === 'policy_evaluation') return 'expert_policy_evaluation';
    return raw || undefined;
  })();
  const evalMode: EvaluationTaskRow['evaluationMode'] = isDatasetOffline
    ? '数据过程评测'
    : isDualArm
      ? 'episode 稳定性评测'
      : '策略评测';
  const successStatsResolved = resolveListItemSuccessStats(item);
  const successRateRaw = metrics.successRate ?? metrics.success_rate;
  const successRate = (() => {
    if (item.status !== 'completed') return null;
    if (successRateRaw != null) return Number(successRateRaw);
    if (
      successStatsResolved?.available &&
      successStatsResolved.totalEpisodes != null &&
      successStatsResolved.totalEpisodes > 0
    ) {
      return (successStatsResolved.successEpisodes ?? 0) / successStatsResolved.totalEpisodes;
    }
    return null;
  })();
  const relatedTask = isDatasetOffline
    ? '—'
    : resolveEvaluationRelatedTaskLabel({
        taskType,
        metrics,
      });
  const displayName = isDatasetOffline
    ? item.taskName
      ? normalizeTaskDisplayName(item.taskName)
      : datasetNameFromTask
        ? `离线数据集评测 · ${datasetNameFromTask}`
        : '离线数据集评测'
    : resolveEvaluationModelDisplayName({
        modelName: metrics.modelName as string | undefined,
        taskName: item.taskName,
        metadata: metrics,
        metrics,
        relatedTaskLabel: relatedTask,
      });
  const episodeCount = Number(
    metrics.numEpisodes ?? metrics.episodes ?? metrics.totalEpisodes ?? 1
  );
  const resolvedModelType = String(
    metrics.modelType ?? metrics.policyType ?? metrics.policy ?? ''
  ).trim();
  const modelType: EvaluationTaskRow['modelType'] = isDatasetOffline
    ? '—'
    : resolvedModelType && !['scripted', 'robomimic'].includes(resolvedModelType)
      ? resolvedModelType
      : isTrainedModel
        ? '已训练模型'
        : isDualArm && !isTrainedModel
          ? '专家策略'
          : taskType === 'cable_threading'
            ? isTrainedModel
              ? '已训练模型'
              : '专家策略'
            : '—';
  const checkpoint = isTrainedModel
    ? String(metrics.modelAssetId ?? metrics.checkpointPath ?? '—')
    : isDatasetOffline
      ? '—'
      : 'scripted';
  const evalBackend = item.evalJobId.startsWith('isaac_eval_') ? 'Isaac Lab' : 'MuJoCo';

  const evalJobId =
    resolveEvaluationJobId({
      evalJobId: item.evalJobId,
      runtimePath: item.runtimePath,
    }) || String(item.evalJobId ?? '').trim();
  if (!evalJobId && process.env.NODE_ENV === 'development') {
    console.warn('[Evaluation mapper] missing valid evalJobId for list item', item);
  }

  const typeFields = resolveEvaluationTypeFields({
    evaluationType: item.evaluationType,
    evaluationTypeLabel: item.evaluationTypeLabel,
    evaluationObject: item.evaluationObject,
    evaluationModeApi,
    modelAssetId: isTrainedModel ? String(metrics.modelAssetId ?? metrics.checkpointPath ?? '') : null,
    datasetId: isDatasetOffline ? String(metrics.datasetId ?? '') : null,
    datasetName: datasetNameFromTask,
    taskType,
    taskName: item.taskName,
    runner: item.runner,
    metrics,
  });

  return {
    id: evalJobId || (item.workspaceJobId != null ? `ws:${item.workspaceJobId}` : item.evalJobId || ''),
    evalJobId: evalJobId || undefined,
    jobId: evalJobId || undefined,
    workspaceJobId: item.workspaceJobId ?? undefined,
    name: item.taskName ? String(item.taskName) : displayName,
    taskName: item.taskName ? String(item.taskName) : displayName,
    source: 'real',
    evaluationMode: evalMode,
    evaluationModeApi,
    evaluationType: typeFields.evaluationType,
    evaluationTypeLabel: typeFields.evaluationTypeLabel,
    evaluationObject: typeFields.evaluationObject,
    relatedTask,
    checkpoint,
    modelType,
    dataVolume: isDualArm ? `${episodeCount} episode` : `${episodeCount || '—'} 条`,
    evalBackend,
    evalRounds: episodeCount,
    status: mapEvaluationStatus(item.status),
    successRate,
    rawName: item.taskName ? String(item.taskName) : undefined,
    createdAtIso: item.createdAt ?? undefined,
    createdAt: formatEvaluationCreatedAt({
      createdAt: item.createdAt,
      startedAt: item.startedAt,
      updatedAt: item.updatedAt,
      finishedAt: item.finishedAt,
      completedAt: metrics.completedAt as string | undefined,
      created_at: metrics.createdAt as string | undefined,
    }),
    updatedAt: formatIsoToLabel(item.updatedAt),
    startedAt: formatIsoToLabel(item.startedAt),
    finishedAt: formatIsoToLabel(item.finishedAt),
    runner: item.runner ?? undefined,
    runtimePath: item.runtimePath ?? undefined,
    dataName: isDatasetOffline ? datasetNameFromTask || undefined : undefined,
    datasetId: isDatasetOffline ? String(metrics.datasetId ?? '') || undefined : undefined,
    resultSummary: isDualArm
      ? buildDualArmEvalResultSummary({
          jobStatus: item.status,
          evaluationMode: evaluationModeApi,
          message: item.message,
          metrics,
        })
      : item.status === 'completed'
        ? '评测已完成，可查看报告与回放。'
        : item.status === 'failed'
          ? '评测失败'
          : '评测运行中…',
    taskType:
      taskType === 'cable_threading' || taskType === 'dual_arm_cable_manipulation'
        ? taskType
        : undefined,
    backendJobStatus: item.status,
    evalVideoExists: Boolean(item.videoAvailable),
    videoExists: Boolean(item.videoAvailable),
    videoJobId: evalJobId || undefined,
    aggregate: metrics.aggregate as Record<string, unknown> | undefined,
    requestedEpisodes: pickNumber(
      item.requestedEpisodes,
      metrics.requestedEpisodes,
      metrics.totalEpisodes,
      metrics.numEpisodes
    ) ?? undefined,
    completedEpisodes: pickNumber(item.completedEpisodes, metrics.completedEpisodes) ?? undefined,
    currentEpisode: pickNumber(item.currentEpisode, metrics.currentEpisode) ?? undefined,
    totalEpisodes: pickNumber(item.totalEpisodes, metrics.totalEpisodes, metrics.requestedEpisodes) ?? undefined,
    progress: pickNumber(item.progress, metrics.progress) ?? undefined,
    progressPercent: pickNumber(item.progressPercent, metrics.progressPercent) ?? undefined,
    progressLabel:
      typeof item.progressLabel === 'string'
        ? item.progressLabel
        : typeof metrics.progressLabel === 'string'
          ? metrics.progressLabel
          : undefined,
    templateDisplayName:
      typeof item.templateDisplayName === 'string' ? item.templateDisplayName : undefined,
    successStats: fallbackSuccessStats(successStatsResolved),
  };
}

export function mergeUniqueByJobId<T extends { id: string; jobId?: string; backendJobId?: string }>(
  primary: T[],
  secondary: T[]
): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const row of [...primary, ...secondary]) {
    const key = row.jobId ?? row.backendJobId ?? row.id;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(row);
  }
  return out;
}

export function findArtifactByType(
  artifacts: WorkspaceArtifactItem[],
  type: string
): WorkspaceArtifactItem | undefined {
  return artifacts.find((a) => a.artifactType === type);
}
