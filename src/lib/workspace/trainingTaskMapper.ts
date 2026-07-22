import type { TrainingJobStatus } from '@/lib/api/trainingClient';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import { resolveTrainabilityFromJobStatus } from '@/lib/workspace/trainingCapabilityUi';
import { formatTrainingDeviceLabel } from '@/lib/workspace/trainingDevice';
import {
  mapTrainingStatusToDisplay,
  normalizeTrainingJobStatus,
  trainingProgressPercent,
} from '@/lib/workspace/trainingStatus';
import { resolveTrainingTaskDisplayName } from '@/lib/workspace/trainingDisplay';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { parseMetricsLossHistory } from '@/lib/workspace/trainingLossSeries';

function resolveEpochFromStatus(status: TrainingJobStatus & { lossHistory?: unknown }): number {
  const statusEpoch = status.epoch ?? 0;
  const history = parseMetricsLossHistory(
    status.lossHistory ? { lossHistory: status.lossHistory } : null
  );
  const seriesMax = history.length > 0 ? Math.max(...history.map((p) => p.epoch)) : 0;
  return Math.max(statusEpoch, seriesMax);
}

export function mapBackendStatusToDisplay(status: TrainingJobStatus['status']) {
  return mapTrainingStatusToDisplay(status);
}

export function trainingJobStatusToRow(status: TrainingJobStatus & { lossHistory?: unknown }): TrainingTaskRow {
  const checkpointExists = Boolean(status.checkpointExists);
  const currentEpoch = resolveEpochFromStatus(status);
  const totalEpochs = status.totalEpochs ?? 0;
  const normalized = normalizeTrainingJobStatus({
    backendStatus: status.status,
    currentEpoch,
    totalEpochs,
    progress: status.progress,
    checkpointExists: Boolean(status.checkpointExists),
    message: status.message,
  });
  const progressPercent = trainingProgressPercent({
    backendStatus: normalized.backendStatus,
    epoch: currentEpoch,
    totalEpochs,
    progress: status.progress,
  });

  return {
    id: status.trainJobId,
    trainJobId: status.trainJobId,
    source: 'real',
    name: resolveTrainingTaskDisplayName({
      taskName: status.taskName,
      datasetName: status.datasetName,
      modelType: status.downstreamModelType ?? undefined,
      trainingBackend: status.trainingBackend ?? undefined,
      jobId: status.trainJobId,
    }),
    relatedTask: status.datasetName ?? '—',
    modelType: status.downstreamModelType ?? 'unknown',
    dataset: status.datasetId ?? '—',
    datasetName: status.datasetName ?? undefined,
    dataVolume: '—',
    status: normalized.displayStatus,
    trainability: resolveTrainabilityFromJobStatus(status),
    backendStatus: normalized.backendStatus,
    trainingBackend: status.trainingBackend ?? undefined,
    dataFormat: status.dataFormat ?? undefined,
    deviceLabel: formatTrainingDeviceLabel(
      status.deviceLabel,
      status.trainingNodeDisplayName,
      status.trainingNodeId
    ),
    trainingNodeId: status.trainingNodeId ?? undefined,
    trainingNodeDisplayName: formatTrainingDeviceLabel(
      status.deviceLabel,
      status.trainingNodeDisplayName,
      status.trainingNodeId
    ),
    currentEpoch,
    totalEpochs,
    progressPercent,
    loss: status.loss ?? null,
    message: status.message ?? '',
    checkpoint: checkpointExists ? status.modelAssetId ?? status.checkpointPath : null,
    checkpointExists,
    hasModelManifest: Boolean(status.modelAssetId),
    checkpointPath: status.checkpointPath ?? null,
    modelAssetId: status.modelAssetId ?? null,
    createdAt: formatDateTimeMinuteYmdSlash(status.createdAt),
    batchSize: 0,
    learningRate: 0,
    seed: 0,
  };
}
