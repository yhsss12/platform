import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import { parseTrainingLogMetrics, type TrainingMetricPoint } from '@/lib/workspace/trainingLogParser';

function mergePoint(
  map: Map<number, TrainingMetricPoint>,
  point: TrainingMetricPoint
): void {
  const existing = map.get(point.epoch) ?? { epoch: point.epoch };
  map.set(point.epoch, {
    epoch: point.epoch,
    trainLoss: point.trainLoss ?? existing.trainLoss,
    validLoss: point.validLoss ?? existing.validLoss,
  });
}

export function parseMetricsLossHistory(metrics?: Record<string, unknown> | null): TrainingMetricPoint[] {
  if (!metrics) return [];

  const raw =
    metrics.lossHistory ??
    metrics.loss_history ??
    metrics.metricsHistory ??
    metrics.metrics_history;

  if (!Array.isArray(raw)) return [];

  const points: TrainingMetricPoint[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue;
    const row = item as Record<string, unknown>;
    const epoch = Number(row.epoch ?? row.step ?? row.iteration ?? 0);
    if (!Number.isFinite(epoch) || epoch <= 0) continue;
    const validLoss =
      row.validLoss != null
        ? Number(row.validLoss)
        : row.valid_loss != null
          ? Number(row.valid_loss)
          : row.valLoss != null
            ? Number(row.valLoss)
            : undefined;
    const hasExplicitTrain = row.trainLoss != null || row.train_loss != null;
    const trainLoss = hasExplicitTrain
      ? Number(row.trainLoss ?? row.train_loss)
      : row.loss != null
        ? Number(row.loss)
        : undefined;
    points.push({
      epoch,
      trainLoss: Number.isFinite(trainLoss as number) ? (trainLoss as number) : undefined,
      validLoss: Number.isFinite(validLoss as number) ? (validLoss as number) : undefined,
    });
  }
  return points;
}

export function buildTrainingLossSeries(options: {
  log?: string;
  row?: Pick<TrainingTaskRow, 'currentEpoch' | 'loss'> | null;
  metrics?: Record<string, unknown> | null;
  accumulated?: TrainingMetricPoint[];
}): TrainingMetricPoint[] {
  const map = new Map<number, TrainingMetricPoint>();

  for (const point of options.accumulated ?? []) {
    mergePoint(map, point);
  }

  for (const point of parseMetricsLossHistory(options.metrics)) {
    mergePoint(map, point);
  }

  for (const point of parseTrainingLogMetrics(options.log ?? '')) {
    mergePoint(map, point);
  }

  const row = options.row;
  if (row && row.currentEpoch > 0 && row.loss != null && Number.isFinite(row.loss)) {
    mergePoint(map, { epoch: row.currentEpoch, trainLoss: row.loss });
  }

  return Array.from(map.values()).sort((a, b) => a.epoch - b.epoch);
}
