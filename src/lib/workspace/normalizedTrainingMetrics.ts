import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import {
  buildTrainingLossSeries,
  parseMetricsLossHistory,
} from '@/lib/workspace/trainingLossSeries';
import type { TrainingMetricPoint } from '@/lib/workspace/trainingLogParser';
import { normalizeTrainingBackendStatus, trainingProgressPercent } from '@/lib/workspace/trainingStatus';

export interface NormalizedTrainingMetrics {
  currentEpoch: number;
  totalEpochs: number;
  loss: number | null;
  lossSeries: TrainingMetricPoint[];
  progressPercent: number | null;
  bestLoss: number | null;
  finalLoss: number | null;
}

const COMPLETED_STATUSES = new Set(['completed', 'succeeded', 'success']);

function parseMetricsJsonl(text?: string | null): TrainingMetricPoint[] {
  if (!text?.trim()) return [];
  const points: TrainingMetricPoint[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const row = JSON.parse(trimmed) as Record<string, unknown>;
      const epoch = Number(row.epoch ?? row.step ?? 0);
      if (!Number.isFinite(epoch) || epoch <= 0) continue;

      const hasExplicitTrain = row.trainLoss != null || row.train_loss != null;
      const trainLoss = hasExplicitTrain
        ? Number(row.trainLoss ?? row.train_loss)
        : row.loss != null
          ? Number(row.loss)
          : undefined;

      const validLoss =
        row.validLoss != null
          ? Number(row.validLoss)
          : row.valid_loss != null
            ? Number(row.valid_loss)
            : row.valLoss != null
              ? Number(row.valLoss)
              : row.validationLoss != null
                ? Number(row.validationLoss)
                : undefined;

      points.push({
        epoch,
        trainLoss: Number.isFinite(trainLoss as number) ? (trainLoss as number) : undefined,
        validLoss: Number.isFinite(validLoss as number) ? (validLoss as number) : undefined,
      });
    } catch {
      // skip malformed line
    }
  }
  return points;
}

function mergeSeries(...sources: TrainingMetricPoint[][]): TrainingMetricPoint[] {
  const map = new Map<number, TrainingMetricPoint>();
  for (const source of sources) {
    for (const point of source) {
      const existing = map.get(point.epoch) ?? { epoch: point.epoch };
      map.set(point.epoch, {
        epoch: point.epoch,
        trainLoss: point.trainLoss ?? existing.trainLoss,
        validLoss: point.validLoss ?? existing.validLoss,
      });
    }
  }
  return Array.from(map.values()).sort((a, b) => a.epoch - b.epoch);
}

function pickLossAtEpoch(series: TrainingMetricPoint[], epoch: number): number | null {
  const point = series.find((item) => item.epoch === epoch);
  const value = point?.trainLoss ?? point?.validLoss;
  return value != null && Number.isFinite(value) ? value : null;
}

function pickTrainLossAtEpoch(series: TrainingMetricPoint[], epoch: number): number | null {
  const point = series.find((item) => item.epoch === epoch);
  const value = point?.trainLoss;
  return value != null && Number.isFinite(value) ? value : null;
}

function bestTrainLoss(series: TrainingMetricPoint[]): number | null {
  const values = series
    .flatMap((point) => [point.trainLoss, point.validLoss])
    .filter((value): value is number => value != null && Number.isFinite(value));
  if (values.length === 0) return null;
  return Math.min(...values);
}

function isCompletedStatus(status?: string | null): boolean {
  return COMPLETED_STATUSES.has(normalizeTrainingBackendStatus(status));
}

/**
 * 统一训练指标来源：metrics history / metrics.jsonl / train.log / status 行字段。
 * 进度、当前 Loss、Loss 图表均读取此结果。
 */
export function normalizedTrainingMetrics(options: {
  log?: string;
  metricsJsonl?: string | null;
  row?: Pick<
    TrainingTaskRow,
    'currentEpoch' | 'totalEpochs' | 'loss' | 'status' | 'backendStatus'
  > | null;
  metrics?: Record<string, unknown> | null;
  accumulated?: TrainingMetricPoint[];
}): NormalizedTrainingMetrics {
  const totalEpochs = Math.max(0, Number(options.row?.totalEpochs ?? options.metrics?.totalEpochs ?? 0));
  const backendStatus = options.row?.backendStatus ?? options.row?.status ?? null;
  const statusEpoch = Math.max(0, Number(options.row?.currentEpoch ?? options.metrics?.epoch ?? 0));

  const fromMetricsHistory = parseMetricsLossHistory(options.metrics);
  const fromJsonl = parseMetricsJsonl(options.metricsJsonl);
  const fromLog = buildTrainingLossSeries({
    log: options.log ?? '',
    metrics: null,
    row: null,
    accumulated: options.accumulated,
  });

  const lossSeries = mergeSeries(fromMetricsHistory, fromJsonl, fromLog);

  const seriesMaxEpoch =
    lossSeries.length > 0 ? Math.max(...lossSeries.map((point) => point.epoch)) : 0;
  const currentEpoch = Math.max(seriesMaxEpoch, statusEpoch);

  const completed =
    isCompletedStatus(backendStatus) && totalEpochs > 0 && currentEpoch >= totalEpochs;

  const seriesLoss = pickLossAtEpoch(lossSeries, currentEpoch);
  const statusLoss =
    options.row?.loss != null && Number.isFinite(options.row.loss) ? options.row.loss : null;
  const metricsLoss =
    options.metrics?.loss != null && Number.isFinite(Number(options.metrics.loss))
      ? Number(options.metrics.loss)
      : null;

  const loss = seriesLoss ?? (completed ? statusLoss : null) ?? seriesLoss ?? metricsLoss ?? statusLoss;

  const progressPercent = trainingProgressPercent({
    backendStatus: completed ? 'completed' : backendStatus ?? 'running',
    epoch: currentEpoch,
    totalEpochs,
  });

  const computedFinal =
    lossSeries.length > 0
      ? pickTrainLossAtEpoch(lossSeries, seriesMaxEpoch) ?? pickLossAtEpoch(lossSeries, seriesMaxEpoch)
      : loss;

  return {
    currentEpoch,
    totalEpochs,
    loss,
    lossSeries,
    progressPercent,
    bestLoss:
      options.metrics?.bestLoss != null && Number.isFinite(Number(options.metrics.bestLoss))
        ? Number(options.metrics.bestLoss)
        : bestTrainLoss(lossSeries),
    finalLoss: completed
      ? options.metrics?.finalLoss != null && Number.isFinite(Number(options.metrics.finalLoss))
        ? Number(options.metrics.finalLoss)
        : computedFinal
      : null,
  };
}
