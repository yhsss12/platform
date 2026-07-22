export interface TrainingMetricPoint {
  epoch: number;
  trainLoss?: number;
  validLoss?: number;
}

function parseLossNumber(raw: string | undefined): number | undefined {
  if (raw == null) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function mergePoint(points: Map<number, TrainingMetricPoint>, point: TrainingMetricPoint): void {
  const existing = points.get(point.epoch) ?? { epoch: point.epoch };
  points.set(point.epoch, {
    epoch: point.epoch,
    trainLoss: point.trainLoss ?? existing.trainLoss,
    validLoss: point.validLoss ?? existing.validLoss,
  });
}

export function parseTrainingLogMetrics(log: string): TrainingMetricPoint[] {
  if (!log.trim()) return [];

  const points = new Map<number, TrainingMetricPoint>();

  for (const line of log.split('\n')) {
    const validationInline = line.match(
      /Validation\s+Epoch\s+(\d+)\s+Loss:\s*([0-9.eE+-]+)/i
    );
    if (validationInline) {
      const epoch = Number(validationInline[1]);
      const validLoss = parseLossNumber(validationInline[2]);
      if (epoch > 0 && validLoss != null) {
        mergePoint(points, { epoch, validLoss });
      }
      continue;
    }

    const actTrainInline = line.match(/(?<!Validation\s)Epoch\s+(\d+)\s+Loss:\s*([0-9.eE+-]+)/i);
    if (actTrainInline) {
      const epoch = Number(actTrainInline[1]);
      const trainLoss = parseLossNumber(actTrainInline[2]);
      if (epoch > 0 && trainLoss != null) {
        mergePoint(points, { epoch, trainLoss });
      }
      continue;
    }

    const trainMatch = line.match(/Train\s+Epoch\s+(\d+)/i);
    if (trainMatch) {
      const epoch = Number(trainMatch[1]);
      const lossOnLine = line.match(/\bLoss\s*:\s*([0-9.eE+-]+)/i);
      if (epoch > 0) {
        const trainLoss = parseLossNumber(lossOnLine?.[1]);
        mergePoint(points, trainLoss != null ? { epoch, trainLoss } : { epoch });
      }
      continue;
    }

    const validMatch = line.match(/Validation\s+Epoch\s+(\d+)/i);
    if (validMatch) {
      const epoch = Number(validMatch[1]);
      const lossOnLine = line.match(/\bLoss\s*:\s*([0-9.eE+-]+)/i);
      if (epoch > 0) {
        const validLoss = parseLossNumber(lossOnLine?.[1]);
        mergePoint(points, validLoss != null ? { epoch, validLoss } : { epoch });
      }
      continue;
    }

    const inlineEpochLoss = line.match(
      /(?:epoch|Epoch)\s*[:=]?\s*(\d+).*?(?:loss|Loss)\s*[:=]\s*([0-9.eE+-]+)/i
    );
    if (inlineEpochLoss) {
      const epoch = Number(inlineEpochLoss[1]);
      const trainLoss = parseLossNumber(inlineEpochLoss[2]);
      if (epoch > 0 && trainLoss != null) {
        mergePoint(points, { epoch, trainLoss });
      }
    }
  }

  return Array.from(points.values()).sort((a, b) => a.epoch - b.epoch);
}
