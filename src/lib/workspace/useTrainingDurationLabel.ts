'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  buildTrainingDurationInput,
  isTrainingDurationRunning,
  resolveTrainingDurationLabel,
  type TrainingDurationInput,
} from '@/lib/workspace/trainingDuration';

export function useTrainingDurationLabel(input: TrainingDurationInput): string {
  const status = input.status ?? (input.metrics?.status as string | undefined) ?? null;
  const running = isTrainingDurationRunning(status);
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!running) return;
    setNowMs(Date.now());
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, input.startedAt, input.finishedAt, input.completedAt, input.endedAt]);

  return useMemo(
    () => resolveTrainingDurationLabel({ ...input, nowMs: running ? nowMs : undefined }),
    [
      running,
      nowMs,
      input.status,
      input.startedAt,
      input.finishedAt,
      input.completedAt,
      input.endedAt,
      input.metrics,
    ]
  );
}

export function useJobTrainingDurationLabel(options: {
  status?: string | null;
  jobDetail?: {
    startedAt?: string | null;
    finishedAt?: string | null;
    metrics?: Record<string, unknown> | null;
  } | null;
}): string {
  const durationInput = useMemo(
    () => buildTrainingDurationInput(options),
    [
      options.status,
      options.jobDetail?.startedAt,
      options.jobDetail?.finishedAt,
      options.jobDetail?.metrics,
    ]
  );
  return useTrainingDurationLabel(durationInput);
}
