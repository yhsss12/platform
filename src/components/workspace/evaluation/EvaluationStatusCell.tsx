'use client';

import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { evaluationTaskStatusBadge } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { StatusBadge } from '@/components/workspace/workspaceUi';

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, Math.round(value)));
}

function resolveEvaluationProgress(row: EvaluationTaskRow): {
  percent: number;
  label: string | null;
} {
  if (row.status !== '评测中') {
    return { percent: 0, label: null };
  }

  const total =
    row.requestedEpisodes ?? row.totalEpisodes ?? row.evalRounds ?? 0;
  const completed = row.completedEpisodes;

  if (typeof row.progressPercent === 'number' && Number.isFinite(row.progressPercent)) {
    return {
      percent: clampPercent(row.progressPercent),
      label: row.progressLabel ?? (total > 0 && typeof completed === 'number' ? `${completed}/${total}` : null),
    };
  }

  if (total > 0 && typeof completed === 'number') {
    return {
      percent: clampPercent((completed / total) * 100),
      label: row.progressLabel ?? `${completed}/${total}`,
    };
  }

  if (typeof row.progress === 'number' && Number.isFinite(row.progress)) {
    const normalized = row.progress <= 1 ? row.progress * 100 : row.progress;
    return {
      percent: clampPercent(normalized),
      label: row.progressLabel ?? (total > 0 ? `0/${total}` : null),
    };
  }

  if (total > 0) {
    return {
      percent: 0,
      label: row.progressLabel ?? `0/${total}`,
    };
  }

  return {
    percent: 0,
    label: row.progressLabel ?? null,
  };
}

function EvaluationMiniProgressBar({ percent }: { percent: number }) {
  return (
    <div
      style={{
        width: '100%',
        height: 6,
        borderRadius: 3,
        backgroundColor: '#dbeafe',
        overflow: 'hidden',
        marginTop: 4,
      }}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
      role="progressbar"
    >
      <div
        style={{
          width: `${percent}%`,
          height: '100%',
          borderRadius: 3,
          backgroundColor: '#2563eb',
          transition: 'width 0.25s ease',
        }}
      />
    </div>
  );
}

export function EvaluationStatusCell({ row }: { row: EvaluationTaskRow }) {
  const showProgress = row.status === '评测中';
  const { percent, label } = resolveEvaluationProgress(row);

  return (
    <div
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        width: 96,
        maxWidth: 120,
        minWidth: 88,
      }}
    >
      <StatusBadge status={evaluationTaskStatusBadge(row.status)} label={row.status} />
      {showProgress ? (
        <>
          <EvaluationMiniProgressBar percent={percent} />
          {label ? (
            <span style={{ marginTop: 4, fontSize: 11, color: '#64748b', lineHeight: 1.2 }}>{label}</span>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
