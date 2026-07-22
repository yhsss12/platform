'use client';

import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import { trainingTaskStatusBadge } from '@/lib/mock/workspaceTrainingMock';
import { StatusBadge } from '@/components/workspace/workspaceUi';
import {
  resolveTrainingDisplayState,
  type TrainingDisplayState,
} from '@/lib/workspace/trainingDisplayState';

/** 列表状态列：仅 badge + 可选 tooltip，不渲染 subLabel / 进度条 */
export function buildTrainingListStatusView(row: TrainingTaskRow): {
  badgeLabel: TrainingDisplayState['badgeLabel'];
  badgeStatus: ReturnType<typeof trainingTaskStatusBadge>;
  tooltip: string | undefined;
} {
  const display = resolveTrainingDisplayState({
    backendStatus: row.backendStatus,
    status: row.status,
    currentEpoch: row.currentEpoch,
    totalEpochs: row.totalEpochs,
    progressPercent: row.progressPercent,
    message: row.message,
  });
  return {
    badgeLabel: display.badgeLabel,
    badgeStatus: trainingTaskStatusBadge(display.badgeLabel),
    tooltip: display.subLabel ?? display.progressHint ?? undefined,
  };
}

export function TrainingStatusCell({
  row,
  showDetailHint = false,
}: {
  row: TrainingTaskRow;
  /** 详情抽屉头部可显示副文案；列表表格保持 false */
  showDetailHint?: boolean;
}) {
  const display = resolveTrainingDisplayState({
    backendStatus: row.backendStatus,
    status: row.status,
    currentEpoch: row.currentEpoch,
    totalEpochs: row.totalEpochs,
    progressPercent: row.progressPercent,
    message: row.message,
  });
  const listView = buildTrainingListStatusView(row);

  if (showDetailHint) {
    return (
      <div
        style={{
          display: 'inline-flex',
          flexDirection: 'column',
          alignItems: 'stretch',
          width: 'fit-content',
          maxWidth: '100%',
        }}
      >
        <StatusBadge status={listView.badgeStatus} label={listView.badgeLabel} />
        {display.subLabel ? (
          <span
            style={{
              marginTop: 4,
              fontSize: 11,
              lineHeight: 1.4,
              color: display.phase === 'failed' ? '#991b1b' : '#92400e',
              maxWidth: 220,
            }}
            title={display.subLabel}
          >
            {display.subLabel}
          </span>
        ) : null}
      </div>
    );
  }

  return (
    <div title={listView.tooltip} style={{ display: 'inline-block', maxWidth: '100%' }}>
      <StatusBadge status={listView.badgeStatus} label={listView.badgeLabel} />
    </div>
  );
}
