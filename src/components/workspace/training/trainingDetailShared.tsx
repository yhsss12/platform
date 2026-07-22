'use client';

import type { CSSProperties, ReactNode } from 'react';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import type { WorkspaceJobDetail } from '@/lib/api/workspaceJobClient';
import { formatTrainingRecipeLabel } from '@/lib/workspace/trainingRecipe';
import { formatTrainingDeviceLabel } from '@/lib/workspace/trainingDevice';
import { extractStoredTrainingJobConfig } from '@/lib/workspace/trainingJobConfig';
import {
  resolveTrainingInitWeightLabel,
  resolveTrainingTaskDisplayName,
} from '@/lib/workspace/trainingDisplay';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { formatLossValue } from '@/lib/workspace/chartFormat';
import { resolveTrainingLossFieldLabel } from '@/lib/workspace/trainingLossDisplay';
import { useJobTrainingDurationLabel } from '@/lib/workspace/useTrainingDurationLabel';
import {
  buildTrainingArtifactDisplayItems,
  type TrainingArtifactDisplayItem,
} from '@/lib/workspace/trainingArtifactsDisplay';

export const trainingSectionTitleStyle: CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  marginBottom: 12,
  letterSpacing: '0.02em',
};

export function TrainingInfoItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>{label}</div>
      <div
        style={{
          fontSize: 13,
          color: '#111827',
          lineHeight: 1.45,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={typeof children === 'string' ? children : undefined}
      >
        {children}
      </div>
    </div>
  );
}

export function TrainingTaskSummaryGrid({
  row,
  jobDetail,
  displayName,
}: {
  row: TrainingTaskRow;
  jobDetail?: WorkspaceJobDetail | null;
  displayName?: string;
}) {
  const metadata = jobDetail?.metadata ?? {};
  const trainConfig = extractStoredTrainingJobConfig(metadata);
  const backendStatus = jobDetail?.status ?? row.backendStatus ?? null;
  const lossFieldLabel = resolveTrainingLossFieldLabel(backendStatus);
  const displayLoss =
    row.loss != null
      ? row.loss
      : jobDetail?.metrics?.loss != null
        ? Number(jobDetail.metrics.loss)
        : null;
  const durationLabel = useJobTrainingDurationLabel({ status: backendStatus, jobDetail });
  const name =
    displayName ??
    resolveTrainingTaskDisplayName({
      taskName: row.name,
      metaTaskName: typeof metadata.taskName === 'string' ? metadata.taskName : null,
      trainConfigTaskName:
        typeof trainConfig?.taskName === 'string' ? trainConfig.taskName : null,
      datasetName: row.datasetName ?? row.relatedTask,
      trainingBackend: row.trainingBackend,
      modelType: row.modelType,
      jobId: row.trainJobId,
    });
  const initWeightLabel = resolveTrainingInitWeightLabel(trainConfig);
  const seedValue =
    trainConfig?.seed != null && trainConfig.seed > 0
      ? String(trainConfig.seed)
      : row.seed > 0
        ? String(row.seed)
        : '—';

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        gap: '14px 20px',
        padding: '14px 16px',
        borderRadius: 8,
        border: '1px solid #e5e7eb',
        backgroundColor: '#f9fafb',
      }}
    >
      <TrainingInfoItem label="训练任务名称">{name}</TrainingInfoItem>
      <TrainingInfoItem label="数据集">{row.datasetName ?? row.relatedTask ?? '—'}</TrainingInfoItem>
      <TrainingInfoItem label="模型类型">
        {formatTrainingRecipeLabel(row.trainingBackend, row.modelType)}
      </TrainingInfoItem>
      <TrainingInfoItem label="训练节点">
        {formatTrainingDeviceLabel(row.deviceLabel, row.trainingNodeDisplayName, row.trainingNodeId)}
      </TrainingInfoItem>
      <TrainingInfoItem label="训练耗时">{durationLabel}</TrainingInfoItem>
      <TrainingInfoItem label="初始化权重">{initWeightLabel}</TrainingInfoItem>
      <TrainingInfoItem label="Seed">{seedValue}</TrainingInfoItem>
      <TrainingInfoItem label="创建时间">
        {formatDateTimeMinuteYmdSlash(jobDetail?.createdAt ?? row.createdAt)}
      </TrainingInfoItem>
      <TrainingInfoItem label={lossFieldLabel}>
        {displayLoss != null ? (
          <span style={{ fontFamily: 'ui-monospace, monospace' }}>{formatLossValue(displayLoss)}</span>
        ) : (
          '—'
        )}
      </TrainingInfoItem>
    </div>
  );
}

export function TrainingArtifactsInfo({
  items,
  title = '产物信息',
}: {
  items: TrainingArtifactDisplayItem[];
  title?: string;
}) {
  return (
    <>
      <div style={trainingSectionTitleStyle}>{title}</div>
      <div
        style={{
          padding: '12px 14px',
          borderRadius: 8,
          border: '1px solid #e5e7eb',
          backgroundColor: '#f9fafb',
          fontSize: 13,
          color: '#374151',
          lineHeight: 1.55,
        }}
      >
        {items.length === 0 ? (
          <div style={{ color: '#9ca3af' }}>暂无产物记录</div>
        ) : (
          <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
            {items.map((item) => (
              <li
                key={item.id}
                style={{
                  display: 'flex',
                  gap: 8,
                  padding: '6px 0',
                  borderBottom: '1px solid #f3f4f6',
                }}
              >
                <span style={{ fontWeight: 500, minWidth: 108, flexShrink: 0 }}>{item.label}</span>
                <span
                  style={{
                    minWidth: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    color: '#6b7280',
                    fontFamily: 'ui-monospace, monospace',
                    fontSize: 12,
                  }}
                  title={item.name}
                >
                  {item.name}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

export { buildTrainingArtifactDisplayItems };

export interface AttachmentSideChannelDisplay {
  enabled: boolean;
  inputMode?: string;
  controlMode?: string;
}

export function resolveAttachmentSideChannelDisplay(
  metadata: Record<string, unknown> | null | undefined
): AttachmentSideChannelDisplay | null {
  if (!metadata) return null;
  const trainConfig = metadata.trainConfig;
  if (!trainConfig || typeof trainConfig !== 'object') return null;
  const snapshot = (trainConfig as Record<string, unknown>).adaptationSnapshot;
  if (!snapshot || typeof snapshot !== 'object') return null;
  const profile = (snapshot as Record<string, unknown>).datasetProfile;
  const adaptation = (snapshot as Record<string, unknown>).modelAdaptation;
  const training =
    adaptation && typeof adaptation === 'object'
      ? (adaptation as Record<string, unknown>).trainingConfig
      : null;
  const sideChannel =
    profile && typeof profile === 'object'
      ? Boolean((profile as Record<string, unknown>).attachmentSideChannel)
      : false;
  if (!sideChannel && training && typeof training === 'object') {
    if (!(training as Record<string, unknown>).attachmentSideChannel) return null;
  }
  const inputMode =
    training && typeof training === 'object'
      ? String((training as Record<string, unknown>).attachmentInputMode || '')
      : '';
  const controlMode =
    training && typeof training === 'object'
      ? String((training as Record<string, unknown>).attachmentControlMode || '')
      : '';
  return {
    enabled: sideChannel || Boolean((training as Record<string, unknown>)?.attachmentSideChannel),
    inputMode: inputMode || undefined,
    controlMode: controlMode || undefined,
  };
}

export function formatAttachmentInputModeLabel(mode?: string): string {
  if (!mode) return '—';
  if (mode === 'not_used_by_policy') return '未使用 attachment';
  if (mode === 'low_dim_obs') return '作为 low_dim 观测';
  return mode;
}

export function formatAttachmentControlModeLabel(mode?: string): string {
  if (!mode) return '—';
  if (mode === 'eval_controller') return 'policy controller';
  if (mode === 'recorded') return 'recorded replay';
  if (mode === 'none') return '关闭';
  return mode;
}
