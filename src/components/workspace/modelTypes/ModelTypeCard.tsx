'use client';

import type { CSSProperties } from 'react';
import { GhostButton, SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';
import type { ModelTypeDefinition } from '@/types/modelType';
import { MODEL_TYPE_STATUS_LABELS } from '@/types/modelType';
import {
  baseAlgorithmLabel,
  robotTypeLabel,
  simulatorLabel,
  structureConfigSummary,
} from '@/lib/workspace/modelTypeDisplay';
import { modelTypeTrainingCapabilityLabel } from '@/lib/workspace/modelTypeTrainingCapability';

function modelTypeBadgeStatus(status: string): 'active' | 'draft' | 'archived' {
  if (status === 'available') return 'active';
  if (status === 'draft') return 'draft';
  return 'archived';
}

function trainingCapabilityBadgeStatus(
  item: Pick<ModelTypeDefinition, 'trainingReady' | 'trainingReadinessStatus'>
): 'active' | 'draft' | 'archived' {
  if (item.trainingReadinessStatus === 'pending') return 'draft';
  if (item.trainingReadinessStatus === 'unknown') return 'draft';
  return item.trainingReady ? 'active' : 'archived';
}

const cardStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  minHeight: 320,
  height: '100%',
  padding: 16,
  borderRadius: 10,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
  boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
  boxSizing: 'border-box',
};

const metaGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: '10px 16px',
  marginTop: 14,
};

const metaItemStyle: CSSProperties = {
  minWidth: 0,
};

const metaLabelStyle: CSSProperties = {
  fontSize: 11,
  color: '#94a3b8',
  marginBottom: 2,
  whiteSpace: 'nowrap',
};

const metaValueStyle: CSSProperties = {
  fontSize: 13,
  color: '#334155',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const summaryStyle: CSSProperties = {
  marginTop: 12,
  fontSize: 12,
  color: '#64748b',
  lineHeight: 1.45,
  display: '-webkit-box',
  WebkitLineClamp: 2,
  WebkitBoxOrient: 'vertical',
  overflow: 'hidden',
  minHeight: 34,
};

const tagRowStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  marginTop: 'auto',
  paddingTop: 12,
  minHeight: 28,
};

const tagStyle: CSSProperties = {
  fontSize: 11,
  padding: '2px 8px',
  borderRadius: 999,
  background: '#f1f5f9',
  color: '#64748b',
  whiteSpace: 'nowrap',
  maxWidth: '100%',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const actionsStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'flex-end',
  gap: 8,
  marginTop: 12,
  paddingTop: 12,
  borderTop: '1px solid #f1f5f9',
};

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={metaItemStyle}>
      <div style={metaLabelStyle}>{label}</div>
      <div style={metaValueStyle} title={value}>
        {value}
      </div>
    </div>
  );
}

export function ModelTypeCard({
  item,
  onViewDetail,
  onEnable,
}: {
  item: ModelTypeDefinition;
  onViewDetail: () => void;
  onEnable?: () => void;
}) {
  return (
    <article style={cardStyle}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 600,
              color: '#111827',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={item.name}
          >
            {item.name}
          </div>
          <div
            style={{
              marginTop: 4,
              fontSize: 12,
              color: '#64748b',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={`${item.modelTypeId} · ${item.adapterKey}`}
          >
            {item.modelTypeId} · {item.adapterKey}
          </div>
        </div>
        <div style={{ display: 'flex', flexShrink: 0, flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
          <StatusBadge
            status={modelTypeBadgeStatus(item.status)}
            label={MODEL_TYPE_STATUS_LABELS[item.status] ?? item.status}
          />
          <StatusBadge
            status={trainingCapabilityBadgeStatus(item)}
            label={modelTypeTrainingCapabilityLabel(item)}
          />
        </div>
      </div>

      <div style={metaGridStyle}>
        <MetaItem label="基础算法" value={baseAlgorithmLabel(item.baseAlgorithm)} />
        <MetaItem label="适配器" value={item.adapterKey} />
        <MetaItem label="仿真环境" value={simulatorLabel(item.simulator)} />
        <MetaItem label="机器人类型" value={robotTypeLabel(item.robotType)} />
      </div>

      <div style={summaryStyle} title={structureConfigSummary(item)}>
        结构摘要：{structureConfigSummary(item)}
      </div>

      <div style={tagRowStyle}>
        {item.tags.length > 0 ? (
          item.tags.map((tag) => (
            <span key={tag} style={tagStyle} title={tag}>
              {tag}
            </span>
          ))
        ) : (
          <span style={{ ...tagStyle, visibility: 'hidden' }}>—</span>
        )}
      </div>

      <div style={actionsStyle}>
        <SecondaryButton onClick={onViewDetail}>查看详情</SecondaryButton>
        {!item.isBuiltin && item.status === 'draft' && onEnable ? (
          <GhostButton onClick={onEnable}>启用</GhostButton>
        ) : null}
      </div>
    </article>
  );
}
