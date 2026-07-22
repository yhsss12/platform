'use client';

import { useEffect, useState } from 'react';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  formatCalculationModeSummary,
  formatSourceFieldsLabel,
  type GenericMetricGroup,
  type MetricTaskMapping,
} from '@/lib/workspace/evaluationMetricRegistry';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 640,
  maxWidth: '100vw',
  backgroundColor: '#fff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  marginBottom: 10,
  letterSpacing: '0.02em',
};

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '96px 1fr', gap: 12, marginBottom: 10, fontSize: 13 }}>
      <span style={{ color: '#9ca3af' }}>{label}</span>
      <span style={{ color: '#111827', lineHeight: 1.55 }}>{children}</span>
    </div>
  );
}

function StatusBadge({ implemented }: { implemented: boolean }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '2px 8px',
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 500,
        background: implemented ? '#ecfdf5' : '#f1f5f9',
        color: implemented ? '#047857' : '#64748b',
      }}
    >
      {implemented ? '已接入' : '规划中'}
    </span>
  );
}

function TaskMappingList({ mappings }: { mappings: MetricTaskMapping[] }) {
  if (mappings.length === 0) {
    return (
      <p style={{ margin: 0, fontSize: 13, color: '#9ca3af' }}>
        暂无任务映射，该指标尚未绑定真实评测字段。
      </p>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {mappings.map((mapping) => (
        <div
          key={`${mapping.metricId}-${mapping.taskType}-${mapping.evaluationMode}`}
          style={{
            padding: '12px 14px',
            borderRadius: 10,
            border: '1px solid #e5eaf2',
            background: '#fafbfc',
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '88px 1fr',
              gap: '6px 12px',
              fontSize: 12,
              lineHeight: 1.55,
            }}
          >
            <span style={{ color: '#94a3b8' }}>任务</span>
            <span style={{ color: '#111827', fontWeight: 500 }}>{mapping.taskLabel}</span>
            <span style={{ color: '#94a3b8' }}>评测模式</span>
            <span style={{ color: '#374151' }}>{mapping.evaluationModeLabel}</span>
            <span style={{ color: '#94a3b8' }}>来源字段</span>
            <span style={{ color: '#374151', fontFamily: 'ui-monospace, monospace', fontSize: 11 }}>
              sourceField: {formatSourceFieldsLabel(mapping)}
            </span>
            <span style={{ color: '#94a3b8' }}>计算方式</span>
            <span style={{ color: '#374151' }}>
              {formatCalculationModeSummary(mapping.calculationMode)}
            </span>
            <span style={{ color: '#94a3b8' }}>接入状态</span>
            <span>
              <StatusBadge implemented={mapping.implemented} />
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function TechnicalMappingSection({ mappings }: { mappings: MetricTaskMapping[] }) {
  const [open, setOpen] = useState(false);

  if (mappings.length === 0) {
    return null;
  }

  return (
    <div style={{ marginTop: 20 }}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: 0,
          border: 'none',
          background: 'none',
          fontSize: 12,
          fontWeight: 600,
          color: '#6b7280',
          cursor: 'pointer',
        }}
      >
        <span style={{ fontSize: 10, color: '#9ca3af' }}>{open ? '▼' : '▶'}</span>
        技术映射
      </button>
      {open ? (
        <div
          style={{
            marginTop: 10,
            padding: '12px 14px',
            borderRadius: 10,
            border: '1px solid #e5eaf2',
            background: '#f8fafc',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}
        >
          {mappings.map((mapping) => (
            <div
              key={`tech-${mapping.metricId}-${mapping.taskType}-${mapping.evaluationMode}`}
              style={{
                paddingBottom: 12,
                borderBottom: '1px solid #eef2f7',
                fontSize: 11,
                lineHeight: 1.6,
                color: '#475569',
                fontFamily: 'ui-monospace, monospace',
              }}
            >
              <div>metricId: {mapping.metricId}</div>
              <div>
                sourceField: {mapping.sourceField ?? '—'}
                {mapping.sourceFields?.length ? ` | sourceFields: ${mapping.sourceFields.join(', ')}` : ''}
              </div>
              <div>calculationMode: {mapping.calculationMode}</div>
              <div>applicableTaskTypes: {mapping.taskType || '—'}</div>
              <div>applicableEvaluationModes: {mapping.evaluationMode || '—'}</div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function GenericMetricDetailDrawer({
  metric,
  onClose,
}: {
  metric: GenericMetricGroup | null;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!metric) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [metric, onClose]);

  if (!metric) return null;

  const valueTypeLabel =
    metric.valueType === 'ratio'
      ? 'ratio'
      : metric.valueType === 'integer'
        ? 'integer'
        : metric.valueType === 'composite'
          ? 'composite'
          : 'number';

  const unitLabel =
    metric.unit === '%'
      ? '%'
      : metric.unit?.trim()
        ? metric.unit
        : '无';

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="generic-metric-drawer-title">
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h2
              id="generic-metric-drawer-title"
              style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827' }}
            >
              {metric.displayName}
            </h2>
            <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>通用评测指标</div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <div style={sectionTitleStyle}>指标定义</div>
          <div
            style={{
              padding: '14px 16px',
              marginBottom: 20,
              borderRadius: 10,
              border: '1px solid #e5eaf2',
              backgroundColor: '#fafbfc',
            }}
          >
            <InfoRow label="指标名称">{metric.displayName}</InfoRow>
            <InfoRow label="指标状态">
              <StatusBadge implemented={metric.status === 'implemented'} />
            </InfoRow>
            <InfoRow label="值类型">{valueTypeLabel}</InfoRow>
            <InfoRow label="单位">{unitLabel}</InfoRow>
            <InfoRow label="指标说明">{metric.description}</InfoRow>
          </div>

          <div style={sectionTitleStyle}>适用任务</div>
          <TaskMappingList mappings={metric.mappings} />

          <TechnicalMappingSection mappings={metric.mappings} />

          <p
            style={{
              margin: '16px 0 0',
              fontSize: 12,
              color: '#6b7280',
              lineHeight: 1.6,
            }}
          >
            该指标为通用指标定义，具体数值由不同任务评测产物中的真实字段映射得到。
          </p>
        </div>

        <div style={{ padding: '14px 20px', borderTop: '1px solid #e5e7eb' }}>
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
      </aside>
    </>
  );
}
