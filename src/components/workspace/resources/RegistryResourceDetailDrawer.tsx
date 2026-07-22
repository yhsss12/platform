'use client';

import { useEffect, useState } from 'react';
import type { ResourceItem } from '@/lib/mock/workspacePagesMock';
import {
  getRegistryResource,
  registryStatusLabel,
  type RegistryResource,
} from '@/lib/api/resourceRegistryClient';
import {
  extractStringList,
  formatThresholds,
  metricImplementationLabel,
  registryAssetTypeLabel,
  resolveRegistryScenarioLabel,
} from '@/lib/workspace/registryResourceDisplay';
import {
  formatMetricEvaluationModes,
  formatMetricTaskTypes,
} from '@/lib/workspace/evaluationMetricRegistry';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';

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
  width: 520,
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

function InfoBlock({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 13, color: '#111827', lineHeight: 1.55 }}>{children}</div>
    </div>
  );
}

function ListBlock({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <InfoBlock label={label}>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        {items.map((item) => (
          <li key={item} style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
            {item}
          </li>
        ))}
      </ul>
    </InfoBlock>
  );
}

export function RegistryResourceDetailDrawer({
  resource,
  fallback,
  onClose,
}: {
  resource: RegistryResource | null;
  fallback?: ResourceItem | null;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<RegistryResource | null>(resource);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showTechnical, setShowTechnical] = useState(false);

  const assetId = resource?.assetId ?? fallback?.id ?? null;

  useEffect(() => {
    if (!assetId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [assetId, onClose]);

  useEffect(() => {
    setShowTechnical(false);
    setLoadError(null);
    if (resource) {
      setDetail(resource);
      return;
    }
    if (!fallback?.id) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    void getRegistryResource(fallback.id)
      .then((row) => {
        if (!cancelled) setDetail(row);
      })
      .catch(() => {
        if (!cancelled) {
          setDetail(null);
          setLoadError('无法从注册表加载详情，将展示本地摘要信息');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [resource, fallback?.id]);

  if (!assetId) return null;

  const metadata = detail?.metadata ?? {};
  const name = detail?.name ?? fallback?.name ?? assetId;
  const description = detail?.description ?? fallback?.description ?? '—';
  const tags = detail?.tags ?? fallback?.tags ?? [];
  const status = detail?.status ?? (fallback?.status === 'active' ? 'available' : fallback?.status ?? 'unknown');
  const scenario = detail ? resolveRegistryScenarioLabel(detail) : fallback?.category ?? '—';
  const thresholds = formatThresholds(metadata);
  const isMetric = detail?.assetType === 'metric';
  const applicableTaskTypes = extractStringList(metadata.applicableTaskTypes);
  const applicableEvaluationModes = extractStringList(metadata.applicableEvaluationModes);
  const calculationMode = metadata.calculationMode ? String(metadata.calculationMode) : null;
  const sourceField = metadata.sourceField ? String(metadata.sourceField) : null;
  const sourceFields = extractStringList(metadata.sourceFields);
  const valueType = metadata.valueType ? String(metadata.valueType) : null;
  const unit = metadata.unit != null ? String(metadata.unit) : null;
  const inputFiles = extractStringList(metadata.input_files);
  const outputFields = extractStringList(metadata.output_fields);
  const reportFields = extractStringList(metadata.report_fields);
  const version = detail?.version ?? fallback?.version ?? '—';
  const updatedAt = detail?.lastModifiedAt
    ? formatDateTimeMinuteYmdSlash(detail.lastModifiedAt)
    : fallback?.updatedAt ?? '—';

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="registry-resource-drawer-title">
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
              id="registry-resource-drawer-title"
              style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827', lineHeight: 1.35 }}
            >
              {name}
            </h2>
            <div
              style={{
                marginTop: 6,
                fontSize: 12,
                color: '#6b7280',
                fontFamily: 'ui-monospace, monospace',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={assetId}
            >
              {assetId}
            </div>
            <div style={{ marginTop: 10 }}>
              <StatusBadge status="active" label={registryStatusLabel(status)} />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          {loading ? (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>加载详情…</p>
          ) : null}
          {loadError ? (
            <p style={{ fontSize: 13, color: '#b45309', marginBottom: 12 }}>{loadError}</p>
          ) : null}

          <div style={sectionTitleStyle}>{isMetric ? '指标定义' : '指标信息'}</div>
          <div
            style={{
              padding: '14px 16px',
              marginBottom: 20,
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              backgroundColor: '#f9fafb',
            }}
          >
            <InfoBlock label="描述">{description}</InfoBlock>
            {isMetric ? (
              <>
                <InfoBlock label="实现状态">
                  {detail ? metricImplementationLabel(detail) : '—'}
                </InfoBlock>
                <InfoBlock label="适用任务">
                  {formatMetricTaskTypes(applicableTaskTypes)}
                </InfoBlock>
                <InfoBlock label="适用评测模式">
                  {formatMetricEvaluationModes(applicableEvaluationModes)}
                </InfoBlock>
                {calculationMode ? <InfoBlock label="计算方式">{calculationMode}</InfoBlock> : null}
                {sourceField ? <InfoBlock label="来源字段">{sourceField}</InfoBlock> : null}
                <ListBlock label="来源字段（复合）" items={sourceFields} />
                {valueType ? <InfoBlock label="值类型">{valueType}</InfoBlock> : null}
                {unit != null && unit !== '' ? <InfoBlock label="单位">{unit || '无'}</InfoBlock> : null}
              </>
            ) : (
              <InfoBlock label="适用任务 / 场景">{scenario}</InfoBlock>
            )}
            {!isMetric ? (
              <InfoBlock label="资源类型">{registryAssetTypeLabel(detail?.assetType ?? fallback?.category)}</InfoBlock>
            ) : null}
            {tags.length > 0 ? (
              <InfoBlock label="标签">
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        borderRadius: 4,
                        backgroundColor: '#eff6ff',
                        color: '#1d4ed8',
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </InfoBlock>
            ) : null}
            {thresholds ? <InfoBlock label="统计口径 / 阈值">{thresholds}</InfoBlock> : null}
            <ListBlock label="输入文件 / 字段" items={inputFiles} />
            <ListBlock label="输出字段" items={outputFields} />
            <ListBlock label="报告字段" items={reportFields} />
            <InfoBlock label="版本">{version}</InfoBlock>
            <InfoBlock label="更新时间">{updatedAt}</InfoBlock>
          </div>

          <button
            type="button"
            onClick={() => setShowTechnical((v) => !v)}
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
            <span style={{ fontSize: 10, color: '#9ca3af' }}>{showTechnical ? '▼' : '▶'}</span>
            技术信息
          </button>

          {showTechnical ? (
            <div
              style={{
                marginTop: 10,
                padding: '12px 14px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                fontSize: 12,
                color: '#475569',
                lineHeight: 1.6,
                wordBreak: 'break-all',
              }}
            >
              <div>assetId：{assetId}</div>
              {detail?.manifestPath ? <div>manifestPath：{detail.manifestPath}</div> : null}
              {detail?.simBackend ? <div>simBackend：{detail.simBackend}</div> : null}
              {detail?.taskType ? <div>taskType：{detail.taskType}</div> : null}
              {Object.keys(detail?.files ?? {}).length > 0 ? (
                <div style={{ marginTop: 8 }}>
                  files：
                  <pre style={{ margin: '4px 0 0', whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(detail?.files, null, 2)}
                  </pre>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div style={{ padding: '14px 20px', borderTop: '1px solid #e5e7eb' }}>
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
      </aside>
    </>
  );
}
