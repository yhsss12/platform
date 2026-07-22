'use client';

import type { RegistryResource } from '@/lib/api/resourceRegistryClient';
import { resolveEvaluationReportRobotDisplay } from '@/lib/workspace/evaluationReport';

interface TaskResourceConfigSectionProps {
  metadata: Record<string, unknown> | null | undefined;
  /** 默认折叠高级配置 */
  collapsed?: boolean;
  taskType?: string | null;
  reportPayload?: Record<string, unknown> | null;
}

function formatList(value: unknown): string {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.join('、') : '未记录';
  }
  return value != null && value !== '' ? String(value) : '未记录';
}

function formatResolvedGroup(group: unknown): string {
  if (!group) return '未记录';
  if (Array.isArray(group)) {
    return group.length > 0
      ? group
          .map((item) =>
            typeof item === 'object' && item && 'name' in item
              ? String((item as RegistryResource).name)
              : String(item)
          )
          .join('\n')
      : '未记录';
  }
  if (typeof group === 'object' && group && 'name' in group) {
    return String((group as RegistryResource).name);
  }
  return String(group);
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div
        style={{
          fontSize: 14,
          color: value === '未记录' ? '#9ca3af' : '#111827',
          wordBreak: 'break-word',
          whiteSpace: 'pre-line',
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function TaskResourceConfigSection({
  metadata,
  collapsed = true,
  taskType,
  reportPayload,
}: TaskResourceConfigSectionProps) {
  const meta = metadata ?? {};
  const snapshot = (meta.manifestSnapshot as Record<string, unknown> | undefined) ?? {};
  const resolved = (meta.resolvedResources as Record<string, unknown> | undefined) ?? {};
  const robotLabel = resolveEvaluationReportRobotDisplay({
    metadata: meta,
    taskType,
    reportPayload,
  });

  const summaryItems = [
    { label: '任务配置', value: formatList(meta.taskConfigId) },
    { label: '仿真后端', value: formatList(meta.simBackend) },
    { label: '任务版本', value: formatList(meta.taskVersion ?? meta.resourceRegistryVersion) },
    {
      label: '机器人',
      value: robotLabel,
    },
    { label: '场景', value: formatResolvedGroup(resolved.scenes) },
    {
      label: '操作对象',
      value: formatResolvedGroup(resolved.objects ?? resolved.endEffectors),
    },
    { label: '评测指标', value: formatList(meta.metricIds) },
    { label: '策略', value: formatResolvedGroup(resolved.policies) },
  ];

  const snapshotText =
    snapshot && Object.keys(snapshot).length > 0
      ? JSON.stringify(snapshot, null, 2)
      : '未记录';

  return (
    <div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: 12,
          marginBottom: collapsed ? 0 : 12,
        }}
      >
        {summaryItems.map((item) => (
          <InfoItem key={item.label} label={item.label} value={item.value} />
        ))}
      </div>

      <details
        style={{ marginTop: 12 }}
        open={!collapsed ? true : undefined}
      >
        <summary
          style={{
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 600,
            color: '#374151',
            userSelect: 'none',
            listStylePosition: 'outside',
          }}
        >
          任务配置详情
        </summary>
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>配置快照</div>
          <pre
            style={{
              margin: 0,
              padding: '12px 14px',
              borderRadius: 8,
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              fontSize: 11,
              lineHeight: 1.55,
              overflow: 'auto',
              maxHeight: 240,
              color: '#334155',
            }}
          >
            {snapshotText}
          </pre>
        </div>
      </details>
    </div>
  );
}
