'use client';

import type { CreateTaskFormState } from '@/lib/mock/workspaceTaskCreateOptions';
import { MockActionHint } from '@/components/workspace/workspaceUi';

const panelStyle: React.CSSProperties = {
  backgroundColor: '#ffffff',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
  padding: '20px 20px 16px',
  position: 'sticky',
  top: 24,
};

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, color: '#111827', lineHeight: 1.45, wordBreak: 'break-word' }}>
        {value || '—'}
      </div>
    </div>
  );
}

export function CreateTaskSummaryPanel({ form }: { form: CreateTaskFormState }) {
  const objects =
    form.objects.length > 0 ? form.objects.join('、') : '—';
  const metrics =
    form.metrics.length > 0 ? form.metrics.join('、') : '—';
  const dataGen = form.generateData
    ? `${form.dataTypes.join('、') || '—'} · ${form.exportFormat}`
    : '不生成数据';

  return (
    <aside style={panelStyle} aria-label="配置摘要">
      <div style={{ fontSize: 15, fontWeight: 600, color: '#111827', marginBottom: 4 }}>
        配置摘要
      </div>
      <p style={{ fontSize: 12, color: '#9ca3af', margin: '0 0 16px' }}>
        随表单填写实时更新
      </p>

      <SummaryRow label="任务名称" value={form.name || '（未填写）'} />
      <SummaryRow label="任务领域" value={form.domain} />
      <SummaryRow label="任务类型" value={form.type} />
      <SummaryRow label="场景" value={form.scene} />
      <SummaryRow label="操作对象" value={objects} />
      <SummaryRow label="机器人" value={form.robot} />
      <SummaryRow label="策略模型" value={form.policy} />
      <SummaryRow label="评测指标" value={metrics} />
      <SummaryRow label="数据生成" value={dataGen} />

      <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid #f3f4f6' }}>
        <MockActionHint />
      </div>
    </aside>
  );
}
