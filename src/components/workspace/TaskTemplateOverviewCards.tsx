'use client';

import type { TaskTemplateOverviewStats } from '@/lib/workspace/taskTemplatePresentation';

interface TaskTemplateOverviewCardsProps {
  stats: TaskTemplateOverviewStats;
}

function OverviewCard({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div
      style={{
        height: 72,
        padding: '12px 16px',
        borderRadius: 12,
        border: '1px solid #e5eaf2',
        backgroundColor: '#fff',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 2,
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 12, color: '#6b7280' }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 22, fontWeight: 600, color: '#111827', lineHeight: 1.1 }}>{value}</span>
        {hint ? <span style={{ fontSize: 12, color: '#9ca3af' }}>{hint}</span> : null}
      </div>
    </div>
  );
}

export function TaskTemplateOverviewCards({ stats }: TaskTemplateOverviewCardsProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: 12,
        marginBottom: 16,
      }}
    >
      <OverviewCard label="已接入模板" value={stats.connectedCount} />
      <OverviewCard label="仿真后端" value={stats.backendCount} hint={stats.backendSummary} />
      <OverviewCard label="可训练模板" value={stats.trainableCount} />
      <OverviewCard label="可评测模板" value={stats.evaluableCount} />
    </div>
  );
}
