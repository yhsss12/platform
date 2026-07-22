'use client';

import type { ReportBasicInfoField } from '@/lib/workspace/evaluationReportBasicInfo';

export function EvaluationReportBasicInfoSection({
  items,
}: {
  items: ReportBasicInfoField[];
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: '12px 24px',
      }}
    >
      {items.map((item) => (
        <div key={item.label}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
          <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}
