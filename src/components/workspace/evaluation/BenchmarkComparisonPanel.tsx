'use client';

import {
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import type { BenchmarkComparisonGroup } from '@/lib/mock/workspaceEvaluationMock';
import { BenchmarkResultsTable } from '@/components/workspace/evaluation/BenchmarkResultsTable';
import type { BenchmarkResultRow } from '@/lib/mock/workspaceEvaluationMock';

function ComparisonCard({ group }: { group: BenchmarkComparisonGroup }) {
  const bestRate = group.maxSuccessRate;

  return (
    <div
      style={{
        padding: '14px 16px',
        borderRadius: 12,
        border: '1px solid #dbeafe',
        background: 'linear-gradient(135deg, #ffffff 0%, #f8fbff 100%)',
        boxShadow: '0 4px 14px rgba(37, 99, 235, 0.06)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>{group.taskName}</div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>
            {group.dataVolume} · {group.evalBackend} · {group.evalRounds} 轮
          </div>
        </div>
        <div
          style={{
            flexShrink: 0,
            padding: '4px 8px',
            borderRadius: 6,
            fontSize: 12,
            fontWeight: 600,
            color: '#047857',
            backgroundColor: '#ecfdf5',
            border: '1px solid #a7f3d0',
          }}
        >
          最高 {bestRate}%
        </div>
      </div>

      <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#475569' }}>模型表现</div>
        {group.entries.map((entry) => {
          const isBest = entry.successRate === bestRate;
          return (
            <div key={`${entry.modelType}-${entry.modelVersion}`} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span
                style={{
                  width: 108,
                  flexShrink: 0,
                  fontSize: 12,
                  color: '#334155',
                  fontWeight: isBest ? 600 : 400,
                }}
              >
                {entry.modelType}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 6,
                  borderRadius: 999,
                  backgroundColor: '#e2e8f0',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${entry.successRate}%`,
                    height: '100%',
                    borderRadius: 999,
                    backgroundColor: isBest ? '#059669' : '#60a5fa',
                  }}
                />
              </div>
              <span
                style={{
                  width: 48,
                  flexShrink: 0,
                  textAlign: 'right',
                  fontSize: 12,
                  fontWeight: 600,
                  color: isBest ? '#047857' : '#2563eb',
                }}
              >
                {entry.successRate}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function BenchmarkComparisonPanel({
  groups,
  rows,
  onDetail,
  onExportReport,
}: {
  groups: BenchmarkComparisonGroup[];
  rows: BenchmarkResultRow[];
  onDetail: (row: BenchmarkResultRow) => void;
  onExportReport: (row: BenchmarkResultRow) => void;
}) {
  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 15, fontWeight: 600, color: '#111827' }}>
        Benchmark 对比
      </div>
      <p style={{ margin: '0 0 14px', fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
        同一任务、同一样本数量、同一评测后端和评测轮次下，不同模型版本的成功率对比。
      </p>

      {groups.length > 0 ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: 12,
            marginBottom: 16,
          }}
        >
          {groups.map((group) => (
            <ComparisonCard key={group.key} group={group} />
          ))}
        </div>
      ) : null}

      <ModulePageTableCard>
        <BenchmarkResultsTable rows={rows} onDetail={onDetail} onExportReport={onExportReport} />
      </ModulePageTableCard>
    </div>
  );
}
