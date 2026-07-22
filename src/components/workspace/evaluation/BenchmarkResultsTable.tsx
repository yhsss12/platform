'use client';

import Link from 'next/link';
import type { BenchmarkResultRow } from '@/lib/mock/workspaceEvaluationMock';
import { formatBenchmarkSampleCount } from '@/lib/mock/workspaceEvaluationMock';

const thStyle: React.CSSProperties = {
  padding: '12px 16px',
  textAlign: 'left',
  borderBottom: '1px solid #e5e7eb',
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  backgroundColor: '#f9fafb',
  whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  fontSize: 13,
  color: '#111827',
  borderBottom: '1px solid #f3f4f6',
  verticalAlign: 'top',
};

const btnLink: React.CSSProperties = {
  padding: '4px 6px',
  fontSize: 12,
  color: '#2563eb',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  textDecoration: 'none',
};

export const EVAL_ROUTES = {
  replay: '/workspace/replay',
} as const;

interface BenchmarkResultsTableProps {
  rows: BenchmarkResultRow[];
  onDetail: (row: BenchmarkResultRow) => void;
  onExportReport: (row: BenchmarkResultRow) => void;
}

export function BenchmarkResultsTable({
  rows,
  onDetail,
  onExportReport,
}: BenchmarkResultsTableProps) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 1180, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
        <thead>
          <tr>
            {[
              '任务',
              '样本数量',
              '评测后端',
              '评测轮次',
              '模型类型',
              '模型版本',
              '成功率',
              '平均耗时',
              '碰撞次数',
              '失败摘要',
              '操作',
            ].map((h) => (
              <th key={h} style={thStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={11} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280', padding: 40 }}>
                暂无匹配评测结果
              </td>
            </tr>
          ) : (
            rows.map((row) => (
              <tr
                key={row.id}
                style={{ transition: 'background-color 0.15s' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <td style={{ ...tdStyle, fontWeight: 500 }}>{row.taskName}</td>
                <td style={tdStyle}>{formatBenchmarkSampleCount(row.dataVolume)}</td>
                <td style={tdStyle}>{row.evalBackend}</td>
                <td style={tdStyle}>{row.evalRounds}</td>
                <td style={tdStyle}>{row.modelType}</td>
                <td style={{ ...tdStyle, fontFamily: 'ui-monospace, monospace', fontSize: 11 }}>
                  {row.checkpoint}
                </td>
                <td style={{ ...tdStyle, fontWeight: 600, color: '#059669' }}>{row.successRate}%</td>
                <td style={tdStyle}>{row.avgDurationSec}s</td>
                <td style={tdStyle}>{row.collisionCount}</td>
                <td style={{ ...tdStyle, fontSize: 12, color: '#6b7280', maxWidth: 180 }}>
                  {row.failureSummary}
                </td>
                <td style={tdStyle}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
                    <button type="button" style={btnLink} onClick={() => onDetail(row)}>
                      详情
                    </button>
                    <Link href={EVAL_ROUTES.replay} style={btnLink}>
                      回放
                    </Link>
                    <button type="button" style={btnLink} onClick={() => onExportReport(row)}>
                      报告
                    </button>
                  </div>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
