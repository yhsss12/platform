'use client';

import Link from 'next/link';
import type { FailureCaseRow } from '@/lib/mock/workspaceEvaluationMock';
import { WS } from '@/components/workspace/workspaceUi';

export function FailureCasesPanel({ cases }: { cases: FailureCaseRow[] }) {
  return (
    <div style={{ ...WS.card, padding: 16, marginBottom: 24 }}>
      <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#111827' }}>
        最近失败案例
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
            {['任务', '失败阶段', '失败原因', '关联数据', '操作'].map((h) => (
              <th
                key={h}
                style={{
                  padding: '10px 12px',
                  textAlign: 'left',
                  fontWeight: 600,
                  color: '#374151',
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cases.map((c) => (
            <tr key={c.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
              <td style={{ padding: '10px 12px', fontWeight: 500 }}>{c.taskName}</td>
              <td style={{ padding: '10px 12px', color: '#d97706' }}>{c.failedStage}</td>
              <td style={{ padding: '10px 12px', color: '#6b7280' }}>{c.reason}</td>
              <td
                style={{
                  padding: '10px 12px',
                  fontFamily: 'ui-monospace, monospace',
                  fontSize: 12,
                }}
              >
                {c.dataName}
              </td>
              <td style={{ padding: '10px 12px' }}>
                <Link href="/workspace/replay" style={{ fontSize: 12, color: '#2563eb', marginRight: 12 }}>
                  查看回放
                </Link>
                <Link href="/workspace/data" style={{ fontSize: 12, color: '#2563eb' }}>
                  查看数据
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
