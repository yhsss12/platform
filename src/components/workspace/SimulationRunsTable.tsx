'use client';

import Link from 'next/link';
import type { SimulationTaskRun } from '@/lib/mock/workspaceMockFlowStore';

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

const statusColor: Record<string, string> = {
  待运行: '#6b7280',
  运行中: '#2563eb',
  已完成: '#059669',
  失败: '#dc2626',
};

export function SimulationRunsTable({
  runs,
  onEnterConsole,
  onDelete,
}: {
  runs: SimulationTaskRun[];
  onEnterConsole: (run: SimulationTaskRun) => void;
  onDelete: (run: SimulationTaskRun) => void;
}) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 960, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
        <thead>
          <tr>
            <th style={thStyle}>任务模板</th>
            <th style={thStyle}>场景</th>
            <th style={thStyle}>机器人</th>
            <th style={thStyle}>策略模型</th>
            <th style={thStyle}>状态</th>
            <th style={thStyle}>进度</th>
            <th style={thStyle}>轮次</th>
            <th style={thStyle}>seed</th>
            <th style={thStyle}>创建时间</th>
            <th style={{ ...thStyle, minWidth: 160 }}>操作</th>
          </tr>
        </thead>
        <tbody>
          {runs.length === 0 ? (
            <tr>
              <td colSpan={10} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280', padding: 40 }}>
                暂无仿真任务，点击右上角「新建任务」创建
              </td>
            </tr>
          ) : (
            runs.map((run) => (
              <tr
                key={run.id}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <td style={{ ...tdStyle, fontWeight: 500 }}>{run.template}</td>
                <td style={tdStyle}>{run.scene}</td>
                <td style={{ ...tdStyle, fontSize: 12 }}>{run.robot}</td>
                <td style={{ ...tdStyle, fontSize: 12 }}>{run.policy}</td>
                <td style={{ ...tdStyle, color: statusColor[run.status] ?? '#374151', fontWeight: 500 }}>
                  {run.status}
                </td>
                <td style={tdStyle}>{run.progressPercent}%</td>
                <td style={tdStyle}>{run.rounds}</td>
                <td style={tdStyle}>{run.seed}</td>
                <td style={{ ...tdStyle, fontSize: 12, color: '#6b7280' }}>{run.createdAt}</td>
                <td style={tdStyle}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    <button
                      type="button"
                      onClick={() => onEnterConsole(run)}
                      style={{
                        padding: '4px 10px',
                        fontSize: 12,
                        borderRadius: 6,
                        border: 'none',
                        backgroundColor: '#2563eb',
                        color: '#fff',
                        cursor: 'pointer',
                      }}
                    >
                      控制台
                    </button>
                    <Link
                      href="/workspace/data"
                      style={{ padding: '4px 8px', fontSize: 12, color: '#2563eb', textDecoration: 'none' }}
                    >
                      数据
                    </Link>
                    <button
                      type="button"
                      onClick={() => onDelete(run)}
                      style={{
                        padding: '4px 8px',
                        fontSize: 12,
                        color: '#dc2626',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                      }}
                    >
                      删除
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
