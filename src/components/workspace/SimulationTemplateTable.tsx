'use client';

import Link from 'next/link';
import type { WorkspaceTask } from '@/lib/mock/workspaceTasksMock';
import { formatWorkspaceDataCell, isWorkspaceDataMuted } from '@/lib/mock/workspaceTasksMock';

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

const tdCompact: React.CSSProperties = {
  ...tdStyle,
  padding: '12px 10px',
  fontSize: 12,
  color: '#6b7280',
  whiteSpace: 'nowrap',
};

const btnPrimary: React.CSSProperties = {
  padding: '4px 8px',
  backgroundColor: '#2563eb',
  border: 'none',
  borderRadius: 6,
  color: '#fff',
  fontSize: 12,
  fontWeight: 500,
  cursor: 'pointer',
  textDecoration: 'none',
  display: 'inline-flex',
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

const btnDanger: React.CSSProperties = {
  padding: '4px 6px',
  fontSize: 12,
  color: '#dc2626',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
};

export const SIMULATION_ROUTES = {
  console: '/workspace/simulation/console',
  config: '/workspace/task-generation',
  data: '/workspace/data',
  evaluation: '/workspace/evaluation',
} as const;

interface SimulationTemplateTableProps {
  templates: WorkspaceTask[];
  onOpenDetail: (task: WorkspaceTask) => void;
  onDelete: (task: WorkspaceTask) => void;
}

export function SimulationTemplateTable({
  templates,
  onOpenDetail,
  onDelete,
}: SimulationTemplateTableProps) {
  const colCount = 10;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 1000, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, minWidth: 140 }}>任务模板</th>
            <th style={thStyle}>领域</th>
            <th style={thStyle}>类型</th>
            <th style={{ ...thStyle, minWidth: 120 }}>默认场景</th>
            <th style={{ ...thStyle, minWidth: 110 }}>机器人</th>
            <th style={{ ...thStyle, minWidth: 100 }}>策略</th>
            <th style={thStyle}>数据生成</th>
            <th style={{ ...thStyle, width: 96 }}>最近仿真</th>
            <th style={{ ...thStyle, width: 72 }}>创建人</th>
            <th style={{ ...thStyle, minWidth: 220 }}>操作</th>
          </tr>
        </thead>
        <tbody>
          {templates.length === 0 ? (
            <tr>
              <td colSpan={colCount} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280', padding: 40 }}>
                暂无匹配仿真任务
              </td>
            </tr>
          ) : (
            templates.map((task) => {
              const dataText = formatWorkspaceDataCell(task);
              return (
                <tr
                  key={task.id}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f9fafb';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <td style={tdStyle}>
                    <div style={{ fontWeight: 500 }}>{task.name}</div>
                  </td>
                  <td style={tdStyle}>{task.domain}</td>
                  <td style={tdStyle}>{task.type}</td>
                  <td style={tdStyle}>{task.scene}</td>
                  <td style={{ ...tdStyle, fontSize: 12 }}>{task.robot}</td>
                  <td style={{ ...tdStyle, fontSize: 12 }}>{task.policy}</td>
                  <td
                    style={{
                      ...tdStyle,
                      fontSize: 12,
                      color: isWorkspaceDataMuted(dataText) ? '#9ca3af' : '#374151',
                    }}
                  >
                    {dataText}
                  </td>
                  <td style={tdCompact}>{task.lastRunTime}</td>
                  <td style={tdCompact}>{task.creator}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      <Link href={SIMULATION_ROUTES.console} style={btnPrimary}>
                        仿真
                      </Link>
                      <button type="button" style={btnLink} onClick={() => onOpenDetail(task)}>
                        详情
                      </button>
                      <Link href={SIMULATION_ROUTES.config} style={btnLink} title="配置仿真">
                        配置
                      </Link>
                      <Link href={SIMULATION_ROUTES.data} style={btnLink}>
                        数据
                      </Link>
                      <Link href={SIMULATION_ROUTES.evaluation} style={btnLink}>
                        评测
                      </Link>
                      <button type="button" style={btnDanger} onClick={() => onDelete(task)}>
                        删除
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
