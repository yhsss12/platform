'use client';

import Link from 'next/link';
import type { WorkspaceTask } from '@/lib/mock/workspaceTasksMock';
import {
  formatWorkspaceDataCell,
  formatWorkspaceEvalCell,
  isWorkspaceDataMuted,
  isWorkspaceEvalMuted,
  isWorkspaceEvalSuccess,
} from '@/lib/mock/workspaceTasksMock';

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
  color: '#ffffff',
  fontSize: 12,
  fontWeight: 500,
  cursor: 'pointer',
  textDecoration: 'none',
  display: 'inline-flex',
  alignItems: 'center',
};

const btnPrimaryDisabled: React.CSSProperties = {
  ...btnPrimary,
  backgroundColor: '#e5e7eb',
  color: '#9ca3af',
  cursor: 'not-allowed',
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

export const WORKSPACE_TASK_ROUTES = {
  config: '/workspace/task-generation',
  simulation: '/workspace/simulation',
  data: '/workspace/data',
  evaluation: '/workspace/evaluation',
} as const;

interface WorkspaceTaskTableProps {
  tasks: WorkspaceTask[];
  onOpenDetail: (task: WorkspaceTask) => void;
  onDelete: (task: WorkspaceTask) => void;
  emptyMessage?: string;
}

export function WorkspaceTaskTable({
  tasks,
  onOpenDetail,
  onDelete,
  emptyMessage = '暂无匹配任务，请调整筛选条件',
}: WorkspaceTaskTableProps) {
  const colCount = 11;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 1080, borderCollapse: 'collapse', backgroundColor: '#ffffff' }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, minWidth: 140 }}>任务</th>
            <th style={thStyle}>领域</th>
            <th style={thStyle}>类型</th>
            <th style={{ ...thStyle, minWidth: 120 }}>场景</th>
            <th style={{ ...thStyle, minWidth: 110 }}>机器人</th>
            <th style={{ ...thStyle, minWidth: 100 }}>策略</th>
            <th style={thStyle}>数据</th>
            <th style={thStyle}>评测</th>
            <th style={{ ...thStyle, width: 96 }}>最近运行</th>
            <th style={{ ...thStyle, width: 72 }}>创建人</th>
            <th style={{ ...thStyle, minWidth: 220 }}>操作</th>
          </tr>
        </thead>
        <tbody>
          {tasks.length === 0 ? (
            <tr>
              <td colSpan={colCount} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280', padding: 40 }}>
                {emptyMessage}
              </td>
            </tr>
          ) : (
            tasks.map((task) => {
              const canSimulate = task.status !== 'pending_config';
              const dataText = formatWorkspaceDataCell(task);
              const evalText = formatWorkspaceEvalCell(task);

              return (
                <tr
                  key={task.id}
                  style={{ transition: 'background-color 0.15s' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f9fafb';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <td style={tdStyle}>
                    <div style={{ fontWeight: 500, color: '#111827' }}>{task.name}</div>
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
                  <td
                    style={{
                      ...tdStyle,
                      fontSize: 12,
                      fontWeight: isWorkspaceEvalSuccess(evalText) ? 600 : 400,
                      color: isWorkspaceEvalSuccess(evalText)
                        ? '#059669'
                        : isWorkspaceEvalMuted(evalText)
                          ? '#9ca3af'
                          : '#374151',
                    }}
                  >
                    {evalText}
                  </td>
                  <td style={tdCompact}>{task.lastRunTime}</td>
                  <td style={tdCompact}>{task.creator}</td>
                  <td style={tdStyle} onClick={(e) => e.stopPropagation()}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4 }}>
                      {canSimulate ? (
                        <Link href={WORKSPACE_TASK_ROUTES.simulation} style={btnPrimary}>
                          仿真
                        </Link>
                      ) : (
                        <span style={btnPrimaryDisabled} title="请先完成仿真配置">
                          需配置
                        </span>
                      )}
                      <button type="button" style={btnLink} onClick={() => onOpenDetail(task)}>
                        详情
                      </button>
                      <Link href={WORKSPACE_TASK_ROUTES.config} style={btnLink}>
                        配置
                      </Link>
                      <Link href={WORKSPACE_TASK_ROUTES.data} style={btnLink}>
                        数据
                      </Link>
                      <Link href={WORKSPACE_TASK_ROUTES.evaluation} style={btnLink}>
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
