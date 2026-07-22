'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import type { WorkspaceTask } from '@/lib/mock/workspaceTasksMock';
import { getStaticGenerateDataTemplateOptions } from '@/lib/workspace/generateDataTemplateOptions';
import { evaluationTemplateOptions } from '@/lib/mock/workspaceEvaluationRecordsMock';

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

const btnDanger: React.CSSProperties = { ...btnLink, color: '#dc2626' };

function pickTemplateOption(taskName: string, options: readonly string[]): string {
  if (options.includes(taskName)) return taskName;
  const partial = options.find(
    (option) =>
      taskName.includes(option.replace(/任务$/, '')) || option.includes(taskName.replace(/任务$/, ''))
  );
  return partial ?? options[0];
}

interface TaskTemplateLibraryTableProps {
  templates: WorkspaceTask[];
  onOpenDetail: (task: WorkspaceTask) => void;
  onDelete: (task: WorkspaceTask) => void;
}

export function TaskTemplateLibraryTable({
  templates,
  onOpenDetail,
  onDelete,
}: TaskTemplateLibraryTableProps) {
  const router = useRouter();

  const handleGenerateData = (task: WorkspaceTask) => {
    const template = pickTemplateOption(task.name, getStaticGenerateDataTemplateOptions());
    router.push(`/workspace/data?openGenerate=1&template=${encodeURIComponent(template)}`);
  };

  const handleEvaluation = (task: WorkspaceTask) => {
    const template = pickTemplateOption(task.name, evaluationTemplateOptions);
    router.push(`/workspace/evaluation?openCreate=1&template=${encodeURIComponent(template)}`);
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 960, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
        <thead>
          <tr>
            {[
              '模板名称',
              '领域',
              '类型',
              '默认场景',
              '默认机器人',
              '默认策略',
              '默认指标',
              '创建人',
              '操作',
            ].map((h) => (
              <th key={h} style={thStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {templates.map((task) => (
            <tr
              key={task.id}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f9fafb';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <td style={{ ...tdStyle, fontWeight: 500 }}>{task.name}</td>
              <td style={tdStyle}>{task.domain}</td>
              <td style={tdStyle}>{task.type}</td>
              <td style={tdStyle}>{task.scene}</td>
              <td style={{ ...tdStyle, fontSize: 12 }}>{task.robot}</td>
              <td style={{ ...tdStyle, fontSize: 12 }}>{task.policy}</td>
              <td style={{ ...tdStyle, fontSize: 12 }}>{task.metrics.slice(0, 2).join(' · ')}</td>
              <td style={tdStyle}>{task.creator}</td>
              <td style={tdStyle}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  <button type="button" style={btnLink} onClick={() => onOpenDetail(task)}>
                    详情
                  </button>
                  <button type="button" style={btnLink} onClick={() => handleGenerateData(task)}>
                    生成数据
                  </button>
                  <button type="button" style={btnLink} onClick={() => handleEvaluation(task)}>
                    评测
                  </button>
                  <Link href="/workspace/task-build/template" style={btnLink}>
                    任务配置
                  </Link>
                  <button type="button" style={btnDanger} onClick={() => onDelete(task)}>
                    删除
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
