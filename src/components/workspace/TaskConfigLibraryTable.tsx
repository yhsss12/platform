'use client';

import { useRouter } from 'next/navigation';
import type { TaskConfigSummary } from '@/lib/api/resourceRegistryClient';
import { buildTaskBuildTemplateHref } from '@/lib/workspace/taskBuildNavigation';

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

interface TaskConfigLibraryTableProps {
  taskConfigs: TaskConfigSummary[];
  taskTemplateIds?: Array<{ registryId: string; templateId: string }>;
  onShowDependencies: (taskConfigId: string) => void;
}

export function TaskConfigLibraryTable({
  taskConfigs,
  taskTemplateIds = [],
  onShowDependencies,
}: TaskConfigLibraryTableProps) {
  const router = useRouter();

  const resolveTemplateId = (task: TaskConfigSummary) => {
    const hit = taskTemplateIds.find((m) => m.registryId === task.assetId);
    return hit?.templateId ?? task.assetId;
  };

  const handleGenerateData = (task: TaskConfigSummary) => {
    const templateId = resolveTemplateId(task);
    router.push(
      `/workspace/data?openGenerate=1&taskTemplateId=${encodeURIComponent(templateId)}&template=${encodeURIComponent(task.name)}&taskConfigId=${encodeURIComponent(task.assetId)}`
    );
  };

  const handleEvaluation = (task: TaskConfigSummary) => {
    const templateId = resolveTemplateId(task);
    router.push(
      `/workspace/evaluation?openCreate=1&taskTemplateId=${encodeURIComponent(templateId)}&template=${encodeURIComponent(task.name)}&taskConfigId=${encodeURIComponent(task.assetId)}`
    );
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', minWidth: 880, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
        <thead>
          <tr>
            {['任务名称', 'taskType', 'simBackend', '版本', '状态', '依赖资源', '指标', '操作'].map((h) => (
              <th key={h} style={thStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {taskConfigs.map((task) => (
            <tr
              key={task.assetId}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f9fafb';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <td style={{ ...tdStyle, fontWeight: 500 }}>{task.name}</td>
              <td style={tdStyle}>{task.taskType ?? '—'}</td>
              <td style={tdStyle}>{task.simBackend}</td>
              <td style={tdStyle}>{task.version}</td>
              <td style={tdStyle}>{task.status}</td>
              <td style={tdStyle}>{task.requiredAssetsCount}</td>
              <td style={tdStyle}>{task.metricsCount}</td>
              <td style={tdStyle}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  <button type="button" style={btnLink} onClick={() => handleGenerateData(task)}>
                    生成数据
                  </button>
                  <button type="button" style={btnLink} onClick={() => handleEvaluation(task)}>
                    启动评测
                  </button>
                  <button
                    type="button"
                    style={btnLink}
                    onClick={() => router.push(buildTaskBuildTemplateHref(resolveTemplateId(task)))}
                  >
                    创建任务配置
                  </button>
                  <button
                    type="button"
                    style={btnLink}
                    onClick={() => onShowDependencies(task.assetId)}
                  >
                    查看详情
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
