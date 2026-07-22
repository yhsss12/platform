'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  statusBadgeStyle,
  trajectoryQualityBadgeStyle,
  type TaskTemplateAssetRow,
} from '@/lib/workspace/taskTemplatePresentation';
import { buildTaskBuildTemplateHref } from '@/lib/workspace/taskBuildNavigation';

const thStyle: React.CSSProperties = {
  padding: '14px 16px',
  textAlign: 'left',
  borderBottom: '1px solid #e5e7eb',
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  backgroundColor: '#f9fafb',
  whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '16px',
  fontSize: 13,
  borderBottom: '1px solid #f3f4f6',
  verticalAlign: 'middle',
};

const tagStyle: React.CSSProperties = {
  display: 'inline-block',
  padding: '2px 8px',
  borderRadius: 999,
  fontSize: 11,
  lineHeight: 1.5,
  border: '1px solid transparent',
  marginRight: 4,
  marginBottom: 4,
};

const btnPrimary: React.CSSProperties = {
  padding: '5px 10px',
  fontSize: 12,
  fontWeight: 500,
  color: '#fff',
  backgroundColor: '#2563eb',
  border: 'none',
  borderRadius: 6,
  cursor: 'pointer',
};

const btnLink: React.CSSProperties = {
  padding: '4px 6px',
  fontSize: 12,
  color: '#2563eb',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
};

const btnDisabled: React.CSSProperties = {
  ...btnPrimary,
  backgroundColor: '#e5e7eb',
  color: '#9ca3af',
  cursor: 'not-allowed',
};

const btnGhost: React.CSSProperties = {
  padding: '4px 8px',
  fontSize: 12,
  color: '#374151',
  background: '#fff',
  border: '1px solid #d1d5db',
  borderRadius: 6,
  cursor: 'pointer',
};

function CapabilityTag({ label }: { label: string }) {
  return (
    <span
      style={{
        ...tagStyle,
        backgroundColor: '#eff6ff',
        color: '#1d4ed8',
        borderColor: '#bfdbfe',
      }}
    >
      {label}
    </span>
  );
}

function RowMoreMenu({
  row,
  onShowDetail,
}: {
  row: TaskTemplateAssetRow;
  onShowDetail: (registryId: string) => void;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  const items = [
    {
      label: '启动评测',
      action: () => {
        router.push(
          `/workspace/evaluation?openCreate=1&taskTemplateId=${encodeURIComponent(row.templateId)}&template=${encodeURIComponent(row.name)}&taskConfigId=${encodeURIComponent(row.registryId)}`
        );
      },
    },
    {
      label: '创建任务配置',
      action: () => router.push(buildTaskBuildTemplateHref(row.templateId)),
    },
    {
      label: '查看历史数据',
      action: () => {
        router.push(`/workspace/data?taskTemplateId=${encodeURIComponent(row.templateId)}`);
      },
    },
    {
      label: '查看指标',
      action: () => router.push('/workspace/resources/metrics'),
    },
  ];

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-block' }}>
      <button type="button" style={btnGhost} onClick={() => setOpen((v) => !v)}>
        更多
      </button>
      {open ? (
        <div
          style={{
            position: 'absolute',
            right: 0,
            top: 'calc(100% + 4px)',
            minWidth: 148,
            backgroundColor: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            boxShadow: '0 8px 24px rgba(15,23,42,0.12)',
            zIndex: 20,
            padding: 4,
          }}
        >
          {items
            .filter((item) => item.label !== '启动评测' || (row.template.hasEvaluationRunner ?? row.supportsEvaluation))
            .map((item) => (
            <button
              key={item.label}
              type="button"
              onClick={() => {
                setOpen(false);
                item.action();
              }}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                padding: '8px 10px',
                fontSize: 13,
                color: '#374151',
                background: 'none',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f3f4f6';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              {item.label}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              onShowDetail(row.registryId);
            }}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              padding: '8px 10px',
              fontSize: 13,
              color: '#374151',
              background: 'none',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
            }}
          >
            查看详情
          </button>
        </div>
      ) : null}
    </div>
  );
}

interface TaskTemplateAssetTableProps {
  rows: TaskTemplateAssetRow[];
  onShowDetail: (registryId: string) => void;
  /** 主表默认不展示 taskType；实验区可开启 */
  showTaskType?: boolean;
}

export function TaskTemplateAssetTable({
  rows,
  onShowDetail,
  showTaskType = false,
}: TaskTemplateAssetTableProps) {
  const router = useRouter();

  const handleGenerateData = (row: TaskTemplateAssetRow) => {
    router.push(
      `/workspace/data?openGenerate=1&taskTemplateId=${encodeURIComponent(row.templateId)}&template=${encodeURIComponent(row.name)}&taskConfigId=${encodeURIComponent(row.registryId)}`
    );
  };

  return (
    <div
      style={{
        overflowX: 'auto',
        borderRadius: 14,
        border: '1px solid #e5e7eb',
        backgroundColor: '#fff',
      }}
    >
      <table style={{ width: '100%', minWidth: 1120, borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {[
              '任务模板',
              '仿真后端',
              '专家策略',
              '数据能力',
              '训练/评测',
              '轨迹质量',
              '状态',
              '操作',
            ].map((h) => (
              <th key={h} style={thStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const qualityStyle = trajectoryQualityBadgeStyle(row.trajectoryQualitySeverity);
            const statusStyle = statusBadgeStyle(row.statusKey);
            return (
              <tr
                key={row.registryId}
                style={{ minHeight: 84 }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#fafbfc';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ fontWeight: 600, color: '#111827', marginBottom: 4 }}>{row.name}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.5, marginBottom: 6 }}>
                    {row.shortDescription}
                  </div>
                  {showTaskType ? (
                    <div style={{ fontSize: 11, color: '#9ca3af', fontFamily: 'monospace' }}>
                      {row.taskTypeLabel}
                    </div>
                  ) : null}
                </td>
                <td style={{ ...tdStyle, minWidth: 120 }}>
                  <span
                    style={{
                      ...tagStyle,
                      backgroundColor: '#f1f5f9',
                      color: '#334155',
                      borderColor: '#e2e8f0',
                      fontWeight: 500,
                    }}
                  >
                    {row.simulatorLabel}
                  </span>
                  <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 6 }}>{row.simulatorSubtitle}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 130 }}>
                  <div
                    style={{ fontWeight: 500, color: '#111827' }}
                    title={row.expertStrategyTooltip}
                  >
                    {row.expertStrategyLabel}
                  </div>
                  {row.expertStrategyReady ? (
                    <span
                      style={{
                        ...tagStyle,
                        marginTop: 6,
                        backgroundColor: '#ecfdf5',
                        color: '#047857',
                        borderColor: '#a7f3d0',
                      }}
                    >
                      已接入
                    </span>
                  ) : (
                    <span
                      style={{
                        ...tagStyle,
                        marginTop: 6,
                        backgroundColor: '#f3f4f6',
                        color: '#6b7280',
                        borderColor: '#e5e7eb',
                      }}
                    >
                      待接入
                    </span>
                  )}
                </td>
                <td style={{ ...tdStyle, minWidth: 160 }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {row.dataCapabilityTags.map((tag) => (
                      <CapabilityTag key={tag} label={tag} />
                    ))}
                  </div>
                </td>
                <td style={{ ...tdStyle, minWidth: 160 }}>
                  <div style={{ fontSize: 12, color: '#374151', lineHeight: 1.6 }}>
                    <div>
                      <span style={{ color: '#9ca3af' }}>训练：</span>
                      {row.trainingLabel}
                    </div>
                    <div>
                      <span style={{ color: '#9ca3af' }}>评测：</span>
                      {row.evaluationLabel}
                    </div>
                  </div>
                </td>
                <td style={{ ...tdStyle, minWidth: 120 }}>
                  <span
                    style={{
                      ...tagStyle,
                      backgroundColor: qualityStyle.backgroundColor,
                      color: qualityStyle.color,
                      borderColor: qualityStyle.borderColor,
                    }}
                  >
                    {row.trajectoryQualityLabel}
                  </span>
                </td>
                <td style={{ ...tdStyle, minWidth: 88 }}>
                  <span
                    style={{
                      ...tagStyle,
                      backgroundColor: statusStyle.backgroundColor,
                      color: statusStyle.color,
                      borderColor: statusStyle.borderColor,
                    }}
                  >
                    {row.statusLabel}
                  </span>
                </td>
                <td style={{ ...tdStyle, minWidth: 180 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'nowrap' }}>
                    {(row.template.hasExpertPolicy ?? row.supportsDataGeneration) ? (
                      <button type="button" style={btnPrimary} onClick={() => handleGenerateData(row)}>
                        生成数据
                      </button>
                    ) : (
                      <button
                        type="button"
                        style={btnDisabled}
                        disabled
                        title="当前任务模板未接入专家策略，暂不支持数据生成"
                      >
                        生成数据
                      </button>
                    )}
                    <button type="button" style={btnLink} onClick={() => onShowDetail(row.registryId)}>
                      详情
                    </button>
                    <RowMoreMenu row={row} onShowDetail={onShowDetail} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
