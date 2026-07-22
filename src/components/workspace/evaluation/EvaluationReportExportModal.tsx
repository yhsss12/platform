'use client';

import { useMemo, useState } from 'react';
import {
  WorkspaceCenteredModal,
  workspaceModalSectionLabel,
} from '@/components/workspace/WorkspaceCenteredModal';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  downloadEvaluationReport,
  type EvaluationReportExportFormat,
  type EvaluationReportExportOptions,
} from '@/lib/api/evaluationClient';
import { resolveExportEvaluationJobId } from '@/lib/workspace/evaluationJobId';

const FORMAT_OPTIONS: { id: EvaluationReportExportFormat; label: string }[] = [
  { id: 'pdf', label: 'PDF' },
  { id: 'docx', label: 'Word / DOCX' },
  { id: 'json', label: 'JSON' },
  { id: 'markdown', label: 'Markdown' },
  { id: 'xlsx', label: 'Excel / XLSX' },
  { id: 'csv', label: 'CSV（ZIP 多表）' },
  { id: 'latex', label: 'LaTeX' },
  { id: 'zip', label: 'ZIP 报告包' },
];

const DEFAULT_OPTIONS: EvaluationReportExportOptions = {
  includeBasicInfo: true,
  includeConfig: true,
  includeMetrics: true,
  includeEpisodes: true,
  includeVideoInfo: true,
  includeDiagnostics: true,
  includeRuntimeIndex: true,
  includeUnavailableMetricReasons: true,
};

export function EvaluationReportExportModal({
  open,
  evalJobId,
  onClose,
}: {
  open: boolean;
  evalJobId: string;
  onClose: () => void;
}) {
  const [format, setFormat] = useState<EvaluationReportExportFormat>('zip');
  const [contentOptions, setContentOptions] = useState(DEFAULT_OPTIONS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const contentFields = useMemo(
    () =>
      [
        ['includeBasicInfo', '基础信息'],
        ['includeConfig', '评测配置'],
        ['includeMetrics', '已选指标'],
        ['includeEpisodes', 'Episode 明细'],
        ['includeVideoInfo', '视频信息'],
        ['includeDiagnostics', '失败诊断'],
        ['includeRuntimeIndex', '原始文件索引'],
        ['includeUnavailableMetricReasons', '不可计算指标原因'],
      ] as const,
    []
  );

  const resolvedEvalJobId = useMemo(
    () => resolveExportEvaluationJobId({ evalJobId }),
    [evalJobId]
  );

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    const result = await downloadEvaluationReport(resolvedEvalJobId || evalJobId, {
      format,
      template: 'standard',
      ...contentOptions,
    });
    setLoading(false);
    if (!result.ok) {
      const message = result.error || '导出失败';
      setError(result.hint ? `${message}。${result.hint}` : message);
      return;
    }
    onClose();
  };

  return (
    <WorkspaceCenteredModal
      open={open}
      title="导出评测报告"
      titleId="evaluation-report-export-title"
      width={640}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={onClose} disabled={loading}>
            取消
          </SecondaryButton>
          <PrimaryButton onClick={() => void handleExport()} disabled={loading || !resolvedEvalJobId}>
            {loading ? '导出中…' : '开始导出'}
          </PrimaryButton>
        </div>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <div style={workspaceModalSectionLabel}>导出格式</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
            {FORMAT_OPTIONS.map((item) => (
              <label
                key={item.id}
                style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#374151' }}
              >
                <input
                  type="radio"
                  name="evaluation-report-format"
                  checked={format === item.id}
                  onChange={() => setFormat(item.id)}
                />
                {item.label}
              </label>
            ))}
          </div>
        </div>

        <div>
          <div style={workspaceModalSectionLabel}>报告内容</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
            {contentFields.map(([key, label]) => (
              <label
                key={key}
                style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#374151' }}
              >
                <input
                  type="checkbox"
                  checked={contentOptions[key]}
                  onChange={(event) =>
                    setContentOptions((prev) => ({ ...prev, [key]: event.target.checked }))
                  }
                />
                {label}
              </label>
            ))}
          </div>
        </div>

        <p style={{ margin: 0, fontSize: 12, color: '#6b7280', lineHeight: 1.6 }}>
          所有格式基于同一份统一 ReportData 渲染；ZIP 报告包包含 PDF、DOCX、JSON、Markdown、XLSX、LaTeX 与 CSV 子包。
        </p>

        {error ? (
          <p style={{ margin: 0, fontSize: 13, color: '#b45309', lineHeight: 1.6 }}>{error}</p>
        ) : null}
      </div>
    </WorkspaceCenteredModal>
  );
}

export function EvaluationReportExportButton({
  evalJobId,
  label = '导出报告',
}: {
  evalJobId: string;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <SecondaryButton onClick={() => setOpen(true)} disabled={!evalJobId}>
        {label}
      </SecondaryButton>
      <EvaluationReportExportModal open={open} evalJobId={evalJobId} onClose={() => setOpen(false)} />
    </>
  );
}
