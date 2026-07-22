'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import type { BenchmarkResultRow } from '@/lib/mock/workspaceEvaluationMock';
import {
  benchmarkStatusBadge,
  benchmarkStatusLabel,
  formatBenchmarkSampleCount,
} from '@/lib/mock/workspaceEvaluationMock';
import { formatEvalConfig } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 420,
  maxWidth: '100vw',
  backgroundColor: '#fff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, color: '#111827' }}>{children}</div>
    </div>
  );
}

export function EvaluationDetailDrawer({
  row,
  onClose,
  onExportReport,
}: {
  row: BenchmarkResultRow | null;
  onClose: () => void;
  onExportReport: (row: BenchmarkResultRow) => void;
}) {
  useEffect(() => {
    if (!row) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [row, onClose]);

  if (!row) return null;

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal>
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
          }}
        >
          <div>
            <h2 style={{ margin: 0, fontSize: 17, fontWeight: 600 }}>Benchmark 详情</h2>
            <div style={{ marginTop: 8 }}>
              <StatusBadge
                status={benchmarkStatusBadge(row.status)}
                label={benchmarkStatusLabel[row.status]}
              />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>
        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <Row label="任务">{row.taskName}</Row>
          <Row label="样本数量">{formatBenchmarkSampleCount(row.dataVolume)}</Row>
          <Row label="评测配置">{formatEvalConfig(row)}</Row>
          <Row label="模型类型">{row.modelType}</Row>
          <Row label="模型版本">
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{row.checkpoint}</span>
          </Row>
          <Row label="关联数据">{row.dataName ?? '—'}</Row>
          <Row label="成功率">{row.successRate}%</Row>
          <Row label="平均耗时">{row.avgDurationSec} s</Row>
          <Row label="碰撞次数">{row.collisionCount}</Row>
          <Row label="失败摘要">{row.failureSummary}</Row>
          {row.trajectoryErrorMm != null ? (
            <Row label="轨迹误差">{row.trajectoryErrorMm} mm</Row>
          ) : null}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 16 }}>
            <Link href="/workspace/replay" style={{ textDecoration: 'none' }}>
              <SecondaryButton>回放</SecondaryButton>
            </Link>
            <SecondaryButton onClick={() => onExportReport(row)}>导出报告</SecondaryButton>
          </div>
        </div>
      </aside>
    </>
  );
}
