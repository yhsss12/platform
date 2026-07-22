'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { PrimaryButton, SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';
import {
  normalizeSimRunStatus,
  runStatusBadgeStatus,
  runStatusHint,
  runStatusLabel,
  type SimRunDisplayStatus,
} from '@/lib/workspace/simulationRunStatus';

export type { SimRunDisplayStatus } from '@/lib/workspace/simulationRunStatus';
export {
  normalizeSimRunStatus,
  runStatusBadgeStatus,
  runStatusHint,
  runStatusLabel,
} from '@/lib/workspace/simulationRunStatus';

export const simConsoleCardStyle: React.CSSProperties = {
  backgroundColor: '#fff',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
  boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
  padding: 16,
};

export function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '96px 1fr',
        gap: 12,
        fontSize: 13,
        lineHeight: 1.5,
        alignItems: 'start',
      }}
    >
      <span style={{ color: '#6b7280' }}>{label}</span>
      <span style={{ color: '#111827', fontWeight: 400, wordBreak: 'break-word' }}>{value}</span>
    </div>
  );
}

export function SidePanelSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 10 }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>{children}</div>
    </div>
  );
}

export function CollapsiblePanel({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div style={{ marginTop: 16, borderTop: '1px solid #f3f4f6', paddingTop: 12 }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%',
          padding: 0,
          border: 'none',
          background: 'none',
          cursor: 'pointer',
          fontSize: 12,
          fontWeight: 500,
          color: '#6b7280',
        }}
      >
        <span>{title}</span>
        <span style={{ fontSize: 11, fontWeight: 400 }}>{open ? '收起' : '展开'}</span>
      </button>
      {open ? <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 10 }}>{children}</div> : null}
    </div>
  );
}

function RunProgressBar({ percent, status }: { percent: number; status: SimRunDisplayStatus }) {
  const barColor =
    status === 'failed' ? '#dc2626' : status === 'completed' ? '#059669' : '#2563eb';

  return (
    <div style={{ marginTop: 12 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 11,
          color: '#6b7280',
          marginBottom: 4,
        }}
      >
        <span>进度</span>
        <span>{status === 'completed' ? '100%' : `${percent}%`}</span>
      </div>
      <div style={{ height: 4, borderRadius: 999, backgroundColor: '#e5e7eb', overflow: 'hidden' }}>
        <div
          style={{
            width: `${status === 'completed' ? 100 : percent}%`,
            height: '100%',
            backgroundColor: barColor,
            borderRadius: 999,
            transition: 'width 0.35s ease',
          }}
        />
      </div>
    </div>
  );
}

export function RunSummaryCard({
  taskName,
  taskTypeLabel,
  runStatus,
  progressPercent,
}: {
  taskName: string;
  taskTypeLabel: string;
  runStatus: SimRunDisplayStatus;
  progressPercent: number;
}) {
  const normalized = normalizeSimRunStatus(runStatus);

  return (
    <div style={{ ...simConsoleCardStyle, padding: '14px 16px' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '8px 20px',
          fontSize: 13,
        }}
      >
        <div>
          <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 2 }}>任务名称</div>
          <div style={{ color: '#111827', fontWeight: 400 }}>{taskName}</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 2 }}>任务类型</div>
          <div style={{ color: '#111827', fontWeight: 400 }}>{taskTypeLabel}</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 2 }}>运行状态</div>
          <StatusBadge status={runStatusBadgeStatus(normalized)} label={runStatusLabel(normalized)} />
        </div>
      </div>
      <RunProgressBar percent={progressPercent} status={normalized} />
      <p style={{ margin: '10px 0 0', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
        {runStatusHint(normalized)}
      </p>
    </div>
  );
}

export function SimulationViewportSection({
  children,
  frameStatusLine,
  accentColor,
  backendLabel = 'MuJoCo',
}: {
  children: ReactNode;
  frameStatusLine: string;
  accentColor?: string;
  backendLabel?: string;
}) {
  return (
    <div style={simConsoleCardStyle}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 500, color: '#374151' }}>仿真画面</span>
        <span
          style={{
            fontSize: 11,
            color: '#6b7280',
            padding: '1px 6px',
            borderRadius: 4,
            backgroundColor: '#f3f4f6',
          }}
        >
          {backendLabel}
        </span>
      </div>
      {children}
      <FrameStatusBar line={frameStatusLine} accentColor={accentColor} />
    </div>
  );
}

export function FrameStatusBar({
  line,
  accentColor = '#374151',
}: {
  line: string;
  accentColor?: string;
}) {
  const parts = line.split(' · ');
  const last = parts.length > 1 ? parts[parts.length - 1] : '';
  const prefix = parts.length > 1 ? parts.slice(0, -1).join(' · ') : line;

  return (
    <div
      style={{
        marginTop: 10,
        padding: '8px 10px',
        borderRadius: 8,
        backgroundColor: '#f9fafb',
        fontSize: 12,
        color: '#6b7280',
        lineHeight: 1.4,
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}
      title={`画面状态：${line}`}
    >
      画面状态：
      {parts.length > 1 ? (
        <>
          {prefix} · <span style={{ color: accentColor }}>{last}</span>
        </>
      ) : (
        <span style={{ color: accentColor }}>{line}</span>
      )}
    </div>
  );
}

export function SideActionsRow({
  onViewLog,
  onViewDataRecord,
  showViewDataRecord,
}: {
  onViewLog?: () => void;
  onViewDataRecord?: () => void;
  showViewDataRecord?: boolean;
}) {
  return (
    <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {onViewLog ? (
        <button
          type="button"
          onClick={onViewLog}
          style={{
            padding: '4px 0',
            fontSize: 12,
            color: '#2563eb',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          查看日志
        </button>
      ) : null}
      {showViewDataRecord && onViewDataRecord ? (
        <button
          type="button"
          onClick={onViewDataRecord}
          style={{
            padding: '4px 0',
            fontSize: 12,
            color: '#2563eb',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          查看数据记录
        </button>
      ) : null}
    </div>
  );
}

export function SimulationRunConsoleLayout({
  summary,
  viewport,
  sidePanel,
  logDrawer,
}: {
  summary: ReactNode;
  viewport: ReactNode;
  sidePanel: ReactNode;
  logDrawer?: ReactNode;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {summary}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.4fr) minmax(280px, 0.9fr)',
          gap: 12,
        }}
      >
        {viewport}
        {sidePanel}
      </div>
      {logDrawer}
    </div>
  );
}

export function RunLogDrawer({
  open,
  logTail,
  loading,
  onClose,
  title = '运行日志',
}: {
  open: boolean;
  logTail: string;
  loading: boolean;
  onClose: () => void;
  title?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <div
        role="presentation"
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15, 23, 42, 0.4)', zIndex: 1500 }}
      />
      <aside
        role="dialog"
        aria-modal
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          bottom: 0,
          width: 560,
          maxWidth: '100vw',
          backgroundColor: '#fff',
          boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
          zIndex: 1501,
          display: 'flex',
          flexDirection: 'column',
          borderLeft: '1px solid #e5e7eb',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '14px 18px',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#111827' }}>{title}</h2>
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
          {loading && !logTail ? (
            <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>正在加载日志…</p>
          ) : null}
          <pre
            style={{
              margin: 0,
              fontFamily: 'ui-monospace, monospace',
              fontSize: 12,
              lineHeight: 1.65,
              color: '#111827',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {logTail || '暂无日志输出，任务启动后将持续刷新。'}
          </pre>
        </div>
      </aside>
    </>
  );
}

export function ConsoleHeaderActions({
  canViewReplay,
  onBackToData,
  onViewReplay,
  replayDisabledTitle = '当前暂无回放视频',
}: {
  canViewReplay: boolean;
  onBackToData: () => void;
  onViewReplay: () => void;
  replayDisabledTitle?: string;
}) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <SecondaryButton onClick={onBackToData}>返回数据中心</SecondaryButton>
      <span title={canViewReplay ? undefined : replayDisabledTitle}>
        <PrimaryButton
          disabled={!canViewReplay}
          onClick={() => {
            if (canViewReplay) onViewReplay();
          }}
        >
          查看回放
        </PrimaryButton>
      </span>
    </div>
  );
}

export function fileReadyLabel(exists: boolean | undefined, pendingLabel = '未生成'): string {
  if (exists) return '已生成';
  return pendingLabel;
}

export interface SimConsoleHeaderState {
  canViewReplay: boolean;
  openReplay: () => void;
  replayDisabledTitle?: string;
}
