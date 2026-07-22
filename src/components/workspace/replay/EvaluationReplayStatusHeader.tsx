'use client';

export function EvaluationReplayStatusHeader({
  taskName,
  statusLabel,
  actions,
}: {
  taskName: string;
  statusLabel: string;
  actions?: React.ReactNode;
}) {
  const statusColor = statusLabel === '失败' ? '#dc2626' : '#374151';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        width: '100%',
      }}
    >
      <span className="replay-header-title">
        当前任务：{taskName}
        <span style={{ marginLeft: 8, color: statusColor }}>{statusLabel}</span>
      </span>
      {actions ? <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>{actions}</div> : null}
    </div>
  );
}
