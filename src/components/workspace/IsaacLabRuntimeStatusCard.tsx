'use client';

import type { IsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';

function statusBadge(available: boolean) {
  return {
    label: available ? '可用' : '未配置',
    color: available ? '#047857' : '#b45309',
    background: available ? '#ecfdf5' : '#fffbeb',
    border: available ? '#a7f3d0' : '#fde68a',
  };
}

function row(label: string, value: React.ReactNode) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 13 }}>
      <span style={{ color: '#6b7280', flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#111827', textAlign: 'right', wordBreak: 'break-word' }}>{value}</span>
    </div>
  );
}

export function IsaacLabRuntimeStatusCard({
  status,
  loading,
  error,
}: {
  status: IsaacLabRuntimeStatus | null;
  loading?: boolean;
  error?: string | null;
}) {
  if (loading) {
    return (
      <div style={{ padding: 14, borderRadius: 10, border: '1px solid #e5e7eb', background: '#fafafa' }}>
        <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>正在检测 Isaac Lab 运行环境…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 14, borderRadius: 10, border: '1px solid #fde68a', background: '#fffbeb' }}>
        <p style={{ margin: 0, fontSize: 13, color: '#b45309' }}>{error}</p>
      </div>
    );
  }

  const badge = statusBadge(Boolean(status?.available));
  return (
    <div style={{ padding: 14, borderRadius: 10, border: '1px solid #e5e7eb', background: '#fafafa' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>Isaac Lab Runtime</div>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: 999,
            color: badge.color,
            backgroundColor: badge.background,
            border: `1px solid ${badge.border}`,
          }}
        >
          {badge.label}
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {row('Isaac Lab Version', status?.isaacLabVersion ?? '—')}
        {row('Default Env', status?.defaultTask ?? '—')}
        {row('GPU', status?.gpuAvailable ? '可用' : '不可用')}
        {row('Task Registered', status?.taskRegistered ? '是' : '否')}
        {row('Configured', status?.configured ? '是' : '否')}
        {row('Runtime Mode', status?.runtimeMode ?? '—')}
        {status?.isaacLabRoot ? row('Root', status.isaacLabRoot) : null}
      </div>
      {status?.issues?.length ? (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 6 }}>Issues</div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
            {status.issues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
