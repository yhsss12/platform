'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { useI18n } from '@/components/common/I18nProvider';

export interface DonutSegment {
  name: string;
  value: number;
  color: string;
}

export function DonutCard({
  title,
  segments,
  centerLabel,
  centerSub,
  topRightMetrics,
  emptyMessage,
}: {
  title: string;
  segments: DonutSegment[];
  centerLabel: string;
  centerSub: string;
  topRightMetrics?: string;
  emptyMessage?: string;
}) {
  const { t } = useI18n();
  const data = segments.filter((s) => s.value > 0);
  const isEmpty = data.length === 0;
  const emptyText = emptyMessage ?? t('dashboard.noData');

  return (
    <div
      style={{
        borderRadius: 16,
        padding: 16,
        backgroundColor: '#fff',
        border: '1px solid rgba(15, 23, 42, 0.06)',
        boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
        minHeight: 260,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>{title}</span>
        {topRightMetrics && (
          <span style={{ fontSize: 11, color: '#6B7280' }}>{topRightMetrics}</span>
        )}
      </div>
      {isEmpty ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af', fontSize: 13 }}>
          {emptyText}
        </div>
      ) : (
        <>
          <div style={{ flex: 1, minHeight: 180, position: 'relative' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius="58%"
                  outerRadius="78%"
                  paddingAngle={1}
                  dataKey="value"
                  stroke="none"
                >
                  {data.map((entry, index) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => [v, '']} contentStyle={{ fontSize: 12, border: '1px solid rgba(15,23,42,0.06)', borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
            <div
              style={{
                position: 'absolute',
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                pointerEvents: 'none',
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 700, color: '#111827' }}>{centerLabel}</div>
              <div style={{ fontSize: 11, color: '#6B7280', marginTop: 2 }}>{centerSub}</div>
            </div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '12px 16px', marginTop: 4 }}>
            {data.map((s) => (
              <span key={s.name} style={{ fontSize: 11, color: '#6B7280' }}>
                <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: 2, backgroundColor: s.color, marginRight: 4, verticalAlign: 'middle' }} />
                {s.name} {s.value}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
