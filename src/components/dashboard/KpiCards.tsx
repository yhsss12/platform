'use client';

const CARD_HEIGHT = 100;
const BORDER = '1px solid rgba(15, 23, 42, 0.06)';
const SHADOW = '0 1px 2px rgba(0,0,0,0.04)';
const ACCENT_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

export interface KpiCardItem {
  title: string;
  value: number;
}

export function KpiCards({ items }: { items: KpiCardItem[] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {items.map((item, i) => (
        <div
          key={item.title}
          style={{
            position: 'relative',
            height: CARD_HEIGHT,
            borderRadius: 16,
            padding: '16px 20px',
            backgroundColor: '#fff',
            border: BORDER,
            boxShadow: SHADOW,
            overflow: 'hidden',
          }}
        >
          <div style={{ position: 'relative', zIndex: 1 }}>
            <div style={{ fontSize: 13, color: '#6B7280', fontWeight: 500, marginBottom: 8 }}>{item.title}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#111827' }}>{item.value}</div>
          </div>
          <div
            style={{
              position: 'absolute',
              right: -30,
              top: -20,
              width: 120,
              height: 120,
              borderRadius: 999,
              backgroundColor: ACCENT_COLORS[i % 4],
              opacity: 0.14,
              pointerEvents: 'none',
            }}
          />
        </div>
      ))}
    </div>
  );
}
