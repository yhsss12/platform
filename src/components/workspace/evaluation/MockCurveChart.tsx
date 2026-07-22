'use client';

import type { CurvePoint } from '@/lib/mock/workspaceEvaluationMock';

interface MockCurveChartProps {
  title: string;
  points: CurvePoint[];
  color: string;
  fillColor: string;
  yMax?: number;
  currentPercent?: number;
}

export function MockCurveChart({
  title,
  points,
  color,
  fillColor,
  yMax = 1,
  currentPercent,
}: MockCurveChartProps) {
  const w = 100;
  const h = 48;
  const pad = 2;

  const coords = points.map((p) => {
    const x = pad + (p.t / 100) * (w - pad * 2);
    const y = h - pad - (p.value / yMax) * (h - pad * 2);
    return `${x},${y}`;
  });
  const linePath = `M ${coords.join(' L ')}`;
  const areaPath = `${linePath} L ${w - pad},${h - pad} L ${pad},${h - pad} Z`;

  const markerX =
    currentPercent != null
      ? pad + (currentPercent / 100) * (w - pad * 2)
      : null;

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>{title}</span>
        {currentPercent != null ? (
          <span style={{ fontSize: 12, color: '#6b7280' }}>当前 {currentPercent}%</span>
        ) : null}
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        style={{
          width: '100%',
          height: 72,
          display: 'block',
          backgroundColor: '#f9fafb',
          borderRadius: 6,
          border: '1px solid #e5e7eb',
        }}
        preserveAspectRatio="none"
      >
        <path d={areaPath} fill={fillColor} stroke="none" />
        <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        {markerX != null ? (
          <line
            x1={markerX}
            y1={pad}
            x2={markerX}
            y2={h - pad}
            stroke="#94a3b8"
            strokeWidth="1"
            strokeDasharray="2 2"
          />
        ) : null}
      </svg>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: '#9ca3af',
          marginTop: 4,
        }}
      >
        <span>0%</span>
        <span>任务进度</span>
        <span>100%</span>
      </div>
    </div>
  );
}
