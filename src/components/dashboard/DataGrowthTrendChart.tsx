'use client';

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useI18n } from '@/components/common/I18nProvider';

interface Point {
  label: string;
  cumulative: number;
  new: number;
}

const GRID_STROKE = 'rgba(15, 23, 42, 0.06)';
const TICK_FILL = '#9ca3af';

export function DataGrowthTrendChart({ points }: { points: Point[] }) {
  const { t } = useI18n();
  const data = points.map((p) => ({ name: p.label, value: p.cumulative, new: p.new }));

  if (data.length === 0) {
    return (
      <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af', fontSize: 13 }}>
        {t('dashboard.waitingData')}
      </div>
    );
  }

  return (
    <div style={{ height: 200 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="dataGrowthGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: TICK_FILL }} axisLine={{ stroke: GRID_STROKE }} tickLine={false} />
          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: TICK_FILL }} width={28} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ fontSize: 12, border: '1px solid rgba(15,23,42,0.06)', borderRadius: 8 }} />
          <Area type="monotone" dataKey="value" stroke="#3b82f6" fill="url(#dataGrowthGrad)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
