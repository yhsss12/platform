'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import Link from 'next/link';
import { useI18n } from '@/components/common/I18nProvider';
import type { ProjectLoadItem } from '@/lib/dashboard/types';

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b'];

export function ProjectTopChart(props: { items: ProjectLoadItem[] }) {
  const { t } = useI18n();
  const { items } = props;
  const data = items.slice(0, 5).map((i) => ({
    name: i.projectName.length > 10 ? i.projectName.slice(0, 10) + '…' : i.projectName,
    fullName: i.projectName,
    tasks: i.taskCount,
    assets: i.dataCount,
    projectId: i.projectId,
  }));

  if (data.length === 0) {
    return <div style={{ padding: 24, color: '#9ca3af', fontSize: 13 }}>{t('dashboard.noProjectData')}</div>;
  }

  return (
    <div style={{ height: 200 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart layout="vertical" data={data} margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={{ stroke: 'rgba(15,23,42,0.06)' }} tickLine={false} />
          <YAxis type="category" dataKey="name" width={64} tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ fontSize: 12, border: '1px solid rgba(15,23,42,0.06)', borderRadius: 8 }} />
          <Bar dataKey="tasks" name={t('dashboard.barTaskCount')} fill="#3b82f6" radius={[0, 4, 4, 0]} maxBarSize={24}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
        {items.slice(0, 5).map((i) => (
          <Link key={i.projectId} href={`/admin/projects/${i.projectId}`} style={{ fontSize: 11, color: '#3b82f6' }}>
            {i.projectName}
          </Link>
        ))}
      </div>
    </div>
  );
}
