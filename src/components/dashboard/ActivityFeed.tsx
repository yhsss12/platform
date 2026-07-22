'use client';

import Link from 'next/link';
import { FileText, Database, User, FolderOpen } from 'lucide-react';
import { useI18n } from '@/components/common/I18nProvider';
import type { ProjectActivity } from '@/lib/projects/projectActivity';
import { formatRelativeTime } from '@/lib/formatRelativeTime';

const iconMap: Record<string, React.ComponentType<{ size?: number }>> = {
  TASK_CREATED: FileText,
  TASK_UPDATED: FileText,
  TASK_DELETED: FileText,
  DATA_IMPORTED: Database,
  DATA_DELETED: Database,
  MEMBER_ADDED: User,
  MEMBER_REMOVED: User,
  MEMBER_ROLE_CHANGED: User,
  PROJECT_UPDATED: FolderOpen,
};

export function ActivityFeed({ activities, projectNameById }: { activities: ProjectActivity[]; projectNameById: (id: string) => string }) {
  const { t } = useI18n();
  const list = activities.slice(0, 10);
  if (list.length === 0) {
    return <div style={{ padding: 16, color: '#9ca3af', fontSize: 13 }}>{t('dashboard.noActivity')}</div>;
  }

  return (
    <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
      {list.map((a) => {
        const Icon = iconMap[a.type] ?? FileText;
        const projectName = projectNameById(a.projectId);
        const label = projectName ? t('dashboard.activityProjectSuffix', { name: projectName }) : '';
        const href = a.type.startsWith('TASK_') ? '/daq/tasks' : a.type.startsWith('DATA_') ? '/data' : a.type.startsWith('MEMBER_') ? '/admin/projects/' + a.projectId : '/admin/projects/' + a.projectId;
        return (
          <li key={a.id} style={{ padding: '10px 0', borderBottom: '1px solid #f3f4f6' }}>
            <Link href={href} style={{ display: 'flex', gap: 10, alignItems: 'flex-start', color: '#374151', textDecoration: 'none' }}>
              <span style={{ flexShrink: 0, marginTop: 2 }}><Icon size={16} /></span>
              <div>
                <span style={{ fontSize: 13 }}>{a.operator} {a.message}{label}</span>
                <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 2 }}>{formatRelativeTime(a.createdAt)}</div>
              </div>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
