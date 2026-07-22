'use client';

import { AlertTriangle } from 'lucide-react';
import { useI18n } from '@/components/common/I18nProvider';

export interface AlertItem {
  id: string;
  type: 'device' | 'task' | 'storage';
  message: string;
}

export function AlertPanel({ alerts }: { alerts: AlertItem[] }) {
  const { t } = useI18n();
  const list = alerts.slice(0, 5);
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 12 }}>{t('dashboard.alertPanelTitle')}</div>
      {list.length === 0 ? (
        <div style={{ padding: 16, color: '#9ca3af', fontSize: 13 }}>{t('dashboard.noAlerts')}</div>
      ) : (
        <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
          {list.map((a) => (
            <li key={a.id} style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #f3f4f6' }}>
              <span style={{ color: '#f59e0b', flexShrink: 0, display: 'inline-flex' }}><AlertTriangle size={14} /></span>
              <span style={{ fontSize: 13, color: '#374151' }}>{a.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
