'use client';

import { useEffect, useState } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import PageHeader from '@/features/data-platform/components/PageHeader';
import EmptyState from '@/features/data-platform/components/EmptyState';
import { listJobs } from '@/features/data-platform/api';
import type { Job } from '@/features/data-platform/models';

function getStatusLabel(status: string, t: (path: string) => string): string {
  const key = status?.toUpperCase();
  if (key === 'PENDING') return t('status.pending');
  if (key === 'RUNNING') return t('status.running');
  if (key === 'COMPLETED') return t('status.completed');
  if (key === 'PAUSED') return t('status.paused');
  if (key === 'FAILED') return t('status.failed');
  if (key === 'SUCCESS') return t('status.success');
  return status || '—';
}

export default function JobsPage() {
  const { t } = useI18n();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listJobs().then((res) => {
      if (res.ok && res.data) {
        setJobs(res.data);
      }
      setLoading(false);
    });
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <PageHeader title={t('jobsPage.title')} />
      <div style={{ marginTop: '24px' }}>
        <div style={{
          display: 'flex',
          gap: '12px',
          marginBottom: '24px',
          padding: '12px',
          backgroundColor: '#1a1a1a',
          borderRadius: '4px',
        }}>
          <button style={{
            padding: '8px 16px',
            backgroundColor: '#4a9eff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}>
            {t('jobsPage.newJob')}
          </button>
          <input
            type="text"
            placeholder={t('jobsPage.searchPlaceholder')}
            style={{
              flex: 1,
              padding: '8px 12px',
              backgroundColor: '#252525',
              border: '1px solid #333',
              borderRadius: '4px',
              color: '#fff',
            }}
            disabled
          />
        </div>
        {loading ? (
          <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>{t('common.loading')}</div>
        ) : jobs.length === 0 ? (
          <EmptyState message={t('jobsPage.emptyMessage')} hint={t('jobsPage.emptyHint')} />
        ) : (
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {jobs.map((job) => (
              <li
                key={job.id}
                style={{
                  padding: '16px',
                  marginBottom: '12px',
                  backgroundColor: '#1a1a1a',
                  borderRadius: '4px',
                  border: '1px solid #333',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>
                      {job.type} Job
                    </div>
                    <div style={{ fontSize: '14px', color: '#999' }}>
                      {t('jobsPage.statusLabel')}: {getStatusLabel(job.status, t)} | {t('jobsPage.progressLabel')}: {job.progress.percent}%
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
