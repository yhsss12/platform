'use client';

import { useEffect, useState } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import PageHeader from '@/features/data-platform/components/PageHeader';
import EmptyState from '@/features/data-platform/components/EmptyState';
import { listDatasets } from '@/features/data-platform/api';
import type { Dataset } from '@/features/data-platform/models';

export default function DatasetsPage() {
  const { t } = useI18n();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listDatasets().then((res) => {
      if (res.ok && res.data) {
        setDatasets(res.data);
      }
      setLoading(false);
    });
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <PageHeader title={t('datasetsPage.title')} />
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
            {t('datasetsPage.newDataset')}
          </button>
          <input
            type="text"
            placeholder={t('datasetsPage.searchPlaceholder')}
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
        ) : datasets.length === 0 ? (
          <EmptyState message={t('datasetsPage.emptyMessage')} hint={t('datasetsPage.emptyHint')} />
        ) : (
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {datasets.map((dataset) => (
              <li
                key={dataset.id}
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
                      {dataset.name}
                    </div>
                    <div style={{ fontSize: '14px', color: '#999' }}>
                      {t('datasetsPage.runsCount')}: {dataset.runIds.length} | {t('datasetsPage.updated')}: {new Date(dataset.updatedAt).toLocaleString()}
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
