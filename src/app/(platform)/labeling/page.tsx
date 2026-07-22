'use client';

import { useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { getDataAsset, type DataAssetItem } from '@/features/data-platform/api/dataAssetsApi';
import { useI18n } from '@/components/common/I18nProvider';

export default function LabelingPage() {
  const searchParams = useSearchParams();
  const datasetId = searchParams.get('datasetId');
  const [dataset, setDataset] = useState<DataAssetItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (!datasetId) {
      setError(t('labelPage.missingDatasetId'));
      setLoading(false);
      return;
    }

    const fetchDataset = async () => {
      try {
        const id = Number(datasetId);
        if (!Number.isFinite(id)) {
          setError(t('labelPage.invalidDatasetId'));
          setLoading(false);
          return;
        }
        const res = await getDataAsset(id);
        if (res.ok && res.data) {
          setDataset(res.data);
        } else {
          setError(res.error || t('labelPage.datasetNotFound'));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : t('feedback.requestFailed'));
      } finally {
        setLoading(false);
      }
    };

    fetchDataset();
  }, [datasetId]);

  if (loading) {
    return (
      <div style={{ padding: '24px' }}>
        <div style={{ textAlign: 'center', padding: '48px', color: '#6b7280' }}>
          {t('common.loading')}
        </div>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div style={{ padding: '24px' }}>
        <div style={{
          padding: '24px',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          color: '#dc2626',
        }}>
          {error || t('labelPage.datasetNotFound')}
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
        padding: '24px',
      }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px', color: '#111827' }}>
          {t('labelPage.title')}
        </h2>
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: '8px' }}>
            {t('labelPage.currentAsset')}：
          </div>
          <div style={{ fontSize: '16px', fontWeight: 500, color: '#111827' }}>
            {dataset.filename}
          </div>
        </div>
          <div style={{ fontSize: '14px', color: '#6b7280' }}>
          <div>{t('labelPage.assetId')}: {dataset.id}</div>
          {(dataset.project_name || dataset.project_id) && (
            <div>{t('labelPage.project')}: {dataset.project_name ?? dataset.project_id}</div>
          )}
        </div>
      </div>
    </div>
  );
}
