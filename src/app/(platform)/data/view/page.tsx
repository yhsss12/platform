'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { FrameCacheProvider } from '@/features/asset-viewer/context/FrameCacheContext';
import EpisodeViewerLayout from '@/features/asset-viewer/components/EpisodeViewerLayout';
import { useI18n } from '@/components/common/I18nProvider';
import { getDataAsset } from '@/features/data-platform/api/dataAssetsApi';

function DataViewContent() {
  const searchParams = useSearchParams();
  const assetId = searchParams.get('assetId') || '';
  const { t } = useI18n();
  const [header, setHeader] = useState<{ title: string; subtitle?: string } | undefined>(undefined);

  useEffect(() => {
    if (!assetId) return;
    const idNum = Number(assetId);
    if (!Number.isFinite(idNum)) return;
    getDataAsset(idNum).then((res) => {
      if (res.ok && res.data) {
        setHeader({
          title: `${t('labelPage.currentAsset')}：${res.data.filename || assetId}`,
        });
      }
    }).catch(() => {});
  }, [assetId, t]);

  return (
    <FrameCacheProvider>
      <EpisodeViewerLayout
        source={{ type: 'asset', assetId }}
        header={header}
        missingContextMessage={t('labelPage.missingDatasetId')}
        t={t}
      />
    </FrameCacheProvider>
  );
}

function DataViewFallback() {
  const { t } = useI18n();
  return (
    <div
      style={{
        height: 'calc(100vh - 60px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#6b7280',
      }}
    >
      {t('common.loading')}
    </div>
  );
}

export default function DataViewPage() {
  return (
    <Suspense fallback={<DataViewFallback />}>
      <DataViewContent />
    </Suspense>
  );
}
