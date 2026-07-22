'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import {
  buildResourceHubOverviewItems,
  buildResourceHubSections,
  ResourceHubView,
} from '@/components/workspace/resources/ResourceHubView';
import { getResourceOverview } from '@/lib/api/resourceRegistryClient';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { useI18n } from '@/components/common/I18nProvider';
import {
  RESOURCE_HUB_OVERVIEW_ITEMS,
  RESOURCE_HUB_SECTIONS,
  type ResourceHubCountKey,
} from '@/lib/workspace/resourceHubSections';

const FAILED_COUNT_KEYS = [
  'taskTemplates',
  'modelAssets',
  'metrics',
  'scenes',
  'robots',
  'objects',
  'policyAssets',
  'physicsProxies',
  'modelTypes',
  'craftConfig',
  'simAssets',
] as const satisfies readonly ResourceHubCountKey[];

function failedCountMap(): Partial<Record<ResourceHubCountKey, null>> {
  return Object.fromEntries(FAILED_COUNT_KEYS.map((key) => [key, null])) as Partial<
    Record<ResourceHubCountKey, null>
  >;
}

export default function ResourcesHubPage() {
  const { t } = useI18n();
  const [counts, setCounts] = useState<Partial<Record<ResourceHubCountKey, number | null>>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [partialWarning, setPartialWarning] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setPartialWarning(null);
    try {
      const overview = await getResourceOverview();
      setCounts({
        taskTemplates: overview.taskTemplates,
        modelAssets: overview.modelAssets,
        metrics: overview.metrics,
        scenes: overview.scenes,
        robots: overview.robots,
        objects: overview.objects,
        policyAssets: overview.policyAssets,
        physicsProxies: overview.physicsProxies,
        modelTypes: overview.modelTypes,
        craftConfig: overview.craftConfig,
        simAssets: overview.simAssets,
      });
      if (overview.partialFailure && overview.warnings?.length) {
        setPartialWarning('部分资源加载失败，失败类别显示为 --');
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载资源总览失败');
      setCounts(failedCountMap());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const sections = useMemo(
    () => buildResourceHubSections(RESOURCE_HUB_SECTIONS, counts),
    [counts]
  );

  const overviewItems = useMemo(
    () => buildResourceHubOverviewItems(RESOURCE_HUB_OVERVIEW_ITEMS, counts),
    [counts]
  );

  return (
    <ModulePageContainer>
      <ModulePageHeader title={t('resourcesHub.title')} subtitle={t('resourcesHub.subtitle')} />

      {partialWarning && !loadError ? (
        <div
          style={{
            marginBottom: 16,
            padding: '10px 14px',
            borderRadius: 10,
            border: '1px solid #fde68a',
            background: '#fffbeb',
            color: '#92400e',
            fontSize: 13,
          }}
        >
          {partialWarning}
        </div>
      ) : null}

      {loadError ? (
        <div
          style={{
            marginBottom: 16,
            padding: '10px 14px',
            borderRadius: 10,
            border: '1px solid #fde68a',
            background: '#fffbeb',
            color: '#92400e',
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <span>{loadError}</span>
          <SecondaryButton onClick={() => void refresh()}>重试</SecondaryButton>
        </div>
      ) : null}

      <ResourceHubView
        overviewItems={overviewItems}
        overviewTitle={t('resourcesHub.overviewTitle')}
        sections={sections}
        enterLabel={t('resourcesHub.enter')}
        loading={loading}
      />
    </ModulePageContainer>
  );
}
