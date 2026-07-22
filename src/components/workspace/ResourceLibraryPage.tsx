'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import {
  listRegistryResources,
  registryStatusLabel,
  type ListResourcesParams,
  type RegistryAssetType,
  type RegistryResource,
} from '@/lib/api/resourceRegistryClient';
import {
  FilterInput,
  GhostButton,
  PrimaryButton,
  SecondaryButton,
  SectionCard,
  StatusBadge,
  WS,
} from './workspaceUi';
import { RegistryResourceDetailDrawer } from '@/components/workspace/resources/RegistryResourceDetailDrawer';
import {
  registryResourceSummaryLine,
  resolveRegistryScenarioLabel,
  metricImplementationLabel,
} from '@/lib/workspace/registryResourceDisplay';

interface DisplayResource {
  id: string;
  name: string;
  category: string;
  version: string;
  status: string;
  updatedAt: string;
  description: string;
  tags: string[];
}

export interface ResourceLibraryPageProps {
  title: string;
  subtitle: string;
  assetTypes?: RegistryAssetType | RegistryAssetType[];
  resourceType?: ListResourcesParams['resourceType'];
  createLabel?: string;
  importLabel?: string;
  resourceFilter?: (resource: RegistryResource) => boolean;
}

function mapRegistryToDisplay(resource: RegistryResource): DisplayResource {
  return {
    id: resource.assetId,
    name: resource.name,
    category: resource.assetType,
    version: resource.version,
    status: resource.status === 'available' ? 'active' : 'draft',
    updatedAt: resource.lastModifiedAt?.slice(0, 10) ?? '—',
    description: resource.description,
    tags: resource.tags ?? [],
  };
}

export default function ResourceLibraryPage({
  title,
  subtitle,
  assetTypes,
  resourceType,
  createLabel = '新建',
  importLabel = '导入',
  resourceFilter,
}: ResourceLibraryPageProps) {
  const types = useMemo(
    () => (assetTypes ? (Array.isArray(assetTypes) ? assetTypes : [assetTypes]) : []),
    [assetTypes]
  );
  const [search, setSearch] = useState('');
  const [tagFilter, setTagFilter] = useState('');
  const [registryResources, setRegistryResources] = useState<RegistryResource[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const hasLoadedOnceRef = useRef(false);
  const [detailRegistry, setDetailRegistry] = useState<RegistryResource | null>(null);

  const refresh = useCallback(async () => {
    if (!hasLoadedOnceRef.current) {
      setLoading(true);
    }
    setLoadError(null);
    try {
      let merged: RegistryResource[] = [];
      if (resourceType) {
        const response = await listRegistryResources({ resourceType });
        merged = response.resources;
      } else {
        const responses = await Promise.all(
          types.map((assetType) => listRegistryResources({ assetType }))
        );
        merged = responses.flatMap((r) => r.resources);
      }
      const deduped = [...new Map(merged.map((item) => [item.assetId, item])).values()];
      setRegistryResources(resourceFilter ? deduped.filter(resourceFilter) : deduped);
      hasLoadedOnceRef.current = true;
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载资源失败');
      setRegistryResources([]);
    } finally {
      setLoading(false);
    }
  }, [types, resourceType, resourceFilter]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const displayResources = useMemo(
    () => registryResources.map(mapRegistryToDisplay),
    [registryResources]
  );

  const allTags = useMemo(() => {
    const set = new Set<string>();
    displayResources.forEach((r) => r.tags.forEach((t) => set.add(t)));
    return [...set];
  }, [displayResources]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return displayResources.filter((r) => {
      if (tagFilter && !r.tags.includes(tagFilter)) return false;
      if (!q) return true;
      return (
        r.name.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)
      );
    });
  }, [displayResources, search, tagFilter]);

  const mockToast = (action: string) => window.alert(`${action}：规划中`);

  const openDetail = useCallback((display: DisplayResource) => {
    const registry = registryResources.find((item) => item.assetId === display.id) ?? null;
    setDetailRegistry(registry);
  }, [registryResources]);

  const closeDetail = useCallback(() => {
    setDetailRegistry(null);
  }, []);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={title}
        subtitle={subtitle}
        actions={
          <>
            <SecondaryButton onClick={() => mockToast(importLabel)}>{importLabel}</SecondaryButton>
            <PrimaryButton onClick={() => mockToast(createLabel)}>{createLabel}</PrimaryButton>
          </>
        }
      />

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <FilterInput value={search} onChange={setSearch} placeholder="搜索资源名称、说明…" />
          <select
            value={tagFilter}
            onChange={(e) => setTagFilter(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              fontSize: 14,
              backgroundColor: '#fff',
            }}
          >
            <option value="">全部标签</option>
            {allTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <span style={{ fontSize: 13, color: '#6b7280' }}>
            {loading ? '加载中…' : `共 ${filtered.length} 项`}
          </span>
          {loadError ? (
            <>
              <span style={{ fontSize: 13, color: '#b45309' }}>{loadError}</span>
              <SecondaryButton onClick={() => void refresh()}>重试</SecondaryButton>
            </>
          ) : null}
        </div>
      </ModulePageFilterCard>

      {loading && !hasLoadedOnceRef.current ? (
        <SectionCard>
          <p style={{ textAlign: 'center', color: '#6b7280', margin: 24 }}>正在加载资源…</p>
        </SectionCard>
      ) : loadError && registryResources.length === 0 ? (
        <SectionCard>
          <p style={{ textAlign: 'center', color: '#b45309', margin: 24, lineHeight: 1.6 }}>
            {loadError}
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
            <SecondaryButton onClick={() => void refresh()}>重试</SecondaryButton>
          </div>
        </SectionCard>
      ) : filtered.length === 0 ? (
        <SectionCard>
          <p style={{ textAlign: 'center', color: '#6b7280', margin: 24, lineHeight: 1.6 }}>
            暂无资源。请执行资源 reindex 同步，或联系管理员添加资源定义。
          </p>
        </SectionCard>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: WS.gap,
          }}
        >
          {filtered.map((r) => {
            const registry = registryResources.find((item) => item.assetId === r.id);
            const status = registry?.status ?? (r.status === 'active' ? 'available' : r.status);
            const scenario = registry ? resolveRegistryScenarioLabel(registry) : r.category;
            const summaryLine = registry
              ? registryResourceSummaryLine(registry)
              : `${r.category} · ${r.version} · 更新 ${r.updatedAt}`;

            return (
              <SectionCard
                key={r.id}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  minHeight: 220,
                  height: '100%',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: 8,
                    marginBottom: 8,
                    minWidth: 0,
                  }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div
                      style={{
                        fontSize: 16,
                        fontWeight: 600,
                        color: '#111827',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={r.name}
                    >
                      {r.name}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        color: '#6b7280',
                        marginTop: 4,
                        fontFamily: 'ui-monospace, monospace',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={r.id}
                    >
                      {r.id}
                    </div>
                  </div>
                  <StatusBadge
                    status="active"
                    label={
                      registry?.assetType === 'metric'
                        ? metricImplementationLabel(registry)
                        : registryStatusLabel(status)
                    }
                  />
                </div>
                <p
                  style={{
                    fontSize: 13,
                    color: '#4b5563',
                    lineHeight: 1.5,
                    margin: '0 0 10px',
                    overflow: 'hidden',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    minHeight: 40,
                  }}
                  title={r.description}
                >
                  {r.description}
                </p>
                <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 10 }}>
                  适用：{scenario}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12, minHeight: 26 }}>
                  {r.tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        borderRadius: 4,
                        backgroundColor: '#eff6ff',
                        color: '#1d4ed8',
                        maxWidth: '100%',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={tag}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: '#9ca3af',
                    marginTop: 'auto',
                    marginBottom: 12,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={summaryLine}
                >
                  {summaryLine}
                </div>
                <div style={{ display: 'flex', gap: 8, borderTop: '1px solid #f3f4f6', paddingTop: 12 }}>
                  <GhostButton onClick={() => openDetail(r)}>详情</GhostButton>
                </div>
              </SectionCard>
            );
          })}
        </div>
      )}

      <RegistryResourceDetailDrawer resource={detailRegistry} onClose={closeDetail} />
    </ModulePageContainer>
  );
}
