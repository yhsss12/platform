'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import {
  FilterInput,
  GhostButton,
  PrimaryButton,
  SectionCard,
} from '@/components/workspace/workspaceUi';
import { CreateModelTypeModal } from '@/components/workspace/modelTypes/CreateModelTypeModal';
import { ModelTypeCard } from '@/components/workspace/modelTypes/ModelTypeCard';
import { ModelTypeDetailDrawer } from '@/components/workspace/modelTypes/ModelTypeDetailDrawer';
import {
  deleteModelType,
  listModelTypes,
  refreshModelTypeTrainingCapabilities,
  updateModelType,
} from '@/lib/api/modelTypesClient';
import { modelTypeHasPendingReadiness } from '@/lib/workspace/modelTypeTrainingCapability';
import type { ModelTypeDefinition } from '@/types/modelType';

export default function ModelTypesPage() {
  const [modelTypes, setModelTypes] = useState<ModelTypeDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshingCapabilities, setRefreshingCapabilities] = useState(false);
  const [search, setSearch] = useState('');
  const [createOpen, setCreateOpen] = useState(false);
  const [detailItem, setDetailItem] = useState<ModelTypeDefinition | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const initialLoadDone = useRef(false);

  const refresh = useCallback(async (options?: { silent?: boolean }) => {
    if (!options?.silent) {
      setLoading(true);
    }
    setLoadError(null);
    try {
      const response = await listModelTypes();
      setModelTypes(response.modelTypes ?? []);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载模型类型失败');
      if (!options?.silent) {
        setModelTypes([]);
      }
    } finally {
      if (!options?.silent) {
        setLoading(false);
      }
      initialLoadDone.current = true;
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const hasPendingReadiness = useMemo(
    () => modelTypeHasPendingReadiness(modelTypes),
    [modelTypes]
  );

  useEffect(() => {
    if (!initialLoadDone.current || !hasPendingReadiness || loadError) return undefined;
    const timer = window.setInterval(() => {
      void refresh({ silent: true });
    }, 3000);
    return () => window.clearInterval(timer);
  }, [hasPendingReadiness, loadError, refresh]);

  const handleRefreshCapabilities = useCallback(async () => {
    setRefreshingCapabilities(true);
    try {
      await refreshModelTypeTrainingCapabilities();
      await refresh({ silent: true });
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '刷新训练能力失败');
    } finally {
      setRefreshingCapabilities(false);
    }
  }, [refresh]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return modelTypes;
    return modelTypes.filter((item) =>
      [item.name, item.modelTypeId, item.baseAlgorithm, item.adapterKey, item.description ?? '', ...(item.tags ?? [])]
        .join(' ')
        .toLowerCase()
        .includes(q)
    );
  }, [modelTypes, search]);

  const subtitle = loading
    ? '管理 BC、ACT、Diffusion Policy、pi0 等模型结构定义。正在加载模型类型…'
    : loadError
      ? '管理 BC、ACT、Diffusion Policy、pi0 等模型结构定义。加载失败，请重试。'
      : hasPendingReadiness
        ? `管理 BC、ACT、Diffusion Policy、pi0 等模型结构定义。当前共 ${modelTypes.length} 项，pi0 runner 检测中…`
        : refreshingCapabilities
          ? `管理 BC、ACT、Diffusion Policy、pi0 等模型结构定义。当前共 ${modelTypes.length} 项，正在刷新训练能力…`
          : `管理 BC、ACT、Diffusion Policy、pi0 等模型结构定义。当前共 ${modelTypes.length} 项（不含已删除）。`;

  const handleDisable = async (item: ModelTypeDefinition) => {
    await updateModelType(item.modelTypeId, { status: 'disabled' });
    await refresh({ silent: true });
    setDetailItem(null);
  };

  const handleDelete = async (item: ModelTypeDefinition) => {
    if (!window.confirm(`确定删除模型类型「${item.name}」？`)) return;
    await deleteModelType(item.modelTypeId);
    await refresh({ silent: true });
    setDetailItem(null);
  };

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="模型类型"
        subtitle={subtitle}
        actions={
          <PrimaryButton onClick={() => setCreateOpen(true)}>新建模型类型</PrimaryButton>
        }
      />

      <ModulePageFilterCard>
        <FilterInput
          placeholder="搜索模型名称、标识、算法、标签…"
          value={search}
          onChange={setSearch}
        />
        <GhostButton
          disabled={refreshingCapabilities}
          onClick={() => void handleRefreshCapabilities()}
        >
          {refreshingCapabilities ? '刷新中…' : '刷新'}
        </GhostButton>
      </ModulePageFilterCard>

      {loadError ? (
        <SectionCard>
          <p style={{ margin: 0, color: '#dc2626', fontSize: 13 }}>{loadError}</p>
          <div style={{ marginTop: 12 }}>
            <PrimaryButton onClick={() => void refresh()}>重试</PrimaryButton>
          </div>
        </SectionCard>
      ) : null}

      {loading ? (
        <SectionCard>
          <p style={{ margin: 0, color: '#64748b', fontSize: 13 }}>正在加载模型类型…</p>
        </SectionCard>
      ) : loadError ? null : filtered.length === 0 ? (
        <SectionCard>
          <p style={{ margin: 0, color: '#64748b', fontSize: 13 }}>
            暂无模型类型。请点击「新建模型类型」创建，或等待系统初始化内置模型类型。
          </p>
        </SectionCard>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: 16,
            alignItems: 'stretch',
          }}
        >
          {filtered.map((item) => (
            <ModelTypeCard
              key={item.modelTypeId}
              item={item}
              onViewDetail={() => setDetailItem(item)}
              onEnable={
                !item.isBuiltin && item.status === 'draft'
                  ? () =>
                      void updateModelType(item.modelTypeId, { status: 'available' }).then(() =>
                        refresh({ silent: true })
                      )
                  : undefined
              }
            />
          ))}
        </div>
      )}

      <CreateModelTypeModal
        open={createOpen}
        submitting={submitting}
        onClose={() => setCreateOpen(false)}
        onCreated={async () => {
          setSubmitting(true);
          try {
            await refresh({ silent: true });
          } finally {
            setSubmitting(false);
          }
        }}
      />

      <ModelTypeDetailDrawer
        open={Boolean(detailItem)}
        modelType={detailItem}
        onClose={() => setDetailItem(null)}
        onDisable={detailItem ? () => void handleDisable(detailItem) : undefined}
        onDelete={detailItem ? () => void handleDelete(detailItem) : undefined}
      />
    </ModulePageContainer>
  );
}
