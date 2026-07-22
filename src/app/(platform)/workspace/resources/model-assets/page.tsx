'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import ListFooterBar from '@/components/common/ListFooterBar';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import {
  deleteModelAsset,
  deleteModelAssetsBatch,
  MODEL_ASSET_DELETE_CONFIRM,
  modelAssetBatchDeleteConfirm,
} from '@/lib/api/modelAssetsClient';
import { usePagePerfLog } from '@/lib/perf/pagePerfLog';
import {
  useInvalidateWorkspaceLists,
  useModelAssetFilterOptionsQuery,
  useModelAssetsQuery,
} from '@/lib/query/workspaceQueries';
import type { ModelAsset } from '@/types/benchmark';
import { PrimaryButton } from '@/components/workspace/workspaceUi';
import { ModelAssetsTable } from '@/components/workspace/resources/ModelAssetsTable';
import { ModelAssetDetailDrawer } from '@/components/workspace/resources/ModelAssetDetailDrawer';
import { ImportPretrainedModelModal } from '@/components/workspace/resources/ImportPretrainedModelModal';
import {
  ModelAssetFilterBar,
  type ModelAssetFilterOption,
} from '@/components/workspace/resources/ModelAssetFilterBar';

export default function ModelAssetsPage() {
  const [search, setSearch] = useState('');
  const [modelTypeFilter, setModelTypeFilter] = useState('');
  const [datasetFilter, setDatasetFilter] = useState('');
  const [sourceTaskFilter, setSourceTaskFilter] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [detailAsset, setDetailAsset] = useState<ModelAsset | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ModelAsset | null>(null);
  const [batchDeleteOpen, setBatchDeleteOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);

  const listParams = useMemo(
    () => ({
      limit: pageSize,
      offset: (page - 1) * pageSize,
      search: search.trim() || undefined,
      modelType: modelTypeFilter || undefined,
      dataset: datasetFilter || undefined,
      sourceTask: sourceTaskFilter || undefined,
    }),
    [search, modelTypeFilter, datasetFilter, sourceTaskFilter, page, pageSize]
  );

  const { invalidateModelAssets } = useInvalidateWorkspaceLists();
  const {
    data: assetResp,
    isLoading: assetsLoading,
    isFetching: assetsFetching,
    isError: assetsError,
    error: assetsQueryError,
  } = useModelAssetsQuery(listParams);
  const { data: filterOptionsResp } = useModelAssetFilterOptionsQuery();

  const assets = assetResp?.modelAssets ?? [];
  const listTotal = assetResp?.total ?? 0;
  const loading = assetsLoading && !assetResp;
  const loadError = assetsError
    ? assetsQueryError instanceof Error
      ? assetsQueryError.message
      : '加载模型资产失败'
    : null;

  usePagePerfLog('ModelAssets', {
    loading,
    apiRequestCount: loading ? 1 : 0,
  });

  const refresh = useCallback(async () => {
    await invalidateModelAssets();
  }, [invalidateModelAssets]);

  const filterOptions: ModelAssetFilterOption[] = useMemo(
    () => [
      {
        key: 'modelType',
        value: modelTypeFilter,
        placeholder: '模型类型',
        options: (filterOptionsResp?.modelTypes ?? []).map((label) => ({ value: label, label })),
        onChange: setModelTypeFilter,
      },
      {
        key: 'dataset',
        value: datasetFilter,
        placeholder: '数据集',
        options: (filterOptionsResp?.datasets ?? []).map((label) => ({ value: label, label })),
        onChange: setDatasetFilter,
      },
      {
        key: 'sourceTask',
        value: sourceTaskFilter,
        placeholder: '来源任务',
        options: (filterOptionsResp?.sourceTasks ?? []).map((label) => ({ value: label, label })),
        onChange: setSourceTaskFilter,
      },
    ],
    [modelTypeFilter, datasetFilter, sourceTaskFilter, filterOptionsResp]
  );

  const handleResetFilters = useCallback(() => {
    setSearch('');
    setModelTypeFilter('');
    setDatasetFilter('');
    setSourceTaskFilter('');
    setPage(1);
  }, []);

  useEffect(() => {
    setPage(1);
  }, [search, modelTypeFilter, datasetFilter, sourceTaskFilter]);

  const paged = assets;
  const tableTotal = listTotal;

  useEffect(() => {
    setSelectedIds(new Set());
  }, [page]);

  const toggleRow = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const allPageSelected = paged.length > 0 && paged.every((asset) => selectedIds.has(asset.id));

  const toggleSelectAll = useCallback(() => {
    const pageIds = paged.map((asset) => asset.id);
    if (allPageSelected) {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        pageIds.forEach((id) => next.delete(id));
        return next;
      });
    } else {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        pageIds.forEach((id) => next.add(id));
        return next;
      });
    }
  }, [paged, allPageSelected]);

  const handleDelete = useCallback(async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const result = await deleteModelAsset(deleteTarget.id);
      setDeleteTarget(null);
      if (detailAsset?.id === deleteTarget.id) {
        setDetailAsset(null);
      }
      await refresh();
      const warning = result.warnings?.length ? `（${result.warnings.join('；')}）` : '';
      setToast(`模型资产已删除${warning}`);
    } catch (err) {
      setToast(err instanceof Error ? err.message : '删除失败');
    } finally {
      setDeleting(false);
    }
  }, [deleteTarget, detailAsset?.id, refresh]);

  const handleConfirmBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setDeleting(true);
    try {
      const targets = assets.filter((asset) => selectedIds.has(asset.id));
      const { deleted, failed } = await deleteModelAssetsBatch(targets.map((asset) => asset.id));
      const deletedIdSet = new Set(deleted);
      if (detailAsset && deletedIdSet.has(detailAsset.id)) {
        setDetailAsset(null);
      }
      setSelectedIds(new Set());
      setBatchDeleteOpen(false);
      await refresh();
      if (failed.length === 0) {
        setToast(`已删除 ${deleted.length} 个模型资产`);
      } else {
        const detail = failed
          .slice(0, 2)
          .map((item) => `${item.modelAssetId}: ${item.error}`)
          .join('；');
        const suffix = failed.length > 2 ? ` 等 ${failed.length} 个失败` : '';
        setToast(`已删除 ${deleted.length} 个，失败 ${failed.length} 个。${detail}${suffix}`);
      }
    } catch (err) {
      setToast(err instanceof Error ? err.message : '批量删除失败');
    } finally {
      setDeleting(false);
    }
  }, [selectedIds, assets, detailAsset, refresh]);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="模型资产"
        subtitle="平台训练产出与外部导入的 checkpoint，供训练初始化与策略评测选用。"
        actions={
          <PrimaryButton onClick={() => setImportOpen(true)}>导入预训练模型</PrimaryButton>
        }
      />

      {toast ? (
        <p style={{ marginBottom: 12, fontSize: 13, color: toast.includes('失败') ? '#b45309' : '#047857' }}>
          {toast}
        </p>
      ) : null}

      <div
        style={{
          marginBottom: 12,
          padding: '12px 16px',
          backgroundColor: '#ffffff',
          borderRadius: 8,
          border: '1px solid #e5e7eb',
        }}
      >
        <ModelAssetFilterBar
          searchValue={search}
          onSearchChange={setSearch}
          filters={filterOptions}
          onReset={handleResetFilters}
        />
      </div>

      <ModulePageTableCard>
        {loadError ? (
          <p style={{ padding: 24, textAlign: 'center', color: '#b45309' }}>{loadError}</p>
        ) : (
          <ModelAssetsTable
            rows={paged}
            trainingByJobId={new Map()}
            loading={loading}
            emptyMessage={
              loading
                ? '正在加载模型资产…'
                : assets.length === 0 && !assetsFetching
                  ? '暂无可用模型资产，请先完成训练任务。'
                  : '没有符合筛选条件的模型资产。'
            }
            selectedIds={selectedIds}
            onToggleRow={toggleRow}
            onToggleSelectAll={toggleSelectAll}
            allPageSelected={allPageSelected}
            onDetail={setDetailAsset}
            onDelete={setDeleteTarget}
          />
        )}

        {!loading && !loadError && tableTotal > 0 ? (
          <ListFooterBar
            variant="inline"
            total={tableTotal}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size);
              setPage(1);
            }}
            selectedCount={selectedIds.size}
            batchActions={[
              {
                key: 'batch-delete',
                label: '批量删除',
                onClick: () => {
                  if (selectedIds.size === 0) return;
                  setBatchDeleteOpen(true);
                },
                danger: true,
              },
            ]}
          />
        ) : null}
      </ModulePageTableCard>

      <ModelAssetDetailDrawer
        asset={detailAsset}
        trainingRow={null}
        onClose={() => setDetailAsset(null)}
        onDelete={setDeleteTarget}
      />

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="删除模型资产"
        description={MODEL_ASSET_DELETE_CONFIRM}
        confirmText="删除"
        loading={deleting}
        onCancel={() => {
          if (!deleting) setDeleteTarget(null);
        }}
        onConfirm={() => void handleDelete()}
      />

      <ConfirmDialog
        open={batchDeleteOpen}
        title="批量删除"
        description={modelAssetBatchDeleteConfirm(selectedIds.size)}
        confirmText="删除"
        cancelText="取消"
        loading={deleting}
        onCancel={() => {
          if (!deleting) setBatchDeleteOpen(false);
        }}
        onConfirm={() => void handleConfirmBatchDelete()}
      />

      <ImportPretrainedModelModal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImported={(result) => {
          void refresh();
          setToast(`已导入模型资产：${result.modelName}`);
        }}
      />
    </ModulePageContainer>
  );
}
