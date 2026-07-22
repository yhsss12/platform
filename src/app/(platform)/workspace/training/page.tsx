'use client';

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import ListFooterBar from '@/components/common/ListFooterBar';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import TaskFilterBar, { type TaskFilterOption } from '@/components/tasks/TaskFilterBar';
import { useI18n } from '@/components/common/I18nProvider';
import { PrimaryButton } from '@/components/workspace/workspaceUi';
import { TrainingTasksTable } from '@/components/workspace/training/TrainingTasksTable';
import {
  CreateTrainingTaskModal,
  type CreateTrainingTaskInput,
} from '@/components/workspace/training/CreateTrainingTaskModal';
import { resolveTrainingDatasetManifests } from '@/lib/workspace/resolveTrainingDatasetManifest';
import { TrainingTaskDetailDrawer } from '@/components/workspace/training/TrainingTaskDetailDrawer';
import { listWorkspaceDataItemsForUi } from '@/lib/workspace/workspaceDataSources';
import { formatTrainingRecipeLabel, normalizeTrainingRecipeFilterValue } from '@/lib/workspace/trainingRecipe';
import {
  trainingStatusOptions,
  type TrainingTaskRow,
} from '@/lib/mock/workspaceTrainingMock';
import { startTrainingJob } from '@/lib/mock/workspaceTrainingStore';
import {
  deleteTrainingJob,
  deleteTrainingJobsBatch,
  TRAINING_JOB_DELETE_CONFIRM,
  trainingJobBatchDeleteConfirm,
} from '@/lib/api/trainingClient';
import { usePagePerfLog } from '@/lib/perf/pagePerfLog';
import {
  useInvalidateWorkspaceLists,
  useTrainingJobsQuery,
} from '@/lib/query/workspaceQueries';
import { usePollingRefresh } from '@/lib/workspace/usePollingRefresh';
import { isTrainingJobInProgress } from '@/lib/workspace/trainingStatus';
import { trainingListItemToRow } from '@/lib/workspace/workspaceJobMapper';

function isRealTrainingRow(row: TrainingTaskRow): boolean {
  return row.source === 'real';
}

export default function TrainingCenterPage() {
  return (
    <Suspense fallback={null}>
      <TrainingCenterPageContent />
    </Suspense>
  );
}

function TrainingCenterPageContent() {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const urlDataset = searchParams.get('dataset') ?? undefined;
  const shouldOpenCreate = searchParams.get('openCreate') === '1';
  const urlJobId = searchParams.get('jobId') ?? undefined;

  const [search, setSearch] = useState('');
  const [modelFilter, setModelFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [createOpen, setCreateOpen] = useState(false);
  const [initialDataset, setInitialDataset] = useState<string | undefined>();
  const [detailRow, setDetailRow] = useState<TrainingTaskRow | null>(null);
  const [rowToDelete, setRowToDelete] = useState<TrainingTaskRow | null>(null);
  const [batchDeleteOpen, setBatchDeleteOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const listParams = useMemo(
    () => ({
      limit: pageSize,
      offset: (page - 1) * pageSize,
      search: search.trim() || undefined,
      status: statusFilter || undefined,
      model: modelFilter || undefined,
    }),
    [page, pageSize, search, statusFilter, modelFilter]
  );

  const { invalidateTrainingJobs } = useInvalidateWorkspaceLists();
  const {
    data: trainingResponse,
    isLoading,
    isError,
    refetch,
  } = useTrainingJobsQuery(listParams);

  const tasks = useMemo(
    () =>
      (trainingResponse?.jobs ?? [])
        .map((job) => trainingListItemToRow(job))
        .filter(isRealTrainingRow),
    [trainingResponse?.jobs]
  );
  const apiUnavailable = isError;
  const listTotal = trainingResponse?.total ?? 0;

  usePagePerfLog('TrainingCenter', {
    loading: isLoading,
    apiRequestCount: isLoading ? 1 : 0,
  });

  const refreshTasks = useCallback(async () => {
    await invalidateTrainingJobs();
  }, [invalidateTrainingJobs]);

  const hasInProgressTasks = useMemo(
    () => tasks.some((t) => isTrainingJobInProgress(t.status)),
    [tasks]
  );

  usePollingRefresh(hasInProgressTasks, () => void refetch(), 4000);

  const dataCenterItems = useMemo(() => listWorkspaceDataItemsForUi(), [createOpen]);

  const showToast = useCallback((text: string) => {
    setToastMsg(text);
    setTimeout(() => setToastMsg(null), 3200);
  }, []);

  useEffect(() => {
    setDetailRow((prev) => {
      if (!prev) return null;
      return tasks.find((row) => row.id === prev.id) ?? null;
    });
  }, [tasks]);

  useEffect(() => {
    if (urlDataset) {
      setInitialDataset(urlDataset);
      setCreateOpen(true);
    } else if (shouldOpenCreate) {
      setCreateOpen(true);
    }
  }, [urlDataset, shouldOpenCreate]);

  useEffect(() => {
    if (!urlJobId || tasks.length === 0) return;
    const match = tasks.find((row) => row.trainJobId === urlJobId || row.id === urlJobId);
    if (match) setDetailRow(match);
  }, [urlJobId, tasks]);

  const modelFilterOptions = useMemo(() => {
    const models = Array.from(
      new Set(tasks.map((t) => normalizeTrainingRecipeFilterValue(t)).filter(Boolean))
    );
    return models;
  }, [tasks]);

  const paged = tasks;

  useEffect(() => {
    setSelectedIds(new Set());
  }, [page]);

  useEffect(() => {
    setPage(1);
  }, [search, modelFilter, statusFilter]);

  const toggleRow = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const allPageSelected = paged.length > 0 && paged.every((row) => selectedIds.has(row.id));

  const toggleSelectAll = useCallback(() => {
    const pageIds = paged.map((row) => row.id);
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

  const filterOptions: TaskFilterOption[] = useMemo(
    () => [
      {
        key: 'model',
        value: modelFilter,
        placeholder: '模型类型',
        options: [{ value: '', label: '全部' }, ...modelFilterOptions.map((m) => ({ value: m, label: m }))],
        onChange: setModelFilter,
      },
      {
        key: 'status',
        value: statusFilter,
        placeholder: '状态',
        options: [{ value: '', label: '全部' }, ...trainingStatusOptions.map((s) => ({ value: s, label: s }))],
        onChange: setStatusFilter,
      },
    ],
    [modelFilter, statusFilter, modelFilterOptions]
  );

  const handleStart = useCallback(
    async (input: CreateTrainingTaskInput) => {
      const { manifests, missingIds } = await resolveTrainingDatasetManifests(input.datasets);
      if (missingIds.length > 0 || manifests.length === 0) {
        showToast('未找到数据集 manifest，请确认生成任务已完成并包含 dataset.manifest.json');
        return;
      }

      let mergedManifest = manifests[0];
      try {
        if (manifests.length > 1) {
          const { mergeTrainingManifestsClient } = await import('@/lib/workspace/trainingDatasetCompat');
          mergedManifest = mergeTrainingManifestsClient(manifests);
        }
      } catch (err) {
        showToast(err instanceof Error ? err.message : '数据集结构不一致，无法合并训练');
        return;
      }

      setSubmitting(true);
      try {
        const { row, raw } = await startTrainingJob(
          mergedManifest,
          input,
          manifests.length > 1 ? manifests : undefined
        );

        await refreshTasks();
        setCreateOpen(false);
        setInitialDataset(undefined);

        if (raw.status === 'running' || raw.status === 'queued') {
          showToast(`训练任务已创建：${row.trainJobId}`);
        } else if (raw.status === 'completed' && raw.checkpointExists) {
          showToast(`训练任务已完成，模型资产：${raw.modelAssetId ?? '已生成 checkpoint'}`);
        } else if (raw.status === 'failed' || raw.status === 'backend_unavailable') {
          showToast(`训练任务创建失败：${raw.message || '请查看 train.log'}`);
        } else {
          showToast(`训练任务已创建：${row.trainJobId}`);
        }
      } catch (err) {
        showToast(err instanceof Error ? err.message : '训练任务创建失败');
      } finally {
        setSubmitting(false);
      }
    },
    [refreshTasks, showToast]
  );

  const handleConfirmDelete = useCallback(async () => {
    if (!rowToDelete) return;
    setDeleteLoading(true);
    try {
      await deleteTrainingJob(rowToDelete.trainJobId);
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(rowToDelete.id);
        return next;
      });
      if (detailRow?.id === rowToDelete.id) setDetailRow(null);
      setRowToDelete(null);
      await refreshTasks();
      showToast('训练任务及其产物已删除');
    } catch (err) {
      showToast(err instanceof Error ? err.message : '删除失败');
    } finally {
      setDeleteLoading(false);
    }
  }, [rowToDelete, detailRow, showToast]);

  const handleConfirmBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setDeleteLoading(true);
    try {
      const targets = tasks.filter((row) => selectedIds.has(row.id));
      const { deleted, failed } = await deleteTrainingJobsBatch(
        targets.map((row) => row.trainJobId)
      );
      const deletedJobIds = new Set(deleted);
      const deletedIds = new Set(
        targets.filter((row) => deletedJobIds.has(row.trainJobId)).map((row) => row.id)
      );
      if (detailRow && deletedIds.has(detailRow.id)) setDetailRow(null);
      setSelectedIds(new Set());
      setBatchDeleteOpen(false);
      await refreshTasks();
      if (failed.length === 0) {
        showToast(`已删除 ${deleted.length} 条`);
      } else {
        const detail = failed
          .slice(0, 2)
          .map((item) => `${item.jobId}: ${item.error}`)
          .join('；');
        const suffix = failed.length > 2 ? ` 等 ${failed.length} 条失败` : '';
        showToast(`已删除 ${deleted.length} 条，失败 ${failed.length} 条。${detail}${suffix}`);
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : '批量删除失败');
    } finally {
      setDeleteLoading(false);
    }
  }, [selectedIds, tasks, detailRow, showToast]);

  const emptyMessage = isLoading
    ? '加载训练任务…'
    : '暂无训练任务。请先从数据中心完成仿真数据生成，或通过新建任务启动训练。';

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('workspacePages.trainingCenterTitle')}
        subtitle={t('workspacePages.trainingCenterSubtitle')}
        subtitleSingleLine
        actions={<PrimaryButton onClick={() => setCreateOpen(true)}>新建任务</PrimaryButton>}
      />

      {apiUnavailable ? (
        <div
          style={{
            marginBottom: 10,
            padding: '8px 12px',
            borderRadius: 8,
            background: '#fffbeb',
            border: '1px solid #fcd34d',
            color: '#92400e',
            fontSize: 12,
          }}
        >
          无法连接 workspace job 服务，请稍后刷新重试。
        </div>
      ) : null}

      <ModulePageFilterCard>
          <TaskFilterBar
            searchValue={search}
            onSearchChange={setSearch}
            searchPlaceholder="搜索训练任务"
            filters={filterOptions}
            onReset={() => {
              setSearch('');
              setModelFilter('');
              setStatusFilter('');
            }}
          />
        </ModulePageFilterCard>

      <ModulePageTableCard>
        <TrainingTasksTable
          rows={paged}
          dataCenterItems={dataCenterItems}
          loading={isLoading}
          emptyMessage={emptyMessage}
          selectedIds={selectedIds}
          onToggleRow={toggleRow}
          onToggleSelectAll={toggleSelectAll}
          allPageSelected={allPageSelected}
          onDetail={(row) => setDetailRow(row)}
          onDelete={setRowToDelete}
        />
        <ListFooterBar
          variant="inline"
          total={listTotal}
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
      </ModulePageTableCard>

      <CreateTrainingTaskModal
        open={createOpen}
        submitting={submitting}
        onClose={() => {
          setCreateOpen(false);
          setInitialDataset(undefined);
        }}
        onStart={handleStart}
        initialDataset={initialDataset}
      />

      <TrainingTaskDetailDrawer
        row={detailRow}
        onClose={() => setDetailRow(null)}
        onRefresh={refreshTasks}
        onDelete={setRowToDelete}
      />

      <ConfirmDialog
        open={batchDeleteOpen}
        title="批量删除"
        description={trainingJobBatchDeleteConfirm(selectedIds.size)}
        confirmText="删除"
        cancelText="取消"
        loading={deleteLoading}
        onCancel={() => {
          if (!deleteLoading) setBatchDeleteOpen(false);
        }}
        onConfirm={handleConfirmBatchDelete}
      />

      <ConfirmDialog
        open={rowToDelete !== null}
        title="删除"
        description={TRAINING_JOB_DELETE_CONFIRM}
        confirmText="删除"
        cancelText="取消"
        loading={deleteLoading}
        onCancel={() => {
          if (!deleteLoading) setRowToDelete(null);
        }}
        onConfirm={handleConfirmDelete}
      />

      {toastMsg ? (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 14,
            zIndex: 1700,
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#fff',
            maxWidth: 'min(92vw, 560px)',
            textAlign: 'center',
            lineHeight: 1.5,
          }}
        >
          {toastMsg}
        </div>
      ) : null}
    </ModulePageContainer>
  );
}
