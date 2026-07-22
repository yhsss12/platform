'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  createSyncBatchJob,
  getDataAssets,
  getDataAsset,
  deleteDataAsset,
  deleteDataAssetsBatch,
  dataAssetRequiresAgentSync,
  isDataAssetSynced,
  getSyncBatchJobStatus,
  type DataAssetItem,
  type DataAssetQueryParams,
} from '@/features/data-platform/api/dataAssetsApi';
import { useTaskCenter } from '@/components/task-center/TaskCenterContext';
import ImportDataDialog from '@/components/assets/ImportDataDialog';
import DatasetTable from '@/components/data/DatasetTable';
import AssetDetailModal from '@/components/data/AssetDetailModal';
import ExportConfigModal from '@/components/data/ExportConfigModal';
import FiltersBar from '@/components/data/FiltersBar';
import ListFooterBar from '@/components/common/ListFooterBar';
import * as projectService from '@/lib/projects/projectService';
import { recordProjectActivityAndTouch } from '@/lib/projects/projectService';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageFilterCard,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import { useI18n } from '@/components/common/I18nProvider';
import type { Project } from '@/lib/projects/types';
import { useAuthStore } from '@/store/authStore';
import { normalizeRole } from '@/lib/api/roleLabels';

export default function DataPage() {
  return <DataPageContent />;
}

function normalizeDataAssetFormat(item: DataAssetItem): string {
  let fmt = (item.format || '').toLowerCase();
  if (!fmt && item.filename) {
    const lower = item.filename.toLowerCase();
    if (/\.(hdf5|h5)$/.test(lower)) fmt = 'hdf5';
    else if (/\.mcap$/.test(lower)) fmt = 'mcap';
    else if (/\.zip$/.test(lower)) fmt = 'lerobot';
  }
  return fmt || '';
}

function resolveDeleteTargets(
  ids: number[],
  selectedItemsMap: Map<number, DataAssetItem>,
  datasets: DataAssetItem[],
): DataAssetItem[] {
  const out: DataAssetItem[] = [];
  for (const id of ids) {
    const a = selectedItemsMap.get(id) ?? datasets.find((d) => d.id === id);
    if (a) out.push(a);
  }
  return out;
}

/** 批量删除：不允许同一批里混选「已同步」与「未同步」的采集数据 */
function collectSyncMixedError(targets: DataAssetItem[]): string | null {
  const collect = targets.filter((a) => String(a.source || '').trim().toLowerCase() === 'collect');
  if (collect.length <= 1) return null;
  const syncedFlags = collect.map((a) => isDataAssetSynced(a));
  if (new Set(syncedFlags).size > 1) {
    return '不能同时选择已同步与未同步的采集数据，请分开删除';
  }
  return null;
}

type DeleteDialogVariant = 'synced_collect' | 'unsynced_collect' | 'generic';

function deleteDialogVariantForTargets(targets: DataAssetItem[]): DeleteDialogVariant {
  const collect = targets.filter((a) => String(a.source || '').trim().toLowerCase() === 'collect');
  if (collect.length === 0) return 'generic';
  const anyUnsynced = collect.some((a) => !isDataAssetSynced(a));
  if (anyUnsynced) return 'unsynced_collect';
  return 'synced_collect';
}

function DataPageContent() {
  const router = useRouter();
  const { t } = useI18n();
  const user = useAuthStore((s) => s.user);
  const canDeleteDataAssets = normalizeRole(user?.role) !== 'USER';
  const [datasets, setDatasets] = useState<DataAssetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [selectedItemsMap, setSelectedItemsMap] = useState<Map<number, DataAssetItem>>(new Map());
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteTargetIds, setDeleteTargetIds] = useState<number[]>([]);
  const [deleteLoading, setDeleteLoading] = useState(false);
  /** 已同步采集数据删除：分项勾选 */
  const [deleteCloud, setDeleteCloud] = useState(true);
  const [deleteRemote, setDeleteRemote] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [exportingAssetId, setExportingAssetId] = useState<number | null>(null);
  const [exportConfigModalOpen, setExportConfigModalOpen] = useState(false);
  const [exportModalAssetIds, setExportModalAssetIds] = useState<number[]>([]);
  const [exportModalCount, setExportModalCount] = useState(0);
  const [exportModalFormatLabel, setExportModalFormatLabel] = useState('');
  const [exportModalFormatSummary, setExportModalFormatSummary] = useState('');
  const [exportModalProjectName, setExportModalProjectName] = useState('');
  const [exportModalAssetNamesPreview, setExportModalAssetNamesPreview] = useState('');
  const [projectList, setProjectList] = useState<Project[]>([]);
  const [detailAsset, setDetailAsset] = useState<DataAssetItem | null>(null);

  const [batchSyncing, setBatchSyncing] = useState(false);
  const batchSyncPollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const syncingAssetIdsRef = useRef<Set<number>>(new Set());

  const [filters, setFilters] = useState<DataAssetQueryParams>({
    keyword: '',
    project: '',
    format: '',
    source: '',
    task_name: '',
    created_from: '',
    created_to: '',
  });

  useEffect(() => {
    let active = true;
    projectService.listAsync(false)
      .then((result) => {
        if (!active) return;
        const projects = Array.isArray(result) ? result : result.projects;
        setProjectList(projects.filter((p) => p.status !== '已归档'));
      })
      .catch(() => {
        if (!active) return;
        setProjectList([]);
      });
    return () => {
      active = false;
    };
  }, []);

  // 任务选项：根据项目、数据类型、数据来源滚动更新（范围逐步缩小）
  const [taskOptions, setTaskOptions] = useState<{ value: string; label: string }[]>([]);
  useEffect(() => {
    let cancelled = false;
    import('@/features/data-platform/api/dataAssetsApi').then(({ getDataAssetTaskOptions }) => {
      getDataAssetTaskOptions({
        project: filters.project || undefined,
        format: filters.format || undefined,
        source: filters.source || undefined,
      })
        .then((res) => {
          if (cancelled) return;
          if (res.ok && res.data?.items) {
            const seen = new Set<string>();
            const items = res.data.items.filter((opt) => {
              if (seen.has(opt.value)) return false;
              seen.add(opt.value);
              return true;
            });
            setTaskOptions(items);
          }
          else setTaskOptions([]);
        })
        .catch(() => {
          if (!cancelled) setTaskOptions([]);
        });
    });
    return () => { cancelled = true; };
  }, [filters.project, filters.source, filters.format]);

  useEffect(() => {
    if (!toastMessage) return;
    const t = setTimeout(() => setToastMessage(''), 2200);
    return () => clearTimeout(t);
  }, [toastMessage]);

  useEffect(() => {
    return () => {
      if (batchSyncPollTimerRef.current) {
        clearTimeout(batchSyncPollTimerRef.current);
        batchSyncPollTimerRef.current = null;
      }
    };
  }, []);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const res = await getDataAssets({
        keyword: filters.keyword,
        project: filters.project,
        format: filters.format,
        source: filters.source,
        task_name: filters.task_name || undefined,
        created_from: (filters.created_from || '').trim() || undefined,
        created_to: (filters.created_to || '').trim() || undefined,
        page,
        page_size: pageSize,
        reconcile_collect_disk:
          String(filters.source || '').trim().toLowerCase() === 'collect' ? true : undefined,
      });
      if (res.ok && res.data) {
        setDatasets(res.data.items);
        setTotal(res.data.total);
      } else {
        console.error('加载数据资产失败:', res.error);
        setDatasets([]);
        setTotal(0);
      }
    } catch (error) {
      console.error('加载数据资产异常:', error);
      setDatasets([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    page,
    pageSize,
    filters.keyword,
    filters.project,
    filters.format,
    filters.source,
    filters.task_name,
    filters.created_from,
    filters.created_to,
  ]);

  // 筛选条件变化时清除选择（避免选中项与当前列表不一致）
  useEffect(() => {
    setSelectedIds(new Set());
    setSelectedItemsMap(new Map());
  }, [
    filters.keyword,
    filters.project,
    filters.format,
    filters.source,
    filters.task_name,
    filters.created_from,
    filters.created_to,
  ]);

  // 支持跨页累积勾选：翻页时不清空选择，仅在筛选条件变化时清空。

  // 数据列表刷新后，同步已选项的字段（尤其是 sync_status），避免批量按钮状态不一致
  useEffect(() => {
    if (selectedIds.size === 0) return;
    setSelectedItemsMap((prev) => {
      if (prev.size === 0) return prev;
      let changed = false;
      const next = new Map(prev);
      for (const [id] of next) {
        const d = datasets.find((x) => x.id === id);
        if (d && next.get(id) !== d) {
          next.set(id, d);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [datasets, selectedIds]);

  const handleFilterChange = (newFilters: Partial<DataAssetQueryParams>) => {
    const next = { ...filters, ...newFilters };
    // 项目/数据类型/数据来源变化时清空任务选择，避免无效任务 ID
    if ('project' in newFilters || 'format' in newFilters || 'source' in newFilters) {
      next.task_name = '';
    }
    setFilters(next);
    setPage(1);
  };

  const handleResetFilters = () => {
    setFilters({
      keyword: '',
      project: '',
      format: '',
      source: '',
      task_name: '',
      created_from: '',
      created_to: '',
    });
    setPage(1);
  };

  // 导入成功后刷新列表，并通知项目列表页刷新数据数（文案由 ImportDataDialog 传入）
  const handleImportSuccess = (message?: string) => {
    setImportDialogOpen(false);
    setPage(1);
    setToastMessage((message && message.trim()) || '导入成功');
    setTimeout(() => {
      loadDatasets();
      window.dispatchEvent(new Event('datasets-changed'));
    }, 300);
  };

  const resolveProject = useCallback((asset: DataAssetItem) => {
    const pv = asset.project_id ?? '';
    if (!pv) return { projectId: '', projectName: asset.project_name ?? '' };
    const byId = projectList.find((p) => p.id === pv);
    if (byId) return { projectId: byId.id, projectName: byId.name };
    return { projectId: pv, projectName: asset.project_name ?? pv };
  }, [projectList]);

  const selectedDatasets = useMemo(
    () => Array.from(selectedItemsMap.values()),
    [selectedItemsMap]
  );
  const hasUnsyncedSelection = useMemo(
    () => selectedDatasets.some((d) => !isDataAssetSynced(d)),
    [selectedDatasets]
  );
  const hasLerobotSelection = useMemo(
    () =>
      selectedDatasets.some((d) => {
        const fmt = (d.format || '').toLowerCase();
        return fmt === 'lerobot' || (!!d.filename && d.filename.toLowerCase().endsWith('.zip'));
      }),
    [selectedDatasets]
  );

  const handleSelectionChange = useCallback((ids: Set<number>) => {
    setSelectedIds(ids);
    setSelectedItemsMap((prev) => {
      const next = new Map(prev);
      for (const id of prev.keys()) {
        if (!ids.has(id)) next.delete(id);
      }
      for (const id of ids) {
        if (!next.has(id)) {
          const d = datasets.find((x) => x.id === id);
          if (d) next.set(id, d);
        }
      }
      return next;
    });
  }, [datasets]);

  const handleClearSelection = useCallback(() => {
    setSelectedIds(new Set());
    setSelectedItemsMap(new Map());
  }, []);

  const openDeleteDialogForSingle = useCallback(
    (id: number) => {
      const targets = resolveDeleteTargets([id], selectedItemsMap, datasets);
      const mixErr = collectSyncMixedError(targets);
      if (mixErr) {
        setToastMessage(mixErr);
        return;
      }
      setDeleteTargetIds([id]);
      setDeleteCloud(true);
      setDeleteRemote(false);
      setDeleteDialogOpen(true);
    },
    [selectedItemsMap, datasets],
  );

  const openDeleteDialogForBatch = useCallback(() => {
    const count = selectedIds.size;
    if (count === 0) return;
    const ids = Array.from(selectedIds);
    const targets = resolveDeleteTargets(ids, selectedItemsMap, datasets);
    const mixErr = collectSyncMixedError(targets);
    if (mixErr) {
      setToastMessage(mixErr);
      return;
    }
    setDeleteTargetIds(ids);
    setDeleteCloud(true);
    setDeleteRemote(false);
    setDeleteDialogOpen(true);
  }, [selectedIds, selectedItemsMap, datasets]);

  const deleteDialogVariant = useMemo((): DeleteDialogVariant => {
    if (!deleteDialogOpen || deleteTargetIds.length === 0) return 'generic';
    const targets = resolveDeleteTargets(deleteTargetIds, selectedItemsMap, datasets);
    return deleteDialogVariantForTargets(targets);
  }, [deleteDialogOpen, deleteTargetIds, selectedItemsMap, datasets]);

  const deleteDialogDescription = useMemo(() => {
    switch (deleteDialogVariant) {
      case 'synced_collect':
        return '可分别勾选删除对象存储/平台中的副本与采集端磁盘上的原始文件，至少选择一项。';
      case 'unsynced_collect':
        return '将删除平台记录并尝试删除采集端上的未同步文件，操作不可恢复。确认删除？';
      default:
        return '删除后将无法恢复，确定要删除该数据吗？';
    }
  }, [deleteDialogVariant]);

  const handleConfirmDelete = useCallback(async () => {
    if (deleteTargetIds.length === 0 || deleteLoading) return;
    setDeleteLoading(true);
    const ids = [...deleteTargetIds];
    const targets = resolveDeleteTargets(ids, selectedItemsMap, datasets);
    const variant = deleteDialogVariantForTargets(targets);

    if (variant === 'synced_collect' && !deleteCloud && !deleteRemote) {
      setDeleteLoading(false);
      setToastMessage('请至少勾选删除云端或采集端一项');
      return;
    }

    if (variant === 'synced_collect' && !deleteCloud) {
      const hasNonCollect = targets.some((a) => String(a.source || '').trim().toLowerCase() !== 'collect');
      if (hasNonCollect) {
        setDeleteLoading(false);
        setToastMessage('所选数据中包含非采集来源时，必须勾选「删除云端副本」');
        return;
      }
    }

    let opts: { deleteRemote?: boolean; deleteCloud?: boolean } = {};
    if (variant === 'synced_collect') {
      opts = { deleteCloud, deleteRemote };
    } else if (variant === 'unsynced_collect') {
      opts = { deleteCloud: true, deleteRemote: true };
    } else {
      opts = { deleteCloud: true, deleteRemote: false };
    }

    let successCount = 0;
    let firstError: string | null = null;
    let firstWarning: string | null = null;

    if (ids.length > 1) {
      try {
        const res = await deleteDataAssetsBatch(ids, opts);
        if (res.ok && res.data) {
          successCount = res.data.deleted?.length ?? 0;
          if (res.data.errors?.length) {
            firstError = res.data.errors.join('；');
          }
          const ws = res.data.warnings;
          if (ws?.length) {
            firstWarning = res.warning || ws.join('；');
          } else if (res.warning) {
            firstWarning = res.warning;
          }
          for (const id of res.data.deleted ?? []) {
            const asset = datasets.find((d) => d.id === id);
            if (asset?.project_id) {
              recordProjectActivityAndTouch(
                asset.project_id,
                'DATA_DELETED',
                '删除了数据资产',
                '当前用户',
                String(id),
              );
            }
          }
        } else if (!res.ok) {
          firstError = res.error || '未知错误';
        }
      } catch (error) {
        firstError = error instanceof Error ? error.message : String(error);
      }
    } else {
      for (const id of ids) {
        try {
          const res = await deleteDataAsset(id, opts);
          if (res.ok) {
            successCount += 1;
            const w = res.warning;
            if (w && !firstWarning) {
              firstWarning = w;
            }
            const asset = datasets.find((d) => d.id === id);
            if (asset?.project_id) {
              recordProjectActivityAndTouch(
                asset.project_id,
                'DATA_DELETED',
                '删除了数据资产',
                '当前用户',
                String(id),
              );
            }
          } else if (!firstError) {
            firstError = res.error || '未知错误';
          }
        } catch (error) {
          if (!firstError) {
            firstError = error instanceof Error ? error.message : String(error);
          }
        }
      }
    }

    setDeleteLoading(false);
    setDeleteDialogOpen(false);
    setDeleteTargetIds([]);
    setDeleteCloud(true);
    setDeleteRemote(false);

    if (successCount > 0) {
      const newTotal = Math.max(0, total - successCount);
      const newTotalPages = Math.max(1, Math.ceil(newTotal / pageSize));
      setPage((p) => Math.min(p, newTotalPages));
      handleClearSelection();
      loadDatasets();
      window.dispatchEvent(new Event('datasets-changed'));

      if (firstError) {
        setToastMessage(
          firstWarning
            ? `已删除 ${successCount} 条。采集端提示：${firstWarning} 另有失败：${firstError}`
            : `已删除 ${successCount} 条；另有失败：${firstError}`,
        );
      } else if (firstWarning) {
        setToastMessage(`已删除 ${successCount} 条。提示：${firstWarning}`);
      } else {
        setToastMessage(successCount === ids.length ? '删除成功' : `已删除 ${successCount} 条`);
      }
    } else if (firstError) {
      setToastMessage(`删除失败：${firstError}`);
    } else if (firstWarning) {
      setToastMessage(firstWarning);
    }
  }, [
    deleteTargetIds,
    deleteLoading,
    datasets,
    total,
    pageSize,
    handleClearSelection,
    loadDatasets,
    deleteCloud,
    deleteRemote,
    selectedItemsMap,
  ]);

  const handlePageSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setPage(1);
  }, []);

  const formatLabelFromKey = useCallback((fmt: string) => {
    if (fmt === 'hdf5') return 'HDF5';
    if (fmt === 'mcap') return 'MCAP';
    if (fmt === 'lerobot') return 'LeRobot';
    if (fmt === 'directory') return 'Directory';
    return fmt ? fmt.toUpperCase() : '—';
  }, []);

  const openExportConfigModal = useCallback(
    (
      ids: number[],
      fmt: string,
      extra?: { projectName?: string; assetNamesPreview?: string; formatSummary?: string }
    ) => {
      setExportModalAssetIds(ids);
      setExportModalCount(ids.length);
      const fl = formatLabelFromKey(fmt);
      setExportModalFormatLabel(fl);
      setExportModalFormatSummary(extra?.formatSummary ?? fl);
      setExportModalProjectName(extra?.projectName ?? '');
      setExportModalAssetNamesPreview(extra?.assetNamesPreview ?? '');
      setExportConfigModalOpen(true);
    },
    [formatLabelFromKey]
  );

  const handleSingleExport = useCallback(
    (asset: DataAssetItem) => {
      if (!isDataAssetSynced(asset)) {
        setToastMessage('该数据尚未同步，暂不可导出');
        return;
      }
      if (exportConfigModalOpen) return;
      const fmt = normalizeDataAssetFormat(asset);
      setExportingAssetId(asset.id);
      openExportConfigModal([asset.id], fmt, {
        projectName: asset.project_name || asset.project_id || undefined,
        assetNamesPreview: asset.filename,
      });
    },
    [exportConfigModalOpen, openExportConfigModal]
  );
  const { addSyncTask, addBatchSyncTask, updateTask } = useTaskCenter();

  const handleBatchExport = useCallback(() => {
    if (exportConfigModalOpen) return;
    if (selectedDatasets.length === 0) {
      setToastMessage('请先勾选要导出的数据');
      return;
    }
    if (selectedDatasets.some((d) => !isDataAssetSynced(d))) {
      setToastMessage('包含未同步数据，暂不可导出');
      return;
    }
    const formats = selectedDatasets.map(normalizeDataAssetFormat).filter(Boolean);
    const uniqueFormats = new Set(formats);
    if (uniqueFormats.size > 1) {
      setToastMessage('当前仅支持同一种数据格式的批量导出，请按 HDF5、MCAP 或 LeRobot 分别导出。');
      return;
    }
    const fmt = formats[0] || '';
    setExportingAssetId(null);
    const projectIds = new Set(selectedDatasets.map((d) => d.project_id || '').filter(Boolean));
    let batchProjectName = '';
    if (projectIds.size === 1) {
      const first = selectedDatasets[0];
      batchProjectName = (first?.project_name || first?.project_id || '').trim();
    } else if (projectIds.size > 1) {
      batchProjectName = '多个项目';
    }
    const head = selectedDatasets
      .slice(0, 3)
      .map((d) => d.filename)
      .join('、');
    const tail = selectedDatasets.length > 3 ? ` 等 ${selectedDatasets.length} 条` : '';
    openExportConfigModal(selectedDatasets.map((d) => d.id), fmt, {
      projectName: batchProjectName,
      assetNamesPreview: head ? `${head}${tail}` : '',
    });
  }, [exportConfigModalOpen, selectedDatasets, openExportConfigModal]);

  const handleSyncDataset = useCallback(
    async (asset: DataAssetItem) => {
      if (!dataAssetRequiresAgentSync(asset)) {
        return;
      }
      if (isDataAssetSynced(asset)) {
        return;
      }
      if (syncingAssetIdsRef.current.has(asset.id)) {
        setToastMessage('该数据正在同步中，请勿重复发起同步');
        return;
      }
      const st = String(asset.sync_status || '').trim().toLowerCase();
      if (st === 'syncing') {
        setToastMessage('该数据正在同步中，请勿重复发起同步');
        return;
      }
      syncingAssetIdsRef.current.add(asset.id);
      setDatasets((prev) =>
        prev.map((d) =>
          d.id === asset.id
            ? {
                ...d,
                sync_status: 'syncing',
                sync_error: null,
              }
            : d
        )
      );
      setSelectedItemsMap((prev) => {
        if (!prev.has(asset.id)) return prev;
        const next = new Map(prev);
        const cur = next.get(asset.id);
        if (cur) {
          next.set(asset.id, { ...cur, sync_status: 'syncing', sync_error: null });
        }
        return next;
      });
      try {
        const res = await addSyncTask({ assetId: asset.id, filename: asset.filename });
        if (res.ok) {
          setDatasets((prev) =>
            prev.map((d) =>
              d.id === asset.id
                ? {
                    ...d,
                    sync_status: 'synced',
                    sync_error: null,
                  }
                : d
            )
          );
          setSelectedItemsMap((prev) => {
            if (!prev.has(asset.id)) return prev;
            const next = new Map(prev);
            const cur = next.get(asset.id);
            if (cur) {
              next.set(asset.id, { ...cur, sync_status: 'synced', sync_error: null });
            }
            return next;
          });
        } else if (res.error) {
          setToastMessage(res.error);
          setDatasets((prev) =>
            prev.map((d) =>
              d.id === asset.id
                ? {
                    ...d,
                    sync_status: 'failed',
                    sync_error: res.error,
                  }
                : d
            )
          );
          setSelectedItemsMap((prev) => {
            if (!prev.has(asset.id)) return prev;
            const next = new Map(prev);
            const cur = next.get(asset.id);
            if (cur) {
              next.set(asset.id, { ...cur, sync_status: 'failed', sync_error: res.error });
            }
            return next;
          });
        }
      } catch (error) {
        setToastMessage(error instanceof Error ? error.message : '发起同步失败');
        setDatasets((prev) =>
          prev.map((d) =>
            d.id === asset.id
              ? {
                  ...d,
                  sync_status: 'failed',
                  sync_error: error instanceof Error ? error.message : '发起同步失败',
                }
              : d
          )
        );
        setSelectedItemsMap((prev) => {
          if (!prev.has(asset.id)) return prev;
          const next = new Map(prev);
          const cur = next.get(asset.id);
          if (cur) {
            next.set(asset.id, { ...cur, sync_status: 'failed', sync_error: error instanceof Error ? error.message : '发起同步失败' });
          }
          return next;
        });
      } finally {
        syncingAssetIdsRef.current.delete(asset.id);
      }
    },
    [addSyncTask, loadDatasets]
  );

  const handleBatchSync = useCallback(async () => {
    if (batchSyncing) return;
    if (selectedIds.size === 0) return;
    if (hasLerobotSelection) {
      setToastMessage('LeRobot 数据不支持同步');
      return;
    }

    const syncingAssets = selectedDatasets.filter((d) => String(d.sync_status || '').trim().toLowerCase() === 'syncing');
    if (syncingAssets.length > 0) {
      setToastMessage(`所选数据包含同步中的条目（${syncingAssets.length} 个），请等待同步完成后再试`);
      return;
    }

    const unsyncedAssets = selectedDatasets.filter(
      (d) => dataAssetRequiresAgentSync(d) && !isDataAssetSynced(d),
    );
    if (unsyncedAssets.length === 0) {
      setToastMessage('所选数据无需同步或均已同步');
      return;
    }

    try {
      if (batchSyncPollTimerRef.current) {
        clearTimeout(batchSyncPollTimerRef.current);
        batchSyncPollTimerRef.current = null;
      }

      setBatchSyncing(true);
      const syncTaskId = addBatchSyncTask({ count: unsyncedAssets.length });

      setDatasets((prev) =>
        prev.map((d) =>
          unsyncedAssets.some((u) => u.id === d.id)
            ? {
                ...d,
                sync_status: 'syncing',
                sync_error: null,
              }
            : d
        )
      );
      setSelectedItemsMap((prev) => {
        if (prev.size === 0) return prev;
        const next = new Map(prev);
        for (const u of unsyncedAssets) {
          const cur = next.get(u.id);
          if (cur) next.set(u.id, { ...cur, sync_status: 'syncing', sync_error: null });
        }
        return next;
      });

      const res = await createSyncBatchJob({
        asset_ids: unsyncedAssets.map((d) => d.id),
      });

      if (!res.ok || !res.data?.jobId) {
        updateTask(syncTaskId, {
          status: 'failed',
          progress: 100,
          currentStep: res.error || '发起批量同步失败',
          errorMessage: res.error || '发起批量同步失败',
        });
        setBatchSyncing(false);
        return;
      }

      const jobId = res.data.jobId;
      updateTask(syncTaskId, {
        syncJobId: jobId,
        status: 'running',
        progress: 3,
        currentStep: `已发起批量同步（${unsyncedAssets.length} 个）`,
      });

      const pollOnce = async () => {
        const stRes = await getSyncBatchJobStatus(jobId);
        if (!stRes.ok || !stRes.data) {
          updateTask(syncTaskId, {
            status: 'failed',
            progress: 100,
            currentStep: stRes.error || '批量同步状态查询失败',
            errorMessage: stRes.error || '批量同步状态查询失败',
          });
          setBatchSyncing(false);
          return;
        }

        const data = stRes.data;
        const status = String(data.status || '').toLowerCase();
        const total = data.total || unsyncedAssets.length;
        const succeeded = data.succeeded || 0;
        const failed = data.failed || 0;
        updateTask(syncTaskId, {
          progress: typeof data.progress === 'number' ? data.progress : 0,
          currentStep:
            data.currentStep ||
            `进度 ${Math.round(typeof data.progress === 'number' ? data.progress : 0)}% · 成功 ${succeeded} · 失败 ${failed} / ${total}`,
        });

        if (status === 'succeeded') {
          setBatchSyncing(false);
          updateTask(syncTaskId, {
            status: 'success',
            progress: 100,
            currentStep: `批量同步完成：成功 ${succeeded}，失败 ${failed}`,
          });
          setDatasets((prev) =>
            prev.map((d) =>
              unsyncedAssets.some((u) => u.id === d.id)
                ? {
                    ...d,
                    sync_status: 'synced',
                    sync_error: null,
                  }
                : d
            )
          );
          setSelectedItemsMap((prev) => {
            if (prev.size === 0) return prev;
            const next = new Map(prev);
            for (const u of unsyncedAssets) {
              const cur = next.get(u.id);
              if (cur) next.set(u.id, { ...cur, sync_status: 'synced', sync_error: null });
            }
            return next;
          });
          return;
        }
        if (status === 'failed') {
          setBatchSyncing(false);
          updateTask(syncTaskId, {
            status: 'failed',
            progress: 100,
            currentStep: `批量同步完成但部分失败：${data.errorMessage || `成功 ${succeeded}，失败 ${failed}`}`,
            errorMessage: data.errorMessage || `成功 ${succeeded}，失败 ${failed}`,
          });
          const items = Array.isArray(data.items) ? data.items : [];
          const itemMap = new Map<number, { status: string; errorMessage: string }>();
          for (const it of items) {
            const id = Number((it as any).assetId);
            if (!Number.isFinite(id)) continue;
            itemMap.set(id, { status: String((it as any).status || ''), errorMessage: String((it as any).errorMessage || '') });
          }
          setDatasets((prev) =>
            prev.map((d) => {
              const st = itemMap.get(d.id);
              if (!st) return d;
              const s = st.status.trim().toLowerCase();
              if (s === 'succeeded' || s === 'success') return { ...d, sync_status: 'synced', sync_error: null };
              if (s === 'failed') return { ...d, sync_status: 'failed', sync_error: st.errorMessage || '同步失败' };
              return d;
            })
          );
          setSelectedItemsMap((prev) => {
            if (prev.size === 0) return prev;
            const next = new Map(prev);
            for (const [id, st] of itemMap.entries()) {
              const cur = next.get(id);
              if (!cur) continue;
              const s = st.status.trim().toLowerCase();
              if (s === 'succeeded' || s === 'success') next.set(id, { ...cur, sync_status: 'synced', sync_error: null });
              else if (s === 'failed') next.set(id, { ...cur, sync_status: 'failed', sync_error: st.errorMessage || '同步失败' });
            }
            return next;
          });
          return;
        }
        if (status === 'canceled' || status === 'cancelled') {
          setBatchSyncing(false);
          updateTask(syncTaskId, {
            status: 'cancelled',
            progress: 100,
            currentStep: `批量同步已取消：${data.errorMessage || '已取消'}`,
          });
          setDatasets((prev) =>
            prev.map((d) =>
              unsyncedAssets.some((u) => u.id === d.id)
                ? {
                    ...d,
                    sync_status: 'unsynced',
                  }
                : d
            )
          );
          return;
        }

        batchSyncPollTimerRef.current = setTimeout(pollOnce, 2000);
      };

      await pollOnce();
    } catch (error) {
      setToastMessage(error instanceof Error ? error.message : '发起批量同步失败');
      setBatchSyncing(false);
    }
  }, [addBatchSyncTask, batchSyncing, hasLerobotSelection, loadDatasets, selectedDatasets, selectedIds.size, updateTask]);

  const handleBatchDelete = useCallback(async () => {
    openDeleteDialogForBatch();
  }, [openDeleteDialogForBatch]);

  return (
    <ModulePageContainer>
      <ModulePageHeader title={t('dataPage.title')} />

      <ModulePageFilterCard>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            flexWrap: 'wrap',
          }}
        >
          <FiltersBar
            filters={filters}
            onFilterChange={handleFilterChange}
            onReset={handleResetFilters}
            projectList={projectList}
            taskOptions={taskOptions}
          />
          <button
            onClick={() => setImportDialogOpen(true)}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2563eb',
              border: 'none',
              borderRadius: 6,
              color: '#ffffff',
              fontSize: 14,
              cursor: 'pointer',
              fontWeight: 500,
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            }}
          >
            {t('dataPage.importData')}
          </button>
        </div>
      </ModulePageFilterCard>

      <ModulePageTableCard>
        <DatasetTable
          datasets={datasets}
          loading={loading}
          total={total}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onDelete={canDeleteDataAssets ? openDeleteDialogForSingle : undefined}
          onExport={handleSingleExport}
          onSync={handleSyncDataset}
          onDetail={(a) => setDetailAsset(a)}
          exportingAssetId={exportingAssetId}
          selectedIds={selectedIds}
          onSelectionChange={handleSelectionChange}
          projectList={projectList}
        />
        <ListFooterBar
          variant="inline"
          total={total}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={handlePageSizeChange}
          selectedCount={selectedIds.size}
          batchActions={[
            {
              key: 'sync',
              label: t('dataPage.batchSync'),
              onClick: handleBatchSync,
              disabled: selectedIds.size === 0 || batchSyncing || hasLerobotSelection || !hasUnsyncedSelection,
            },
            { key: 'export', label: t('dataPage.batchExport'), onClick: handleBatchExport, disabled: selectedIds.size === 0 || exportConfigModalOpen || hasUnsyncedSelection },
            {
              key: 'delete',
              label: t('dataPage.batchDelete'),
              onClick: handleBatchDelete,
              disabled: selectedIds.size === 0 || !canDeleteDataAssets,
              danger: true,
            },
          ]}
        />
      </ModulePageTableCard>

      {importDialogOpen && (
        <ImportDataDialog
          open={importDialogOpen}
          onClose={() => setImportDialogOpen(false)}
          onSuccess={handleImportSuccess}
        />
      )}

      <ConfirmDialog
        open={deleteDialogOpen}
        title={t('common.delete')}
        description={deleteDialogDescription}
        extraContent={
          deleteDialogVariant === 'synced_collect' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: '#4b5563' }}>
              <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={deleteCloud}
                  onChange={(e) => setDeleteCloud(e.target.checked)}
                />
                删除云端副本（对象存储 MinIO 与平台数据资产记录）
              </label>
              <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={deleteRemote}
                  onChange={(e) => setDeleteRemote(e.target.checked)}
                />
                删除采集端原始文件（需 Agent 隧道在线）
              </label>
            </div>
          ) : undefined
        }
        confirmText={t('common.delete')}
        cancelText="取消"
        loading={deleteLoading}
        onCancel={() => {
          if (deleteLoading) return;
          setDeleteDialogOpen(false);
          setDeleteTargetIds([]);
          setDeleteCloud(true);
          setDeleteRemote(false);
        }}
        onConfirm={handleConfirmDelete}
      />

      <AssetDetailModal
        open={detailAsset != null}
        asset={detailAsset}
        projectList={projectList}
        onClose={() => setDetailAsset(null)}
      />

      <ExportConfigModal
        open={exportConfigModalOpen}
        assetIds={exportModalAssetIds}
        exportCount={exportModalCount}
        formatLabel={exportModalFormatLabel}
        formatSummary={exportModalFormatSummary}
        projectName={exportModalProjectName}
        assetNamesPreview={exportModalAssetNamesPreview}
        onClose={() => {
          setExportConfigModalOpen(false);
          setExportingAssetId(null);
        }}
      />

      {toastMessage && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: '22px',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#ffffff',
            padding: '10px 14px',
            borderRadius: '10px',
            fontSize: '13px',
            boxShadow: '0 18px 60px rgba(15,23,42,0.25)',
            zIndex: 1500,
          }}
        >
          {toastMessage}
        </div>
      )}
    </ModulePageContainer>
  );
}
