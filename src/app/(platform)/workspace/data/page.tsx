'use client';

import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
} from '@/lib/workspace/taskDisplayNames';
import { buildIsaacBlockStackingReplayHref } from '@/lib/workspace/isaacReplayNavigation';
import {
  buildIsaacGenerateJobHref,
  deleteIsaacLabDataset,
  getIsaacLabRuntimeStatus,
  startIsaacLabGenerateDataset,
  startIsaacLabReplayFromDataset,
} from '@/lib/api/isaacLabClient';
import dynamic from 'next/dynamic';
import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useQueryClient } from '@tanstack/react-query';
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
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { DataCenterEntryCards } from '@/components/workspace/data/DataCenterEntryCards';
import { WorkspaceDatasetTable } from '@/components/workspace/WorkspaceDatasetTable';
import { WorkspaceDatasetDetailDrawer } from '@/components/workspace/WorkspaceDatasetDetailDrawer';
import {
  type GenerateDataPayload,
} from '@/lib/workspace/generateDataPayloadTypes';
import { ImportIsaacDemoModal } from '@/components/workspace/data/ImportIsaacDemoModal';
import { ImportDatasetModal } from '@/components/workspace/data/ImportDatasetModal';
import { BuildDatasetModal } from '@/components/workspace/data/BuildDatasetModal';
import {
  DATASET_SOURCE_FILTER_OPTIONS,
  resolveDatasetSourceLabel,
} from '@/lib/workspace/datasetDisplay';
import { normalizeDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import { resolveDatasetFormatLabel } from '@/lib/workspace/taskTemplateMapping';
import { buildSimulationConsoleHref } from '@/lib/workspace/simulationConsole';
import {
  appendMockDataItem,
  bindCableThreadingBackendJobToDataItem,
  createCableThreadingGenerateRun,
  createDataFromGeneration,
  createDualArmCableGenerateRun,
  setActiveDataGeneration,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  deleteWorkspaceJob,
  workspaceJobBatchDeleteConfirm,
} from '@/lib/api/workspaceJobClient';
import { generateNutAssemblyDataAsync } from '@/lib/api/nutAssemblyClient';
import {
  buildNutAssemblyGenerateRequest,
  nutAssemblyUsesMimicgenProgress,
} from '@/lib/workspace/nutAssemblyGeneratePayload';
import { NUT_ASSEMBLY_PATH_DEFAULTS } from '@/lib/workspace/generateDataTaskParams';
import {
  buildNutAssemblyConsoleHref,
  createPendingNutAssemblyDataItem,
  isNutAssemblyTask,
  makeNutAssemblyLocalRunId,
} from '@/lib/workspace/nutAssembly';
import { generateCableThreadingDataAsync } from '@/lib/api/cableThreadingClient';
import { buildCableThreadingGenerateRequest } from '@/lib/workspace/cableThreadingGeneratePayload';
import { generateDualArmCableDataAsync } from '@/lib/api/dualArmCableClient';
import {
  buildCableThreadingConsoleHref,
  createPendingCableThreadingDataItem,
  isCableThreadingTask,
  makeCableThreadingLocalRunId,
} from '@/lib/workspace/cableThreading';
import {
  buildDualArmCableConsoleHref,
  createPendingDualArmCableDataItem,
  isDualArmCableTask,
} from '@/lib/workspace/dualArmCable';
import { ISAAC_BLOCK_STACKING_TEMPLATE_ID } from '@/lib/workspace/isaacBlockStacking';
import {
  ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import { deleteIsaacLabFrankaStackCubeDataset } from '@/lib/api/isaaclabFrankaStackCubeClient';
import {
  buildIsaacSimFrankaPickPlaceReplayHref,
  ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS,
  isIsaacSimFrankaPickPlaceTask,
} from '@/lib/workspace/isaacsimFrankaPickPlace';
import { generateIsaacSimFrankaPickPlaceDataAsync } from '@/lib/api/isaacsimFrankaPickPlaceClient';
import { isDatasetGenerationEnabled } from '@/lib/workspace/taskTemplateCapabilities';
import { sortDatasetsByCreatedAtDesc } from '@/lib/workspace/datasetSort';
import { listWorkspaceDatasets, deleteImportedWorkspaceDataset, deleteBuiltWorkspaceDataset } from '@/lib/api/datasetsClient';
import {
  isFrankStackCubeDataset,
  isFrankStackCubeProductTask,
} from '@/lib/workspace/isaacStackCubeProduct';
import { isIsaacLabFrankaStackCubeDataset } from '@/lib/workspace/isaaclabFrankaStackCube';
import { isLegacyIsaacLabRegistryDataset } from '@/lib/workspace/datasetTableActions';
import { isImportedWorkspaceDataset } from '@/lib/workspace/datasetDisplay';
import { isBuiltWorkspaceDataset } from '@/lib/workspace/datasetImportWorkflow';
import type { Dataset } from '@/types/benchmark';
import { usePagePerfLog } from '@/lib/perf/pagePerfLog';
import {
  type DatasetListResponse,
  useInvalidateWorkspaceLists,
  useWorkspaceDatasetsQuery,
  workspaceQueryKeys,
} from '@/lib/query/workspaceQueries';

const GenerateDataModal = dynamic(
  () => import('@/components/workspace/data/GenerateDataModal').then((m) => m.GenerateDataModal),
  { ssr: false }
);

const NutAssemblyGenerationProgressModal = dynamic(
  () =>
    import('@/components/workspace/data/NutAssemblyGenerationProgressModal').then(
      (m) => m.NutAssemblyGenerationProgressModal
    ),
  { ssr: false }
);

export default function WorkspaceDataPage() {
  return (
    <Suspense fallback={null}>
      <WorkspaceDataPageInner />
    </Suspense>
  );
}

function WorkspaceDataPageInner() {
  const { t } = useI18n();
  const router = useRouter();
  const searchParams = useSearchParams();
  const shouldOpenGenerate = searchParams.get('openGenerate') === '1';
  const urlTemplate = searchParams.get('template') ?? undefined;
  const urlTaskConfigId = searchParams.get('taskConfigId') ?? undefined;

  const [taskFilter, setTaskFilter] = useState('');
  const [search, setSearch] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [formatFilter, setFormatFilter] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [detailDataset, setDetailDataset] = useState<Dataset | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [generateModalOpen, setGenerateModalOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [generateSubmitting, setGenerateSubmitting] = useState(false);
  const [generateStatusMsg, setGenerateStatusMsg] = useState<string | null>(null);
  const [importIsaacModalOpen, setImportIsaacModalOpen] = useState(false);
  const [importDatasetModalOpen, setImportDatasetModalOpen] = useState(false);
  const [buildModalOpen, setBuildModalOpen] = useState(false);
  const [preselectedBuildDatasetId, setPreselectedBuildDatasetId] = useState<string | null>(null);
  const [preferredIsaacSeedDatasetId, setPreferredIsaacSeedDatasetId] = useState<string | null>(null);
  const [nutAssemblyProgress, setNutAssemblyProgress] = useState<{
    jobId: string;
    dataId: string;
    payload: GenerateDataPayload;
  } | null>(null);

  const { invalidateDatasets } = useInvalidateWorkspaceLists();
  const queryClient = useQueryClient();
  const {
    data: dsResponse,
    isLoading,
    isError,
    error,
  } = useWorkspaceDatasetsQuery();

  const datasets = dsResponse?.datasets ?? [];
  const apiUnavailable = isError;
  const apiErrorMessage = isError
    ? error instanceof Error
      ? `后端 Dataset 索引请求失败：${error.message}`
      : '后端 Dataset 索引暂不可用，请检查服务连接后刷新。'
    : datasets.length === 0
      ? '当前索引为空，暂无已注册数据集。'
      : null;

  usePagePerfLog('DataCenter', {
    loading: isLoading,
    apiRequestCount: isLoading ? 1 : 0,
  });

  const refreshItems = useCallback(async () => {
    await invalidateDatasets();
  }, [invalidateDatasets]);

  const patchDatasets = useCallback(
    (updater: Dataset[] | ((prev: Dataset[]) => Dataset[])) => {
      queryClient.setQueriesData<DatasetListResponse>(
        { queryKey: workspaceQueryKeys.datasetsIndex },
        (current) => {
          if (!current) return current;
          const nextDatasets =
            typeof updater === 'function' ? updater(current.datasets) : updater;
          return {
            ...current,
            datasets: nextDatasets,
            total: Math.max(0, current.total - (current.datasets.length - nextDatasets.length)),
          };
        }
      );
    },
    [queryClient]
  );

  useEffect(() => {
    if (shouldOpenGenerate) {
      setGenerateModalOpen(true);
    }
  }, [shouldOpenGenerate]);

  const showToast = useCallback((text: string) => {
    setToastMsg(text);
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  const filteredDatasets = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = datasets.filter((dataset) => {
      if (taskFilter) {
        const isCable = dataset.sourceJobId.startsWith('ct_gen_');
        const isDual = dataset.sourceJobId.startsWith('dac_gen_');
        if (taskFilter === '单臂线缆穿杆' && !isCable) return false;
        if (taskFilter === '双臂线缆操控' && !isDual) return false;
      }
      if (sourceFilter && resolveDatasetSourceLabel(dataset) !== sourceFilter) return false;
      if (formatFilter && resolveDatasetFormatLabel(dataset) !== formatFilter) return false;
      if (!q) return true;
      const haystack = [
        dataset.id,
        dataset.name,
        dataset.displayName,
        dataset.taskDisplayName,
        normalizeDatasetDisplayName({
          displayName: dataset.displayName,
          name: dataset.name,
          taskType: dataset.taskType,
          createdAt: dataset.createdAt,
          sourceJobId: dataset.sourceJobId,
        }),
        dataset.sourceJobId,
        dataset.manifestPath,
        dataset.storagePath,
        dataset.sourceType,
        dataset.format,
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(q);
    });
    return sortDatasetsByCreatedAtDesc(filtered);
  }, [datasets, search, taskFilter, sourceFilter, formatFilter]);

  const totalFiltered = filteredDatasets.length;
  const pagedDatasets = useMemo(
    () => filteredDatasets.slice((page - 1) * pageSize, page * pageSize),
    [filteredDatasets, page, pageSize]
  );

  useEffect(() => {
    setSelectedIds(new Set());
  }, [page]);

  useEffect(() => {
    setPage(1);
  }, [search, taskFilter, sourceFilter, formatFilter]);

  const toggleRow = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const allPageSelected =
    pagedDatasets.length > 0 && pagedDatasets.every((dataset) => selectedIds.has(dataset.id));

  const toggleSelectAll = useCallback(() => {
    const pageIds = pagedDatasets.map((d) => d.id);
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
  }, [pagedDatasets, allPageSelected]);

  const handleResetFilters = () => {
    setSearch('');
    setTaskFilter('');
    setSourceFilter('');
    setFormatFilter('');
  };

  const handlePageSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setPage(1);
  }, []);

  const handleGenerateData = useCallback(
    async (payload: GenerateDataPayload) => {
      if (isIsaacSimFrankaPickPlaceTask(payload.template)) {
        if (payload.launch !== 'start') {
          showToast('Franka 物体搬运数据生成仅支持立即启动');
          return;
        }
        setGenerateSubmitting(true);
        setGenerateStatusMsg('正在启动 Franka 物体搬运数据生成…');
        try {
          const response = await generateIsaacSimFrankaPickPlaceDataAsync({
            taskId: 'isaacsim_franka_pick_place',
            episodes: payload.episodes ?? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.episodes,
            seed: payload.seed ?? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.seed,
            saveVideo: payload.saveVideo ?? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.saveVideo,
            saveTrajectory: payload.saveTrajectory ?? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.saveTrajectory,
            headless: payload.isaacsimHeadless ?? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.headless,
            taskConfigId: urlTaskConfigId,
          });
          setGenerateModalOpen(false);
          showToast(response.message || '数据生成任务已启动');
          void refreshItems();
          router.push(
            buildIsaacSimFrankaPickPlaceReplayHref({
              jobId: response.jobId,
            })
          );
        } catch (err) {
          showToast(err instanceof Error ? err.message : 'Franka 物体搬运数据生成启动失败');
        } finally {
          setGenerateSubmitting(false);
          setGenerateStatusMsg(null);
        }
        return;
      }

      if (isFrankStackCubeProductTask(payload.template)) {
        if (payload.launch !== 'start') {
          showToast('Isaac Lab 数据生成仅支持立即启动');
          return;
        }
        setGenerateSubmitting(true);
        setGenerateStatusMsg('正在启动 Isaac Lab 数据生成…');
        try {
          const started = await startIsaacLabGenerateDataset({
            datasetName: payload.outputName,
            numDemos: payload.isaacNumDemos ?? 10,
            seed: payload.seed,
            headless: payload.isaacHeadless ?? true,
            enableCameras: payload.isaacEnableCameras ?? true,
            generationMode: payload.isaacGenerationMode ?? 'mimic_auto',
            seedDatasetId: payload.isaacSeedDatasetId,
            seedDatasetFile: payload.isaacSeedDatasetFile,
            numEnvs: payload.isaacParallelNumEnvs,
          });
          setGenerateModalOpen(false);
          showToast(`Isaac 数据生成任务已创建：${started.jobId}`);
          router.push(buildIsaacGenerateJobHref(started.jobId));
        } catch (err) {
          showToast(err instanceof Error ? err.message : 'Isaac Lab 数据生成启动失败');
        } finally {
          setGenerateSubmitting(false);
          setGenerateStatusMsg(null);
        }
        return;
      }

      if (!isDatasetGenerationEnabled(payload.template)) {
        showToast('该任务暂不支持数据生成');
        return;
      }

      if (isDualArmCableTask(payload.template)) {
        if (payload.launch === 'save') {
          const pendingItem = createPendingDualArmCableDataItem(payload, `dac-pending-save-${Date.now()}`);
          pendingItem.status = 'pending';
          pendingItem.id = `dac-save-${Date.now()}`;
          appendMockDataItem(pendingItem);
          setGenerateModalOpen(false);
          showToast('数据生成任务已保存');
          return;
        }

        setGenerateSubmitting(true);
        setGenerateStatusMsg('正在启动线缆整理 episode…');
        try {
          const response = await generateDualArmCableDataAsync({
            taskType: 'dual_arm_cable_manipulation',
            taskName: payload.template,
            maxCables: payload.dualArmMaxCables ?? 1,
            seed: payload.seed,
            record: payload.dualArmRecord ?? true,
            headless: payload.dualArmHeadless ?? true,
            stretchMode:
              (payload.dualArmStretchMode as 'fixed_distance' | 'fixed_force' | 'ema_jump') ??
              'fixed_distance',
            releaseMode:
              (payload.dualArmReleaseMode as 'three_phase' | 'direct_open' | 'slow_open') ??
              'three_phase',
            taskConfigId: urlTaskConfigId,
          });
          const pendingItem = createPendingDualArmCableDataItem(payload, response.jobId);
          createDualArmCableGenerateRun(pendingItem, payload, response.jobId);
          appendMockDataItem(pendingItem);
          setGenerateModalOpen(false);
          setActiveDataGeneration(pendingItem, {
            episodes: 1,
            seed: payload.seed,
            template: payload.template,
          });
          showToast('已创建运行任务，正在跳转控制台…');
          router.push(
            buildDualArmCableConsoleHref({
              jobId: response.jobId,
              dataId: pendingItem.id,
            })
          );
        } catch (err) {
          showToast(err instanceof Error ? err.message : '线缆整理任务启动失败');
        } finally {
          setGenerateSubmitting(false);
          setGenerateStatusMsg(null);
        }
        return;
      }

      if (isNutAssemblyTask(payload.template)) {
        if (payload.launch !== 'start') {
          showToast('生成任务数据仅支持立即启动');
          return;
        }

        setGenerateSubmitting(true);
        setGenerateStatusMsg('正在启动生成任务数据…');
        try {
          const request = buildNutAssemblyGenerateRequest(payload, urlTaskConfigId);
          const response = await generateNutAssemblyDataAsync(request);
          const dataItemKey = makeNutAssemblyLocalRunId();
          const pendingItem = createPendingNutAssemblyDataItem(payload, dataItemKey, response.jobId);
          appendMockDataItem(pendingItem);
          setActiveDataGeneration(pendingItem, {
            episodes: payload.episodes,
            seed: payload.seed ?? 0,
            template: payload.template,
          });
          const generationPath =
            request.generationPath ??
            payload.generationPath ??
            NUT_ASSEMBLY_PATH_DEFAULTS.generationPath;
          setGenerateModalOpen(false);
          if (nutAssemblyUsesMimicgenProgress(generationPath)) {
            setNutAssemblyProgress({
              jobId: response.jobId,
              dataId: pendingItem.id,
              payload,
            });
            showToast('MimicGen 数据生成已启动，请查看进度弹窗');
          } else {
            showToast('已创建运行任务，正在跳转控制台…');
            router.push(
              buildNutAssemblyConsoleHref({
                jobId: response.jobId,
                dataId: pendingItem.id,
              })
            );
          }
        } catch (err) {
          showToast(err instanceof Error ? err.message : '生成任务数据启动失败');
        } finally {
          setGenerateSubmitting(false);
          setGenerateStatusMsg(null);
        }
        return;
      }

      if (isCableThreadingTask(payload.template)) {
        if (payload.launch === 'save') {
          const localRunId = makeCableThreadingLocalRunId();
          const pendingItem = createPendingCableThreadingDataItem(payload, localRunId);
          pendingItem.status = 'pending';
          pendingItem.size = '—';
          pendingItem.frameOrTrajectoryCount = '待生成';
          appendMockDataItem(pendingItem);
          setGenerateModalOpen(false);
          showToast('数据生成任务已保存');
          return;
        }

        setGenerateSubmitting(true);
        setGenerateStatusMsg('正在启动线缆穿杆数据生成…');
        try {
          const response = await generateCableThreadingDataAsync({
            episodes: payload.episodes,
            robot: payload.cableThreadingRobot,
            cableModel: payload.cableThreadingCableModel,
            difficulty: payload.cableThreadingDifficulty,
            horizon: payload.cableThreadingHorizon,
            seed: payload.seed,
            outputFormat: payload.cableThreadingSaveHdf5 ? 'hdf5' : 'npz',
            saveHdf5: payload.cableThreadingSaveHdf5,
            saveProcessVideo: payload.cableThreadingSaveProcessVideo ?? true,
            taskConfigId: urlTaskConfigId,
          });
          const dataItemKey = makeCableThreadingLocalRunId();
          const pendingItem = createPendingCableThreadingDataItem(
            payload,
            dataItemKey,
            response.jobId
          );
          createCableThreadingGenerateRun(pendingItem, payload, response.jobId);
          appendMockDataItem(pendingItem);
          bindCableThreadingBackendJobToDataItem(pendingItem.id, response.jobId);
          setGenerateModalOpen(false);
          setActiveDataGeneration(pendingItem, {
            episodes: payload.episodes,
            seed: payload.seed,
            template: payload.template,
          });
          showToast('已创建运行任务，正在跳转控制台…');
          router.push(
            buildCableThreadingConsoleHref({
              jobId: response.jobId,
              dataId: pendingItem.id,
            })
          );
        } catch (err) {
          showToast(err instanceof Error ? err.message : '线缆穿杆启动失败');
        } finally {
          setGenerateSubmitting(false);
          setGenerateStatusMsg(null);
        }
        return;
      }

      const item = createDataFromGeneration(payload);
      if (payload.launch === 'start') {
        item.status = 'generating';
      }
      appendMockDataItem(item);
      setGenerateModalOpen(false);
      if (payload.launch === 'start') {
        setActiveDataGeneration(item, {
          episodes: payload.episodes,
          seed: payload.seed,
          template: payload.template,
        });
        router.push(
          buildSimulationConsoleHref({
            mode: 'data-generation',
            task: payload.template,
            dataset: item.name,
            dataId: item.id,
            backend: payload.simBackend.toLowerCase(),
            physicsProxyMode: payload.physicsProxyMode,
            physicsProxyModel: payload.physicsProxyModel ?? undefined,
          })
        );
        return;
      }
      showToast('数据生成任务已保存');
    },
    [router, showToast, urlTaskConfigId]
  );

  const handleConfirmBatchDelete = useCallback(async () => {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    setDeleteLoading(true);
    try {
      const targets = datasets.filter((dataset) => ids.includes(dataset.id));
      let deletedCount = 0;
      let failedCount = 0;

      for (const dataset of targets) {
        try {
          if (isImportedWorkspaceDataset(dataset)) {
            await deleteImportedWorkspaceDataset(dataset.id);
          } else if (isBuiltWorkspaceDataset(dataset)) {
            await deleteBuiltWorkspaceDataset(dataset.id);
          } else if (isIsaacLabFrankaStackCubeDataset(dataset)) {
            await deleteIsaacLabFrankaStackCubeDataset(dataset.sourceJobId);
          } else if (isLegacyIsaacLabRegistryDataset(dataset)) {
            await deleteIsaacLabDataset(dataset.id);
          } else {
            await deleteWorkspaceJob(dataset.sourceJobId);
          }
          deletedCount += 1;
        } catch {
          failedCount += 1;
        }
      }

      const removedIds = new Set(targets.map((d) => d.id));
      patchDatasets((prev) => prev.filter((d) => !removedIds.has(d.id)));
      setSelectedIds(new Set());
      if (detailDataset && removedIds.has(detailDataset.id)) setDetailDataset(null);
      setDeleteConfirmOpen(false);

      if (failedCount === 0) {
        showToast(`已删除 ${deletedCount} 条数据集`);
      } else {
        showToast(`已删除 ${deletedCount} 条，失败 ${failedCount} 条`);
      }
      void refreshItems();
    } catch (err) {
      showToast(err instanceof Error ? err.message : '批量删除失败');
    } finally {
      setDeleteLoading(false);
    }
  }, [selectedIds, datasets, detailDataset, showToast, refreshItems, patchDatasets]);

  const filterOptions: TaskFilterOption[] = useMemo(
    () => [
      {
        key: 'task',
        value: taskFilter,
        placeholder: '来源任务',
        options: [
          { value: '', label: '全部' },
          { value: '单臂线缆穿杆', label: '单臂线缆穿杆' },
          { value: '双臂线缆操控', label: '双臂线缆操控' },
        ],
        onChange: setTaskFilter,
      },
      {
        key: 'source',
        value: sourceFilter,
        placeholder: '数据来源',
        options: [
          { value: '', label: '全部' },
          ...DATASET_SOURCE_FILTER_OPTIONS.map((label) => ({ value: label, label })),
        ],
        onChange: setSourceFilter,
      },
      {
        key: 'format',
        value: formatFilter,
        placeholder: '数据格式',
        options: [
          { value: '', label: '全部' },
          { value: 'HDF5', label: 'HDF5' },
          { value: 'NPZ', label: 'NPZ' },
          { value: 'Manifest', label: 'Manifest' },
        ],
        onChange: setFormatFilter,
      },
    ],
    [taskFilter, sourceFilter, formatFilter]
  );

  const handleOpenBuild = useCallback((dataset?: Dataset) => {
    setPreselectedBuildDatasetId(dataset?.id ?? null);
    setBuildModalOpen(true);
  }, []);

  const handleDatasetDelete = useCallback(
    (dataset: Dataset) => {
      void (async () => {
        setDeleteLoading(true);
        try {
          if (isIsaacLabFrankaStackCubeDataset(dataset)) {
            await deleteIsaacLabFrankaStackCubeDataset(dataset.sourceJobId);
          } else if (isLegacyIsaacLabRegistryDataset(dataset)) {
            await deleteIsaacLabDataset(dataset.id);
          } else {
            await deleteWorkspaceJob(dataset.sourceJobId);
          }
          patchDatasets((prev) => prev.filter((d) => d.id !== dataset.id));
          if (detailDataset?.id === dataset.id) setDetailDataset(null);
          showToast('数据集记录已删除');
          void refreshItems();
        } catch (err) {
          showToast(err instanceof Error ? err.message : '删除失败');
        } finally {
          setDeleteLoading(false);
        }
      })();
    },
    [detailDataset, showToast, refreshItems, patchDatasets]
  );

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('workspacePages.dataCenterTitle')}
        subtitle={t('workspacePages.dataCenterSubtitle')}
        actions={
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <SecondaryButton onClick={() => setImportDatasetModalOpen(true)}>导入</SecondaryButton>
            <SecondaryButton onClick={() => showToast('导出')}>导出</SecondaryButton>
          </div>
        }
      />

      {apiErrorMessage ? (
        <div
          style={{
            marginBottom: 12,
            padding: '10px 14px',
            borderRadius: 8,
            background: apiUnavailable ? '#fffbeb' : '#f0f9ff',
            border: apiUnavailable ? '1px solid #fcd34d' : '1px solid #bae6fd',
            color: apiUnavailable ? '#92400e' : '#0369a1',
            fontSize: 13,
          }}
        >
          {apiErrorMessage}
        </div>
      ) : null}

      <DataCenterEntryCards
        onStartGenerate={() => setGenerateModalOpen(true)}
        onStartBuild={() => handleOpenBuild()}
      />

      <ModulePageFilterCard>
        <TaskFilterBar
          searchValue={search}
          onSearchChange={setSearch}
          searchPlaceholder="搜索"
          filters={filterOptions}
          onReset={handleResetFilters}
        />
      </ModulePageFilterCard>

      <ModulePageTableCard>
        <div
          style={{
            padding: '14px 16px 0',
            fontSize: 15,
            fontWeight: 600,
            color: '#111827',
          }}
        >
          数据集列表
        </div>
        <WorkspaceDatasetTable
          datasets={pagedDatasets}
          selectedIds={selectedIds}
          onToggleRow={toggleRow}
          onToggleSelectAll={toggleSelectAll}
          allPageSelected={allPageSelected}
          onOpenDetail={setDetailDataset}
          onDelete={handleDatasetDelete}
          onBuild={handleOpenBuild}
        />
        <ListFooterBar
          variant="inline"
          total={totalFiltered}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={handlePageSizeChange}
          selectedCount={selectedIds.size}
          batchActions={[
            {
              key: 'batch-delete',
              label: '批量删除',
              onClick: () => {
                if (selectedIds.size === 0) return;
                setDeleteConfirmOpen(true);
              },
              danger: true,
            },
          ]}
        />
      </ModulePageTableCard>

      <WorkspaceDatasetDetailDrawer
        dataset={detailDataset}
        onClose={() => setDetailDataset(null)}
        onTrain={(dataset) => router.push(`/workspace/training?dataset=${encodeURIComponent(dataset.id)}`)}
        onBuilt={() => void refreshItems()}
        onBuild={handleOpenBuild}
      />

      <BuildDatasetModal
        open={buildModalOpen}
        onClose={() => {
          setBuildModalOpen(false);
          setPreselectedBuildDatasetId(null);
        }}
        datasets={datasets}
        preselectedSourceDatasetId={preselectedBuildDatasetId}
        onBuilt={(dataset) => {
          patchDatasets((prev) => {
            const without = prev.filter((row) => row.id !== dataset.id);
            return [dataset, ...without];
          });
          setPage(1);
          showToast('标准训练数据集构建完成，已登记到数据中心');
          void refreshItems();
        }}
      />

      <ImportDatasetModal
        open={importDatasetModalOpen}
        onClose={() => setImportDatasetModalOpen(false)}
        onImported={(dataset) => {
          patchDatasets((prev) => {
            const without = prev.filter((row) => row.id !== dataset.id);
            return [dataset, ...without];
          });
          setPage(1);
          showToast(
            '数据集已导入。系统已完成结构解析，若满足训练规范可直接用于训练，否则可通过数据构建完成适配。'
          );
          void refreshItems();
        }}
      />

      <ImportIsaacDemoModal
        open={importIsaacModalOpen}
        onClose={() => setImportIsaacModalOpen(false)}
        onImported={(dataset) => {
          patchDatasets((prev) => {
            const without = prev.filter((row) => row.id !== dataset.id);
            return [dataset, ...without];
          });
          setPreferredIsaacSeedDatasetId(dataset.id);
          setGenerateModalOpen(true);
          setPage(1);
          showToast(`已登记数据集：${dataset.name}`);
          void refreshItems();
        }}
      />

      <GenerateDataModal
        open={generateModalOpen}
        onClose={() => {
          if (!generateSubmitting) {
            setGenerateModalOpen(false);
            setPreferredIsaacSeedDatasetId(null);
          }
        }}
        onSubmit={handleGenerateData}
        initialTemplate={urlTemplate}
        submitting={generateSubmitting}
        submittingMessage={generateStatusMsg ?? undefined}
        isaacSeedDatasets={datasets}
        preferredSeedDatasetId={preferredIsaacSeedDatasetId}
        onImportIsaacDemo={() => {
          setImportIsaacModalOpen(true);
        }}
        onViewIsaacTaskTemplate={() => {
          setGenerateModalOpen(false);
          router.push(
            `/workspace/resources/task-templates?templateId=${encodeURIComponent(ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID)}`
          );
        }}
      />

      <ConfirmDialog
        open={deleteConfirmOpen}
        title="批量删除"
        description={workspaceJobBatchDeleteConfirm(selectedIds.size)}
        confirmText="删除"
        cancelText="取消"
        loading={deleteLoading}
        onCancel={() => {
          if (!deleteLoading) setDeleteConfirmOpen(false);
        }}
        onConfirm={handleConfirmBatchDelete}
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
            fontWeight: 500,
            zIndex: 1700,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#fff',
          }}
        >
          {toastMsg}
        </div>
      ) : null}

      <NutAssemblyGenerationProgressModal
        open={nutAssemblyProgress != null}
        jobId={nutAssemblyProgress?.jobId ?? ''}
        dataId={nutAssemblyProgress?.dataId}
        onClose={() => setNutAssemblyProgress(null)}
        onCompleted={() => void refreshItems()}
        onRetryConfig={() => setGenerateModalOpen(true)}
      />
    </ModulePageContainer>
  );
}
