'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import ListFooterBar from '@/components/common/ListFooterBar';
import TaskFilterBar, { type TaskFilterOption } from '@/components/tasks/TaskFilterBar';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import { TaskTemplateAssetTable } from '@/components/workspace/TaskTemplateAssetTable';
import { TaskTemplateOverviewCards } from '@/components/workspace/TaskTemplateOverviewCards';
import { TaskConfigDependenciesDrawer } from '@/components/workspace/TaskConfigDependenciesDrawer';
import { type TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import { usePagePerfLog } from '@/lib/perf/pagePerfLog';
import { useInvalidateWorkspaceLists, useTaskTemplatesQuery } from '@/lib/query/workspaceQueries';
import { ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID, isIsaacLabFrankaStackCubeTask } from '@/lib/workspace/isaaclabFrankaStackCube';
import {
  buildTaskTemplatesPathWithParams,
  clearIsaacReplayQueryParams,
  readIsaacGenerateJobId,
  readIsaacReplayJobId,
  readTaskTemplateIdFromQuery,
} from '@/lib/workspace/isaacReplayNavigation';
import { buildIsaacLabFrankaStackCubeConsoleHref } from '@/lib/workspace/isaaclabFrankaStackCube';
import { isValidIsaacGenerateJobId } from '@/lib/workspace/backendJobIds';
import { isFrankStackCubeProductTask } from '@/lib/workspace/isaacStackCubeProduct';
import {
  buildTaskTemplateAssetRow,
  computeTaskTemplateOverviewStats,
  filterTaskTemplateRows,
  partitionTaskTemplateRows,
  type CapabilityFilter,
  type SimulatorBackendFilter,
  type TaskTemplateStatusKey,
} from '@/lib/workspace/taskTemplatePresentation';

function resolveDrawerTaskConfigId(
  templateParam: string,
  taskTemplates: TaskTemplateDto[]
): string | null {
  const matched =
    taskTemplates.find(
      (item) => item.id === templateParam || item.registryTaskConfigId === templateParam
    ) ?? null;
  if (matched) {
    return matched.registryTaskConfigId ?? matched.id;
  }
  if (isFrankStackCubeProductTask(templateParam)) {
    const frankTemplate = taskTemplates.find(
      (item) => item.id === ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID
    );
    return frankTemplate?.registryTaskConfigId ?? frankTemplate?.id ?? null;
  }
  return null;
}

export default function TaskTemplatesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [search, setSearch] = useState('');
  const [backendFilter, setBackendFilter] = useState<'' | SimulatorBackendFilter>('');
  const [statusFilter, setStatusFilter] = useState<'' | TaskTemplateStatusKey>('');
  const [capabilityFilter, setCapabilityFilter] = useState<'' | CapabilityFilter>('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [dependencyTaskId, setDependencyTaskId] = useState<string | null>(null);

  const isaacReplayJobId = readIsaacReplayJobId(searchParams);
  const isaacGenerateJobId = readIsaacGenerateJobId(searchParams);

  const clearIsaacReplayQuery = useCallback(() => {
    const next = clearIsaacReplayQueryParams(new URLSearchParams(searchParams.toString()));
    router.replace(buildTaskTemplatesPathWithParams(next));
  }, [router, searchParams]);

  const handleCloseDrawer = useCallback(() => {
    setDependencyTaskId(null);
    const templateParam = readTaskTemplateIdFromQuery(searchParams);
    if (isaacReplayJobId || isaacGenerateJobId || templateParam) {
      clearIsaacReplayQuery();
    }
  }, [clearIsaacReplayQuery, isaacGenerateJobId, isaacReplayJobId, searchParams]);

  const { invalidateTaskTemplates } = useInvalidateWorkspaceLists();
  const {
    data: templatesResponse,
    isLoading: loading,
    isError,
    error,
  } = useTaskTemplatesQuery({ limit: 500, offset: 0 });

  const taskTemplates = templatesResponse?.taskTemplates ?? [];
  const loadError = isError
    ? error instanceof Error
      ? error.message
      : '加载任务模板失败'
    : null;

  usePagePerfLog('TaskTemplates', {
    loading,
    apiRequestCount: loading ? 1 : 0,
  });

  const refresh = useCallback(async () => {
    await invalidateTaskTemplates();
  }, [invalidateTaskTemplates]);

  useEffect(() => {
    if (isaacGenerateJobId && isValidIsaacGenerateJobId(isaacGenerateJobId)) {
      router.replace(buildIsaacLabFrankaStackCubeConsoleHref({ jobId: isaacGenerateJobId }));
    }
  }, [isaacGenerateJobId, router]);

  useEffect(() => {
    if (loading || taskTemplates.length === 0) return;
    if (isaacGenerateJobId && isValidIsaacGenerateJobId(isaacGenerateJobId)) return;
    const templateParam = readTaskTemplateIdFromQuery(searchParams);
    let drawerId: string | null = null;
    if (templateParam) {
      drawerId = resolveDrawerTaskConfigId(templateParam, taskTemplates);
    } else if (isaacReplayJobId) {
      drawerId = resolveDrawerTaskConfigId(ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID, taskTemplates);
    }
    if (drawerId) {
      setDependencyTaskId(drawerId);
    }
  }, [isaacGenerateJobId, isaacReplayJobId, loading, searchParams, taskTemplates]);

  const assetRows = useMemo(
    () => taskTemplates.map(buildTaskTemplateAssetRow),
    [taskTemplates]
  );

  const primaryRows = useMemo(
    () => partitionTaskTemplateRows(assetRows).primary,
    [assetRows]
  );

  const overviewStats = useMemo(() => computeTaskTemplateOverviewStats(assetRows), [assetRows]);

  const filteredPrimary = useMemo(
    () =>
      filterTaskTemplateRows(primaryRows, {
        search,
        backendFilter,
        statusFilter,
        capabilityFilter,
      }),
    [primaryRows, backendFilter, capabilityFilter, search, statusFilter]
  );

  const paged = useMemo(
    () => filteredPrimary.slice((page - 1) * pageSize, page * pageSize),
    [filteredPrimary, page, pageSize]
  );

  const filterOptions: TaskFilterOption[] = [
    {
      key: 'backend',
      value: backendFilter,
      placeholder: '全部后端',
      options: [
        { value: 'mujoco', label: 'MuJoCo' },
        { value: 'isaac_lab', label: 'Isaac Lab' },
      ],
      onChange: (value) => setBackendFilter(value as '' | SimulatorBackendFilter),
    },
    {
      key: 'status',
      value: statusFilter,
      placeholder: '全部状态',
      options: [
        { value: 'available', label: '已接入' },
        { value: 'pending', label: '待配置' },
        { value: 'maintenance', label: '维护中' },
      ],
      onChange: (value) => setStatusFilter(value as '' | TaskTemplateStatusKey),
    },
    {
      key: 'capability',
      value: capabilityFilter,
      placeholder: '全部能力',
      options: [
        { value: 'data_generation', label: '可生成数据' },
        { value: 'training', label: '可训练' },
        { value: 'evaluation', label: '可评测' },
      ],
      onChange: (value) => setCapabilityFilter(value as '' | CapabilityFilter),
    },
  ];

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="任务模板"
        subtitle="管理平台已接入的标准任务模板，支持数据生成、策略训练、模型评测与回放分析。"
        actions={
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <SecondaryButton onClick={() => router.push('/workspace/task-build/template')}>
              创建任务配置
            </SecondaryButton>
            <PrimaryButton onClick={() => router.push('/workspace/data?openGenerate=1')}>
              生成数据
            </PrimaryButton>
          </div>
        }
      />

      {!loading && assetRows.length > 0 ? (
        <TaskTemplateOverviewCards stats={overviewStats} />
      ) : null}

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <TaskFilterBar
            searchValue={search}
            onSearchChange={setSearch}
            searchPlaceholder="搜索任务名称、后端或策略"
            filters={filterOptions}
            onReset={() => {
              setSearch('');
              setBackendFilter('');
              setStatusFilter('');
              setCapabilityFilter('');
            }}
          />
          {loadError ? <span style={{ fontSize: 13, color: '#b45309' }}>{loadError}</span> : null}
        </div>
      </ModulePageFilterCard>

      <ModulePageTableCard>
        {loading ? (
          <p style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>正在加载任务模板…</p>
        ) : paged.length === 0 ? (
          <p style={{ padding: 24, textAlign: 'center', color: '#6b7280', lineHeight: 1.6 }}>
            暂无匹配的任务模板。
          </p>
        ) : (
          <TaskTemplateAssetTable rows={paged} onShowDetail={setDependencyTaskId} />
        )}
        {!loading && filteredPrimary.length > 0 ? (
          <ListFooterBar
            variant="inline"
            total={filteredPrimary.length}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size);
              setPage(1);
            }}
          />
        ) : null}
      </ModulePageTableCard>

      <TaskConfigDependenciesDrawer
        taskConfigId={dependencyTaskId}
        taskTemplates={taskTemplates}
        onClose={handleCloseDrawer}
        isaacReplayJobId={isaacReplayJobId}
        isaacGenerateJobId={isaacGenerateJobId}
        onClearIsaacReplayQuery={clearIsaacReplayQuery}
      />
    </ModulePageContainer>
  );
}
