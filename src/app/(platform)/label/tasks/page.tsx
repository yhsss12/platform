'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';
import type { LabelTask } from '@/features/data-platform/models/labelTask';
import { getLabelTasks, labelTaskRowToTask } from '@/features/label-runner/api/labelApi';
import * as projectService from '@/lib/projects/projectService';
import type { Project } from '@/lib/projects/types';
import ListFooterBar from '@/components/common/ListFooterBar';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageFilterCard,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import TaskFilterBar, { type TaskFilterOption } from '@/components/tasks/TaskFilterBar';
import { useI18n } from '@/components/common/I18nProvider';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import ListCreateButton from '@/components/common/ListCreateButton';
import { useAuthStore } from '@/store/authStore';
import { normalizeRole } from '@/lib/api/roleLabels';
import {
  canAnnotateLabelTask,
  canReviewLabelTask,
  labelTaskToActorPayload,
} from '@/lib/label/labelTaskActorPermissions';
import { formatDateTimeMinute } from '@/utils/format';

// 动态导入模态组件，避免 SSR 问题
const CreateLabelTaskModal = dynamic(
  () => import('@/features/data-platform/components/label/CreateLabelTaskModal'),
  { ssr: false }
);

export default function LabelTasksPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);

  // i18n 兜底：当 key 未命中时，t(key) 会回传 key 本身
  const tr = useCallback(
    (key: string, fallback: string) => {
      const v = t(key as any);
      return typeof v === 'string' && v === key ? fallback : v;
    },
    [t]
  );
  const [projectList, setProjectList] = useState<Project[]>([]);
  const ownedProjectIds = useMemo(
    () =>
      new Set(
        projectList
          .filter((p) => (p.ownerId || '').trim() === (authUser?.id || '').trim())
          .map((p) => p.id)
      ),
    [projectList, authUser?.id]
  );
  const canManageLabelTasks =
    normalizeRole(authUser?.role) !== 'USER' || ownedProjectIds.size > 0;
  const [tasks, setTasks] = useState<LabelTask[]>([]);
  const [query, setQuery] = useState('');
  const [filterProjectId, setFilterProjectId] = useState('');
  const [filterLabeler, setFilterLabeler] = useState('');
  const [filterReviewer, setFilterReviewer] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [initialDatasetIds, setInitialDatasetIds] = useState<number[] | null>(null);
  const [initialProjectId, setInitialProjectId] = useState<string | null>(null);
  const [initialFilePaths, setInitialFilePaths] = useState<string[]>([]);
  const [initialFromDataAssets, setInitialFromDataAssets] = useState(false);
  const [tasksLoading, setTasksLoading] = useState(true);
  const [tasksError, setTasksError] = useState<string | null>(null);
  const [creatingTask, setCreatingTask] = useState(false);
  const [toast, setToast] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [batchDeleteConfirmOpen, setBatchDeleteConfirmOpen] = useState(false);
  const [batchDeleteLoading, setBatchDeleteLoading] = useState(false);
  const [deleteTaskConfirm, setDeleteTaskConfirm] = useState<LabelTask | null>(null);
  const [toggleCompletedTask, setToggleCompletedTask] = useState<LabelTask | null>(null);
  const [toggleVerifiedTask, setToggleVerifiedTask] = useState<LabelTask | null>(null);
  const [executingTaskId, setExecutingTaskId] = useState<string | null>(null);
  const [executeStage, setExecuteStage] = useState<'idle' | 'checking' | 'preparing'>('idle');

  const showToast = useCallback((type: 'success' | 'error' | 'info', message: string) => {
    setToast({ type, message });
    setTimeout(() => {
      setToast((prev) => (prev && prev.message === message ? null : prev));
    }, 2200);
  }, []);

  const fetchTasks = useCallback(async () => {
    setTasksLoading(true);
    setTasksError(null);
    try {
      const res = await getLabelTasks({ limit: 500 });
      if (res.ok && Array.isArray(res.data)) {
        // 按创建时间倒序，使新建任务出现在列表最前，创建后刷新时能立即看到
        const sorted = [...res.data].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        const mapped: LabelTask[] = sorted.map((row, i) => labelTaskRowToTask(row, i + 1));
        setTasks(mapped);
      } else {
        setTasks([]);
        setTasksError(res.error || t('feedback.requestFailed'));
        showToast('error', res.error || t('feedback.requestFailed'));
      }
    } catch (e) {
      setTasks([]);
      const msg = e instanceof Error ? e.message : t('feedback.requestFailed');
      setTasksError(msg);
      showToast('error', msg);
    } finally {
      setTasksLoading(false);
    }
  }, [showToast, t]);

  // 从数据库加载任务列表（严格按 label_tasks 表显示）
  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  // 从 URL 读取 datasetIds / projectId / filePath（数据资产页批量标注跳转）
  useEffect(() => {
    const datasetIdsStr = searchParams.get('datasetIds') || searchParams.get('datasetId');
    const projectId = searchParams.get('projectId');
    const filePathEnc = searchParams.get('filePath');
    const fromDataAssets = searchParams.get('fromDataAssets') === '1';
    const fromSelectionStore = searchParams.get('fromSelectionStore') === '1';
    if (datasetIdsStr && canManageLabelTasks) {
      const ids = datasetIdsStr.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      if (ids.length > 0) {
        setInitialDatasetIds(ids);
        setInitialProjectId(projectId || null);
        setInitialFromDataAssets(!!fromDataAssets);
        const paths: string[] = [];
        if (filePathEnc) {
          try {
            paths.push(decodeURIComponent(filePathEnc));
          } catch {
            /* ignore */
          }
        }
        setInitialFilePaths(paths);
        setShowCreateModal(true);
        router.replace('/label/tasks', { scroll: false });
      }
    }
    // 批量场景优先从 sessionStorage 恢复（规避 URL 过长）
    if (!datasetIdsStr && fromSelectionStore && canManageLabelTasks && typeof window !== 'undefined') {
      try {
        const raw = window.sessionStorage.getItem('label:pendingSelection');
        if (!raw) return;
        const parsed = JSON.parse(raw) as {
          projectId?: string;
          datasetIds?: number[];
          filePaths?: string[];
          createdAt?: number;
        };
        const ids = Array.isArray(parsed?.datasetIds)
          ? parsed.datasetIds.map((v) => Number(v)).filter((v) => Number.isFinite(v))
          : [];
        if (ids.length === 0) return;
        // 30 分钟有效期，避免陈旧选择误触发
        const createdAt = Number(parsed?.createdAt || 0);
        if (createdAt > 0 && Date.now() - createdAt > 30 * 60 * 1000) {
          window.sessionStorage.removeItem('label:pendingSelection');
          return;
        }
        setInitialDatasetIds(ids);
        setInitialProjectId((parsed?.projectId || projectId || '').trim() || null);
        setInitialFromDataAssets(true);
        setInitialFilePaths(Array.isArray(parsed?.filePaths) ? parsed.filePaths.filter((p) => !!p) : []);
        setShowCreateModal(true);
        window.sessionStorage.removeItem('label:pendingSelection');
        router.replace('/label/tasks', { scroll: false });
      } catch {
        // ignore parse/storage errors
      }
    }
  }, [searchParams, router, canManageLabelTasks]);

  // 动态生成筛选选项：所属项目来自项目列表
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

  const projectOptions = useMemo(() => projectList, [projectList]);
  const projectNameById = useMemo(() => new Map(projectList.map((p) => [p.id, p.name] as const)), [projectList]);

  const labelerOptions = useMemo(() => {
    const labelers = new Set<string>();
    tasks.forEach(t => {
      if (t.labeler) labelers.add(t.labeler);
    });
    return Array.from(labelers).sort();
  }, [tasks]);

  const reviewerOptions = useMemo(() => {
    const reviewers = new Set<string>();
    tasks.forEach((t) => {
      if (t.reviewer) reviewers.add(t.reviewer);
    });
    return Array.from(reviewers).sort();
  }, [tasks]);

  // 过滤任务
  const filteredTasks = useMemo(() => {
    let filtered = [...tasks];

    // 搜索过滤（任务名称）
    if (query.trim()) {
      const lowerQuery = query.toLowerCase();
      filtered = filtered.filter(t => t.name.toLowerCase().includes(lowerQuery));
    }

    // 所属项目过滤
    if (filterProjectId) {
      filtered = filtered.filter(t => t.projectId === filterProjectId);
    }

    // 标注员过滤
    if (filterLabeler) {
      filtered = filtered.filter(t => t.labeler === filterLabeler);
    }

    // 审核员过滤（与标注员一致：按任务表 reviewer 字段）
    if (filterReviewer) {
      filtered = filtered.filter((t) => t.reviewer === filterReviewer);
    }

    // 按任务编号从小到大排序
    filtered.sort((a, b) => {
      const na = parseInt(a.id, 10);
      const nb = parseInt(b.id, 10);
      return (isNaN(na) ? 0 : na) - (isNaN(nb) ? 0 : nb);
    });

    return filtered;
  }, [tasks, query, filterProjectId, filterLabeler, filterReviewer]);

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const totalFiltered = filteredTasks.length;
  const pagedTasks = useMemo(
    () => filteredTasks.slice((page - 1) * pageSize, page * pageSize),
    [filteredTasks, page, pageSize]
  );

  useEffect(() => {
    setSelectedIds(new Set());
  }, [page]);

  const handlePageSizeChange = useCallback((size: number) => {
    setPageSize(size);
    setPage(1);
  }, []);

  const toggleRow = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    const currentPageIds = pagedTasks.map((t) => t.id);
    const allSelected = currentPageIds.length > 0 && currentPageIds.every((id) => selectedIds.has(id));
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (allSelected) currentPageIds.forEach((id) => next.delete(id));
      else currentPageIds.forEach((id) => next.add(id));
      return next;
    });
  }, [pagedTasks, selectedIds]);

  const handleBatchDelete = useCallback(() => {
    if (selectedIds.size === 0) return;
    setBatchDeleteConfirmOpen(true);
  }, [selectedIds]);

  const handleBatchDeleteConfirm = useCallback(async () => {
    if (selectedIds.size === 0) {
      setBatchDeleteConfirmOpen(false);
      return;
    }
    setBatchDeleteLoading(true);
    try {
      const toDelete = tasks.filter((t) => selectedIds.has(t.id));
      const { deleteLabelTask } = await import('@/features/label-runner/api/labelApi');
      let hasError = false;
      for (const task of toDelete) {
        const backendTaskId = task.backendTaskId || task.id;
        const res = await deleteLabelTask(backendTaskId);
        if (!res.ok) {
          hasError = true;
        }
      }
      setSelectedIds(new Set());
      await fetchTasks();
      showToast(hasError ? 'error' : 'success', hasError ? t('feedback.deleteFailed') : t('feedback.deleteSuccess'));
    } catch (e) {
      console.error('批量删除任务失败:', e);
      showToast('error', t('feedback.deleteFailed'));
    } finally {
      setBatchDeleteLoading(false);
      setBatchDeleteConfirmOpen(false);
    }
  }, [selectedIds, tasks, fetchTasks, showToast, t]);

  // 重置筛选
  const handleReset = () => {
    setQuery('');
    setFilterProjectId('');
    setFilterLabeler('');
    setFilterReviewer('');
  };

  // 创建新任务
  const handleCreateTask = async (
    taskData: Omit<LabelTask, 'id' | 'createdAt' | 'updatedAt'>
  ): Promise<void> => {
    setCreatingTask(true);
    try {
      // 调用后端 API 创建任务（写入 backend/data/assets/assets.db 的 label_tasks 表）
      const { createLabelTask } = await import('@/features/label-runner/api/labelApi');

      // 构建请求参数：优先使用 dataset_ids，否则使用 dataset_path
      const requestParams: Record<string, unknown> = {
        name: taskData.name?.trim() ?? '',
        data_count: taskData.dataCount,
        project_id: taskData.projectId?.trim() || undefined,
        labeler: taskData.labeler?.trim() || undefined,
        reviewer: taskData.reviewer?.trim() || undefined,
        collector: taskData.collector?.trim() || '默认采集员',
      };

      const datasetIds = (taskData as { datasetIds?: unknown }).datasetIds;
      const hasDatasetIds = Array.isArray(datasetIds) && datasetIds.length > 0;
      const hasDatasetPath = Boolean(taskData.datasetDir?.trim());

      if (hasDatasetIds) {
        requestParams.dataset_ids = datasetIds;
        // 弹窗内选择的数据集均来自数据资产，必须带 dataset_source 否则后端会按 hdf5_datasets 查导致失败
        requestParams.dataset_source = 'data_assets';
      } else if (hasDatasetPath) {
        requestParams.dataset_path = taskData.datasetDir!.trim();
      } else {
        setCreatingTask(false);
        showToast('error', t('feedback.requestFailed'));
        return;
      }

      const response = await createLabelTask(
        requestParams as unknown as import('@/features/label-runner/api/labelApi').CreateLabelTaskRequest
      );

      if (!response.ok || !response.data) {
        showToast('error', t('feedback.error'));
        throw new Error(response.error || 'create task failed');
      }

      setShowCreateModal(false);
      setEditTask(null);
      setInitialDatasetIds(null);
      setInitialProjectId(null);
      setInitialFilePaths([]);
      setInitialFromDataAssets(false);
      await fetchTasks();
      setPage(1);
      router.replace('/label/tasks', { scroll: false });
    } catch (error) {
      console.error('创建任务失败:', error);
      showToast('error', t('feedback.error'));
      throw error;
    } finally {
      setCreatingTask(false);
    }
  };

  // 删除任务（同步删除后端数据库与配置文件）
  const handleDelete = async (task: LabelTask) => {
    setDeleteTaskConfirm(task);
  };

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTaskConfirm) return;
    const backendTaskId = deleteTaskConfirm.backendTaskId || deleteTaskConfirm.id;
    try {
      const { deleteLabelTask } = await import('@/features/label-runner/api/labelApi');
      const res = await deleteLabelTask(backendTaskId);
      if (!res.ok) {
        showToast('error', t('feedback.deleteFailed'));
        setDeleteTaskConfirm(null);
        return;
      }
    } catch (e) {
      showToast('error', t('feedback.deleteFailed'));
      setDeleteTaskConfirm(null);
      return;
    }

    await fetchTasks();
    setDeleteTaskConfirm(null);
    showToast('success', t('feedback.deleteSuccess'));
  }, [deleteTaskConfirm, fetchTasks, showToast, t]);

  // 切换「已完成」状态
  const handleToggleCompleted = (task: LabelTask) => {
    const payload = labelTaskToActorPayload(task, projectList);
    if (!canAnnotateLabelTask(authUser, payload)) {
      showToast('error', tr('labelTasksPage.noAnnotatePermission', '无权标注该任务'));
      return;
    }
    if (task.completed) {
      setToggleCompletedTask(task);
      return;
    }
    setToggleCompletedTask(task);
  };

  const handleToggleCompletedConfirm = useCallback(async () => {
    if (!toggleCompletedTask) return;
    try {
      const backendTaskId = toggleCompletedTask.backendTaskId || toggleCompletedTask.id;
      const { updateLabelTask } = await import('@/features/label-runner/api/labelApi');
      const res = await updateLabelTask(backendTaskId, { completed: !toggleCompletedTask.completed });
      if (!res.ok) {
        showToast('error', t('feedback.requestFailed'));
        return;
      }
      await fetchTasks();
    } catch {
      showToast('error', t('feedback.requestFailed'));
    } finally {
      setToggleCompletedTask(null);
    }
  }, [toggleCompletedTask, fetchTasks, showToast, t]);

  // 切换「已校验」状态
  const handleToggleVerified = (task: LabelTask) => {
    const payload = labelTaskToActorPayload(task, projectList);
    if (!canReviewLabelTask(authUser, payload)) {
      showToast('error', t('labelTasksPage.noReviewPermission') || '无权审核该任务');
      return;
    }
    if (task.verified) {
      setToggleVerifiedTask(task);
      return;
    }
    setToggleVerifiedTask(task);
  };

  const handleToggleVerifiedConfirm = useCallback(async () => {
    if (!toggleVerifiedTask) return;
    try {
      const backendTaskId = toggleVerifiedTask.backendTaskId || toggleVerifiedTask.id;
      const { updateLabelTask } = await import('@/features/label-runner/api/labelApi');
      const res = await updateLabelTask(backendTaskId, { verified: !toggleVerifiedTask.verified });
      if (!res.ok) {
        showToast('error', t('feedback.requestFailed'));
        return;
      }
      await fetchTasks();
    } catch {
      showToast('error', t('feedback.requestFailed'));
    } finally {
      setToggleVerifiedTask(null);
    }
  }, [toggleVerifiedTask, fetchTasks, showToast, t]);

  // 编辑任务：打开任务定义弹窗，可修改之前的定义
  const [editTask, setEditTask] = useState<LabelTask | null>(null);
  const [viewInstructionsTask, setViewInstructionsTask] = useState<LabelTask | null>(null);
  const [viewInstructionsContent, setViewInstructionsContent] = useState<string>('');
  const [viewInstructionsLoading, setViewInstructionsLoading] = useState(false);

  const handleEdit = (task: LabelTask) => {
    setEditTask(task);
    setShowCreateModal(true);
  };

  const handleViewInstructions = useCallback(async (task: LabelTask) => {
    const backendTaskId = task.backendTaskId || task.id;
    setViewInstructionsTask(task);
    setViewInstructionsContent('');
    setViewInstructionsLoading(true);
    try {
      const { getEpisodes, getTaskInstructions, getTaskInstructionsFile } = await import(
        '@/features/label-runner/api/labelApi'
      );
      // 与执行页左侧「已标注」一致：来自 episodes 接口合并的 data_assets.instruction_text
      const epRes = await getEpisodes(backendTaskId);
      if (epRes.ok && epRes.data && epRes.data.length > 0) {
        const instructions = epRes.data.map((ep) => ({
          episode_id: ep.id,
          episode_name: ep.name,
          instruction: String(ep.instruction_text ?? '').trim(),
        }));
        setViewInstructionsContent(JSON.stringify({ instructions }, null, 2));
        return;
      }
      // 回退：任务目录 instructions.json（与保存接口写入的任务级文件一致）
      const tr = await getTaskInstructions(backendTaskId);
      if (tr.ok && tr.data?.instructions && tr.data.instructions.length > 0) {
        setViewInstructionsContent(JSON.stringify({ instructions: tr.data.instructions }, null, 2));
        return;
      }
      // 再回退：数据集目录下旧版 instructions.json（可能与 DB 不同步）
      const res = await getTaskInstructionsFile(backendTaskId);
      if (res.ok && res.data?.content != null) {
        setViewInstructionsContent(res.data.content);
      } else {
        setViewInstructionsContent(JSON.stringify({ instructions: [] }, null, 2));
      }
    } catch {
      setViewInstructionsContent(JSON.stringify({ instructions: [] }, null, 2));
    } finally {
      setViewInstructionsLoading(false);
    }
  }, []);

  const handleSaveEdit = useCallback((taskId: string, patch: { name: string; dataCount: number; projectId: string; labeler?: string; reviewer?: string; collector?: string; datasetIds?: number[]; datasetDir?: string }) => {
    const backendTaskId = tasks.find(t => t.id === taskId)?.backendTaskId || taskId;
    const updatedTasks = tasks.map(t =>
      t.id === taskId
        ? {
            ...t,
            name: patch.name,
            dataCount: patch.dataCount,
            projectId: patch.projectId,
            labeler: patch.labeler ?? '',
            reviewer: patch.reviewer ?? '',
            collector: patch.collector ?? t.collector,
            datasetIds: Array.isArray(patch.datasetIds) ? [...patch.datasetIds] : t.datasetIds,
            datasetDir: patch.datasetDir ?? t.datasetDir,
            updatedAt: new Date().toISOString(),
          }
        : t
    );
    setTasks(updatedTasks);
    setShowCreateModal(false);
    setEditTask(null);
    // 调用后端 API 更新任务配置
    import('@/features/label-runner/api/labelApi').then(({ updateLabelTask }) => {
      updateLabelTask(backendTaskId, {
        name: patch.name,
        data_count: patch.dataCount,
        labeler: patch.labeler || undefined,
        reviewer: patch.reviewer || undefined,
        collector: patch.collector || undefined,
        dataset_ids: Array.isArray(patch.datasetIds) ? patch.datasetIds : undefined,
        dataset_source: Array.isArray(patch.datasetIds) && patch.datasetIds.length > 0 ? 'data_assets' : undefined,
        dataset_path: patch.datasetDir || undefined,
        project_id: patch.projectId || undefined,
      }).catch((err: unknown) => console.warn('更新后端任务配置失败:', err));
    });
  }, [tasks]);

  // 执行任务
  const handleExecute = async (task: LabelTask) => {
    const backendTaskId = task.backendTaskId || task.id;
    if (executingTaskId && executingTaskId !== backendTaskId) {
      showToast('info', t('labelTasksPage.preparingAnotherTask') || '已有任务正在准备中，请稍候');
      return;
    }
    if (executingTaskId === backendTaskId) {
      return;
    }
    const payload = labelTaskToActorPayload(task, projectList);
    if (!canAnnotateLabelTask(authUser, payload)) {
      showToast('error', t('labelTasksPage.noAnnotatePermission') || '无权标注该任务');
      return;
    }
    let keepBusyUntilUnmount = false;
    try {
      setExecutingTaskId(backendTaskId);
      setExecuteStage('checking');
      
      // 调用后端 API 检查导入状态
      const { getTaskImportStatus, loadTaskDataset } = await import('@/features/label-runner/api/labelApi');
      
      // 先检查是否已导入
      const statusResponse = await getTaskImportStatus(backendTaskId);
      
      if (!statusResponse.ok || !statusResponse.data) {
        // 如果检查失败，尝试直接加载
        console.warn('检查导入状态失败，尝试直接加载数据集');
      } else if (!statusResponse.data.imported) {
        // 如果未导入，先加载数据集
        setExecuteStage('preparing');
        const loadingMsg = '正在扫描数据集目录，请稍候...';
        console.log(loadingMsg);
        
        const loadResponse = await loadTaskDataset(backendTaskId);
        
        if (!loadResponse.ok || !loadResponse.data) {
          showToast('error', `扫描数据集失败: ${loadResponse.error || '未知错误'}`);
          return;
        }
        
        console.log(`扫描成功: 找到 ${loadResponse.data.count || 0} 个 HDF5 文件`);
      }
      
      // 跳转到执行页，使用后端真实的task_id
      keepBusyUntilUnmount = true;
      router.push(`/label/execute?taskId=${backendTaskId}`);
    } catch (error) {
      console.error('执行任务失败:', error);
      showToast('error', `执行任务失败: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      // 成功触发路由跳转后，保持“进入中”状态直到页面卸载，避免按钮短暂回弹造成闪烁
      if (!keepBusyUntilUnmount) {
        setExecuteStage('idle');
        setExecutingTaskId(null);
      }
    }
  };

  const tableColCount = canManageLabelTasks ? 10 : 9;

  return (
      <ModulePageContainer>
      <ModulePageHeader title={t('labelTasksPage.title')} />

      {/* 搜索和筛选 */}
      <ModulePageFilterCard>
        <TaskFilterBar
          searchValue={query}
          onSearchChange={setQuery}
          searchPlaceholder={t('labelTasksPage.searchPlaceholder')}
          filters={[
            {
              key: 'projectId',
              value: filterProjectId,
              placeholder: t('labelTasksPage.projectFilter'),
              options: projectOptions.map((p) => ({ value: p.id, label: p.name })),
              onChange: setFilterProjectId,
            },
            {
              key: 'labeler',
              value: filterLabeler,
              placeholder: t('labelTasksPage.labelerFilter') || '标注员',
              options: labelerOptions.map((v) => ({ value: v, label: v })),
              onChange: setFilterLabeler,
            },
            {
              key: 'reviewer',
              value: filterReviewer,
              placeholder: t('labelTasksPage.reviewerFilter') || '审核员',
              options: reviewerOptions.map((v) => ({ value: v, label: v })),
              onChange: setFilterReviewer,
            },
          ]}
          onReset={handleReset}
          rightAction={
            canManageLabelTasks ? (
              <ListCreateButton
                onClick={() => {
                  setEditTask(null);
                  setShowCreateModal(true);
                }}
              >
                + {t('common.new')}
              </ListCreateButton>
            ) : undefined
          }
        />
      </ModulePageFilterCard>

      {/* 表格区域 */}
      <ModulePageTableCard>
        {tasksError && (
          <div style={{ padding: 16, color: '#b91c1c', fontSize: 14 }}>
            {tasksError}
          </div>
        )}
        {tasksLoading ? (
          <div style={{ padding: 40, textAlign: 'center', color: '#6b7280', fontSize: 14 }}>
            {t('common.loading')}
          </div>
        ) : (
        <>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
          }}
        >
          <thead>
            <tr style={{ backgroundColor: '#f9fafb' }}>
              {canManageLabelTasks && (
                <th
                  style={{
                    padding: '12px 16px',
                    textAlign: 'center',
                    borderBottom: '1px solid #e5e7eb',
                    fontSize: '13px',
                    fontWeight: '600',
                    color: '#374151',
                    width: 40,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={pagedTasks.length > 0 && pagedTasks.every((t) => selectedIds.has(t.id))}
                    onChange={toggleSelectAll}
                    style={{ cursor: 'pointer', width: 16, height: 16 }}
                  />
                </th>
              )}
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableName')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableDataCount')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableLabeler')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableReviewer')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableProject')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableCreatedAt')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableUpdatedAt')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('labelTasksPage.tableStatus')}
              </th>
              <th
                style={{
                  padding: '12px 16px',
                  textAlign: 'left',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                }}
              >
                {t('common.actions')}
              </th>
            </tr>
          </thead>
          <tbody>
            {pagedTasks.length === 0 ? (
              <tr>
                <td
                  colSpan={tableColCount}
                  style={{
                    padding: '40px',
                    textAlign: 'center',
                    color: '#6b7280',
                    fontSize: '14px',
                  }}
                >
                  {t('labelTasksPage.empty')}
                </td>
              </tr>
            ) : (
              pagedTasks.map((task) => {
                const projectId = task.projectId;
                const projectName = projectId ? (projectNameById.get(projectId) ?? projectId) : '—';
                const actorPayload = labelTaskToActorPayload(task, projectList);
                const canReviewThis = canReviewLabelTask(authUser, actorPayload);
                const canAnnotateThis = canAnnotateLabelTask(authUser, actorPayload);
                const backendTaskId = task.backendTaskId || task.id;
                const isExecutingThis = executingTaskId === backendTaskId;
                const isBlockedByOtherTask = Boolean(executingTaskId && !isExecutingThis);
                const executeButtonDisabled = !canAnnotateThis || isBlockedByOtherTask || isExecutingThis;
                const executeButtonText = isExecutingThis
                  ? (executeStage === 'preparing' ? '准备中...' : '进入中...')
                  : t('common.run');

                return (
                  <tr
                    key={task.id}
                    style={{
                      transition: 'background-color 0.15s',
                      borderBottom: '1px solid #f3f4f6',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f9fafb';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    {canManageLabelTasks && (
                      <td
                        style={{
                          padding: '12px 16px',
                          textAlign: 'center',
                          fontSize: '13px',
                          color: '#111827',
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={selectedIds.has(task.id)}
                          onChange={() => toggleRow(task.id)}
                          style={{ cursor: 'pointer', width: 16, height: 16 }}
                        />
                      </td>
                    )}
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {task.name}
                    </td>
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {task.dataCount ?? '—'}
                    </td>
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {task.labeler || '—'}
                    </td>
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {task.reviewer || '—'}
                    </td>
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {projectName}
                    </td>
                    <td
                      style={{
                        padding: '12px 16px',
                        fontSize: '13px',
                        color: '#111827',
                      }}
                    >
                      {formatDateTimeMinute(task.createdAt)}
                    </td>
                  <td
                    style={{
                      padding: '12px 16px',
                      fontSize: '13px',
                      color: '#111827',
                    }}
                  >
                    {formatDateTimeMinute(task.updatedAt)}
                  </td>
                  <td
                    style={{
                      padding: '12px 16px',
                      fontSize: '13px',
                      color: '#111827',
                    }}
                  >
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {canAnnotateThis ? (
                        <button
                          type="button"
                          onClick={() => handleToggleCompleted(task)}
                          title={t('labelTasksPage.completedToggleTooltip') || ''}
                          style={{
                            padding: '4px 10px',
                            border: 'none',
                            borderRadius: '4px',
                            fontSize: '12px',
                            cursor: 'pointer',
                            fontWeight: '500',
                            backgroundColor: task.completed ? '#059669' : '#e5e7eb',
                            color: task.completed ? '#ffffff' : '#6b7280',
                            transition: 'all 0.2s',
                          }}
                        >
                          {task.completed ? `✓ ${t('status.completed')}` : t('labelTasksPage.statusNotStarted')}
                        </button>
                      ) : (
                        <span
                          style={{
                            padding: '4px 10px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '500',
                            backgroundColor: task.completed ? '#059669' : '#e5e7eb',
                            color: task.completed ? '#ffffff' : '#6b7280',
                          }}
                        >
                          {task.completed ? `✓ ${t('status.completed')}` : t('labelTasksPage.statusNotStarted')}
                        </span>
                      )}
                      {canReviewThis ? (
                        <button
                          type="button"
                          onClick={() => handleToggleVerified(task)}
                          title={t('labelTasksPage.verifiedToggleTooltip') || ''}
                          style={{
                            padding: '4px 10px',
                            border: 'none',
                            borderRadius: '4px',
                            fontSize: '12px',
                            cursor: 'pointer',
                            fontWeight: '500',
                            backgroundColor: task.verified ? '#2563eb' : '#e5e7eb',
                            color: task.verified ? '#ffffff' : '#6b7280',
                            transition: 'all 0.2s',
                          }}
                        >
                          {task.verified ? `✓ ${t('labelTasksPage.statusVerified')}` : t('labelTasksPage.statusUnverified')}
                        </button>
                      ) : (
                        <span
                          style={{
                            padding: '4px 10px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '500',
                            backgroundColor: task.verified ? '#2563eb' : '#e5e7eb',
                            color: task.verified ? '#ffffff' : '#6b7280',
                          }}
                        >
                          {task.verified ? `✓ ${t('labelTasksPage.statusVerified')}` : t('labelTasksPage.statusUnverified')}
                        </span>
                      )}
                    </div>
                  </td>
                  <td
                    style={{
                      padding: '12px 16px',
                      fontSize: '13px',
                      color: '#111827',
                    }}
                  >
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {canManageLabelTasks && (
                        <button
                          onClick={() => handleEdit(task)}
                          style={{
                            padding: '4px 12px',
                            backgroundColor: 'transparent',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            color: '#374151',
                            fontSize: '12px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#f9fafb';
                            e.currentTarget.style.borderColor = '#9ca3af';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'transparent';
                            e.currentTarget.style.borderColor = '#d1d5db';
                          }}
                        >
                          {t('common.edit')}
                        </button>
                      )}
                      <button
                        type="button"
                        disabled={executeButtonDisabled}
                        onClick={() => !executeButtonDisabled && handleExecute(task)}
                        title={
                          !canAnnotateThis
                            ? t('labelTasksPage.noAnnotatePermission') || ''
                            : isBlockedByOtherTask
                              ? (t('labelTasksPage.preparingAnotherTask') || '已有任务正在准备中，请稍候')
                              : isExecutingThis
                                ? (executeStage === 'preparing' ? '正在准备标注数据' : '正在检查任务状态')
                            : undefined
                        }
                        style={{
                          padding: '4px 12px',
                          backgroundColor: executeButtonDisabled ? '#9ca3af' : '#2563eb',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#ffffff',
                          fontSize: '12px',
                          cursor: executeButtonDisabled ? 'not-allowed' : 'pointer',
                          fontWeight: '500',
                          transition: 'all 0.2s',
                          opacity: executeButtonDisabled ? 0.85 : 1,
                        }}
                        onMouseEnter={(e) => {
                          if (!executeButtonDisabled) e.currentTarget.style.backgroundColor = '#1d4ed8';
                        }}
                        onMouseLeave={(e) => {
                          if (!executeButtonDisabled) e.currentTarget.style.backgroundColor = '#2563eb';
                        }}
                      >
                        {executeButtonText}
                      </button>
                      <button
                        onClick={() => handleViewInstructions(task)}
                        style={{
                          padding: '4px 12px',
                          backgroundColor: 'transparent',
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          color: '#374151',
                          fontSize: '12px',
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f9fafb';
                          e.currentTarget.style.borderColor = '#9ca3af';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.borderColor = '#d1d5db';
                        }}
                      >
                        {t('common.view')}
                      </button>
                      {canManageLabelTasks && (
                        <button
                          onClick={() => handleDelete(task)}
                          style={{
                            padding: '4px 12px',
                            backgroundColor: 'transparent',
                            border: '1px solid #d1d5db',
                            borderRadius: '4px',
                            color: '#374151',
                            fontSize: '12px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#fef2f2';
                            e.currentTarget.style.borderColor = '#ef4444';
                            e.currentTarget.style.color = '#ef4444';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'transparent';
                            e.currentTarget.style.borderColor = '#d1d5db';
                            e.currentTarget.style.color = '#374151';
                          }}
                        >
                          {t('common.delete')}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
                );
              })
            )}
          </tbody>
        </table>
        <ListFooterBar
          variant="inline"
          total={totalFiltered}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={handlePageSizeChange}
          selectedCount={selectedIds.size}
          batchActions={
            canManageLabelTasks
              ? [
                  {
                    key: 'delete',
                    label: t('labelTasksPage.batchDelete') || t('common.delete'),
                    onClick: handleBatchDelete,
                    danger: true,
                  },
                ]
              : []
          }
        />
        </>
        )}
      </ModulePageTableCard>

      {/* 新建/编辑任务弹窗 */}
      <CreateLabelTaskModal
        open={showCreateModal && canManageLabelTasks}
        onClose={() => {
          setShowCreateModal(false);
          setEditTask(null);
          setInitialDatasetIds(null);
          setInitialProjectId(null);
          setInitialFilePaths([]);
          setInitialFromDataAssets(false);
          router.replace('/label/tasks', { scroll: false });
        }}
        onSubmit={handleCreateTask}
        initialTask={editTask}
        onSave={handleSaveEdit}
        initialDatasetIds={initialDatasetIds ?? undefined}
        initialProjectId={initialProjectId ?? undefined}
        initialFilePaths={initialFilePaths}
        initialFromDataAssets={initialFromDataAssets}
        isSubmitting={creatingTask}
      />

      {/* 批量删除确认 */}
      <ConfirmDialog
        open={batchDeleteConfirmOpen}
        title="删除标注任务"
        description="确认删除选中的标注任务吗？删除后不可恢复。"
        confirmText="确认"
        cancelText="取消"
        loading={batchDeleteLoading}
        onCancel={() => {
          if (batchDeleteLoading) return;
          setBatchDeleteConfirmOpen(false);
        }}
        onConfirm={handleBatchDeleteConfirm}
      />

      {/* 单个删除确认 */}
      <ConfirmDialog
        open={!!deleteTaskConfirm}
        title="删除标注任务"
        description="确认删除该标注任务吗？删除后不可恢复。"
        confirmText="确认"
        cancelText="取消"
        onCancel={() => setDeleteTaskConfirm(null)}
        onConfirm={handleDeleteConfirm}
      />

      {/* 取消「已完成」确认 */}
      <ConfirmDialog
        open={!!toggleCompletedTask}
        title={t('dialog.genericTitle')}
        description={tr('labelTasksPage.confirmToggleCompleted', '确认要切换已完成状态吗？')}
        confirmText="确认"
        cancelText="取消"
        onCancel={() => setToggleCompletedTask(null)}
        onConfirm={handleToggleCompletedConfirm}
      />

      {/* 取消「已校验」确认 */}
      <ConfirmDialog
        open={!!toggleVerifiedTask}
        title={t('dialog.genericTitle')}
        description={tr('labelTasksPage.confirmToggleVerified', '确认要切换已校验状态吗？')}
        confirmText="确认"
        cancelText="取消"
        onCancel={() => setToggleVerifiedTask(null)}
        onConfirm={handleToggleVerifiedConfirm}
      />

      {/* 全局 Toast */}
      {toast && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 18px',
            borderRadius: 10,
            fontSize: 13,
            color: '#ffffff',
            backgroundColor:
              toast.type === 'error'
                ? '#dc2626'
                : toast.type === 'success'
                  ? '#16a34a'
                  : '#111827',
            boxShadow: '0 18px 60px rgba(15,23,42,0.25)',
            zIndex: 1800,
          }}
        >
          {toast.message}
        </div>
      )}

      {/* 查看 instructions.json 弹窗 */}
      {viewInstructionsTask && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setViewInstructionsTask(null)}
        >
          <div
            style={{
              width: '90%',
              maxWidth: '600px',
              maxHeight: '80vh',
              backgroundColor: '#fff',
              borderRadius: '8px',
              boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                padding: '16px 20px',
                borderBottom: '1px solid #e5e7eb',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <span style={{ fontSize: '16px', fontWeight: 600, color: '#111827' }}>
                标注信息 - {viewInstructionsTask.name}
              </span>
              <button
                onClick={() => setViewInstructionsTask(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '20px',
                  color: '#6b7280',
                  cursor: 'pointer',
                  padding: '4px',
                }}
              >
                ✕
              </button>
            </div>
            <pre
              style={{
                flex: 1,
                margin: 0,
                padding: '20px',
                overflow: 'auto',
                fontSize: '13px',
                fontFamily: 'monospace',
                backgroundColor: '#f9fafb',
                color: '#374151',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {viewInstructionsLoading ? '加载中...' : viewInstructionsContent || '{\n  "instructions": []\n}'}
            </pre>
          </div>
        </div>
      )}
    </ModulePageContainer>
  );
}

