import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import {
  hasBuiltDataset,
  isDemoDataCategory,
  isGenerateDataRow,
  isPureDatasetRow,
} from '@/lib/mock/workspaceDataMock';
import {
  buildCableThreadingConsoleHref,
  buildCableThreadingReplayHref,
  resolveCableThreadingConsoleJobId,
} from '@/lib/workspace/cableThreading';
import {
  buildDualArmCableConsoleHref,
  buildDualArmCableReplayHref,
  isDualArmCableDataItem,
  resolveDualArmBackendJobId,
  resolveDualArmConsoleJobId,
} from '@/lib/workspace/dualArmCable';
import {
  buildNutAssemblyConsoleHref,
  buildNutAssemblyReplayHref,
} from '@/lib/workspace/nutAssembly';
import {
  buildSimulationConsoleHref,
  isActiveDataGenerationStatus,
  isFailedDataStatus,
} from '@/lib/workspace/simulationConsole';

export const WORKSPACE_DATA_ROUTES = {
  replay: '/workspace/replay',
  training: '/workspace/training',
  evaluation: '/workspace/evaluation',
} as const;

export type WorkspaceDataActionVariant = 'primary' | 'link' | 'danger';

export type WorkspaceDataAction = {
  key: string;
  label: string;
  href?: string;
  onClick?: () => void;
  variant: WorkspaceDataActionVariant;
};

export type WorkspaceDataActionHandlers = {
  onOpenDetail: (item: WorkspaceDataItem) => void;
  onBuildDataset: (item: WorkspaceDataItem) => void;
  onDelete?: (item: WorkspaceDataItem) => void;
};

function isRunningOrQueued(item: WorkspaceDataItem): boolean {
  if (isActiveDataGenerationStatus(item.status)) return true;
  const backend = item.backendJobStatus;
  return backend === 'queued' || backend === 'running';
}

/** 是否支持从任务数据构建训练数据集 */
export function isDatasetBuildSupported(item: WorkspaceDataItem): boolean {
  if (hasBuiltDataset(item)) return false;
  if (item.datasetBuildSupported != null) return item.datasetBuildSupported;
  if (isDualArmCableDataItem(item) || item.taskType === 'dual_arm_cable_manipulation') {
    return false;
  }
  if (item.taskType === 'cable_threading') {
    if (item.status !== 'completed') return false;
    if (item.qualityStatus === '不可构建') return false;
    return Boolean(
      item.npzPath ||
        item.hdf5Path ||
        item.manifestPath ||
        (item.successfulEpisodes != null && item.successfulEpisodes > 0)
    );
  }
  return item.status === 'completed' && isDemoDataCategory(item.dataCategory);
}

function hasFailedReplayVideo(item: WorkspaceDataItem): boolean {
  return item.generateVideoExists === true || Boolean(item.generateVideoPath);
}

function resolveReplayHref(item: WorkspaceDataItem): string {
  if (item.taskType === 'nut_assembly' || item.sourceJobId?.startsWith('na_gen_')) {
    const jobId = item.sourceJobId ?? item.jobId ?? item.backendJobId;
    return jobId
      ? buildNutAssemblyReplayHref({ jobId, datasetId: item.datasetId ?? item.id })
      : WORKSPACE_DATA_ROUTES.replay;
  }
  if (isDualArmCableDataItem(item)) {
    const jobId = resolveDualArmBackendJobId(item);
    return jobId ? buildDualArmCableReplayHref({ jobId }) : WORKSPACE_DATA_ROUTES.replay;
  }
  const cableJobId =
    item.taskType === 'cable_threading'
      ? item.sourceJobId ?? item.jobId ?? item.backendJobId ?? item.id
      : undefined;
  return cableJobId != null
    ? buildCableThreadingReplayHref({ jobId: cableJobId })
    : WORKSPACE_DATA_ROUTES.replay;
}

function buildViewRunAction(item: WorkspaceDataItem): WorkspaceDataAction | null {
  if (item.taskType === 'nut_assembly' || item.sourceJobId?.startsWith('na_gen_')) {
    const jobId = item.sourceJobId ?? item.jobId ?? item.backendJobId;
    if (!jobId) return null;
    return {
      key: 'view-run',
      label: '查看运行',
      variant: 'primary',
      href: buildNutAssemblyConsoleHref({ jobId, dataId: item.id }),
    };
  }
  if (isDualArmCableDataItem(item)) {
    const jobId = resolveDualArmConsoleJobId(item);
    if (!jobId) return null;
    return {
      key: 'view-run',
      label: '查看运行',
      variant: 'primary',
      href: buildDualArmCableConsoleHref({
        jobId,
        dataId: item.id.startsWith('dac-pending-') ? item.id : undefined,
      }),
    };
  }

  if (item.taskType === 'cable_threading') {
    const jobId = resolveCableThreadingConsoleJobId(item);
    if (!jobId) return null;
    return {
      key: 'view-run',
      label: '查看运行',
      variant: 'primary',
      href: buildCableThreadingConsoleHref({
        jobId,
        dataId: item.id,
      }),
    };
  }

  return {
    key: 'view-run',
    label: '查看运行',
    variant: 'primary',
    href: buildSimulationConsoleHref({
      mode: 'data-generation',
      task: item.taskName,
      dataset: item.name,
      backend: 'mujoco',
    }),
  };
}

function buildViewLogAction(item: WorkspaceDataItem): WorkspaceDataAction | null {
  if (item.taskType === 'nut_assembly' || item.sourceJobId?.startsWith('na_gen_')) {
    const jobId = item.sourceJobId ?? item.jobId ?? item.backendJobId;
    if (!jobId) return null;
    return {
      key: 'view-log',
      label: '查看日志',
      variant: 'link',
      href: buildNutAssemblyConsoleHref({ jobId, dataId: item.id }),
    };
  }
  if (isDualArmCableDataItem(item)) {
    const jobId = resolveDualArmConsoleJobId(item);
    if (!jobId) return null;
    return {
      key: 'view-log',
      label: '查看日志',
      variant: 'link',
      href: buildDualArmCableConsoleHref({ jobId }),
    };
  }

  if (item.taskType === 'cable_threading') {
    const jobId = resolveCableThreadingConsoleJobId(item);
    if (!jobId) return null;
    return {
      key: 'view-log',
      label: '查看日志',
      variant: 'link',
      href: buildCableThreadingConsoleHref({ jobId, dataId: item.id }),
    };
  }

  const viewRun = buildViewRunAction(item);
  if (!viewRun) return null;
  return { ...viewRun, key: 'view-log', label: '查看日志', variant: 'link' };
}

function detailAction(item: WorkspaceDataItem, handlers: WorkspaceDataActionHandlers): WorkspaceDataAction {
  return {
    key: 'detail',
    label: '详情',
    variant: 'link',
    onClick: () => handlers.onOpenDetail(item),
  };
}

function replayAction(item: WorkspaceDataItem): WorkspaceDataAction {
  return {
    key: 'replay',
    label: '回放',
    variant: 'link',
    href: resolveReplayHref(item),
  };
}

function trainAction(item: WorkspaceDataItem): WorkspaceDataAction {
  return {
    key: 'train',
    label: '训练',
    variant: 'link',
    href: `${WORKSPACE_DATA_ROUTES.training}?dataset=${encodeURIComponent(item.datasetId ?? item.id)}`,
  };
}

function deleteAction(item: WorkspaceDataItem, handlers: WorkspaceDataActionHandlers): WorkspaceDataAction | null {
  if (!handlers.onDelete) return null;
  return {
    key: 'delete',
    label: '删除',
    variant: 'danger',
    onClick: () => handlers.onDelete!(item),
  };
}

function withDelete(actions: WorkspaceDataAction[], item: WorkspaceDataItem, handlers: WorkspaceDataActionHandlers) {
  const del = deleteAction(item, handlers);
  return del ? [...actions, del] : actions;
}

/**
 * 统一数据中心任务数据操作列规则。
 * 顺序：详情 → 主操作（构建/训练/查看运行/回放）→ 删除
 */
export function getWorkspaceDataActions(
  item: WorkspaceDataItem,
  handlers: WorkspaceDataActionHandlers
): WorkspaceDataAction[] {
  const detail = detailAction(item, handlers);

  if (isPureDatasetRow(item) && !isGenerateDataRow(item)) {
    return withDelete([detail, trainAction(item)], item, handlers);
  }

  if (item.dataCategory === '评测数据集' && !isGenerateDataRow(item)) {
    return withDelete(
      [
        detail,
        {
          key: 'eval',
          label: '评测',
          variant: 'link',
          href: `${WORKSPACE_DATA_ROUTES.evaluation}?openCreate=1`,
        },
        replayAction(item),
      ],
      item,
      handlers
    );
  }

  if (isRunningOrQueued(item)) {
    const viewRun = buildViewRunAction(item);
    return withDelete(viewRun ? [detail, viewRun] : [detail], item, handlers);
  }

  if (isFailedDataStatus(item.status)) {
    const actions: WorkspaceDataAction[] = [detail];
    const viewLog = buildViewLogAction(item);
    if (viewLog) actions.push(viewLog);
    if (hasFailedReplayVideo(item)) actions.push(replayAction(item));
    return withDelete(actions, item, handlers);
  }

  if (isGenerateDataRow(item)) {
    const actions: WorkspaceDataAction[] = [detail];
    if (hasBuiltDataset(item)) {
      actions.push(trainAction(item));
    } else if (isDatasetBuildSupported(item)) {
      actions.push({
        key: 'build',
        label: '构建数据集',
        variant: 'link',
        onClick: () => handlers.onBuildDataset(item),
      });
    }
    if (item.status === 'completed' || item.status === 'exported' || item.status === 'built') {
      actions.push(replayAction(item));
    }
    return withDelete(actions, item, handlers);
  }

  if (item.status === 'completed' || item.status === 'built' || item.status === 'exported') {
    const actions: WorkspaceDataAction[] = [detail, replayAction(item)];
    if (isDatasetBuildSupported(item)) {
      actions.push({
        key: 'build',
        label: '构建数据集',
        variant: 'link',
        onClick: () => handlers.onBuildDataset(item),
      });
    }
    return withDelete(actions, item, handlers);
  }

  return withDelete([detail], item, handlers);
}
