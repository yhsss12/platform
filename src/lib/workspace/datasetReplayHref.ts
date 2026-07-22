import type { Dataset } from '@/types/benchmark';
import { isIsaacLabDataset } from '@/lib/workspace/datasetTableActions';

export function resolveDatasetReplayTaskType(dataset: Dataset): string {
  if (
    dataset.sourceJobId.startsWith('na_gen_') ||
    dataset.taskType === 'nut_assembly' ||
    dataset.taskTemplateId === 'nut_assembly_single_arm' ||
    dataset.taskTemplateId === 'task_nut_assembly_v1'
  ) {
    return 'nut_assembly';
  }
  if (
    dataset.taskTemplateId === 'isaac_block_stacking' ||
    isIsaacLabDataset(dataset)
  ) {
    return 'isaac_block_stacking';
  }
  if (
    dataset.sourceJobId.startsWith('dac_gen_') ||
    dataset.taskTemplateId === 'dual_arm_cable_manipulation'
  ) {
    return 'dual_arm_cable_manipulation';
  }
  if (
    dataset.sourceJobId.startsWith('ct_gen_') ||
    dataset.taskTemplateId === 'cable_threading_single_arm'
  ) {
    return 'cable_threading';
  }
  return 'cable_threading';
}

export function buildUnifiedDatasetReplayHref(params: {
  taskType: string;
  datasetId: string;
  sourceJobId?: string;
  replayJobId?: string;
  evalId?: string;
}): string {
  const search = new URLSearchParams({
    taskType: params.taskType,
    datasetId: params.datasetId,
  });
  if (params.sourceJobId) search.set('jobId', params.sourceJobId);
  if (params.replayJobId) search.set('replayJobId', params.replayJobId);
  if (params.evalId) search.set('evalId', params.evalId);
  return `/workspace/replay?${search.toString()}`;
}

export function inferReplayTaskTypeFromJobId(jobId?: string | null): string | undefined {
  if (!jobId) return undefined;
  if (jobId.startsWith('na_gen_')) return 'nut_assembly';
  if (jobId.startsWith('isaac_gen_') || jobId.startsWith('isaac_replay_')) {
    return 'isaac_block_stacking';
  }
  if (jobId.startsWith('dac_gen_')) return 'dual_arm_cable_manipulation';
  if (jobId.startsWith('ct_gen_') || jobId.startsWith('ct_eval_')) return 'cable_threading';
  return undefined;
}

export interface UnifiedReplayWorkbenchModeParams {
  replayType?: string | null;
  taskType?: string | null;
  jobId?: string | null;
  datasetId?: string | null;
  replayJobId?: string | null;
  evalId?: string | null;
  evalJobId?: string | null;
  hasUrlTarget?: boolean;
}

/** 是否走 UnifiedReplayWorkbench（新适配器路径），否则使用遗留 cable/dual/isaac 面板 */
export function isUnifiedReplayWorkbenchMode(
  params: UnifiedReplayWorkbenchModeParams
): boolean {
  if (!params.hasUrlTarget) return false;

  const jobId = (params.jobId ?? '').trim();
  const datasetId = (params.datasetId ?? '').trim();
  const taskType = (params.taskType ?? '').trim();
  const replayJobId = (params.replayJobId ?? '').trim();

  if (jobId.startsWith('dac_gen_')) return false;
  if (jobId.startsWith('ct_gen_') || jobId.startsWith('ct_eval_')) return false;
  if (
    jobId.startsWith('isaac_gen_') ||
    replayJobId.startsWith('isaac_replay_') ||
    (taskType === 'isaac_block_stacking' && datasetId.startsWith('isaac_ds_'))
  ) {
    return false;
  }

  if (taskType === 'nut_assembly' || taskType === 'nut_assembly_single_arm') return true;
  if (datasetId && taskType) return true;
  if (params.replayType === 'dataset' && datasetId) return true;

  return false;
}
