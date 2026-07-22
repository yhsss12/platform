import { buildCableThreadingReplayHref } from '@/lib/workspace/cableThreading';
import { buildDualArmCableReplayHref } from '@/lib/workspace/dualArmCable';
import { inferReplayTaskTypeFromJobId } from '@/lib/workspace/datasetReplayHref';
import {
  buildNutAssemblyConsoleHref,
  buildNutAssemblyReplayHref,
} from '@/lib/workspace/nutAssembly';
import { buildIsaacBlockStackingReplayHref } from '@/lib/workspace/isaacBlockStacking';

/** 回放页「返回数据中心」默认链接 */
export const REPLAY_DATA_CENTER_HREF = '/workspace/data';

export type ReplayPanelNavigationTarget = {
  href: string;
  label?: string;
  disabled?: boolean;
  reason?: string;
};

export function resolveReplayPanelNavigationTarget(
  id?: string | null,
  options?: { datasetId?: string }
): ReplayPanelNavigationTarget {
  const trimmed = id?.trim();
  if (!trimmed) {
    return {
      href: REPLAY_DATA_CENTER_HREF,
      disabled: true,
      reason: 'Missing replay id',
    };
  }

  if (trimmed.startsWith('na_gen_')) {
    return {
      href: buildNutAssemblyReplayHref({
        jobId: trimmed,
        datasetId: options?.datasetId,
      }),
    };
  }

  if (trimmed.startsWith('na_console_')) {
    return {
      href: buildNutAssemblyConsoleHref({
        jobId: trimmed,
        dataId: options?.datasetId,
      }),
    };
  }

  if (trimmed.startsWith('ct_gen_') || trimmed.startsWith('ct_eval_')) {
    return {
      href: buildCableThreadingReplayHref({
        jobId: trimmed.startsWith('ct_gen_') ? trimmed : undefined,
        evalId: trimmed.startsWith('ct_eval_') ? trimmed : undefined,
        datasetId: options?.datasetId,
      }),
    };
  }

  if (trimmed.startsWith('dac_gen_')) {
    return {
      href: buildDualArmCableReplayHref({
        jobId: trimmed,
        datasetId: options?.datasetId,
      }),
    };
  }

  if (trimmed.startsWith('isaac_gen_') || trimmed.startsWith('isaac_replay_')) {
    return {
      href: buildIsaacBlockStackingReplayHref({
        jobId: trimmed.startsWith('isaac_gen_') ? trimmed : undefined,
        datasetId: options?.datasetId,
      }),
    };
  }

  const inferredTaskType = inferReplayTaskTypeFromJobId(trimmed);
  if (inferredTaskType === 'cable_threading') {
    return {
      href: buildCableThreadingReplayHref({ jobId: trimmed, datasetId: options?.datasetId }),
    };
  }
  if (inferredTaskType === 'nut_assembly') {
    return {
      href: buildNutAssemblyReplayHref({ jobId: trimmed, datasetId: options?.datasetId }),
    };
  }
  if (inferredTaskType === 'dual_arm_cable_manipulation') {
    return {
      href: buildDualArmCableReplayHref({ jobId: trimmed, datasetId: options?.datasetId }),
    };
  }
  if (inferredTaskType === 'isaac_block_stacking') {
    return {
      href: buildIsaacBlockStackingReplayHref({ jobId: trimmed, datasetId: options?.datasetId }),
    };
  }

  return {
    href: REPLAY_DATA_CENTER_HREF,
    disabled: true,
    reason: 'Replay navigation is not configured for this item',
  };
}

/** 根据 jobId 构造回放面板链接（字符串形式，供旧调用方使用） */
export function buildReplayPanelHref(
  id?: string | null,
  options?: { datasetId?: string }
): string {
  return resolveReplayPanelNavigationTarget(id, options).href;
}
