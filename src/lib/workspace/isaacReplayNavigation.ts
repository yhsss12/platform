import {
  buildIsaacBlockStackingReplayHref as buildUnifiedIsaacReplayHref,
  ISAAC_BLOCK_STACKING_TEMPLATE_ID,
} from '@/lib/workspace/isaacBlockStacking';

export const ISAAC_REPLAY_JOB_QUERY_KEY = 'isaacReplayJob';
export const ISAAC_GENERATE_JOB_QUERY_KEY = 'isaacGenerateJob';
export const ISAAC_TEMPLATE_ID_QUERY_KEY = 'templateId';
export const ISAAC_DATASET_ID_QUERY_KEY = 'datasetId';
export const ISAAC_REPLAY_JOB_ID_QUERY_KEY = 'replayJobId';

const TASK_TEMPLATES_PATH = '/workspace/resources/task-templates';

/** @deprecated 请使用 buildIsaacBlockStackingReplayHref；保留兼容旧调用签名 */
export function buildIsaacBlockStackingReplayHref(jobId: string, datasetId?: string): string {
  if (jobId.startsWith('isaac_replay_')) {
    return buildUnifiedIsaacReplayHref({ replayJobId: jobId, datasetId });
  }
  return buildUnifiedIsaacReplayHref({ jobId, datasetId });
}

export function readIsaacReplayJobId(searchParams: URLSearchParams): string | null {
  const value = searchParams.get(ISAAC_REPLAY_JOB_QUERY_KEY)?.trim();
  return value || null;
}

export function readIsaacGenerateJobId(searchParams: URLSearchParams): string | null {
  const value = searchParams.get(ISAAC_GENERATE_JOB_QUERY_KEY)?.trim();
  return value || null;
}

export function readTaskTemplateIdFromQuery(searchParams: URLSearchParams): string | null {
  const templateId = searchParams.get(ISAAC_TEMPLATE_ID_QUERY_KEY)?.trim();
  if (templateId) return templateId;
  const legacyTemplate = searchParams.get('template')?.trim();
  return legacyTemplate || null;
}

export function clearIsaacReplayQueryParams(searchParams: URLSearchParams): URLSearchParams {
  const next = new URLSearchParams(searchParams.toString());
  next.delete(ISAAC_REPLAY_JOB_QUERY_KEY);
  next.delete(ISAAC_GENERATE_JOB_QUERY_KEY);
  next.delete(ISAAC_TEMPLATE_ID_QUERY_KEY);
  next.delete('template');
  return next;
}

export function buildTaskTemplatesPathWithParams(params: URLSearchParams): string {
  const qs = params.toString();
  return qs ? `${TASK_TEMPLATES_PATH}?${qs}` : TASK_TEMPLATES_PATH;
}
