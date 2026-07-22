import {
  resolveReplaySourceKind,
  type ResolveReplaySourceKindParams,
} from '@/lib/workspace/replayViewModel';

export type ReplayPageKind = 'data_generation' | 'evaluation' | 'generic';

export function resolveReplayPageKind(
  params: ResolveReplaySourceKindParams
): ReplayPageKind {
  const source = resolveReplaySourceKind(params);
  if (source === 'dataset') return 'data_generation';
  if (source === 'evaluation') return 'evaluation';
  return 'generic';
}

export function hasReplayUrlTarget(params: {
  jobId?: string | null;
  evalId?: string | null;
  evalJobId?: string | null;
  datasetId?: string | null;
  taskType?: string | null;
  replayJobId?: string | null;
}): boolean {
  if (params.taskType === 'isaac_block_stacking' && (params.datasetId?.trim() || params.evalId?.startsWith('isaac_eval_'))) {
    return true;
  }
  return Boolean(
    params.jobId?.trim() ||
      params.evalId?.trim() ||
      params.evalJobId?.trim() ||
      params.replayJobId?.trim()
  );
}
