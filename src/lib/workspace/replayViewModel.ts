/**
 * 数据集回放 vs 评测回放的页面语义（标题、返回路径、adapter 边界）。
 * 新入口必须显式传 replayType；无 replayType 时按历史 URL 规则推断。
 */

export type ReplaySourceKind = 'dataset' | 'evaluation';

export type ReplayTypeParam = ReplaySourceKind;

export interface ReplayViewModel {
  replayType: ReplayTypeParam;
  sourceKind: ReplaySourceKind;
  title: string;
  subtitle: string;
  returnPath: string;
  returnLabel: string;
}

export interface ResolveReplaySourceKindParams {
  replayType?: string | null;
  jobId?: string | null;
  evalId?: string | null;
  evalJobId?: string | null;
  datasetId?: string | null;
}

const DATASET_JOB_PREFIXES = ['ct_gen_', 'dac_gen_', 'isaac_gen_', 'isaac_replay_'] as const;
const EVAL_JOB_PREFIXES = ['ct_eval_', 'isaac_eval_'] as const;

function isDatasetJobId(jobId: string | undefined): boolean {
  if (!jobId) return false;
  return DATASET_JOB_PREFIXES.some((p) => jobId.startsWith(p));
}

function isEvalJobId(id: string | undefined): boolean {
  if (!id) return false;
  if (id.startsWith('eval_')) return true;
  return EVAL_JOB_PREFIXES.some((p) => id.startsWith(p));
}

/** 解析 replayType（显式优先，否则兼容历史 URL） */
export function resolveReplaySourceKind(params: ResolveReplaySourceKindParams): ReplaySourceKind {
  const explicit = params.replayType?.trim();
  if (explicit === 'dataset' || explicit === 'evaluation') {
    return explicit;
  }

  const evalId = params.evalId?.trim();
  const evalJobId = params.evalJobId?.trim();
  const jobId = params.jobId?.trim();

  if (isEvalJobId(evalId) || isEvalJobId(evalJobId) || isEvalJobId(jobId)) {
    return 'evaluation';
  }

  if (params.datasetId?.trim() || isDatasetJobId(jobId)) {
    return 'dataset';
  }

  if (evalId || evalJobId) {
    return 'evaluation';
  }

  return 'dataset';
}

export function appendReplayTypeToSearch(
  search: URLSearchParams,
  replayType: ReplaySourceKind
): URLSearchParams {
  search.set('replayType', replayType);
  return search;
}

export function buildReplayPageHref(
  replayType: ReplaySourceKind,
  params: Record<string, string | number | undefined | null>
): string {
  const search = appendReplayTypeToSearch(new URLSearchParams(), replayType);
  for (const [key, value] of Object.entries(params)) {
    if (value == null || value === '') continue;
    search.set(key, String(value));
  }
  return `/workspace/replay?${search.toString()}`;
}

type ReplayCopy = {
  replayDataTitle: string;
  replayDataSubtitle: string;
  replayEvalTitle: string;
  replayEvalSubtitle: string;
};

const DEFAULT_COPY: ReplayCopy = {
  replayDataTitle: '数据集回放',
  replayDataSubtitle: '查看数据生成过程、数据集 episode 与轨迹回放结果。',
  replayEvalTitle: '评测回放',
  replayEvalSubtitle: '查看模型在任务环境中的评测 rollout、评测结果与关键指标。',
};

export function buildReplayViewModel(
  params: ResolveReplaySourceKindParams,
  copy: Partial<ReplayCopy> = {}
): ReplayViewModel {
  const c = { ...DEFAULT_COPY, ...copy };
  const replayType = resolveReplaySourceKind(params);

  if (replayType === 'evaluation') {
    return {
      replayType: 'evaluation',
      sourceKind: 'evaluation',
      title: c.replayEvalTitle,
      subtitle: c.replayEvalSubtitle,
      returnPath: '/workspace/evaluation',
      returnLabel: '返回评测中心',
    };
  }

  return {
    replayType: 'dataset',
    sourceKind: 'dataset',
    title: c.replayDataTitle,
    subtitle: c.replayDataSubtitle,
    returnPath: '/workspace/data',
    returnLabel: '返回数据中心',
  };
}
