import type { EvaluationJobListItem, EvaluationJobListResponse } from '@/lib/api/evaluationClient';

type RawEvaluationJobListResponse = Partial<EvaluationJobListResponse> & {
  items?: EvaluationJobListItem[];
  evaluations?: EvaluationJobListItem[];
  evaluationJobs?: EvaluationJobListItem[];
};

/** 兼容后端/历史字段名：jobs / items / evaluations / evaluationJobs */
export function normalizeEvaluationJobListResponse(
  raw: RawEvaluationJobListResponse | null | undefined
): EvaluationJobListResponse {
  const jobs = Array.isArray(raw?.jobs)
    ? raw.jobs
    : Array.isArray(raw?.items)
      ? raw.items
      : Array.isArray(raw?.evaluations)
        ? raw.evaluations
        : Array.isArray(raw?.evaluationJobs)
          ? raw.evaluationJobs
          : [];
  const total =
    typeof raw?.total === 'number' && Number.isFinite(raw.total) ? raw.total : jobs.length;
  return { jobs, total };
}

export type EvaluationListLoadState = 'loading' | 'error' | 'empty' | 'success';

export function resolveEvaluationListLoadState(input: {
  isPending: boolean;
  isError: boolean;
  hasResponse: boolean;
  total: number;
}): EvaluationListLoadState {
  if (input.isError) return 'error';
  if (input.isPending && !input.hasResponse) return 'loading';
  if (input.total === 0) return 'empty';
  return 'success';
}

export function evaluationListEmptyMessage(
  loadState: EvaluationListLoadState,
  errorMessage?: string | null
): string {
  if (loadState === 'loading') return '加载评测任务…';
  if (loadState === 'error') {
    const detail = errorMessage?.trim();
    return detail ? `评测任务加载失败：${detail}` : '评测任务加载失败，请稍后刷新重试。';
  }
  return '暂无评测任务。请先从数据中心或训练中心启动评测。';
}
