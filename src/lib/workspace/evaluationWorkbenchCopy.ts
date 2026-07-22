/** 评测中心进入的执行 / 回放 / 详情页统一页面标题（非数据集回放） */

export const EVALUATION_WORKBENCH_TITLE = '评测工作台';

export const EVALUATION_WORKBENCH_SUBTITLE =
  '查看评测任务运行状态、评测画面、结果指标与失败诊断信息。';

export function getEvaluationWorkbenchTitle(): string {
  return EVALUATION_WORKBENCH_TITLE;
}

export function getEvaluationWorkbenchSubtitle(): string {
  return EVALUATION_WORKBENCH_SUBTITLE;
}

export function mapEvaluationJobStatusLabel(status: string | null | undefined): string {
  if (!status || status === 'loading') return '加载中…';
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'queued') return '排队中';
  if (status === 'running') return '运行中';
  return status;
}
