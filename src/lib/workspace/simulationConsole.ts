import {
  getEvaluationWorkbenchSubtitle,
  getEvaluationWorkbenchTitle,
} from '@/lib/workspace/evaluationWorkbenchCopy';

export type SimulationConsoleMode = 'data-generation' | 'evaluation' | 'replay';

export function resolveSimulationConsoleMode(
  mode: string | null | undefined
): SimulationConsoleMode | null {
  if (mode === 'data-generation' || mode === 'evaluation' || mode === 'replay') return mode;
  return null;
}

export function buildSimulationConsoleHref(params: {
  mode: SimulationConsoleMode;
  task?: string;
  dataset?: string;
  dataId?: string;
  checkpoint?: string;
  backend?: string;
  rounds?: number;
  physicsProxyMode?: 'off' | 'pinn' | 'hybrid';
  physicsProxyModel?: string;
}): string {
  const q = new URLSearchParams({ mode: params.mode });
  if (params.task) q.set('task', params.task);
  if (params.dataset) q.set('dataset', params.dataset);
  if (params.dataId) q.set('dataId', params.dataId);
  if (params.checkpoint) q.set('checkpoint', params.checkpoint);
  if (params.backend) q.set('backend', params.backend);
  if (params.rounds != null) q.set('rounds', String(params.rounds));
  if (params.physicsProxyMode && params.physicsProxyMode !== 'off') {
    q.set('physicsProxyMode', params.physicsProxyMode);
  }
  if (params.physicsProxyModel) q.set('physicsProxyModel', params.physicsProxyModel);
  return `/workspace/simulation/console?${q.toString()}`;
}

export function getSimulationConsolePageCopy(mode: string | null | undefined) {
  const resolved = resolveSimulationConsoleMode(mode);

  return {
    title: resolved === 'evaluation' ? getEvaluationWorkbenchTitle() : '运行控制台',
    subtitle: resolved === 'evaluation' ? getEvaluationWorkbenchSubtitle() : undefined,
    backLabel:
      resolved === 'evaluation'
        ? '返回评测中心'
        : resolved === 'data-generation'
          ? '返回数据中心'
          : '返回工作台',
    backHref:
      resolved === 'evaluation'
        ? '/workspace/evaluation'
        : resolved === 'data-generation'
          ? '/workspace/data'
          : '/workspace',
    mode: resolved,
  };
}

export function isActiveDataGenerationStatus(status: string): boolean {
  return (
    status === 'generating' ||
    status === 'pending' ||
    status === 'running' ||
    status === 'processing'
  );
}

export function isFailedDataStatus(status: string): boolean {
  return status === 'failed';
}
