'use client';

import { InfoRow } from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import type { EvaluationWorkbenchBasicInfo } from '@/lib/workspace/evaluationWorkbenchBasicInfo';

export function EvaluationWorkbenchBasicInfoRows({ info }: { info: EvaluationWorkbenchBasicInfo }) {
  return (
    <>
      <InfoRow label="任务名称" value={info.taskName || '—'} />
      <InfoRow label="评测类型" value={info.evaluationTypeLabel} />
      {info.associatedTaskName ? (
        <InfoRow label="关联任务" value={info.associatedTaskName} />
      ) : null}
      <InfoRow label="仿真平台" value={info.simulationPlatform || '—'} />
      <InfoRow label="状态" value={info.statusLabel || '—'} />
      <InfoRow label="评测对象" value={info.evaluationObjectLabel || '—'} />
      {info.robotType ? <InfoRow label="机器人" value={info.robotType} /> : null}
      {info.modelAssetName ? <InfoRow label="模型资产" value={info.modelAssetName} /> : null}
      {info.datasetName ? <InfoRow label="数据集" value={info.datasetName} /> : null}
    </>
  );
}
