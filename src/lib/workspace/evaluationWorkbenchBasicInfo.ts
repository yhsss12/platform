import { mapEvaluationJobStatusLabel } from '@/lib/workspace/evaluationWorkbenchCopy';
import {
  normalizeEvaluationTypeKey,
  normalizeEvaluationTypeLabel,
  type EvaluationTypeLabel,
} from '@/lib/workspace/evaluationType';
import {
  getTaskDisplayName,
  getTaskTemplateDisplayName,
} from '@/lib/workspace/taskDisplayNames';
import { resolveEvaluationReportRobotDisplay } from '@/lib/workspace/evaluationReport';

export interface EvaluationWorkbenchBasicInfo {
  taskName: string;
  evaluationTypeLabel: EvaluationTypeLabel;
  evaluationObjectLabel: string;
  simulationPlatform: string;
  statusLabel: string;
  robotType?: string;
  modelAssetName?: string;
  datasetName?: string;
  associatedTaskName?: string;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function pickString(...values: unknown[]): string {
  for (const value of values) {
    if (value == null) continue;
    const text = String(value).trim();
    if (text) return text;
  }
  return '';
}

function formatSimulationPlatform(raw: string, taskType: string): string {
  const normalized = raw.trim().toLowerCase().replace(/-/g, '_').replace(/\s+/g, '_');
  const labels: Record<string, string> = {
    mujoco: 'MuJoCo',
    isaac_lab: 'Isaac Lab',
    isaaclab: 'Isaac Lab',
    isaac_sim: 'Isaac Sim',
    isaacsim: 'Isaac Sim',
    isaac: 'Isaac Lab',
  };
  if (labels[normalized]) return labels[normalized];
  if (raw.trim()) return raw.trim();
  if (['block_stacking', 'isaac_block_stacking', 'isaaclab_franka_stack_cube', 'stacking'].includes(taskType)) {
    return 'Isaac Lab';
  }
  if (['cable_threading', 'dual_arm_cable_manipulation'].includes(taskType)) {
    return 'MuJoCo';
  }
  return '—';
}

function resolveAssociatedTaskName(taskType: string, taskTemplateId: string): string {
  const fromTemplate = getTaskTemplateDisplayName(taskTemplateId);
  if (fromTemplate) return fromTemplate;
  const fromType = getTaskDisplayName(taskType);
  return fromType === '—' ? '' : fromType;
}

function resolveEvaluationObjectLabel(
  evaluationObject: string,
  evaluationTypeLabel: EvaluationTypeLabel
): string {
  const normalized = evaluationObject.toLowerCase();
  if (normalized === 'trained_model' || normalized === 'model') return '已训练模型';
  if (normalized === 'dataset') return '数据集';
  if (normalized === 'expert_policy' || normalized === 'expert') return '专家策略';
  if (evaluationTypeLabel === '模型评测') return '已训练模型';
  if (evaluationTypeLabel === '数据集评测') return '数据集';
  return '专家策略';
}

function mapWorkbenchStatusLabel(status: string | null | undefined): string {
  const normalized = String(status ?? '').trim().toLowerCase();
  if (normalized === 'pending') return '待评测';
  if (normalized === 'canceled' || normalized === 'cancelled') return '已取消';
  if (normalized === 'running') return '评测中';
  return mapEvaluationJobStatusLabel(status);
}

function normalizeWorkbenchBasicInfo(
  workbench: Record<string, unknown>,
  status: Record<string, unknown>
): EvaluationWorkbenchBasicInfo | null {
  const taskName = pickString(workbench.taskName, status.taskName);
  const evaluationTypeLabel = pickString(workbench.evaluationTypeLabel, status.evaluationTypeLabel);
  if (!taskName || !evaluationTypeLabel) return null;
  const label = evaluationTypeLabel as EvaluationTypeLabel;
  const modelAssetName = pickString(workbench.modelAssetName, status.modelAssetName) || undefined;
  const datasetName = pickString(workbench.datasetName, status.datasetName) || undefined;
  return {
    taskName,
    evaluationTypeLabel: label,
    evaluationObjectLabel: pickString(
      workbench.evaluationObjectLabel,
      resolveEvaluationObjectLabel(
        pickString(workbench.evaluationObject, status.evaluationObject),
        label
      )
    ),
    simulationPlatform: pickString(workbench.simulationPlatform, status.simulationPlatform) || '—',
    statusLabel: pickString(
      workbench.statusLabel,
      mapWorkbenchStatusLabel(String(status.status ?? ''))
    ),
    robotType: pickString(workbench.robotType, status.robotType) || undefined,
    modelAssetName: label === '模型评测' ? modelAssetName : undefined,
    datasetName: label === '数据集评测' ? datasetName : undefined,
    associatedTaskName: pickString(workbench.associatedTaskName) || undefined,
  };
}

export function resolveEvaluationWorkbenchBasicInfo(input: {
  evalJobId?: string;
  status?: Record<string, unknown> | null;
  result?: Record<string, unknown> | null;
  aggregate?: Record<string, unknown> | null;
  live?: Record<string, unknown> | null;
  metadata?: Record<string, unknown> | null;
  listItem?: Record<string, unknown> | null;
  fallbackTaskName?: string;
}): EvaluationWorkbenchBasicInfo {
  const status = asRecord(input.status);
  const result = asRecord(input.result);
  const aggregate = asRecord(input.aggregate);
  const live = asRecord(input.live ?? status.live);
  const listItem = asRecord(input.listItem);
  const metadata = asRecord(input.metadata ?? status.metadata ?? result.metadata);

  for (const source of [
    asRecord(status.workbenchBasicInfo),
    asRecord(result.workbenchBasicInfo),
  ]) {
    const normalized = normalizeWorkbenchBasicInfo(source, status);
    if (normalized) return normalized;
  }

  const topLevelTaskName = pickString(
    status.taskName,
    listItem.taskName,
    listItem.name,
    result.taskName
  );
  const topLevelTypeLabel = pickString(
    status.evaluationTypeLabel,
    listItem.evaluationTypeLabel
  );
  if (topLevelTaskName && topLevelTypeLabel) {
    const label = topLevelTypeLabel as EvaluationTypeLabel;
    const evaluationObject = pickString(status.evaluationObject, listItem.evaluationObject);
    const modelAssetName = pickString(status.modelAssetName, listItem.modelAssetName) || undefined;
    return {
      taskName: topLevelTaskName,
      evaluationTypeLabel: label,
      evaluationObjectLabel: resolveEvaluationObjectLabel(evaluationObject, label),
      simulationPlatform:
        pickString(status.simulationPlatform) ||
        formatSimulationPlatform('', pickString(status.taskType, listItem.taskType)),
      statusLabel: mapWorkbenchStatusLabel(String(status.status ?? listItem.status ?? '')),
      robotType: pickString(status.robotType) || undefined,
      modelAssetName: label === '模型评测' ? modelAssetName : undefined,
      datasetName: undefined,
      associatedTaskName:
        resolveAssociatedTaskName(
          pickString(status.taskType, listItem.taskType),
          pickString(status.taskTemplateId, listItem.taskTemplateId)
        ) || undefined,
    };
  }

  const evaluationRequest = asRecord(metadata.evaluationRequest ?? status.evaluationRequest);
  const config = asRecord(metadata.config ?? evaluationRequest.config ?? status.config);
  const cableThreading = asRecord(metadata.cableThreading ?? evaluationRequest.cableThreading);
  const dualArmCable = asRecord(metadata.dualArmCable ?? evaluationRequest.dualArmCable);

  const taskType = pickString(
    status.taskType,
    result.taskType,
    aggregate.taskType,
    listItem.taskType,
    metadata.taskType,
    evaluationRequest.taskType,
    config.taskType
  );

  const taskTemplateId = pickString(
    status.taskTemplateId,
    result.taskTemplateId,
    metadata.taskTemplateId,
    evaluationRequest.taskTemplateId,
    config.taskTemplateId
  );

  const typeInput = {
    evaluationType: pickString(
      status.evaluationType,
      listItem.evaluationType,
      metadata.evaluationType,
      evaluationRequest.evaluationType
    ),
    evaluationTypeLabel: pickString(
      status.evaluationTypeLabel,
      listItem.evaluationTypeLabel,
      metadata.evaluationTypeLabel,
      evaluationRequest.evaluationTypeLabel
    ),
    evaluationMode: pickString(
      status.evaluationMode,
      aggregate.evaluationMode,
      listItem.evaluationMode,
      metadata.evaluationMode,
      evaluationRequest.evaluationMode,
      evaluationRequest.productEvaluationMode
    ),
    evaluationObject: pickString(
      status.evaluationObject,
      listItem.evaluationObject,
      metadata.evaluationObject,
      evaluationRequest.evaluationObject
    ),
    modelAssetId: pickString(
      status.modelAssetId,
      listItem.modelAssetId,
      metadata.modelAssetId,
      evaluationRequest.modelAssetId,
      live.modelAssetId
    ),
    modelAssetName: pickString(
      status.modelAssetName,
      listItem.modelAssetName,
      metadata.modelAssetName,
      evaluationRequest.modelAssetName,
      evaluationRequest.modelName,
      metadata.modelName,
      listItem.modelName
    ),
    datasetId: pickString(
      status.datasetId,
      listItem.datasetId,
      metadata.datasetId,
      evaluationRequest.datasetId
    ),
    datasetName: pickString(
      status.datasetName,
      listItem.datasetName,
      metadata.datasetName,
      evaluationRequest.datasetName
    ),
    taskType,
    taskName: pickString(
      listItem.taskName,
      listItem.name,
      evaluationRequest.taskName,
      evaluationRequest.modelName,
      metadata.taskName,
      metadata.displayName,
      metadata.modelName,
      cableThreading.taskName,
      dualArmCable.taskName
    ),
    metadata,
    metrics: asRecord(status.metrics),
  };

  const evaluationTypeLabel = normalizeEvaluationTypeLabel(typeInput);
  const evaluationTypeKey = normalizeEvaluationTypeKey(typeInput);
  const evaluationObject = pickString(
    typeInput.evaluationObject,
    evaluationTypeKey === 'model'
      ? 'trained_model'
      : evaluationTypeKey === 'dataset'
        ? 'dataset'
        : 'expert_policy'
  );

  const associatedTaskName = resolveAssociatedTaskName(taskType, taskTemplateId);

  const taskName =
    pickString(
      listItem.taskName,
      listItem.name,
      evaluationRequest.taskName,
      evaluationRequest.evaluationTaskName,
      evaluationRequest.modelName,
      metadata.taskName,
      metadata.displayName,
      metadata.templateDisplayName,
      metadata.modelName,
      cableThreading.taskName,
      cableThreading.modelName,
      dualArmCable.taskName,
      dualArmCable.modelName,
      config.taskName,
      config.modelName,
      input.fallbackTaskName,
      input.evalJobId
    ) || '—';

  const simulationPlatform = formatSimulationPlatform(
    pickString(
      status.simulationPlatform,
      metadata.simulationPlatform,
      config.simulationPlatform,
      config.simulatorBackend,
      metadata.simulatorBackend
    ),
    taskType
  );

  const robotType = resolveEvaluationReportRobotDisplay({
    metadata: {
      ...metadata,
      evaluationRequest,
      config,
      cableThreading,
      dualArmCable,
      robot: pickString(live.robot, metadata.robot, config.robot, cableThreading.robot),
    },
    taskType,
  });

  const modelAssetName = pickString(
    status.modelAssetName,
    listItem.modelAssetName,
    evaluationRequest.modelAssetName,
    evaluationRequest.modelName,
    metadata.modelAssetName,
    metadata.modelName
  );

  const datasetName = pickString(
    status.datasetName,
    listItem.datasetName,
    evaluationRequest.datasetName,
    evaluationRequest.datasetId,
    metadata.datasetName,
    metadata.datasetId
  );

  return {
    taskName,
    evaluationTypeLabel,
    evaluationObjectLabel: resolveEvaluationObjectLabel(evaluationObject, evaluationTypeLabel),
    simulationPlatform,
    statusLabel: mapWorkbenchStatusLabel(String(status.status ?? listItem.status ?? '')),
    robotType: robotType !== '—' ? robotType : undefined,
    modelAssetName: evaluationTypeLabel === '模型评测' && modelAssetName ? modelAssetName : undefined,
    datasetName: evaluationTypeLabel === '数据集评测' && datasetName ? datasetName : undefined,
    associatedTaskName: associatedTaskName || undefined,
  };
}
