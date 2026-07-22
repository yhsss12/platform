/** 评测任务 mock — 策略评测为主线 */

import type { EvaluationSuccessStats } from '@/lib/api/evaluationClient';
import type { PhysicsProxyMode } from '@/lib/mock/physicsProxiesMock';
import { formatEvalConfigWithPhysicsProxy } from '@/lib/mock/physicsProxiesMock';
import {
  isCanonicalEvaluationDisplayName,
  normalizeEvaluationDisplayName,
} from '@/lib/workspace/evaluationDisplay';
import {
  normalizeEvaluationTypeLabel,
  type EvaluationTypeLabel,
} from '@/lib/workspace/evaluationType';

export type EvaluationBackend = 'MuJoCo' | 'Isaac Lab';

export type EvaluationDataSource =
  | '仿真数据'
  | '平台数据'
  | '外部数据'
  | '真实采集数据';

export type EvaluationMode =
  | '策略评测'
  | 'episode 稳定性评测'
  | '数据过程评测'
  | '策略对比'
  | '失败分析';

export type EvaluationTaskStatus = '待评测' | '评测中' | '已完成' | '失败';

export interface EvaluationTaskRow {
  /** @deprecated 请使用 evalJobId；保留兼容旧代码 */
  id: string;
  /** 真实评测 job id（ct_eval_* / eval_* / isaac_eval_*），删除/回放/报告必用 */
  evalJobId?: string;
  jobId?: string;
  /** workspace_jobs 主键，待评测记录删除 fallback */
  workspaceJobId?: number | string;
  name: string;
  taskName?: string;
  evaluationMode: EvaluationMode;
  relatedTask: string;
  checkpoint: string;
  modelType: string;
  dataVolume: string;
  evalBackend: EvaluationBackend;
  evalRounds: number;
  status: EvaluationTaskStatus;
  successRate: number | null;
  createdAt: string;
  /** 后端原始 evaluationMode（trained_model_evaluation / expert_policy_evaluation / episode_stability ...） */
  evaluationModeApi?: string;
  /** 统一评测类型 key */
  evaluationType?: 'expert_policy' | 'model' | 'dataset';
  /** 统一评测类型展示 */
  evaluationTypeLabel?: EvaluationTypeLabel;
  evaluationObject?: string;
  /** 后端原始名称（request/taskName 等），仅用于内部信息展示 */
  rawName?: string;
  /** createdAt 原始 ISO（用于派生规范展示名） */
  createdAtIso?: string;
  /** 过程评测详情 query（高级模式） */
  dataName?: string;
  /** 离线数据集评测 ID（workspace job metadata） */
  datasetId?: string;
  metrics?: string[];
  resultSummary?: string;
  /** @deprecated 保留兼容 */
  targetLabel?: string;
  dataSource?: EvaluationDataSource;
  processScore?: string | null;
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string | null;
  physicsProxyError?: string;
  physicsProxySpeedup?: string;
  physicsProxyErrorThreshold?: number;
  highFidelityReviewRatio?: number | string;
  highFidelityReviewEnabled?: boolean;
  /** CableThreading 真实后端记录 */
  taskType?: 'cable_threading' | 'dual_arm_cable_manipulation';
  cableModel?: string;
  difficulty?: string;
  policy?: string;
  robot?: string;
  everSuccessRate?: number;
  resultPath?: string;
  evalCsvPath?: string;
  failuresPath?: string;
  videoPath?: string;
  videoSizeBytes?: number;
  videoExists?: boolean;
  videoJobId?: string;
  evalVideoExists?: boolean;
  evalVideoPath?: string;
  evalVideoSizeBytes?: number;
  timelineExists?: boolean;
  timelinePath?: string;
  aggregate?: Record<string, unknown>;
  backendJobStatus?: string;
  backendCommand?: string;
  /** Dual-arm episode 稳定性评测 */
  dualArmEvalSeeds?: number[];
  dualArmEvalCurrentEpisode?: number;
  dualArmEvalTotalEpisodes?: number;
  dualArmMeanFinalSag?: number;
  dualArmMeanFinalSpan?: number;
  dualArmMaxCables?: number;
  source?: 'real' | 'demo';
  requestedEpisodes?: number;
  completedEpisodes?: number;
  currentEpisode?: number;
  totalEpisodes?: number;
  progress?: number;
  progressPercent?: number;
  progressLabel?: string;
  templateDisplayName?: string;
  runner?: string;
  runtimePath?: string;
  successStats?: EvaluationSuccessStats;
  updatedAt?: string;
  startedAt?: string;
  finishedAt?: string;
}

/** @deprecated 使用 EvaluationTaskRow */
export type EvaluationRecordRow = EvaluationTaskRow;

export function formatSampleCountForDisplay(value: string): string {
  if (/条/.test(value)) return value;
  const match = value.match(/(\d+)\s*demos?/i);
  if (match) return `${match[1]} 条`;
  return value;
}

export interface EvaluationOverviewStats {
  completedCount: number;
  runningCount: number;
  maxSuccessRate: number | null;
}

export function computeEvaluationOverviewStats(tasks: EvaluationTaskRow[]): EvaluationOverviewStats {
  const completedCount = tasks.filter((t) => t.status === '已完成').length;
  const runningCount = tasks.filter((t) => t.status === '评测中').length;
  const rates = tasks.map((t) => t.successRate).filter((r): r is number => r != null);
  const maxSuccessRate = rates.length ? Math.max(...rates) : null;
  return { completedCount, runningCount, maxSuccessRate };
}

export const evaluationTasksMock: EvaluationTaskRow[] = [
  {
    id: 'eval-cable-001',
    name: '线缆穿杆 · 专家策略评测',
    evaluationMode: '策略评测',
    relatedTask: '线缆穿杆',
    checkpoint: '—',
    modelType: 'scripted',
    dataVolume: '50 条',
    evalBackend: 'MuJoCo',
    evalRounds: 50,
    status: '已完成',
    successRate: 88.0,
    createdAt: '2026-05-31 14:22',
    metrics: ['成功率', '平均耗时', '碰撞次数', '轨迹误差', '失败案例', '视频回放'],
    resultSummary: '50 次评测完成，穿线成功率稳定。',
  },
  {
    id: 'eval-dual-arm-001',
    name: '线缆整理 · 稳定性评测',
    evaluationMode: 'episode 稳定性评测',
    relatedTask: '线缆整理',
    checkpoint: '—',
    modelType: 'episode_stability',
    dataVolume: '20 条',
    evalBackend: 'MuJoCo',
    evalRounds: 20,
    status: '已完成',
    successRate: 75.0,
    createdAt: '2026-06-01 11:30',
    metrics: ['成功率', 'episode 稳定性', '失败案例'],
    resultSummary: '双臂协作 episode 稳定性评测完成。',
  },
];

/** @deprecated 使用 evaluationTasksMock */
export const evaluationRecordsMock = evaluationTasksMock;

export const evaluationModeOptions = [
  '专家策略评测',
  '模型评测',
  '数据集评测',
] as const;

export const evaluationBackendOptions: EvaluationBackend[] = ['MuJoCo', 'Isaac Lab'];

export const evaluationDataSourceOptions = [
  '仿真数据',
  '平台数据',
  '外部数据',
  '真实采集数据',
] as const;

export const evaluationStatusOptions = ['待评测', '评测中', '已完成', '失败'] as const;

export const evaluationTemplateOptions = [
  '线缆穿杆',
  '螺母装配',
  '线缆整理',
  '物块堆叠',
] as const;

export const evaluationTaskConfigOptions = ['default', 'randomized', 'hard'] as const;
export const evaluationRoundOptions = [10, 50, 100] as const;

export const evaluationDataOptions = [
  'cable-threading-demo-50',
  'dual-arm-cable-train-20',
  'nut-assembly-mimicgen-20',
] as const;

export const checkpointEvalMetricOptions = [
  '成功率',
  '平均耗时',
  '碰撞次数',
  '轨迹误差',
  '失败案例',
  '视频回放',
] as const;

export const processLevelMetricOptions = [
  '进度预测',
  '成功概率',
  '失败节点',
  '轨迹质量',
] as const;

/** @deprecated */
export const unifiedEvaluationMetricOptions = checkpointEvalMetricOptions;

/** @deprecated */
export const onlineSceneOptions = ['精密装配工位', '线缆插接工作台', '生化实验台'] as const;
export const onlineRobotOptions = ['双臂协作机器人', '六轴工业机械臂'] as const;
export const onlinePolicyOptions = ['ACT', 'DP3', 'Diffusion Policy', 'OpenVLA'] as const;

export function evaluationTaskStatusBadge(
  status: EvaluationTaskStatus
): 'active' | 'running' | 'draft' | 'failed' {
  switch (status) {
    case '已完成':
      return 'active';
    case '评测中':
      return 'running';
    case '待评测':
      return 'draft';
    case '失败':
      return 'failed';
  }
}

export function formatSuccessRate(status: EvaluationTaskStatus, rate: number | null): string {
  if (status === '评测中') return '评测中';
  if (rate == null) return '—';
  return `${rate}%`;
}

export function formatEvalConfigList(row: { evalBackend: string; evalRounds: number }): string {
  return `${row.evalBackend} · ${row.evalRounds}次`;
}

/** 评测中心列表 — 任务业务名称（模型名称 / 评测名称） */
export function formatEvaluationTaskListName(row: EvaluationTaskRow): string {
  const isDatasetEval =
    row.evaluationMode === '数据过程评测' ||
    Boolean(row.datasetId) ||
    Boolean(row.dataName?.trim()) ||
    /离线数据集评测/i.test(row.name);
  if (isDatasetEval) {
    const stripped = row.name.replace(/\s*[·•]\s*(eval_|ct_eval_|isaac_eval_)[^\s]+$/i, '').trim();
    return stripped || row.name.trim() || '离线数据集评测';
  }

  const userTaskName = row.taskName?.trim() || row.rawName?.trim();
  if (userTaskName) return userTaskName;

  const displayName = row.name?.trim();
  if (displayName && !isCanonicalEvaluationDisplayName(displayName)) {
    return displayName;
  }

  return normalizeEvaluationDisplayName({
    displayName: row.name,
    taskType: row.taskType ?? null,
    evaluationMode: row.evaluationModeApi ?? null,
    createdAtIso: row.createdAtIso ?? null,
    evalJobId: row.evalJobId ?? row.jobId ?? row.id,
  });
}

/** 评测中心列表 — 评测类型 */
export function formatEvaluationListType(row: EvaluationTaskRow): EvaluationTypeLabel {
  if (row.evaluationTypeLabel) {
    return row.evaluationTypeLabel;
  }
  return normalizeEvaluationTypeLabel({
    evaluationType: row.evaluationType,
    evaluationMode: row.evaluationModeApi ?? row.evaluationMode,
    evaluationObject: row.evaluationObject,
    modelAssetId: row.checkpoint !== 'scripted' ? row.checkpoint : null,
    datasetId: row.datasetId,
    datasetName: row.dataName,
    taskType: row.taskType,
    taskName: row.name,
  });
}

/** 评测中心列表 — 评测对象：数据集/模型名称，专家策略为 — */
export function formatEvaluationObjectList(row: EvaluationTaskRow): string {
  const type = formatEvaluationListType(row);
  if (type === '专家策略评测') return '—';

  if (type === '数据集评测') {
    const fromDataName = row.dataName?.trim();
    if (fromDataName) return fromDataName;
    const fromName = row.name.replace(/^离线数据集评测\s*[·•]\s*/i, '').trim();
    if (fromName && fromName !== row.name) return fromName;
    return row.name.trim() || '—';
  }

  if (row.modelType && row.modelType !== '—' && row.modelType !== '专家策略' && row.modelType !== '已训练模型') {
    return row.modelType;
  }
  if (row.checkpoint && row.checkpoint !== '—' && row.checkpoint !== 'scripted') {
    return row.checkpoint;
  }
  if (row.modelType === '已训练模型') {
    return row.checkpoint && row.checkpoint !== '—' ? row.checkpoint : '已训练模型';
  }
  return '—';
}

/** @deprecated 使用 formatEvaluationListType */
export function formatEvaluationModelTypeList(row: EvaluationTaskRow): string {
  return formatEvaluationListType(row);
}

/** @deprecated 评测列表已移除模型版本列 */
export function formatEvaluationModelVersionList(row: EvaluationTaskRow): string {
  if (row.evaluationMode === 'episode 稳定性评测' || row.taskType === 'dual_arm_cable_manipulation') {
    return 'episode 稳定性';
  }
  if (!row.checkpoint || row.checkpoint === '—') return '—';
  return row.checkpoint;
}

/** @deprecated 评测列表已移除模型类型列，请用 formatEvaluationListType */
export function formatEvaluationModelTypeListLegacy(row: EvaluationTaskRow): string {
  if (!row.modelType || row.modelType === '—') return '—';
  return row.modelType;
}

/** @deprecated 评测列表已移除样本数量列 */
export function formatEvaluationSampleCountList(row: EvaluationTaskRow): string {
  return formatSampleCountForDisplay(row.dataVolume);
}

/** 评测中心列表 — 成功率列 */
export function formatEvaluationSuccessRateList(
  status: EvaluationTaskStatus,
  rate: number | null
): string {
  if (status === '评测中') return '评测中';
  if (rate == null) return '—';
  return `${rate}%`;
}

/** 评测模式筛选匹配 */
export function matchesEvaluationModeFilter(row: EvaluationTaskRow, modeFilter: string): boolean {
  if (!modeFilter) return true;
  return formatEvaluationListType(row) === modeFilter;
}

export function formatEvalConfig(row: {
  evalBackend: string;
  evalRounds: number;
  physicsProxyMode?: PhysicsProxyMode;
}): string {
  return formatEvalConfigWithPhysicsProxy(row);
}

/** 判断是否为 workspace_jobs 持久化的真实评测 job（非 mock 种子） */
export function isRealEvaluationBackendJob(row: EvaluationTaskRow): boolean {
  if (row.backendJobStatus) return true;
  return /^(eval_|ct_eval_)/.test(row.id);
}

/** 首页 mock 种子：排除过程分析类，保留策略/episode 稳定性演示 */
function isHomeMockEvaluationRow(row: EvaluationTaskRow): boolean {
  if (isRealEvaluationBackendJob(row)) return true;
  if (row.evaluationMode === '数据过程评测' || row.evaluationMode === '失败分析') {
    return false;
  }
  return row.evaluationMode === '策略评测' || row.evaluationMode === 'episode 稳定性评测';
}

/** 首页任务列表：真实 DB job 全量展示；mock 按 evaluationMode 过滤 */
export function filterHomeEvaluationTasks(tasks: EvaluationTaskRow[]): EvaluationTaskRow[] {
  return tasks.filter(isHomeMockEvaluationRow);
}

/** 评测任务类型标签（展示用，不参与过滤） */
export function getEvaluationTaskTypeTags(row: EvaluationTaskRow): string[] {
  const tags: string[] = [];
  if (row.taskType === 'cable_threading' || row.id.startsWith('ct_eval_')) {
    tags.push('cable_threading');
  }
  if (row.taskType === 'dual_arm_cable_manipulation' || row.id.startsWith('eval_')) {
    tags.push('dual_arm_cable_manipulation');
  }
  if (row.evaluationMode === '策略评测') tags.push('策略评测');
  if (row.evaluationMode === 'episode 稳定性评测') tags.push('episode_stability');
  if (row.evaluationMode === 'episode 稳定性评测') tags.push('过程稳定性评测');
  if (isRealEvaluationBackendJob(row)) tags.push('真实 job');
  return [...new Set(tags)];
}

export interface EvaluationReportDetail {
  evalId: string;
  title: string;
  relatedTask: string;
  modelVersion: string;
  modelType: string;
  sampleCount: string;
  evalBackend: string;
  evalRounds: number;
  successRate: number | null;
  avgDurationSec: number;
  collisionCount: number;
  failureSummaries: string[];
  conclusion: string;
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string | null;
  physicsProxyError?: string;
  physicsProxySpeedup?: string;
  physicsProxyErrorThreshold?: number;
  highFidelityReviewRatio?: number | string;
}

const reportMetricsByCheckpoint: Record<
  string,
  Pick<EvaluationReportDetail, 'avgDurationSec' | 'collisionCount' | 'failureSummaries' | 'conclusion'>
> = {};

export function findEvaluationTaskById(
  evalId: string,
  tasks: EvaluationTaskRow[]
): EvaluationTaskRow | undefined {
  return tasks.find((t) => t.id === evalId);
}

export function buildEvaluationReport(
  row: EvaluationTaskRow,
  extraFailures: { failedStage: string; reason: string }[] = []
): EvaluationReportDetail {
  const metrics = reportMetricsByCheckpoint[row.checkpoint];
  const taskFailures = extraFailures.filter((f) => f.failedStage || f.reason);
  const failureSummaries =
    metrics?.failureSummaries ??
    (taskFailures.length
      ? taskFailures.map((f) => `${f.failedStage}：${f.reason}`)
      : row.resultSummary
        ? [row.resultSummary]
        : ['暂无失败案例摘要']);

  return {
    evalId: row.id,
    title: `${row.relatedTask} · 策略评测报告`,
    relatedTask: row.relatedTask,
    modelVersion: row.checkpoint,
    modelType: row.modelType,
    sampleCount: formatSampleCountForDisplay(row.dataVolume),
    evalBackend: row.evalBackend,
    evalRounds: row.evalRounds,
    successRate: row.successRate,
    avgDurationSec: metrics?.avgDurationSec ?? 50,
    collisionCount: metrics?.collisionCount ?? 0,
    failureSummaries,
    conclusion:
      metrics?.conclusion ??
      (row.resultSummary || '评测已完成，详细指标请结合回放与失败案例进一步分析。'),
    physicsProxyMode: row.physicsProxyMode,
    physicsProxyModel: row.physicsProxyModel,
    physicsProxyError: row.physicsProxyError,
    physicsProxySpeedup: row.physicsProxySpeedup,
    physicsProxyErrorThreshold: row.physicsProxyErrorThreshold,
    highFidelityReviewRatio: row.highFidelityReviewRatio,
  };
}
