import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import type { DualArmCableJobStatusResponse } from '@/lib/api/dualArmCableClient';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import { isPendingLocalJobId } from '@/lib/workspace/backendJobIds';

import {
  DUAL_ARM_CABLE_DISPLAY_NAME,
  matchesDualArmCableDisplayName,
} from '@/lib/workspace/taskDisplayNames';

export const DUAL_ARM_CABLE_TASK_NAME = DUAL_ARM_CABLE_DISPLAY_NAME;
export const DUAL_ARM_CABLE_TASK_TYPE = 'dual_arm_cable_manipulation';

export const DUAL_ARM_CABLE_STRETCH_MODES = [
  { value: 'fixed_distance', label: '固定距离' },
  { value: 'fixed_force', label: '固定力控' },
  { value: 'ema_jump', label: 'EMA 跳变' },
] as const;

export const DUAL_ARM_CABLE_RELEASE_MODES = [
  { value: 'three_phase', label: '三阶段释放' },
  { value: 'slow_open', label: '慢速张开' },
  { value: 'direct_open', label: '直接张开' },
] as const;

export const DUAL_ARM_CABLE_DEFAULTS = {
  maxCables: 1,
  seed: 42,
  record: true,
  headless: true,
  stretchMode: 'fixed_distance' as const,
  releaseMode: 'three_phase' as const,
  robot: 'Dual Franka FR3',
  endEffector: 'Robotiq 2F-85',
  scene: '双臂桌面线缆整理工位',
};

export const DUAL_ARM_CABLE_PHASES = [
  '场景初始化',
  '视觉感知',
  '抓取点规划',
  '双臂接近线缆',
  '抓取并拉伸',
  '放置与释放',
  '成功条件判定',
  'episode 完成',
] as const;

export function isDualArmCableTask(taskName: string | undefined | null): boolean {
  return matchesDualArmCableDisplayName(taskName);
}

export function isDualArmCableReplayMode(taskType: string | null | undefined): boolean {
  return taskType === DUAL_ARM_CABLE_TASK_TYPE;
}

/** 判断数据中心记录是否属于双臂线缆真实任务（含缺 taskType 的旧记录） */
export function isDualArmCableDataItem(item: {
  taskType?: string | null;
  taskName?: string | null;
  id: string;
  jobId?: string | null;
  sourceJobId?: string | null;
  simulationId?: string | null;
}): boolean {
  if (item.taskType === DUAL_ARM_CABLE_TASK_TYPE) return true;
  if (isDualArmCableTask(item.taskName)) return true;
  return resolveDualArmBackendJobId(item) != null;
}

const DAC_GEN_JOB_ID_PATTERN = /^dac_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;

/** 从数据中心记录解析真实后端 jobId（dac_gen_*） */
export function resolveDualArmBackendJobId(item: {
  id: string;
  jobId?: string | null;
  sourceJobId?: string | null;
  simulationId?: string | null;
}): string | undefined {
  if (item.jobId && DAC_GEN_JOB_ID_PATTERN.test(item.jobId)) return item.jobId;
  if (DAC_GEN_JOB_ID_PATTERN.test(item.id)) return item.id;
  if (item.sourceJobId && DAC_GEN_JOB_ID_PATTERN.test(item.sourceJobId)) return item.sourceJobId;
  if (item.simulationId && DAC_GEN_JOB_ID_PATTERN.test(item.simulationId)) return item.simulationId;
  const pending = item.id.match(/^dac-pending-(dac_gen_\d{8}_\d{6}_[a-f0-9]{4})$/);
  if (pending) return pending[1];
  return undefined;
}

export function resolveDualArmConsoleJobId(item: WorkspaceDataItem): string | undefined {
  if (item.staleLocalPending) return undefined;
  const realJobId = resolveDualArmBackendJobId(item);
  if (!realJobId || isPendingLocalJobId(realJobId)) return undefined;
  return realJobId;
}

function nowLabel() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function generateDefaultDualArmDataName(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const seq = String(Math.floor(Math.random() * 999) + 1).padStart(3, '0');
  return `${DUAL_ARM_CABLE_TASK_NAME}数据_${date}_${seq}`;
}

export function stretchModeLabel(value?: string): string {
  return DUAL_ARM_CABLE_STRETCH_MODES.find((m) => m.value === value)?.label ?? value ?? '—';
}

export function releaseModeLabel(value?: string): string {
  return DUAL_ARM_CABLE_RELEASE_MODES.find((m) => m.value === value)?.label ?? value ?? '—';
}

export function buildDualArmCableConsoleHref(params: {
  jobId: string;
  dataId?: string;
}): string {
  const search = new URLSearchParams({
    mode: 'data-generation',
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    jobId: params.jobId,
  });
  if (params.dataId) search.set('dataId', params.dataId);
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildDualArmCableReplayHref(params: { jobId: string; datasetId?: string }): string {
  const search = new URLSearchParams({
    replayType: 'dataset',
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    jobId: params.jobId,
  });
  if (params.datasetId) search.set('datasetId', params.datasetId);
  return `/workspace/replay?${search.toString()}`;
}

export function buildDualArmCableReportHref(params: { jobId: string }): string {
  const search = new URLSearchParams({
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    jobId: params.jobId,
    tab: 'metrics',
  });
  return `/workspace/replay?${search.toString()}`;
}

export function createPendingDualArmCableDataItem(
  payload: GenerateDataPayload,
  jobId: string
): WorkspaceDataItem {
  const name = payload.outputName?.trim() || generateDefaultDualArmDataName();
  const maxCables = payload.dualArmMaxCables ?? DUAL_ARM_CABLE_DEFAULTS.maxCables;
  return {
    id: `dac-pending-${jobId}`,
    name,
    taskId: DUAL_ARM_CABLE_TASK_TYPE,
    taskName: DUAL_ARM_CABLE_TASK_NAME,
    jobId,
    backendJobId: jobId,
    simulationId: jobId,
    sourceJobId: jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${maxCables} episode`,
    size: '—',
    status: 'generating',
    generatedAt: nowLabel(),
    creator: '当前用户',
    scene: DUAL_ARM_CABLE_DEFAULTS.scene,
    robot: DUAL_ARM_CABLE_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    saveVideo: payload.saveProcessVideo,
    frameOrTrajectoryCount: `${maxCables} 根线缆 · seed ${payload.seed ?? DUAL_ARM_CABLE_DEFAULTS.seed} · 生成中`,
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    dualArmMaxCables: maxCables,
    dualArmSeed: payload.seed ?? DUAL_ARM_CABLE_DEFAULTS.seed,
    dualArmStretchMode: payload.dualArmStretchMode ?? DUAL_ARM_CABLE_DEFAULTS.stretchMode,
    dualArmReleaseMode: payload.dualArmReleaseMode ?? DUAL_ARM_CABLE_DEFAULTS.releaseMode,
    backendJobStatus: 'queued',
    datasetBuildSupported: undefined,
    qualityStatus: '不可构建',
    contents: [
      ...(payload.saveProcessVideo ? ['过程视频'] : []),
      '运行结果',
      '运行日志',
      '感知结果',
    ],
  };
}

export function dualArmCableDataItemFromJobStatus(
  status: DualArmCableJobStatusResponse,
  payload: GenerateDataPayload
): WorkspaceDataItem {
  const jobId = status.jobId;
  const root = status.runtimePath?.replace(/\/$/, '');
  const maxCables = status.maxCables ?? payload.dualArmMaxCables ?? 1;
  const succeeded = status.succeededCables ?? 0;
  const episodeSuccess = status.episodeSuccess;
  const m = status.metrics ?? {};

  return {
    id: jobId,
    name: payload.outputName?.trim() || generateDefaultDualArmDataName(),
    taskId: DUAL_ARM_CABLE_TASK_TYPE,
    taskName: DUAL_ARM_CABLE_TASK_NAME,
    jobId,
    backendJobId: jobId,
    simulationId: jobId,
    sourceJobId: jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${maxCables} episode`,
    size: status.videoExists ? '含 MP4' : '—',
    status:
      status.status === 'completed'
        ? 'completed'
        : status.status === 'failed'
          ? 'failed'
          : 'generating',
    generatedAt: nowLabel(),
    creator: '当前用户',
    scene: DUAL_ARM_CABLE_DEFAULTS.scene,
    robot: DUAL_ARM_CABLE_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    saveVideo: payload.saveProcessVideo,
    frameOrTrajectoryCount: `${succeeded}/${maxCables} 成功 · seed ${payload.seed ?? DUAL_ARM_CABLE_DEFAULTS.seed}`,
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    dualArmMaxCables: maxCables,
    dualArmSeed: payload.seed ?? DUAL_ARM_CABLE_DEFAULTS.seed,
    dualArmStretchMode: payload.dualArmStretchMode ?? DUAL_ARM_CABLE_DEFAULTS.stretchMode,
    dualArmReleaseMode: payload.dualArmReleaseMode ?? DUAL_ARM_CABLE_DEFAULTS.releaseMode,
    dualArmEpisodeSuccess: episodeSuccess,
    dualArmSucceededCables: succeeded,
    dualArmLeftContact: m.left_contact,
    dualArmRightContact: m.right_contact,
    dualArmStretchReached: m.stretch_reached,
    dualArmSagM: m.sag_m,
    dualArmSpanM: m.span_m,
    dualArmFinalSagM: m.final_sag_m,
    dualArmFinalSpanM: m.final_span_m,
    manifestPath: status.manifestPath ?? (root ? `${root}/results/episode_manifest.json` : undefined),
    episodeResultPath: status.resultPath ?? (root ? `${root}/results/episode_result.json` : undefined),
    generateVideoPath: status.videoPath ?? (root ? `${root}/videos/generate.mp4` : undefined),
    generateVideoExists: status.videoExists,
    backendJobStatus: status.status,
    datasetBuildSupported: undefined,
    qualityStatus: status.status === 'completed' ? undefined : '不可构建',
    contents: [
      ...(payload.saveProcessVideo ? ['过程视频'] : []),
      '运行结果',
      '运行日志',
      '感知结果',
    ],
  };
}

export function dualArmCableFieldsFromTaskParams(
  taskParams: Record<string, string | number>
): {
  dualArmMaxCables: number;
  dualArmStretchMode: string;
  dualArmReleaseMode: string;
} {
  return {
    dualArmMaxCables: Number(taskParams.max_cables ?? DUAL_ARM_CABLE_DEFAULTS.maxCables),
    dualArmStretchMode: String(taskParams.stretch_mode ?? DUAL_ARM_CABLE_DEFAULTS.stretchMode),
    dualArmReleaseMode: String(taskParams.release_mode ?? DUAL_ARM_CABLE_DEFAULTS.releaseMode),
  };
}

export function formatDualArmMetric(value: boolean | number | null | undefined): string {
  if (value === true) return '是';
  if (value === false) return '否';
  if (typeof value === 'number') return value.toFixed(4);
  return '—';
}
