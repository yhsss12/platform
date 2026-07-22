import type { DualArmCableJobStatusResponse } from '@/lib/api/dualArmCableClient';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import { listWorkspaceDataItemsForUi } from '@/lib/workspace/workspaceDataSources';
import {
  DUAL_ARM_CABLE_DEFAULTS,
  DUAL_ARM_CABLE_PHASES,
  DUAL_ARM_CABLE_TASK_NAME,
  DUAL_ARM_CABLE_TASK_TYPE,
  formatDualArmMetric,
  resolveDualArmBackendJobId,
} from '@/lib/workspace/dualArmCable';
import type { CableReplayPhaseView } from '@/lib/workspace/replayCableThreadingAdapter';

export type DualArmReplayStatus = 'completed' | 'failed' | 'generating' | 'running' | 'pending';

export interface DualArmReplayRecord {
  id: string;
  title: string;
  runNumber: string;
  status: DualArmReplayStatus;
  episodeSuccess: boolean | null;
  hasVideo: boolean;
  backendJobId: string;
  createdAt: string;
  dataItem: WorkspaceDataItem;
}

export function mapDualArmDataStatus(status: WorkspaceDataItem['status']): DualArmReplayStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'generating') return 'running';
  return 'pending';
}

export function mapDualArmJobStatus(status: string): DualArmReplayStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'running' || status === 'queued') return 'running';
  return 'pending';
}

export function formatDualArmJobIdTimestamp(jobId: string): string {
  const match = jobId.match(/^dac_gen_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
  if (!match) return '—';
  return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}`;
}

export function dualArmReplayStatusLabel(status: DualArmReplayStatus): string {
  switch (status) {
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    case 'running':
      return '运行中';
    case 'generating':
      return '生成中';
    default:
      return '待处理';
  }
}

function dataItemFromJobStatus(status: DualArmCableJobStatusResponse): WorkspaceDataItem {
  const m = status.metrics ?? {};
  const root = status.runtimePath?.replace(/\/$/, '');
  return {
    id: status.jobId,
    name: `${DUAL_ARM_CABLE_TASK_NAME}_${status.jobId}`,
    taskId: DUAL_ARM_CABLE_TASK_TYPE,
    taskName: DUAL_ARM_CABLE_TASK_NAME,
    jobId: status.jobId,
    backendJobId: status.jobId,
    simulationId: status.jobId,
    sourceJobId: status.jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${status.maxCables ?? 1} episode`,
    size: status.videoExists ? '含 MP4' : '—',
    status:
      status.status === 'completed'
        ? 'completed'
        : status.status === 'failed'
          ? 'failed'
          : 'generating',
    generatedAt: formatDualArmJobIdTimestamp(status.jobId),
    creator: '当前用户',
    scene: DUAL_ARM_CABLE_DEFAULTS.scene,
    robot: DUAL_ARM_CABLE_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    dualArmMaxCables: status.maxCables ?? m.max_cables ?? 1,
    dualArmSeed: undefined,
    dualArmEpisodeSuccess: status.episodeSuccess ?? m.episode_success,
    dualArmSucceededCables: status.succeededCables ?? m.num_cables_succeeded,
    dualArmLeftContact: m.left_contact,
    dualArmRightContact: m.right_contact,
    dualArmStretchReached: m.stretch_reached,
    dualArmSagM: m.sag_m,
    dualArmSpanM: m.span_m,
    dualArmFinalSagM: m.final_sag_m,
    dualArmFinalSpanM: m.final_span_m,
    generateVideoExists: status.videoExists,
    generateVideoPath: status.videoPath ?? (root ? `${root}/videos/generate.mp4` : undefined),
    episodeResultPath: status.resultPath ?? (root ? `${root}/results/episode_result.json` : undefined),
    manifestPath: status.manifestPath ?? (root ? `${root}/results/episode_manifest.json` : undefined),
  };
}

export function dualArmReplayRecordFromStatus(
  status: DualArmCableJobStatusResponse,
  runNumber = '001'
): DualArmReplayRecord {
  const dataItem = dataItemFromJobStatus(status);
  return {
    id: status.jobId,
    title: `${DUAL_ARM_CABLE_TASK_NAME} · 过程回放`,
    runNumber,
    status: mapDualArmJobStatus(status.status),
    episodeSuccess: status.episodeSuccess ?? null,
    hasVideo: status.videoExists,
    backendJobId: status.jobId,
    createdAt: formatDualArmJobIdTimestamp(status.jobId),
    dataItem,
  };
}

export function buildDualArmReplayRecords(): DualArmReplayRecord[] {
  const items = listWorkspaceDataItemsForUi().filter(
    (item) => item.taskType === DUAL_ARM_CABLE_TASK_TYPE || resolveDualArmBackendJobId(item)
  );

  return items
    .map((item, index) => {
      const backendJobId = resolveDualArmBackendJobId(item);
      if (!backendJobId) return null;
      return {
        id: backendJobId,
        title: `${DUAL_ARM_CABLE_TASK_NAME} · 过程回放`,
        runNumber: String(index + 1).padStart(3, '0'),
        status: mapDualArmDataStatus(item.status),
        episodeSuccess: item.dualArmEpisodeSuccess ?? null,
        hasVideo: item.generateVideoExists === true && Boolean(item.generateVideoPath),
        backendJobId,
        createdAt: item.generatedAt || formatDualArmJobIdTimestamp(backendJobId),
        dataItem: item,
      } satisfies DualArmReplayRecord;
    })
    .filter((r): r is DualArmReplayRecord => r != null);
}

export function resolveDualArmReplayRecordId(
  records: DualArmReplayRecord[],
  jobId?: string
): string | null {
  if (!jobId) return records[0]?.id ?? null;
  const match = records.find(
    (r) =>
      r.id === jobId ||
      r.backendJobId === jobId ||
      r.dataItem.id === jobId ||
      r.dataItem.sourceJobId === jobId ||
      r.dataItem.simulationId === jobId
  );
  return match?.id ?? null;
}

export function dualArmReplayStaticPhases(): CableReplayPhaseView {
  return {
    hasVideoSync: false,
    syncFootnote: '展示该任务的标准执行阶段，暂未与视频帧级同步。',
    points: DUAL_ARM_CABLE_PHASES.map((label) => ({
      label,
      active: false,
    })),
  };
}

export function dualArmReplayMetrics(item: WorkspaceDataItem): { label: string; value: string }[] {
  return [
    { label: 'episode_success', value: formatDualArmMetric(item.dualArmEpisodeSuccess) },
    {
      label: 'num_cables_succeeded / max_cables',
      value: `${item.dualArmSucceededCables ?? '—'} / ${item.dualArmMaxCables ?? '—'}`,
    },
    { label: 'left_contact', value: formatDualArmMetric(item.dualArmLeftContact) },
    { label: 'right_contact', value: formatDualArmMetric(item.dualArmRightContact) },
    { label: 'stretch_reached', value: formatDualArmMetric(item.dualArmStretchReached) },
    { label: 'sag_m', value: formatDualArmMetric(item.dualArmSagM) },
    { label: 'span_m', value: formatDualArmMetric(item.dualArmSpanM) },
    { label: 'final_sag_m', value: formatDualArmMetric(item.dualArmFinalSagM) },
    { label: 'final_span_m', value: formatDualArmMetric(item.dualArmFinalSpanM) },
  ];
}

export function dualArmReplayLogPaths(jobId: string, runtimePath?: string): { label: string; path: string }[] {
  const root = runtimePath?.replace(/\/$/, '');
  if (!root) return [{ label: 'jobId', path: jobId }];
  return [
    { label: 'run.log', path: `${root}/logs/run.log` },
    { label: 'episode_result.json', path: `${root}/results/episode_result.json` },
    { label: 'latest_grasp.json', path: `${root}/results/steps/step_00/grasp_output/latest_grasp.json` },
    { label: 'perception_attempt_0.log', path: `${root}/results/steps/step_00/perception_attempt_0.log` },
  ];
}

export function dualArmReplayArtifacts(record: DualArmReplayRecord): { label: string; value: string }[] {
  const jobId = record.backendJobId;
  const root = record.dataItem.episodeResultPath?.replace(/\/results\/episode_result\.json$/, '');
  return [
    {
      label: 'generate.mp4',
      value: record.hasVideo
        ? record.dataItem.generateVideoPath ?? (root ? `${root}/videos/generate.mp4` : '已生成')
        : '未生成',
    },
    {
      label: 'episode_result.json',
      value: record.dataItem.episodeResultPath ?? '—',
    },
    {
      label: 'run.log',
      value: root ? `${root}/logs/run.log` : '—',
    },
  ];
}
