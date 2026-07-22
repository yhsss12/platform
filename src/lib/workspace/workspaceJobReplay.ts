import type { WorkspaceArtifactItem, WorkspaceJobDetail } from '@/lib/api/workspaceJobClient';
import type { CableReplayRecord, CableReplayStatus } from '@/lib/workspace/replayCableThreadingAdapter';
import type { DualArmReplayRecord } from '@/lib/workspace/replayDualArmCableAdapter';
import { DUAL_ARM_CABLE_TASK_NAME, DUAL_ARM_CABLE_TASK_TYPE } from '@/lib/workspace/dualArmCable';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';

function mapWorkspaceStatus(status: string): CableReplayStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed' || status === 'canceled') return 'failed';
  if (status === 'running') return 'running';
  if (status === 'pending' || status === 'queued') return 'pending';
  return 'generating';
}

export function cableReplayRecordFromWorkspaceJob(
  job: WorkspaceJobDetail,
  artifacts: WorkspaceArtifactItem[]
): CableReplayRecord {
  const isEval = job.jobId.startsWith('ct_eval_');
  const metrics = job.metricsSummary ?? {};
  const hasVideoFromArtifacts = artifacts.some(
    (a) => a.artifactType === 'video' && Boolean(a.filePath)
  );
  const hasVideo = job.videoAvailable || hasVideoFromArtifacts;
  const successRate =
    metrics.successRate != null
      ? Number(metrics.successRate)
      : metrics.finalSuccessRate != null
        ? Number(metrics.finalSuccessRate)
        : null;

  return {
    id: job.jobId,
    recordType: isEval ? 'policy_eval' : 'data_generation',
    title: job.taskName ?? job.jobId,
    runNumber: job.jobId,
    status: mapWorkspaceStatus(job.status),
    successRate,
    hasVideo,
    videoJobId: job.jobId,
    frameJobId: job.jobId,
    backendJobId: job.jobId,
    createdAt: job.createdAt,
  };
}

export function dualArmReplayRecordFromWorkspaceJob(
  job: WorkspaceJobDetail,
  artifacts: WorkspaceArtifactItem[]
): DualArmReplayRecord {
  const metrics = job.metricsSummary ?? {};
  const hasVideoFromArtifacts = artifacts.some(
    (a) => a.artifactType === 'video' && Boolean(a.filePath)
  );
  const hasVideo = job.videoAvailable || hasVideoFromArtifacts;
  const dataItem: WorkspaceDataItem = {
    id: job.jobId,
    name: job.taskName ?? job.jobId,
    taskId: job.jobId,
    taskName: DUAL_ARM_CABLE_TASK_NAME,
    simulationId: job.jobId,
    dataCategory: '真实数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: '—',
    size: '—',
    status: job.status === 'completed' ? 'completed' : job.status === 'failed' ? 'failed' : 'generating',
    generatedAt: job.createdAt,
    creator: '平台用户',
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    jobId: job.jobId,
    backendJobId: job.jobId,
    generateVideoExists: hasVideo,
    dualArmEpisodeSuccess: Boolean(metrics.episode_success ?? metrics.episodeSuccess),
    dualArmSucceededCables: Number(metrics.num_cables_succeeded ?? metrics.numCablesSucceeded ?? 0),
  };

  return {
    id: job.jobId,
    title: job.taskName ?? job.jobId,
    runNumber: job.jobId,
    status: job.status === 'completed' ? 'completed' : job.status === 'failed' ? 'failed' : 'running',
    episodeSuccess: Boolean(metrics.episode_success ?? metrics.episodeSuccess),
    hasVideo,
    backendJobId: job.jobId,
    createdAt: job.createdAt,
    dataItem,
  };
}

export function resolveWorkspaceReplayJobId(params: {
  jobId?: string;
  evalId?: string;
  evalJobId?: string;
}): string | null {
  return params.evalJobId ?? params.jobId ?? params.evalId ?? null;
}

export function isPersistedWorkspaceJobId(jobId: string): boolean {
  return (
    jobId.startsWith('ct_gen_') ||
    jobId.startsWith('ct_eval_') ||
    jobId.startsWith('dac_gen_') ||
    jobId.startsWith('eval_') ||
    jobId.startsWith('train_')
  );
}
