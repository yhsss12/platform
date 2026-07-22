'use client';

import { apiGet, apiDelete } from '@/lib/api/authClient';

export interface IsaacLabFrankaStackCubeJobStatusResponse {
  jobId: string;
  taskId: string;
  status: string;
  progress?: number | null;
  totalEpisodes?: number | null;
  completedEpisodes?: number | null;
  successEpisodes?: number | null;
  failedEpisodes?: number | null;
  generationMode?: string | null;
  phase?: string | null;
  phaseLabel?: string | null;
  phaseStartedAt?: string | null;
  phaseUpdatedAt?: string | null;
  phaseTimings?: Record<string, unknown> | null;
  progressMessage?: string | null;
  errorSummary?: string | null;
  requestedDevice?: string | null;
  resolvedDevice?: string | null;
  cudaVisibleDevices?: string | null;
  isGpuRequested?: boolean | null;
  torchCudaAvailable?: boolean | null;
  outputDir?: string | null;
  datasetId?: string | null;
  runtimeMode?: string | null;
  message?: string;
  videoExists: boolean;
  video_status?: string | null;
  videoStatus?: string | null;
  taskIdValidated?: boolean | null;
  validationError?: string | null;
  videoPath?: string | null;
  episodeId?: string | null;
  episodeManifest?: Record<string, unknown> | null;
  datasetManifest?: Record<string, unknown> | null;
  metrics: Record<string, unknown>;
  statusUrl?: string | null;
  videoUrl?: string | null;
  logUrl?: string | null;
  manifestPath?: string | null;
  liveFrameAvailable?: boolean | null;
  liveFrameBlack?: boolean | null;
  liveFrameExists?: boolean | null;
  enableCameras?: boolean | null;
  liveFrameUrl?: string | null;
}

export function buildIsaacLabFrankaStackCubeVideoApiPath(jobId: string, episode = 'ep_000001'): string {
  return `/api/workspace/isaaclab-franka-stack-cube/jobs/${encodeURIComponent(jobId)}/video?episode=${encodeURIComponent(episode)}`;
}

export function buildIsaacLabFrankaStackCubeLiveFrameApiPath(jobId: string): string {
  return `/api/workspace/isaaclab-franka-stack-cube/jobs/${encodeURIComponent(jobId)}/live-frame`;
}

export async function getIsaacLabFrankaStackCubeJobStatus(
  jobId: string
): Promise<IsaacLabFrankaStackCubeJobStatusResponse> {
  return apiGet<IsaacLabFrankaStackCubeJobStatusResponse>(
    `/workspace/isaaclab-franka-stack-cube/jobs/${encodeURIComponent(jobId)}/status`
  );
}

export async function getIsaacLabFrankaStackCubeJobLog(
  jobId: string
): Promise<{ jobId: string; tail: string }> {
  return apiGet<{ jobId: string; tail: string }>(
    `/workspace/isaaclab-franka-stack-cube/jobs/${encodeURIComponent(jobId)}/log`
  );
}

export async function deleteIsaacLabFrankaStackCubeDataset(jobId: string): Promise<{ ok: boolean }> {
  return apiDelete<{ ok: boolean }>(
    `/workspace/isaaclab-franka-stack-cube/datasets/${encodeURIComponent(jobId)}`
  );
}
