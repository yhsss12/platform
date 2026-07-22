'use client';

import { apiGet, apiPost } from '@/lib/api/authClient';

export interface IsaacSimFrankaPickPlaceGenerateRequest {
  taskId?: string;
  episodes?: number;
  seed?: number;
  saveVideo?: boolean;
  saveTrajectory?: boolean;
  headless?: boolean;
  taskConfigId?: string;
}

export interface IsaacSimFrankaPickPlaceGenerateAsyncResponse {
  jobId: string;
  taskId: string;
  status: string;
  message: string;
  statusUrl?: string | null;
  videoUrl?: string | null;
}

export interface IsaacSimFrankaPickPlaceJobStatusResponse {
  jobId: string;
  taskId: string;
  status: string;
  progress?: number | null;
  totalEpisodes?: number | null;
  completedEpisodes?: number | null;
  successEpisodes?: number | null;
  failedEpisodes?: number | null;
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
}

export function buildIsaacSimFrankaPickPlaceVideoApiPath(jobId: string, episode = 'ep_000001'): string {
  return `/api/workspace/isaacsim-franka-pick-place/jobs/${encodeURIComponent(jobId)}/video?episode=${encodeURIComponent(episode)}`;
}

export async function generateIsaacSimFrankaPickPlaceDataAsync(
  payload: IsaacSimFrankaPickPlaceGenerateRequest
): Promise<IsaacSimFrankaPickPlaceGenerateAsyncResponse> {
  return apiPost<IsaacSimFrankaPickPlaceGenerateAsyncResponse>(
    '/workspace/isaacsim-franka-pick-place/generate-async',
    payload
  );
}

export async function getIsaacSimFrankaPickPlaceJobStatus(
  jobId: string
): Promise<IsaacSimFrankaPickPlaceJobStatusResponse> {
  return apiGet<IsaacSimFrankaPickPlaceJobStatusResponse>(
    `/workspace/isaacsim-franka-pick-place/jobs/${encodeURIComponent(jobId)}/status`
  );
}

export async function getIsaacSimFrankaPickPlaceJobLog(
  jobId: string
): Promise<{ jobId: string; tail: string }> {
  return apiGet<{ jobId: string; tail: string }>(
    `/workspace/isaacsim-franka-pick-place/jobs/${encodeURIComponent(jobId)}/log`
  );
}
