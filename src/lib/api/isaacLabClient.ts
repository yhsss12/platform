'use client';

import { apiGet, apiPost, apiDelete } from '@/lib/api/authClient';
import { buildIsaacBlockStackingConsoleHref } from '@/lib/workspace/isaacBlockStacking';

export interface IsaacLabRuntimeStatus {
  enabled: boolean;
  configured: boolean;
  available: boolean;
  runtimeMode: string;
  isaacLabRoot: string | null;
  isaacLabSh: string | null;
  isaacLabPython: string | null;
  isaacLabVersion: string | null;
  defaultTask: string;
  gpuAvailable: boolean;
  taskRegistered: boolean;
  mimicTaskRegistered?: boolean;
  outputRoot: string | null;
  defaultSeedFile?: string | null;
  defaultSeedAvailable?: boolean;
  stackCubeGenerationReady?: boolean;
  stackCubeIssueCodes?: string[];
  scriptedExpertAvailable?: boolean;
  scriptedExpertReady?: boolean;
  scriptedExpertIssueCodes?: string[];
  issues: string[];
}

export interface IsaacLabSmokeTestResponse {
  jobId: string;
  kind: string;
  status: string;
  runtimePath: string;
  statusUrl: string;
  logPaths: {
    stdout?: string;
    stderr?: string;
  };
}

export interface IsaacLabRunJobStatus {
  jobId: string;
  kind?: string;
  status: string;
  phase?: string;
  message?: string;
  command?: string[];
  keyword?: string;
  taskId?: string;
  datasetFile?: string;
  datasetName?: string;
  datasetAvailable?: boolean;
  datasetId?: string;
  generationMode?: string;
  numDemos?: number;
  totalEpisodes?: number;
  completedEpisodes?: number;
  successfulEpisodes?: number;
  currentEpisode?: number;
  episodeCount?: number;
  progress?: number;
  seed?: number;
  headless?: boolean;
  enableCameras?: boolean;
  videoRequested?: boolean;
  videoAvailable?: boolean;
  videoPath?: string;
  videoNote?: string;
  exitCode?: number;
  timedOut?: boolean;
  stackEnvMatches?: number;
  startedAt?: string;
  finishedAt?: string;
  updatedAt?: string;
  seedSource?: string;
  artifactStatus?: {
    seedHdf5?: boolean;
    annotatedHdf5?: boolean;
    datasetHdf5?: boolean;
    generationManifest?: boolean;
    metricsJson?: boolean;
  };
    liveFrameAvailable?: boolean;
  liveFrameBlack?: boolean;
  latestFramePath?: string;
  previewVideoAvailable?: boolean;
  previewStatus?: 'pending' | 'generating' | 'completed' | 'failed' | string;
  previewVideoPath?: string;
  browserPreviewVideoPath?: string;
  visualPhase?: string;
  visualNumEnvs?: number;
  parallelNumEnvs?: number;
  visualMode?: 'single_env' | 'parallel_overview' | 'replay_preview' | string;
  visualEnvIndex?: number;
  paths?: Record<string, string>;
}

export async function getIsaacLabRuntimeStatus(): Promise<IsaacLabRuntimeStatus> {
  return apiGet<IsaacLabRuntimeStatus>('/workspace/isaac-lab/runtime/status');
}

export async function startIsaacLabSmokeTest(keyword = 'Stack'): Promise<IsaacLabSmokeTestResponse> {
  return apiPost<IsaacLabSmokeTestResponse>('/workspace/isaac-lab/smoke-test', { keyword });
}

export async function getIsaacLabRunJobStatus(jobId: string): Promise<IsaacLabRunJobStatus> {
  return apiGet<IsaacLabRunJobStatus>(`/workspace/isaac-lab/jobs/${encodeURIComponent(jobId)}/status`);
}

export async function getIsaacLabRunJobLog(
  jobId: string,
  stream: 'stdout' | 'stderr' = 'stdout',
  lines = 40
): Promise<string> {
  const res = await apiGet<{ tail: string }>(
    `/workspace/isaac-lab/jobs/${encodeURIComponent(jobId)}/log?stream=${stream}&lines=${lines}`
  );
  return res.tail ?? '';
}

export interface IsaacLabReplayDemoRequest {
  taskId: string;
  datasetFile: string;
  headless: boolean;
  enableCameras: boolean;
  video: boolean;
}

export interface IsaacLabReplayDemoResponse {
  jobId: string;
  kind: string;
  status: string;
  runtimePath?: string;
  statusUrl?: string;
  logPaths?: Record<string, string>;
}

export async function startIsaacLabReplayDemo(
  payload: IsaacLabReplayDemoRequest
): Promise<IsaacLabReplayDemoResponse> {
  return apiPost<IsaacLabReplayDemoResponse>('/workspace/isaac-lab/replay-demo', payload);
}

export async function startIsaacLabGenerateDataset(payload: {
  taskId?: string;
  datasetName: string;
  numDemos: number;
  seed?: number;
  headless?: boolean;
  enableCameras?: boolean;
  generationMode?: string;
  seedDatasetFile?: string;
  seedDatasetId?: string;
  video?: boolean;
  numEnvs?: number;
}): Promise<{
  jobId: string;
  kind: string;
  status: string;
  runtimePath?: string;
  statusUrl?: string;
  logPaths?: Record<string, string>;
}> {
  return apiPost('/workspace/isaac-lab/generate-dataset', {
    taskId: payload.taskId ?? 'Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0',
    datasetName: payload.datasetName,
    numDemos: payload.numDemos,
    seed: payload.seed ?? 0,
    headless: payload.headless ?? true,
    enableCameras: payload.enableCameras ?? true,
    generationMode: payload.generationMode ?? 'mimic_auto',
    seedDatasetFile: payload.seedDatasetFile || undefined,
    seedDatasetId: payload.seedDatasetId || undefined,
    video: payload.video ?? true,
    numEnvs: payload.numEnvs,
  });
}

export function buildIsaacGenerateJobHref(jobId: string, dataId?: string): string {
  return buildIsaacBlockStackingConsoleHref({ jobId, dataId });
}

export function getIsaacLabJobVideoUrl(jobId: string): string {
  return `/api/workspace/isaac-lab/jobs/${encodeURIComponent(jobId)}/video`;
}

export function buildIsaacLabLiveFrameApiPath(jobId: string): string {
  return `/api/workspace/isaac-lab/jobs/${encodeURIComponent(jobId)}/live/latest`;
}

export function getIsaacLabReplayVideoUrl(jobId: string): string {
  return getIsaacLabJobVideoUrl(jobId);
}

export interface IsaacLabImportDemoRequest {
  datasetFile: string;
  displayName: string;
  taskId?: string;
}

export interface IsaacLabImportDemoResponse {
  dataset: import('@/types/benchmark').Dataset;
}

export async function importIsaacLabDemoDataset(
  payload: IsaacLabImportDemoRequest
): Promise<IsaacLabImportDemoResponse> {
  return apiPost<IsaacLabImportDemoResponse>('/workspace/isaac-lab/datasets/import-demo', {
    datasetFile: payload.datasetFile,
    displayName: payload.displayName,
    taskId: payload.taskId ?? 'Isaac-Stack-Cube-Franka-IK-Rel-v0',
  });
}

export type IsaacLabVideoSource = 'replay' | 'preview' | 'videos' | 'converted' | 'none';

export interface IsaacLabDatasetPlaybackInfo {
  videoJobId?: string | null;
  videoSource: IsaacLabVideoSource;
  videoSourceKind?: 'replay' | 'preview' | 'videos' | null;
  videoPath?: string | null;
  rawVideoPath?: string | null;
  browserVideoPath?: string | null;
  codec?: string | null;
  browserCompatible?: boolean;
  transcoded?: boolean;
  transcodeNote?: string | null;
  playable?: boolean;
}

export interface IsaacLabDatasetReplayContext {
  dataset: import('@/types/benchmark').Dataset;
  sourceJobId?: string | null;
  sourceJobStatus?: IsaacLabRunJobStatus | null;
  replayJobs: Array<{
    jobId: string;
    status: string;
    phase?: string;
    message?: string;
    videoAvailable?: boolean;
  }>;
  replayJobId?: string | null;
  replayJobStatus?: string | null;
  replayInProgress: boolean;
  replayFailed: boolean;
  playback?: IsaacLabDatasetPlaybackInfo | null;
  usingPreviewFallback: boolean;
  hasDatasetFile: boolean;
  videoSourceLabel: string;
}

export async function getIsaacLabDatasetReplayContext(
  datasetId: string
): Promise<IsaacLabDatasetReplayContext> {
  return apiGet<IsaacLabDatasetReplayContext>(
    `/workspace/isaac-lab/datasets/${encodeURIComponent(datasetId)}/replay-context`
  );
}

export interface IsaacLabReplayFromDatasetResponse {
  datasetId: string;
  jobId: string;
  kind: string;
  status: string;
  runtimePath?: string;
  statusUrl?: string;
  reused?: boolean;
}

export async function startIsaacLabReplayFromDataset(
  datasetId: string
): Promise<IsaacLabReplayFromDatasetResponse> {
  return apiPost<IsaacLabReplayFromDatasetResponse>(
    `/workspace/isaac-lab/datasets/${encodeURIComponent(datasetId)}/replay`,
    {}
  );
}

export async function deleteIsaacLabDataset(datasetId: string): Promise<{ ok: boolean }> {
  return apiDelete<{ ok: boolean }>(
    `/workspace/isaac-lab/datasets/${encodeURIComponent(datasetId)}`
  );
}
