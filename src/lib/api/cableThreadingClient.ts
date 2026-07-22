'use client';

import { apiGet, apiPost } from '@/lib/api/authClient';

export interface CableThreadingPathInfo {
  path: string;
  exists: boolean;
  sizeBytes?: number | null;
}

export interface CableThreadingGenerateRequest {
  episodes?: number;
  robot?: string;
  cableModel?: string;
  difficulty?: string;
  horizon?: number;
  seed?: number;
  outputFormat?: 'npz' | 'hdf5' | 'lerobot';
  saveHdf5?: boolean;
  saveProcessVideo?: boolean;
  lerobotTaskInstruction?: string;
  lerobotRobot?: string;
  lerobotFps?: number;
  taskConfigId?: string;
}

export interface CableThreadingGenerateAsyncResponse {
  jobId: string;
  taskType: string;
  status: 'running';
  frameUrl: string;
  statusUrl: string;
  command: string;
}

export interface CableThreadingJobStatusResponse {
  jobId: string;
  taskType: string;
  status: string;
  live: Record<string, unknown>;
  paths: {
    npz?: CableThreadingPathInfo;
    hdf5?: CableThreadingPathInfo;
    lerobot?: CableThreadingPathInfo;
    manifest?: CableThreadingPathInfo;
    collectCsv?: CableThreadingPathInfo;
    failures?: CableThreadingPathInfo;
    log?: CableThreadingPathInfo;
    liveStatus?: CableThreadingPathInfo;
    liveFrame?: CableThreadingPathInfo;
    generateVideo?: CableThreadingPathInfo;
    evalCsv?: CableThreadingPathInfo;
    resultsJson?: CableThreadingPathInfo;
    failuresJson?: CableThreadingPathInfo;
    evalVideo?: CableThreadingPathInfo;
    evalBrowserVideo?: CableThreadingPathInfo;
  };
  metrics: {
    finalSuccessRate?: number | null;
    successfulEpisodes?: number | null;
    episodes?: number;
    failedEpisodes?: number | null;
    seed?: number | null;
    failureSummary?: Array<{
      episodeIndex?: number | null;
      seed?: number | null;
      success?: boolean;
      failureReason?: string | null;
    }>;
    successRate?: number | null;
    everSuccessRate?: number | null;
    numEpisodes?: number;
    aggregate?: Record<string, unknown>;
  };
  command?: string;
  startedAt?: string | null;
  generateVideoExists?: boolean;
  generateVideoSizeBytes?: number | null;
  generateVideoPath?: string | null;
  evalVideoExists?: boolean;
  evalVideoSizeBytes?: number | null;
  evalVideoPath?: string | null;
  evalBrowserVideoPath?: string | null;
  evalBrowserVideoExists?: boolean;
  browserVideoPath?: string | null;
  videoResolution?: string | null;
  evalVideoStatus?: string | null;
  videoUrl?: string | null;
  timelineExists?: boolean;
  timelinePath?: string | null;
  timelineUrl?: string | null;
  failedStage?: string | null;
  failureReason?: string | null;
  errorMessage?: string | null;
  logPaths?: {
    stdout?: string | null;
    stderr?: string | null;
    run?: string | null;
  };
  requestedEpisodes?: number | null;
  completedEpisodes?: number | null;
  successfulEpisodes?: number | null;
  failedEpisodes?: number | null;
  recordedVideoCount?: number | null;
  replayUri?: string | null;
  replayUris?: Array<{
    episodeIndex?: number | null;
    uri: string;
    label?: string | null;
    fileName?: string | null;
  }>;
  videoAvailable?: boolean;
  isRepresentativeVideo?: boolean;
  successRate?: number | null;
  warning?: string | null;
  currentEpisodeIndex?: number | null;
  taskName?: string | null;
  evaluationMode?: string | null;
  evaluationObject?: string | null;
  evaluationType?: string | null;
  evaluationTypeLabel?: string | null;
  simulationPlatform?: string | null;
  robotType?: string | null;
  modelAssetName?: string | null;
  workbenchBasicInfo?: {
    taskName?: string;
    evaluationTypeLabel?: string;
    evaluationObjectLabel?: string;
    simulationPlatform?: string;
    statusLabel?: string;
    robotType?: string | null;
    modelAssetName?: string | null;
    datasetName?: string | null;
    associatedTaskName?: string | null;
  } | null;
  selectedMetricIds?: string[];
  metricResults?: Record<
    string,
    {
      metricId: string;
      displayName: string;
      value: number | null;
      formattedValue: string;
      unit?: string;
      available: boolean;
      reason?: string;
      source?: string;
    }
  >;
  runMetrics?: Record<string, unknown>;
  replayContent?: {
    replayContentKind: 'dataset_trajectory_replay' | 'generation_process_preview' | 'evaluation_replay';
    hasHdf5Trajectories: boolean;
    trajectoryCount?: number | null;
    totalEpisodes?: number | null;
    failedEpisodes?: number | null;
    hasGenerationPreview: boolean;
    hasFailures: boolean;
    hasEvaluationResult?: boolean;
    primarySource?: string | null;
    tabs: Array<{ id: string; label: string }>;
    trajectories?: string[];
    failureRecords?: Array<{
      episodeIndex?: number | null;
      seed?: number | null;
      failureReason?: string | null;
      writtenToDataset?: boolean;
    }>;
    debug?: Record<string, unknown>;
    hasRgbObservation?: boolean;
    rgbCameras?: string[];
    trajectoryDisplayMode?: 'rgb_frame_replay' | 'state_trajectory';
  };
}

export interface CableThreadingHdf5TrajectoryMeta {
  demoName: string;
  stepCount: number;
  actionDim?: number | null;
  hasRgbObservation: boolean;
  rgbCameras: string[];
  defaultCamera?: string | null;
  lowDimObsKeys: string[];
  trajectoryDisplayMode: 'rgb_frame_replay' | 'state_trajectory';
  hasActions: boolean;
  hasStates: boolean;
  camera?: string | null;
  displayOrientation?: string | null;
  rawStorageOrientation?: string | null;
  displayTransformApplied?: string | null;
  displayOnlyTransform?: boolean;
  cameraDisplayInfo?: Record<
    string,
    {
      camera: string;
      displayOrientation?: string | null;
      rawStorageOrientation?: string | null;
      displayTransformApplied?: string | null;
      displayOnlyTransform?: boolean;
    }
  >;
}

export interface CableThreadingHdf5TrajectoryStep {
  demoName: string;
  stepIndex: number;
  action: number[];
  obs: Record<string, number[]>;
  reward?: number | null;
  done?: boolean | null;
}

export interface CableThreadingTimelineEvent {
  episode: number;
  frameIndex: number;
  step: number;
  timeSec: number;
  phase: string;
  label: string;
}

export interface CableThreadingTimelineResponse {
  events: CableThreadingTimelineEvent[];
  videoFps?: number;
}

export interface CableThreadingGenerateResponse {
  jobId: string;
  taskType: string;
  status: 'completed' | 'failed';
  command: string;
  paths: {
    npz: CableThreadingPathInfo;
    hdf5: CableThreadingPathInfo;
    manifest: CableThreadingPathInfo;
    collectCsv: CableThreadingPathInfo;
    failures: CableThreadingPathInfo;
    log: CableThreadingPathInfo;
  };
  metrics: {
    finalSuccessRate?: number | null;
    successfulEpisodes?: number | null;
    episodes?: number;
    returnCode?: number;
  };
  stdoutTail: string[];
}

export interface CableThreadingEvaluateRequest {
  episodes?: number;
  robot?: string;
  cableModel?: string;
  difficulty?: string;
  horizon?: number;
  seed?: number;
  policy?: string;
  checkpoint?: string;
  device?: string;
  taskConfigId?: string;
}

export interface CableThreadingEvaluateAsyncResponse {
  evalJobId: string;
  jobId: string;
  taskType: string;
  status: 'queued' | 'running';
  statusUrl: string;
  command: string;
}

export interface CableThreadingEvaluateResponse {
  jobId: string;
  taskType: string;
  status: 'completed' | 'failed';
  command: string;
  paths: {
    evalCsv: CableThreadingPathInfo;
    resultsJson: CableThreadingPathInfo;
    failuresJson: CableThreadingPathInfo;
    log: CableThreadingPathInfo;
  };
  metrics: {
    successRate?: number | null;
    everSuccessRate?: number | null;
    numEpisodes?: number;
    aggregate?: Record<string, unknown>;
    returnCode?: number;
  };
  stdoutTail: string[];
}

export interface CableThreadingVideoRequest {
  episodes?: number;
  robot?: string;
  cableModel?: string;
  difficulty?: string;
  horizon?: number;
  seed?: number;
}

export interface CableThreadingVideoResponse {
  jobId: string;
  taskType: string;
  status: 'completed' | 'failed';
  command: string;
  paths: {
    video: CableThreadingPathInfo;
    log: CableThreadingPathInfo;
  };
  videoExists: boolean;
  videoSizeBytes?: number | null;
  stdoutTail: string[];
}

function wrapError(err: unknown, fallback: string): Error {
  if (err instanceof Error && err.message) return err;
  return new Error(fallback);
}

export async function generateCableThreadingData(
  payload: CableThreadingGenerateRequest
): Promise<CableThreadingGenerateResponse> {
  try {
    return await apiPost<CableThreadingGenerateResponse>(
      '/workspace/cable-threading/generate',
      payload
    );
  } catch (err) {
    throw wrapError(err, '线缆穿杆数据生成请求失败，请检查登录状态与后端服务');
  }
}

export async function generateCableThreadingDataAsync(
  payload: CableThreadingGenerateRequest
): Promise<CableThreadingGenerateAsyncResponse> {
  try {
    return await apiPost<CableThreadingGenerateAsyncResponse>(
      '/workspace/cable-threading/generate-async',
      payload
    );
  } catch (err) {
    throw wrapError(err, '线缆穿杆异步数据生成启动失败，请检查登录状态与后端服务');
  }
}

export async function getCableThreadingJobStatus(
  jobId: string
): Promise<CableThreadingJobStatusResponse> {
  try {
    return await apiGet<CableThreadingJobStatusResponse>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/status`
    );
  } catch (err) {
    throw wrapError(err, '读取线缆穿杆状态失败');
  }
}

export async function getCableThreadingJobLog(
  jobId: string
): Promise<{ jobId: string; tail: string }> {
  try {
    return await apiGet<{ jobId: string; tail: string }>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/log`
    );
  } catch (err) {
    throw wrapError(err, '读取线缆穿杆日志失败');
  }
}

export async function getCableThreadingEvalResult(
  jobId: string
): Promise<Record<string, unknown>> {
  try {
    return await apiGet<Record<string, unknown>>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/result`
    );
  } catch (err) {
    throw wrapError(err, '读取线缆穿杆评测报告失败');
  }
}

export function buildCableThreadingTimelineApiPath(jobId: string): string {
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/timeline`;
}

export async function getCableThreadingJobTimeline(
  jobId: string
): Promise<CableThreadingTimelineResponse> {
  try {
    return await apiGet<CableThreadingTimelineResponse>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/timeline`
    );
  } catch (err) {
    throw wrapError(err, '读取线缆穿杆时间线失败');
  }
}

export function buildCableThreadingFrameApiPath(jobId: string): string {
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/frame`;
}

export async function evaluateCableThreadingPolicy(
  payload: CableThreadingEvaluateRequest
): Promise<CableThreadingEvaluateResponse> {
  try {
    return await apiPost<CableThreadingEvaluateResponse>(
      '/workspace/cable-threading/evaluate',
      payload
    );
  } catch (err) {
    throw wrapError(err, '线缆穿杆评测请求失败，请检查登录状态与后端服务');
  }
}

export async function evaluateCableThreadingPolicyAsync(
  payload: CableThreadingEvaluateRequest
): Promise<CableThreadingEvaluateAsyncResponse> {
  try {
    return await apiPost<CableThreadingEvaluateAsyncResponse>(
      '/workspace/cable-threading/evaluate-async',
      payload
    );
  } catch (err) {
    throw wrapError(err, '线缆穿杆评测任务创建失败，请检查登录状态与后端服务');
  }
}

export async function generateCableThreadingVideo(
  payload: CableThreadingVideoRequest
): Promise<CableThreadingVideoResponse> {
  try {
    return await apiPost<CableThreadingVideoResponse>(
      '/workspace/cable-threading/video',
      payload
    );
  } catch (err) {
    throw wrapError(err, '线缆穿杆视频生成请求失败，请检查登录状态与后端服务');
  }
}

export async function getCableThreadingHdf5TrajectoryMeta(
  jobId: string,
  demoName: string
): Promise<CableThreadingHdf5TrajectoryMeta> {
  try {
    return await apiGet<CableThreadingHdf5TrajectoryMeta>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/hdf5-trajectory/${encodeURIComponent(demoName)}`
    );
  } catch (err) {
    throw wrapError(err, '读取 HDF5 轨迹元数据失败');
  }
}

export async function getCableThreadingHdf5TrajectoryStep(
  jobId: string,
  demoName: string,
  index: number
): Promise<CableThreadingHdf5TrajectoryStep> {
  try {
    return await apiGet<CableThreadingHdf5TrajectoryStep>(
      `/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/hdf5-trajectory/${encodeURIComponent(demoName)}/step?index=${index}`
    );
  } catch (err) {
    throw wrapError(err, '读取 HDF5 轨迹步详情失败');
  }
}
