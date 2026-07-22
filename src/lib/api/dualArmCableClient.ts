'use client';

import { apiGet, apiPost } from '@/lib/api/authClient';

export interface DualArmCableGenerateRequest {
  taskType?: string;
  taskName?: string;
  maxCables?: number;
  numEpisodes?: number;
  seed?: number;
  record?: boolean;
  headless?: boolean;
  stretchMode?: 'ema_jump' | 'fixed_distance' | 'fixed_force';
  releaseMode?: 'three_phase' | 'direct_open' | 'slow_open';
  taskConfigId?: string;
}

export interface DualArmCableGenerateAsyncResponse {
  jobId: string;
  taskType: string;
  status: 'queued' | 'running';
  frameUrl?: string | null;
  statusUrl?: string | null;
}

export interface DualArmCableJobStatusResponse {
  jobId: string;
  taskType: string;
  status: string;
  progress: number | null;
  phase?: string | null;
  maxCables: number;
  succeededCables: number;
  episodeSuccess: boolean;
  videoExists: boolean;
  liveFrameExists?: boolean;
  liveFrameSeq?: number | null;
  liveFrameUpdatedAt?: string | null;
  liveFrameSource?: string | null;
  currentStep?: number | null;
  episodeIndex?: number | null;
  videoPath?: string | null;
  resultPath?: string | null;
  runtimePath?: string | null;
  logPath?: string | null;
  manifestPath?: string | null;
  message: string;
  metrics: {
    episode_success?: boolean;
    num_cables_succeeded?: number;
    max_cables?: number;
    left_contact?: boolean;
    right_contact?: boolean;
    stretch_reached?: boolean;
    sag_m?: number;
    span_m?: number;
    final_sag_m?: number;
    final_span_m?: number;
  };
  frameUrl?: string | null;
  videoUrl?: string | null;
  logUrl?: string | null;
  resultUrl?: string | null;
}

export function buildDualArmCableFrameApiPath(jobId: string): string {
  return `/api/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/frame`;
}

export function buildDualArmCableVideoApiPath(jobId: string): string {
  return `/api/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/video`;
}

export async function generateDualArmCableDataAsync(
  payload: DualArmCableGenerateRequest
): Promise<DualArmCableGenerateAsyncResponse> {
  return apiPost<DualArmCableGenerateAsyncResponse>(
    '/workspace/dual-arm-cable/generate-async',
    payload
  );
}

export async function getDualArmCableJobStatus(
  jobId: string
): Promise<DualArmCableJobStatusResponse> {
  return apiGet<DualArmCableJobStatusResponse>(
    `/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/status`
  );
}

export interface DualArmIlExportProbeResponse {
  jobId: string;
  exportReady: boolean;
  failureReason?: string | null;
  reason?: string | null;
  actionAvailable: boolean;
  observationAvailable: boolean;
  missingFields: string[];
  hdf5Exists: boolean;
  manifestExists: boolean;
  hdf5Path?: string | null;
  manifestPath?: string | null;
  trainable: boolean;
  exportReport: Record<string, unknown>;
}

export interface DualArmIlExportBuildResponse {
  jobId: string;
  status: 'built' | 'already_built';
  manifestPath: string;
  hdf5Path: string;
  message: string;
  manifest?: Record<string, unknown>;
  exportReport?: Record<string, unknown>;
}

export async function probeDualArmIlExport(jobId: string): Promise<DualArmIlExportProbeResponse> {
  return apiGet<DualArmIlExportProbeResponse>(
    `/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/il-export/probe`
  );
}

export async function buildDualArmIlExport(jobId: string): Promise<DualArmIlExportBuildResponse> {
  return apiPost<DualArmIlExportBuildResponse>(
    `/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/il-export/build`,
    {}
  );
}

export async function getDualArmCableJobLog(jobId: string): Promise<{ jobId: string; tail: string }> {
  return apiGet<{ jobId: string; tail: string }>(
    `/workspace/dual-arm-cable/jobs/${encodeURIComponent(jobId)}/log`
  );
}
