'use client';

import { apiDelete, apiGet, apiPost } from '@/lib/api/authClient';

export type WorkspaceJobType = 'generate' | 'evaluation' | 'training' | 'dataset_build';
export type WorkspaceTaskType = 'cable_threading' | 'dual_arm_cable_manipulation' | 'unknown';

export interface WorkspaceArtifactCounts {
  video: number;
  log: number;
  manifest: number;
  metrics: number;
  checkpoint: number;
  result: number;
  other: number;
}

export interface WorkspaceJobSummary {
  jobId: string;
  jobType: WorkspaceJobType;
  taskType: WorkspaceTaskType;
  taskName?: string | null;
  status: string;
  source: string;
  runner?: string | null;
  createdAt: string;
  updatedAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  runtimePath: string;
  metricsSummary: Record<string, unknown>;
  videoAvailable: boolean;
  reportAvailable: boolean;
  artifactCounts: WorkspaceArtifactCounts;
}

export interface WorkspaceJobListResponse {
  jobs: WorkspaceJobSummary[];
  total: number;
}

export interface WorkspaceJobDetail extends WorkspaceJobSummary {
  metadata: Record<string, unknown>;
  metrics: Record<string, unknown>;
  errorMessage?: string | null;
}

export interface WorkspaceArtifactItem {
  id: number;
  jobId: string;
  artifactType: string;
  name: string;
  filePath: string;
  urlPath?: string | null;
  episodeIndex?: number | null;
  createdAt: string;
  metadata: Record<string, unknown>;
}

export interface WorkspaceJobArtifactsResponse {
  jobId: string;
  artifacts: WorkspaceArtifactItem[];
}

export interface ListWorkspaceJobsParams {
  jobType?: WorkspaceJobType;
  taskType?: WorkspaceTaskType;
  status?: string;
  source?: string;
  limit?: number;
  offset?: number;
}

export interface WorkspaceReindexRequest {
  taskType?: WorkspaceTaskType;
  jobType?: WorkspaceJobType | 'data_generation' | 'all';
  dryRun?: boolean;
  overwrite?: boolean;
  restoreDeleted?: boolean;
}

export interface WorkspaceReindexResponse {
  scanned: number;
  insertedJobs: number;
  updatedJobs: number;
  insertedArtifacts: number;
  skipped: number;
  skippedDeleted?: number;
  errors: string[];
  syncedTrainingJobs?: number;
  syncedTrainingAssets?: number;
  syncedEvalJobs?: number;
  syncErrors?: string[];
  scannedDatasets?: number;
  insertedHdf5Datasets?: number;
  updatedHdf5Datasets?: number;
  insertedDataAssets?: number;
  updatedDataAssets?: number;
  skippedDatasets?: number;
}

function buildQuery(params: ListWorkspaceJobsParams): string {
  const q = new URLSearchParams();
  if (params.jobType) q.set('jobType', params.jobType);
  if (params.taskType) q.set('taskType', params.taskType);
  if (params.status) q.set('status', params.status);
  if (params.source) q.set('source', params.source);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.offset != null) q.set('offset', String(params.offset));
  const s = q.toString();
  return s ? `?${s}` : '';
}

export async function listWorkspaceJobs(
  params: ListWorkspaceJobsParams = {}
): Promise<WorkspaceJobListResponse> {
  return apiGet<WorkspaceJobListResponse>(`/workspace/jobs${buildQuery(params)}`);
}

export async function getWorkspaceJob(jobId: string): Promise<WorkspaceJobDetail> {
  return apiGet<WorkspaceJobDetail>(`/workspace/jobs/${encodeURIComponent(jobId)}`);
}

export async function getWorkspaceJobArtifacts(
  jobId: string
): Promise<WorkspaceJobArtifactsResponse> {
  return apiGet<WorkspaceJobArtifactsResponse>(
    `/workspace/jobs/${encodeURIComponent(jobId)}/artifacts`
  );
}

export async function reindexWorkspaceJobs(
  payload: WorkspaceReindexRequest = {}
): Promise<WorkspaceReindexResponse> {
  return apiPost<WorkspaceReindexResponse>('/workspace/jobs/reindex', payload);
}

export interface WorkspaceJobDeleteResponse {
  success: boolean;
  jobId: string;
  deletedJob: boolean;
  deletedArtifacts: number;
  runtimeDeleted: boolean;
  runtimePath: string;
  canReindexRecover: boolean;
  reason?: string | null;
}

export const WORKSPACE_JOB_DELETE_CONFIRM =
  '确认删除该记录及其真实产物？此操作将删除平台记录，以及 runs 中的仿真视频、日志、结果文件和 episode 产物。删除后无法恢复。';

export function workspaceJobBatchDeleteConfirm(count: number): string {
  return `确认删除选中的 ${count} 条记录及其真实产物？此操作将删除平台记录，以及 runs 中对应的视频、日志、结果文件和 episode 产物。删除后无法恢复。`;
}

export async function deleteWorkspaceJob(jobId: string): Promise<WorkspaceJobDeleteResponse> {
  return apiDelete<WorkspaceJobDeleteResponse>(
    `/workspace/jobs/${encodeURIComponent(jobId)}`
  );
}

export async function deleteWorkspaceJobsBatch(jobIds: string[]): Promise<{
  deleted: string[];
  failed: Array<{ jobId: string; error: string }>;
}> {
  const deleted: string[] = [];
  const failed: Array<{ jobId: string; error: string }> = [];
  for (const jobId of jobIds) {
    try {
      await deleteWorkspaceJob(jobId);
      deleted.push(jobId);
    } catch (err) {
      failed.push({
        jobId,
        error: err instanceof Error ? err.message : '删除失败',
      });
    }
  }
  return { deleted, failed };
}
