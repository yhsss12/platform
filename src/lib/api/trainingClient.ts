'use client';

import { apiDelete, apiGet, apiPost } from '@/lib/api/authClient';
import type { DatasetManifest } from '@/lib/workspace/datasetManifest';
import type {
  TrainingModelAdvancedParams,
  TrainingPretrainedOptions,
  TrainingSeedMode,
} from '@/lib/mock/workspaceTrainingMock';

export type TrainingBackendRequest =
  | 'auto'
  | 'robomimic'
  | 'robomimic_bc'
  | 'isaac_robomimic_bc'
  | 'torch_bc'
  | 'act'
  | 'dt'
  | 'diffusion_policy';

export type TrainingJobBackendStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'backend_unavailable';

export interface TrainingCapabilities {
  foundTrainingScripts: boolean;
  supportedTrainingBackends: string[];
  recommendedBackend: string;
  evidence: string[];
}

export type TrainingNodeStatus =
  | 'available'
  | 'busy'
  | 'unreachable'
  | 'misconfigured'
  | 'placeholder';

export interface TrainingNodeGpuInfo {
  name?: string | null;
  memoryTotalMb?: number | null;
  memoryUsedMb?: number | null;
  memoryFreeMb?: number | null;
  memoryUsedRatio?: number | null;
}

export interface TrainingNodeListItem {
  nodeId: string;
  label: string;
  deviceLabel: string;
  trainingNodeDisplayName?: string;
  host?: string | null;
  executionMode: string;
  description?: string;
  status: TrainingNodeStatus;
  statusLabel: string;
  message?: string;
  selectable?: boolean;
  sshTarget?: string | null;
  workdir?: string | null;
  gpuModel?: string | null;
  gpuMemoryGb?: number | null;
  gpu?: TrainingNodeGpuInfo | null;
}

export interface CreateTrainingJobRequest {
  datasetId: string;
  datasetIds?: string[];
  datasetManifestPath?: string;
  datasetManifest?: DatasetManifest;
  datasetManifests?: DatasetManifest[];
  modelTypeId?: string;
  downstreamModelType: string;
  trainingBackend: TrainingBackendRequest;
  dataFormat: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  device: string;
  deviceLabel?: string;
  trainingNodeId?: string;
  seed?: number;
  seedMode?: TrainingSeedMode;
  pretrained?: TrainingPretrainedOptions;
  taskName?: string;
  saveFinal?: boolean;
  saveBest?: boolean;
  checkpointIntervalEpochs?: number | null;
}

export interface CreateTrainingJobResponse {
  trainJobId: string;
  status: TrainingJobBackendStatus;
  message: string;
}

export interface TrainingJobStatus {
  trainJobId: string;
  status: TrainingJobBackendStatus;
  progress: number;
  epoch: number;
  totalEpochs: number;
  loss: number | null;
  checkpointExists: boolean;
  checkpointPath: string | null;
  modelAssetId: string | null;
  message: string;
  datasetId?: string | null;
  datasetName?: string | null;
  downstreamModelType?: string | null;
  trainingBackend?: string | null;
  dataFormat?: string | null;
  device?: string | null;
  deviceLabel?: string | null;
  trainingNodeId?: string | null;
  trainingNodeDisplayName?: string | null;
  createdAt?: string | null;
  taskName?: string | null;
}

export interface TrainingJobListItem {
  trainJobId: string;
  status: TrainingJobBackendStatus;
  datasetId?: string | null;
  datasetName?: string | null;
  downstreamModelType?: string | null;
  trainingBackend?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
  checkpointExists?: boolean;
  modelAssetId?: string | null;
  epoch?: number | null;
  totalEpochs?: number | null;
  loss?: number | null;
  message?: string | null;
  dataFormat?: string | null;
  deviceLabel?: string | null;
  trainingNodeId?: string | null;
  trainingNodeDisplayName?: string | null;
  taskName?: string | null;
  progress?: number | null;
  lossHistory?: unknown;
  bestLoss?: number | null;
  finalLoss?: number | null;
  runtimeAvailable?: boolean;
}

export interface TrainingJobDeleteResponse {
  trainJobId: string;
  deleted: boolean;
  deletedAt?: string | null;
}

export const TRAINING_JOB_DELETE_CONFIRM =
  '确认删除该训练任务及其真实产物？此操作将删除平台记录、模型资产索引，以及 runs 中的 checkpoint、日志与产物目录。删除后无法恢复。';

export function trainingJobBatchDeleteConfirm(count: number): string {
  return `确认删除选中的 ${count} 条训练任务及其真实产物？此操作将删除平台记录、模型资产索引，以及 runs 中对应的 checkpoint、日志与产物目录。进行中的任务会先停止再删除。删除后无法恢复。`;
}

export interface TrainingJobModelResponse {
  trainJobId: string;
  ready: boolean;
  modelManifest?: Record<string, unknown> | null;
  checkpointPath?: string | null;
}

export function getTrainingCapabilities(): Promise<TrainingCapabilities> {
  return apiGet<TrainingCapabilities>('/workspace/training/capabilities');
}

export function listTrainingNodes(refresh = false): Promise<{ nodes: TrainingNodeListItem[] }> {
  const query = refresh ? '?refresh=true' : '';
  return apiGet<{ nodes: TrainingNodeListItem[] }>(`/workspace/training/nodes${query}`);
}

export function getTrainingNodeStatus(
  nodeId: string,
  refresh = false
): Promise<{ node: TrainingNodeListItem }> {
  const query = refresh ? '?refresh=true' : '';
  return apiGet<{ node: TrainingNodeListItem }>(
    `/workspace/training/nodes/${encodeURIComponent(nodeId)}${query}`
  );
}

export interface TrainingJobListResponse {
  jobs: TrainingJobListItem[];
  total: number;
}

export interface ListTrainingJobsParams {
  limit?: number;
  offset?: number;
  search?: string;
  status?: string;
  model?: string;
}

export function listTrainingJobs(params: ListTrainingJobsParams = {}): Promise<TrainingJobListResponse> {
  const qs = new URLSearchParams();
  if (params.limit != null) qs.set('limit', String(params.limit));
  if (params.offset != null) qs.set('offset', String(params.offset));
  if (params.search?.trim()) qs.set('search', params.search.trim());
  if (params.status?.trim()) qs.set('status', params.status.trim());
  if (params.model?.trim()) qs.set('model', params.model.trim());
  const query = qs.toString();
  return apiGet<TrainingJobListResponse>(
    query ? `/workspace/training/jobs?${query}` : '/workspace/training/jobs'
  );
}

export function createTrainingJob(
  payload: CreateTrainingJobRequest
): Promise<CreateTrainingJobResponse> {
  return apiPost<CreateTrainingJobResponse>('/workspace/training/jobs', payload);
}

export function getTrainingJobStatus(trainJobId: string): Promise<TrainingJobStatus> {
  return apiGet<TrainingJobStatus>(`/workspace/training/jobs/${encodeURIComponent(trainJobId)}/status`);
}

export function getTrainingJobLog(trainJobId: string): Promise<{ trainJobId: string; log: string }> {
  return apiGet<{ trainJobId: string; log: string }>(
    `/workspace/training/jobs/${encodeURIComponent(trainJobId)}/log`
  );
}

export function getTrainingJobModel(trainJobId: string): Promise<TrainingJobModelResponse> {
  return apiGet<TrainingJobModelResponse>(
    `/workspace/training/jobs/${encodeURIComponent(trainJobId)}/model`
  );
}

export function deleteTrainingJob(trainJobId: string): Promise<TrainingJobDeleteResponse> {
  return apiDelete<TrainingJobDeleteResponse>(
    `/workspace/training/jobs/${encodeURIComponent(trainJobId)}`
  );
}

export async function deleteTrainingJobsBatch(trainJobIds: string[]): Promise<{
  deleted: string[];
  failed: Array<{ jobId: string; error: string }>;
}> {
  const deleted: string[] = [];
  const failed: Array<{ jobId: string; error: string }> = [];
  for (const jobId of trainJobIds) {
    try {
      await deleteTrainingJob(jobId);
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
