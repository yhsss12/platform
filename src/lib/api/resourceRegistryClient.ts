'use client';

import { apiGet, apiPost } from '@/lib/api/authClient';

export type RegistryAssetType =
  | 'robot'
  | 'end_effector'
  | 'object'
  | 'scene'
  | 'task'
  | 'metric'
  | 'policy';

export type ResourceDefinitionType =
  | RegistryAssetType
  | 'task_config'
  | 'physics_proxy';

export interface RegistryResource {
  assetId: string;
  assetType: string;
  resourceType?: string;
  resourceId?: string;
  name: string;
  version: string;
  status: string;
  simBackend: string;
  description: string;
  tags: string[];
  files: Record<string, unknown>;
  metadata: Record<string, unknown>;
  manifestPath: string;
  lastModifiedAt: string;
  taskType?: string | null;
  requiredAssets?: Record<string, unknown> | null;
  metrics?: string[] | null;
  runner?: Record<string, unknown> | null;
  defaultConfig?: Record<string, unknown> | null;
  physicsProxy?: Record<string, unknown>;
  source?: string;
  storageUri?: string | null;
}

export interface RegistryListResponse {
  resources: RegistryResource[];
  total: number;
  source: 'registry' | 'mock' | 'mixed' | 'database';
  stats: {
    total?: number;
    byType?: Record<string, number>;
    byResourceType?: Record<string, number>;
    byBackend?: Record<string, number>;
    lastModifiedAt?: string | null;
    lastScanAt?: string | null;
  };
}

export interface ResourceOverviewWarning {
  category: string;
  message: string;
  path?: string | null;
}

export interface ResourceOverviewResponse {
  taskTemplates: number | null;
  modelAssets: number | null;
  metrics: number | null;
  scenes: number | null;
  robots: number | null;
  objects: number | null;
  policyAssets: number | null;
  physicsProxies: number | null;
  modelTypes: number | null;
  craftConfig: number | null;
  simAssets: number | null;
  source?: string;
  warnings?: ResourceOverviewWarning[];
  partialFailure?: boolean;
}

export interface RegistryReindexResponse {
  scanned: number;
  valid: number;
  invalid: number;
  synced?: number;
  created?: number;
  updated?: number;
  skipped?: number;
  resourcesByType: Record<string, number>;
  errors: string[];
  warnings: string[];
  lastScanAt?: string | null;
  source?: string;
}

export interface TaskConfigSummary {
  assetId: string;
  taskType?: string | null;
  name: string;
  version: string;
  status: string;
  simBackend: string;
  description: string;
  requiredAssetsCount: number;
  metricsCount: number;
  runner: Record<string, unknown>;
  tags: string[];
  lastModifiedAt?: string | null;
}

export interface TaskConfigDetail extends TaskConfigSummary {
  requiredAssets: Record<string, unknown>;
  metrics: string[];
  defaultConfig: Record<string, unknown>;
  resolvedResources: Record<string, RegistryResource | RegistryResource[]>;
  manifestPath?: string | null;
}

export interface ListResourcesParams {
  assetType?: RegistryAssetType | string;
  resourceType?: ResourceDefinitionType | string;
  simBackend?: string;
  status?: string;
  taskType?: string;
  includeMock?: boolean;
}

export async function getResourceOverview(): Promise<ResourceOverviewResponse> {
  return apiGet<ResourceOverviewResponse>('/workspace/resources/overview');
}

export async function listRegistryResources(
  params: ListResourcesParams = {}
): Promise<RegistryListResponse> {
  const query = new URLSearchParams();
  if (params.assetType) query.set('assetType', params.assetType);
  if (params.resourceType) query.set('resourceType', params.resourceType);
  if (params.simBackend) query.set('simBackend', params.simBackend);
  if (params.status) query.set('status', params.status);
  if (params.taskType) query.set('taskType', params.taskType);
  if (params.includeMock) query.set('includeMock', 'true');
  const suffix = query.toString() ? `?${query.toString()}` : '';
  return apiGet<RegistryListResponse>(`/workspace/resources${suffix}`);
}

export async function getRegistryResource(
  assetId: string,
  resourceType?: string
): Promise<RegistryResource> {
  if (resourceType) {
    return apiGet<RegistryResource>(
      `/workspace/resources/${encodeURIComponent(resourceType)}/${encodeURIComponent(assetId)}`
    );
  }
  return apiGet<RegistryResource>(`/workspace/resources/${encodeURIComponent(assetId)}`);
}

export async function reindexRegistryResources(): Promise<RegistryReindexResponse> {
  return apiPost<RegistryReindexResponse>('/workspace/resources/reindex', {});
}

export async function listTaskConfigs(taskType?: string): Promise<{
  taskConfigs: TaskConfigSummary[];
  total: number;
}> {
  const suffix = taskType ? `?taskType=${encodeURIComponent(taskType)}` : '';
  return apiGet<{ taskConfigs: TaskConfigSummary[]; total: number }>(
    `/workspace/task-configs${suffix}`
  );
}

export async function getTaskConfig(taskConfigId: string): Promise<TaskConfigDetail> {
  return apiGet<TaskConfigDetail>(
    `/workspace/task-configs/${encodeURIComponent(taskConfigId)}`
  );
}

export function registryStatusLabel(status: string): string {
  switch (status) {
    case 'available':
      return '可用';
    case 'experimental':
      return '实验';
    case 'deprecated':
      return '已弃用';
    default:
      return status;
  }
}
