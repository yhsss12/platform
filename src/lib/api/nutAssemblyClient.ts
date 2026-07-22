'use client';

import { apiGet, apiPost } from '@/lib/api/authClient';

const NUT_ASSEMBLY_FETCH_INIT: RequestInit = {
  cache: 'no-store',
  headers: {
    'Cache-Control': 'no-cache',
    Pragma: 'no-cache',
  },
};

function nutAssemblyPollPath(path: string): string {
  const separator = path.includes('?') ? '&' : '?';
  return `${path}${separator}_=${Date.now()}`;
}

export type NutAssemblyGenerationMode = 'mimicgen_datagen' | 'robosuite_rollout';

export type NutAssemblySourceDemoSelection = 'official' | 'local' | 'custom' | 'auto';

export interface NutAssemblyPhysicsEnhancementRequest {
  enabled: boolean;
  method?: 'pinn_repair';
  modelId?: string;
  repairStages?: string[];
  candidateSource?: string[];
  maxCandidates?: number;
  maxRepairAttemptsPerCandidate?: number;
  xyErrorThreshold?: number;
  heightErrorThreshold?: number;
  validationMode?: 'mujoco_rollout';
  appendRepairedDemos?: boolean;
}

export interface NutAssemblyPinnModelStatus {
  modelId?: string;
  available?: boolean;
  displayName?: string;
  modelPath?: string | null;
  pipelineVersion?: string | null;
  pinnBackend?: 'heuristic' | 'torch_model' | string | null;
  modelLoaded?: boolean;
  repairStages?: string[];
  error?: string | null;
}

export interface NutAssemblyGenerateRequest {
  taskTemplateId?: string;
  episodes?: number;
  seed?: number;
  renderVideo?: boolean;
  sourceDemoPath?: string | null;
  sourceDemoSelection?: NutAssemblySourceDemoSelection | null;
  sourceDemoDatasetId?: string;
  envName?: string;
  outputName?: string;
  horizon?: number;
  taskConfigId?: string | null;
  generationMode?: NutAssemblyGenerationMode;
  generationPath?: 'expert_policy' | 'demo_augmentation' | 'expert_seed_then_augmentation';
  generationCount?: number;
  maxSteps?: number;
  expertPolicy?: string;
  successFilter?: boolean;
  keepFailedTrajectories?: boolean;
  augmentationAlgorithm?: 'mimicgen';
  targetCount?: number;
  seedGenerationCount?: number;
  seedKeepCount?: number;
  autoSelectBestSeeds?: boolean;
  replayValidation?: boolean;
  enablePinnRepair?: boolean;
  physicsEnhancement?: NutAssemblyPhysicsEnhancementRequest;
}

export interface NutAssemblySourceDemoOption {
  label?: string;
  path?: string;
  exists?: boolean;
  validationPassed?: boolean;
  sourceDemoOrigin?: string;
  demoCount?: number;
  objectPoseKeys?: string[];
  envName?: string;
  md5?: string;
  needsPrepare?: boolean;
  alreadyPrepared?: boolean;
  warning?: string;
  registryRelativePath?: string;
  requiresPath?: boolean;
}

export interface NutAssemblySourceDemoStatus {
  defaultSelection?: NutAssemblySourceDemoSelection;
  defaultWarning?: string | null;
  officialSourceValidated?: boolean;
  manifestPath?: string;
  options?: {
    official?: NutAssemblySourceDemoOption;
    local?: NutAssemblySourceDemoOption;
    custom?: NutAssemblySourceDemoOption;
  };
  coreDataset?: NutAssemblySourceDemoOption & { purpose?: string; notDefaultSourceDemo?: boolean };
  error?: string;
}

export interface NutAssemblyMimicgenEnvStatus {
  overallOk?: boolean;
  checkedAt?: string;
  nutAssemblyMvpExists?: boolean;
  error?: string;
  checks?: Record<string, { ok?: boolean; version?: string; error?: string }>;
}

export interface NutAssemblyGenerateAsyncResponse {
  jobId: string;
  taskType: string;
  status: 'running';
  statusUrl: string;
  resultUrl: string;
  command: string;
}

export interface NutAssemblyPathInfo {
  path: string;
  exists: boolean;
  sizeBytes?: number | null;
}

export interface NutAssemblyJobStatusResponse {
  jobId: string;
  taskType: string;
  status: string;
  live: Record<string, unknown>;
  paths: Record<string, NutAssemblyPathInfo>;
  metrics: Record<string, unknown>;
  command?: string;
  startedAt?: string | null;
  stage?: string | null;
  progress?: number | null;
  message?: string | null;
  lastHeartbeatAt?: string | null;
  elapsedSeconds?: number | null;
  logLastModifiedAt?: string | null;
  generationMode?: string | null;
  policyMode?: string | null;
  sourceEnvName?: string | null;
  runtimeEnvName?: string | null;
  sourceDemoPath?: string | null;
  sourceDemoOrigin?: string | null;
  sourceDemoOriginReason?: string | null;
  successRate?: number | null;
  failureDistribution?: Record<string, number> | null;
  videoUrl?: string | null;
  generateVideoExists?: boolean | null;
  hasDatagenInfo?: boolean | null;
  hasObjectPoses?: boolean | null;
  objectPoseKeys?: string[] | null;
  fallbackFrom?: string | null;
  fallbackReason?: string | null;
  episodesRequested?: number | null;
  episodesGenerated?: number | null;
  datagenFailedTrials?: number | null;
  datagenSuccessRate?: number | null;
  traceback?: string | null;
  hdf5Path?: string | null;
  videoPath?: string | null;
  logTail?: string | null;
}

export async function getNutAssemblyJob(
  jobId: string,
  tail = 20
): Promise<NutAssemblyJobStatusResponse> {
  return apiGet<NutAssemblyJobStatusResponse>(
    nutAssemblyPollPath(`/workspace/nut-assembly/jobs/${encodeURIComponent(jobId)}?tail=${tail}`),
    NUT_ASSEMBLY_FETCH_INIT
  );
}

export async function generateNutAssemblyDataAsync(
  payload: NutAssemblyGenerateRequest
): Promise<NutAssemblyGenerateAsyncResponse> {
  return apiPost<NutAssemblyGenerateAsyncResponse>(
    '/workspace/nut-assembly/generate-async',
    payload
  );
}

export async function getNutAssemblyJobStatus(
  jobId: string
): Promise<NutAssemblyJobStatusResponse> {
  return apiGet<NutAssemblyJobStatusResponse>(
    nutAssemblyPollPath(`/workspace/nut-assembly/jobs/${encodeURIComponent(jobId)}/status`),
    NUT_ASSEMBLY_FETCH_INIT
  );
}

export async function getNutAssemblyJobLog(
  jobId: string,
  tail = 20
): Promise<{ jobId: string; tail: string }> {
  return apiGet<{ jobId: string; tail: string }>(
    nutAssemblyPollPath(`/workspace/nut-assembly/jobs/${encodeURIComponent(jobId)}/log?tail=${tail}`),
    NUT_ASSEMBLY_FETCH_INIT
  );
}

export async function getNutAssemblyJobResult(jobId: string): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(
    nutAssemblyPollPath(`/workspace/nut-assembly/jobs/${encodeURIComponent(jobId)}/result`),
    NUT_ASSEMBLY_FETCH_INIT
  );
}

export async function getNutAssemblySourceDemoStatus(): Promise<NutAssemblySourceDemoStatus> {
  return apiGet<NutAssemblySourceDemoStatus>('/workspace/nut-assembly/source-demo-status');
}

export async function getNutAssemblyPinnModelStatus(
  modelId = 'nut_assembly_pinn_v1'
): Promise<NutAssemblyPinnModelStatus> {
  return apiGet<NutAssemblyPinnModelStatus>(
    `/workspace/nut-assembly/pinn-model-status?modelId=${encodeURIComponent(modelId)}`,
    NUT_ASSEMBLY_FETCH_INIT
  );
}

export async function getNutAssemblyMimicgenEnvStatus(
  refresh = false
): Promise<NutAssemblyMimicgenEnvStatus> {
  const query = refresh ? '?refresh=true' : '';
  return apiGet<NutAssemblyMimicgenEnvStatus>(`/workspace/nut-assembly/mimicgen-env-status${query}`);
}
