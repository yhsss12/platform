'use client';

import { apiGet } from '@/lib/api/authClient';
import type { TaskTemplate } from '@/types/benchmark';

export type WorkspaceEvaluationMode =
  | 'expert_policy_evaluation'
  | 'trained_model_evaluation'
  | 'episode_stability';

export interface TaskTemplateDto extends TaskTemplate {
  englishName?: string | null;
  supportedEvaluationModes?: WorkspaceEvaluationMode[];
  registryTaskConfigId?: string | null;
  physicsBackend?: string | null;
  defaultEnv?: string | null;
  adapterStatus?: string | null;
  requiresExternalRuntime?: boolean;
  simulatorBackendLabel?: string | null;
  simulatorBackend?: 'mujoco' | 'isaac_lab' | 'isaacsim' | null;
  supportsDatasetGeneration?: boolean | 'planned';
  replayAvailable?: boolean;
  supportsImportedDemoReplay?: boolean;
  hasExpertPolicy?: boolean;
  hasEvaluationRunner?: boolean;
  supportsDataGeneration?: boolean;
  supportsEvaluation?: boolean;
  defaultReplayCamera?: string | null;
  defaultMetricIds?: string[];
  availableMetricIds?: string[];
}

export interface TaskTemplateListResponse {
  taskTemplates: TaskTemplateDto[];
  total: number;
}

export async function listTaskTemplates(params?: {
  limit?: number;
  offset?: number;
}): Promise<TaskTemplateListResponse> {
  const qs = new URLSearchParams();
  if (params?.limit != null) qs.set('limit', String(params.limit));
  if (params?.offset != null) qs.set('offset', String(params.offset));
  const query = qs.toString();
  return apiGet<TaskTemplateListResponse>(
    query ? `/workspace/task-templates?${query}` : '/workspace/task-templates'
  );
}
