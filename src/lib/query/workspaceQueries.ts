'use client';

import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  listWorkspaceDatasets,
  type DatasetListResponse,
  type ListWorkspaceDatasetsParams,
} from '@/lib/api/datasetsClient';
import { listTrainingJobs } from '@/lib/api/trainingClient';
import { listEvaluationJobs } from '@/lib/api/evaluationClient';
import { listModelAssetFilterOptions, listModelAssets } from '@/lib/api/modelAssetsClient';
import { listTaskTemplates } from '@/lib/api/taskTemplatesClient';

export interface WorkspaceListParams {
  limit?: number;
  offset?: number;
}

export interface EvaluationJobsListParams extends WorkspaceListParams {
  search?: string;
  status?: string;
  mode?: string;
  backend?: string;
}

export interface ModelAssetListParams extends WorkspaceListParams {
  forEvaluation?: boolean;
  taskType?: string;
  search?: string;
  status?: string;
  modelType?: string;
  trainingJobId?: string;
  datasetId?: string;
  source?: string;
  dataset?: string;
  sourceTask?: string;
}

export const workspaceQueryKeys = {
  datasets: (params: ListWorkspaceDatasetsParams = {}) => ['workspace', 'datasets', params] as const,
  datasetsIndex: ['workspace', 'datasets'] as const,
  trainingJobs: (params: WorkspaceListParams = {}) => ['workspace', 'trainingJobs', params] as const,
  evaluationJobs: (params: WorkspaceListParams = {}) => ['workspace', 'evaluationJobs', params] as const,
  modelAssets: (options: ModelAssetListParams = {}) => ['workspace', 'modelAssets', options] as const,
  modelAssetFilterOptions: ['workspace', 'modelAssets', 'filterOptions'] as const,
  taskTemplates: (params: WorkspaceListParams = {}) => ['workspace', 'taskTemplates', params] as const,
};

export function useWorkspaceDatasetsQuery(
  params: ListWorkspaceDatasetsParams = {},
  enabled = true
) {
  return useQuery({
    queryKey: workspaceQueryKeys.datasets(params),
    queryFn: () => listWorkspaceDatasets(params),
    enabled,
  });
}

export function useTrainingJobsQuery(params: WorkspaceListParams = {}, enabled = true) {
  return useQuery({
    queryKey: workspaceQueryKeys.trainingJobs(params),
    queryFn: () => listTrainingJobs(params),
    enabled,
  });
}

export function useEvaluationJobsQuery(params: EvaluationJobsListParams = {}, enabled = true) {
  return useQuery({
    queryKey: workspaceQueryKeys.evaluationJobs(params),
    queryFn: () => listEvaluationJobs(params),
    enabled,
    placeholderData: keepPreviousData,
  });
}

export function useModelAssetsQuery(
  options: ModelAssetListParams & {
    enabled?: boolean;
  } = {}
) {
  const {
    forEvaluation,
    taskType,
    search,
    status,
    modelType,
    trainingJobId,
    datasetId,
    source,
    dataset,
    sourceTask,
    enabled = true,
    limit,
    offset,
  } = options;
  return useQuery({
    queryKey: workspaceQueryKeys.modelAssets({
      forEvaluation,
      taskType,
      search,
      status,
      modelType,
      trainingJobId,
      datasetId,
      source,
      dataset,
      sourceTask,
      limit,
      offset,
    }),
    queryFn: () =>
      listModelAssets({
        forEvaluation,
        taskType,
        search,
        status,
        modelType,
        trainingJobId,
        datasetId,
        source,
        dataset,
        sourceTask,
        limit,
        offset,
      }),
    enabled,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

export function useModelAssetFilterOptionsQuery(enabled = true) {
  return useQuery({
    queryKey: workspaceQueryKeys.modelAssetFilterOptions,
    queryFn: () => listModelAssetFilterOptions(),
    enabled,
    staleTime: 5 * 60_000,
  });
}

export function useTaskTemplatesQuery(params: WorkspaceListParams = {}, enabled = true) {
  return useQuery({
    queryKey: workspaceQueryKeys.taskTemplates(params),
    queryFn: () => listTaskTemplates(params),
    enabled,
  });
}

export function useInvalidateWorkspaceLists() {
  const queryClient = useQueryClient();
  return {
    invalidateDatasets: () =>
      queryClient.invalidateQueries({ queryKey: workspaceQueryKeys.datasetsIndex }),
    invalidateTrainingJobs: () =>
      queryClient.invalidateQueries({ queryKey: ['workspace', 'trainingJobs'] }),
    invalidateEvaluationJobs: () =>
      queryClient.invalidateQueries({ queryKey: ['workspace', 'evaluationJobs'] }),
    invalidateModelAssets: () =>
      queryClient.invalidateQueries({ queryKey: ['workspace', 'modelAssets'] }),
    invalidateTaskTemplates: () =>
      queryClient.invalidateQueries({ queryKey: ['workspace', 'taskTemplates'] }),
    invalidateAll: () => {
      void queryClient.invalidateQueries({ queryKey: ['workspace'] });
    },
  };
}

export type { DatasetListResponse, ListWorkspaceDatasetsParams };
