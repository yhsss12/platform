/**
 * 训练任务 — 对接后端 workspace_jobs 与 /api/workspace/training
 */

import {
  createTrainingJob,
  getTrainingJobLog,
  getTrainingJobModel,
  getTrainingJobStatus,
  listTrainingJobs,
  type CreateTrainingJobRequest,
  type TrainingJobListResponse,
  type TrainingJobStatus,
} from '@/lib/api/trainingClient';
import {
  listModelAssets,
  modelAssetsToCheckpointOptions,
} from '@/lib/api/modelAssetsClient';
import { listWorkspaceJobs } from '@/lib/api/workspaceJobClient';
import { workspaceTrainingJobToRow } from '@/lib/workspace/workspaceJobMapper';
import { trainingJobStatusToRow } from '@/lib/workspace/trainingTaskMapper';
import type { DatasetManifest } from '@/lib/workspace/datasetManifest';
import type { CreateTrainingTaskInput } from './workspaceTrainingMock';
import type { TrainingTaskRow } from './workspaceTrainingMock';
import { buildTrainingJobRequest } from '@/lib/workspace/prepareTrainingJobManifest';

export async function fetchRealTrainingTasks(): Promise<TrainingTaskRow[]> {
  const response = await listWorkspaceJobs({ jobType: 'training', source: 'real', limit: 100 });
  return response.jobs
    .map((job) => workspaceTrainingJobToRow(job))
    .filter((row) => row.source === 'real');
}

/** 保留供评测弹窗等调用：仅返回真实 training job */
export async function fetchTrainingTasks(): Promise<TrainingTaskRow[]> {
  try {
    return await fetchRealTrainingTasks();
  } catch {
    return [];
  }
}

export async function startTrainingJob(
  manifest: DatasetManifest,
  input: CreateTrainingTaskInput,
  datasetManifests?: DatasetManifest[]
): Promise<{ row: TrainingTaskRow; raw: TrainingJobStatus; request: CreateTrainingJobRequest }> {
  const request = buildTrainingJobRequest(manifest, input, datasetManifests);
  if (process.env.NODE_ENV !== 'production') {
    console.info('[training] create job request', JSON.stringify(request, null, 2));
  }
  const created = await createTrainingJob(request);
  const status = await getTrainingJobStatus(created.trainJobId);
  if (process.env.NODE_ENV !== 'production') {
    console.info('[training] create job response', created);
    console.info('[training] job status', status);
    try {
      const logRes = await getTrainingJobLog(created.trainJobId);
      const preview = (logRes.log || '').split('\n').slice(0, 20).join('\n');
      console.info('[training] train.log preview (first 20 lines)\n', preview);
    } catch (err) {
      console.warn('[training] train.log preview failed', err);
    }
  }
  return { row: trainingJobStatusToRow(status), raw: status, request };
}

export async function refreshTrainingTask(trainJobId: string): Promise<TrainingTaskRow> {
  const status = await getTrainingJobStatus(trainJobId);
  return trainingJobStatusToRow(status);
}

export async function listAvailableCheckpoints(): Promise<string[]> {
  try {
    const { modelAssets } = await listModelAssets();
    return modelAssets.map((asset) => asset.id);
  } catch {
    const tasks = await fetchTrainingTasks();
    return tasks
      .filter((task) => task.checkpointExists && task.modelAssetId)
      .map((task) => task.modelAssetId as string);
  }
}

export interface TrainingCheckpointOption {
  trainJobId: string;
  modelAssetId: string | null;
  label: string;
  ready: boolean;
  checkpointPath: string | null;
}

export async function listAvailableTrainingCheckpoints(): Promise<TrainingCheckpointOption[]> {
  try {
    const { modelAssets } = await listModelAssets();
    const fromRegistry = modelAssetsToCheckpointOptions(modelAssets);
    if (fromRegistry.length > 0) {
      return fromRegistry.map((opt) => ({
        trainJobId: opt.trainJobId,
        modelAssetId: opt.modelAssetId,
        label: opt.label,
        ready: opt.ready,
        checkpointPath: opt.checkpointPath,
      }));
    }
  } catch {
    /* fallback below */
  }

  const response = await listTrainingJobs().catch((): TrainingJobListResponse => ({ jobs: [], total: 0 }));
  const completed = response.jobs.filter((job) => job.status === 'completed' && job.checkpointExists);
  const options = await Promise.all(
    completed.map(async (job) => {
      const name = job.datasetName
        ? `${job.datasetName} · ${job.downstreamModelType ?? '训练'}`
        : job.trainJobId;
      try {
        const model = await getTrainingJobModel(job.trainJobId);
        const checkpointPath = model.checkpointPath ?? null;
        const ready = Boolean(model.ready && checkpointPath);
        const assetLabel = job.modelAssetId ?? job.trainJobId;
        return {
          trainJobId: job.trainJobId,
          modelAssetId: job.modelAssetId ?? null,
          label: `${assetLabel}（${name}）`,
          ready,
          checkpointPath,
        } satisfies TrainingCheckpointOption;
      } catch {
        return {
          trainJobId: job.trainJobId,
          modelAssetId: job.modelAssetId ?? null,
          label: job.modelAssetId ?? job.trainJobId,
          ready: false,
          checkpointPath: null,
        } satisfies TrainingCheckpointOption;
      }
    })
  );
  return options;
}
