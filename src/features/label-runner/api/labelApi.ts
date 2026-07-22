/**
 * 标注任务 API
 */
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from '@/features/data-platform/api/client';

/** 后端返回的标注任务（与 label_tasks 表字段一致） */
export interface LabelTaskRow {
  id: number;
  task_id: string;
  name: string;
  dataset_path: string;
  dataset_ids: string | null;
  dataset_source: string | null;
  data_count: number | null;
  device_type: string | null;
  project_id: string | null;
  labeler: string | null;
  reviewer: string | null;
  collector: string;
  completed: boolean;
  verified: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * 从数据库获取标注任务列表（严格按 label_tasks 表）
 */
export async function getLabelTasks(
  params?: { skip?: number; limit?: number }
): Promise<ApiResponse<LabelTaskRow[]>> {
  const search = new URLSearchParams();
  if (params?.skip != null) search.set('skip', String(params.skip));
  if (params?.limit != null) search.set('limit', String(params.limit));
  const q = search.toString();
  return apiGet<LabelTaskRow[]>(`/api/label/tasks${q ? `?${q}` : ''}`);
}

/**
 * 从数据库返回行转为前端 LabelTask（严格按表字段映射）
 */
export function labelTaskRowToTask(row: LabelTaskRow, displayIndex: number): import('@/features/data-platform/models/labelTask').LabelTask {
  let datasetIds: number[] | undefined;
  if (row.dataset_ids) {
    try {
      const parsed = JSON.parse(row.dataset_ids) as unknown;
      if (Array.isArray(parsed)) {
        datasetIds = parsed.map((v) => Number(v)).filter((v) => !Number.isNaN(v));
      } else {
        datasetIds = undefined;
      }
    } catch {
      datasetIds = undefined;
    }
  }
  return {
    id: String(displayIndex).padStart(4, '0'),
    backendTaskId: row.task_id,
    name: row.name,
    datasetDir: row.dataset_path ?? '',
    datasetIds,
    dataCount: row.data_count ?? undefined,
    deviceType: row.device_type ?? undefined,
    projectId: row.project_id ?? undefined,
    labeler: row.labeler ?? undefined,
    reviewer: row.reviewer ?? undefined,
    collector: row.collector ?? '',
    createdAt: row.created_at ?? '',
    updatedAt: row.updated_at ?? '',
    completed: row.completed,
    verified: row.verified,
  };
}

export interface CreateLabelTaskRequest {
  name: string;
  dataset_path?: string;  // 可选，如果提供了 dataset_ids 则不需要
  dataset_ids?: number[];  // 数据集 ID 列表（可为 data_assets 或 hdf5_datasets 的 ID）
  dataset_source?: 'data_assets';  // 来自数据资产页时传此字段
  data_count?: number;
  device_type?: string;   // 已废弃，可选
  project_id?: string;     // 所属项目 ID
  labeler?: string;
  reviewer?: string;
  collector?: string;
}

export interface CreateLabelTaskResponse {
  task_id: string;
  name: string;
  dataset_path: string;
  data_count?: number;
  device_type: string;
  labeler?: string;
  reviewer?: string;
  collector?: string;
}

/**
 * 创建标注任务
 */
export async function createLabelTask(
  request: CreateLabelTaskRequest
): Promise<ApiResponse<CreateLabelTaskResponse>> {
  return apiPost<CreateLabelTaskResponse>('/api/label/tasks', request);
}

export interface UpdateLabelTaskRequest {
  name?: string;
  data_count?: number;
  device_type?: string;
  labeler?: string;
  reviewer?: string;
  collector?: string;
  dataset_path?: string;
  dataset_ids?: number[];
  dataset_source?: 'data_assets';
  project_id?: string;
  completed?: boolean;
  verified?: boolean;
}

/**
 * 删除标注任务（同步删除数据库与配置文件）
 */
export async function deleteLabelTask(
  taskId: string
): Promise<ApiResponse<unknown>> {
  return apiDelete<unknown>(`/api/label/tasks/${taskId}`);
}

/** GET /api/label/tasks/:id/actor — 前端标注门禁（与后端权限一致） */
export interface LabelTaskActorContext {
  task_id: string;
  name?: string | null;
  project_id?: string | null;
  project_owner_id?: string | null;
  labeler?: string | null;
  reviewer?: string | null;
}

export async function getLabelTaskActorContext(
  taskId: string
): Promise<ApiResponse<LabelTaskActorContext>> {
  return apiGet<LabelTaskActorContext>(
    `/api/label/tasks/${encodeURIComponent(taskId)}/actor`
  );
}

/**
 * 更新标注任务配置
 */
export async function updateLabelTask(
  taskId: string,
  request: UpdateLabelTaskRequest
): Promise<ApiResponse<unknown>> {
  return apiPatch<unknown>(`/api/label/tasks/${taskId}`, request);
}

export interface TaskImportStatusResponse {
  imported: boolean;
  episode_count: number;
}

/**
 * 获取任务导入状态
 */
export async function getTaskImportStatus(
  taskId: string
): Promise<ApiResponse<TaskImportStatusResponse>> {
  return apiGet<TaskImportStatusResponse>(`/api/label/tasks/${taskId}/import_status`);
}

export interface LoadTaskDatasetResponse {
  count: number;
  episodes?: Array<{
    episode_id: string;
    filename: string;
    abs_path: string;
  }>;
}

/**
 * 加载任务数据集（扫描数据集目录）
 */
export async function loadTaskDataset(
  taskId: string
): Promise<ApiResponse<LoadTaskDatasetResponse>> {
  return apiPost<LoadTaskDatasetResponse>(`/api/label/tasks/${taskId}/load_dataset`, {});
}

export interface Episode {
  id: string;
  name: string;
  path: string;
  /** 数据资产表中的标注信息，用于任务描述与「已标注/未标注」展示 */
  instruction_text?: string;
  cameras?: string[];
  frameCount?: number;
}

/**
 * 获取 episode 列表
 */
export async function getEpisodes(
  taskId?: string
): Promise<ApiResponse<Episode[]>> {
  const query = taskId ? `?taskId=${taskId}` : '';
  return apiGet<Episode[]>(`/api/label/episodes${query}`);
}

export interface EpisodeInfo {
  id: string;
  name: string;
  path: string;
  cameras: string[];
  frameCount: number;
  /** MCAP: 首帧时间戳（纳秒），用于进度条时间显示 */
  startTimeNs?: number;
  /** MCAP: 末帧时间戳（纳秒），用于进度条时间显示 */
  endTimeNs?: number;
}

/**
 * 获取 episode 详细信息
 */
export async function getEpisode(
  episodeId: string,
  taskId?: string
): Promise<ApiResponse<EpisodeInfo>> {
  const query = taskId ? `?taskId=${taskId}` : '';
  return apiGet<EpisodeInfo>(`/api/label/episodes/${episodeId}${query}`);
}

/**
 * 获取 Token（与 api/client 保持一致）
 */
function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.sessionStorage.getItem('auth.access_token');
}

/**
 * 获取帧图像
 * @param signal 可选，用于取消请求（播放时切换帧可取消上一帧请求，避免闪屏）
 */
export async function getFrame(
  episodeId: string,
  camera: string,
  frame: number,
  quality: number = 85,
  taskId?: string,
  signal?: AbortSignal
): Promise<Blob> {
  const queryParams = new URLSearchParams({
    camera,
    frame: frame.toString(),
    quality: quality.toString(),
  });
  if (taskId) {
    queryParams.append('taskId', taskId);
  }
  
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  
  const response = await fetch(`/api/label/frames/${episodeId}?${queryParams}`, {
    signal,
    // 改造说明：不再依赖 cookie
    credentials: 'omit',
    headers,
  });
  if (!response.ok) {
    let errorMessage = `获取帧图像失败: ${response.statusText}`;
    try {
      const errorData = await response.json();
      const msg = errorData.detail ?? errorData.error ?? errorData.message;
      if (msg) errorMessage = `获取帧图像失败: ${msg}`;
    } catch {
      // 响应不是 JSON，使用默认信息
    }
    throw new Error(errorMessage);
  }
  return response.blob();
}

/**
 * 按数据资产获取单帧图像（仅供数据资产可视化页使用）
 * 止血策略：401 不自动 refresh，直接抛错让上层处理登录态失效。
 */
export async function getAssetFrame(
  assetId: string | number,
  episodeId: string,
  camera: string,
  frame: number,
  quality: number = 85,
  signal?: AbortSignal
): Promise<Blob> {
  const queryParams = new URLSearchParams({
    camera,
    frame: frame.toString(),
    quality: quality.toString(),
    assetId: String(assetId),
  });
  const token = getAuthToken();
  if (token) queryParams.set('token', token);

  const headers: HeadersInit = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch(`/api/data-assets/frames/${encodeURIComponent(episodeId)}?${queryParams}`, {
    signal,
    credentials: 'omit',
    headers,
  });

  if (!response.ok) {
    let errorMessage = `获取帧图像失败: ${response.statusText}`;
    try {
      const errorData = await response.json();
      const msg = errorData.detail ?? errorData.error ?? errorData.message;
      if (msg) errorMessage = `获取帧图像失败: ${msg}`;
    } catch {
      // ignore
    }
    throw new Error(errorMessage);
  }
  return response.blob();
}

const MAX_CONCURRENT_FRAME_REQUESTS = 3;
let frameRequestInFlight = 0;
const frameRequestQueue: Array<() => void> = [];

function runNextFrameRequest(): void {
  if (frameRequestInFlight >= MAX_CONCURRENT_FRAME_REQUESTS || frameRequestQueue.length === 0) return;
  const task = frameRequestQueue.shift()!;
  frameRequestInFlight += 1;
  task();
}

/**
 * 带并发限制的 getFrame，多视角时避免同时发起过多请求导致卡顿
 */
export function getFrameQueued(
  episodeId: string,
  camera: string,
  frame: number,
  quality: number = 85,
  taskId?: string,
  signal?: AbortSignal
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const task = () => {
      getFrame(episodeId, camera, frame, quality, taskId, signal)
        .then((blob) => {
          frameRequestInFlight -= 1;
          resolve(blob);
          runNextFrameRequest();
        })
        .catch((err) => {
          frameRequestInFlight -= 1;
          reject(err);
          runNextFrameRequest();
        });
    };
    if (frameRequestInFlight < MAX_CONCURRENT_FRAME_REQUESTS) {
      frameRequestInFlight += 1;
      task();
    } else {
      frameRequestQueue.push(task);
    }
  });
}

export interface FramesBatchResponse {
  start: number;
  count: number;
  frames: string[];  // base64 JPEG
}

/** 批量获取帧（MCAP 预加载），一次请求返回多帧 */
export async function getFramesBatch(
  episodeId: string,
  camera: string,
  start: number,
  count: number,
  taskId: string,
  signal?: AbortSignal
): Promise<FramesBatchResponse> {
  const queryParams = new URLSearchParams({
    camera,
    start: start.toString(),
    count: count.toString(),
    taskId,
  });
  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (token) queryParams.set('token', token);

  const doFetch = async (url: string, hdrs: HeadersInit) => {
    return fetch(url, {
      signal,
      credentials: 'omit',
      headers: hdrs,
    });
  };

  const url = `/api/label/frames/${episodeId}/batch?${queryParams}`;
  const response = await doFetch(url, headers);
  if (!response.ok) {
    let detail: string;
    try {
      const err = await response.json().catch(() => ({}));
      detail = err?.detail ?? err?.message ?? response.statusText;
    } catch {
      detail = response.statusText;
    }
    throw new Error(detail || `预加载失败: ${response.statusText}`);
  }
  return response.json();
}

/** MCAP MJPEG 流 URL，用于 <img src> 实现类视频播放 */
export function getMcapStreamUrl(
  episodeId: string,
  camera: string,
  taskId: string,
  startFrame: number,
  fps: number = 10
): string {
  const params = new URLSearchParams({
    camera,
    fps: fps.toString(),
    start_frame: startFrame.toString(),
    taskId,
  });
  return `/api/label/stream/mcap/${episodeId}?${params}`;
}

/**
 * 获取指令
 */
export async function getInstruction(
  episodeId: string,
  taskId?: string
): Promise<ApiResponse<{ instruction: string }>> {
  const query = taskId ? `?taskId=${taskId}` : '';
  return apiGet<{ instruction: string }>(`/api/label/instructions/${episodeId}${query}`);
}

/**
 * 保存指令
 */
export async function saveInstruction(
  episodeId: string,
  instruction: string,
  episodeIndex?: number,
  taskId?: string
): Promise<ApiResponse<void>> {
  const query = taskId ? `?taskId=${taskId}` : '';
  const body: any = { instruction };
  if (episodeIndex !== undefined) {
    body.episode_index = episodeIndex;
  }
  return apiPost<void>(`/api/label/instructions/${episodeId}${query}`, body);
}

export interface GenerateAnnotationRequest {
  episode_id: string;
  camera_name?: string;
  taskId?: string;
  model?: string;
  openai_api_key?: string;
  openai_base_url?: string;
}

export interface GenerateAnnotationResponse {
  jobId: string;
}

/**
 * 生成自动标注（异步）
 */
export async function generateAnnotation(
  episodeId: string,
  cameraName?: string,
  taskId?: string,
  options?: { model?: string; openai_api_key?: string; openai_base_url?: string }
): Promise<ApiResponse<GenerateAnnotationResponse>> {
  const body: GenerateAnnotationRequest = {
    episode_id: episodeId,
  };
  if (cameraName) {
    body.camera_name = cameraName;
  }
  if (taskId) {
    body.taskId = taskId;
  }
  if (options?.model) {
    body.model = options.model;
  }
  if (options?.openai_api_key) {
    body.openai_api_key = options.openai_api_key;
  }
  if (options?.openai_base_url) {
    body.openai_base_url = options.openai_base_url;
  }
  return apiPost<GenerateAnnotationResponse>('/api/label/annotation/generate', body);
}

export interface AnnotationJobStatus {
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  result?: string;
  error?: string;
}

/**
 * 获取标注任务状态
 */
export async function getAnnotationStatus(
  jobId: string
): Promise<ApiResponse<AnnotationJobStatus>> {
  return apiGet<AnnotationJobStatus>(`/api/label/annotation/status/${jobId}`);
}

/**
 * 取消标注任务
 */
export async function cancelAnnotation(
  jobId: string
): Promise<ApiResponse<{ cancelled: boolean }>> {
  return apiPost<{ cancelled: boolean }>(`/api/label/annotation/cancel/${jobId}`, {});
}

/**
 * 获取任务下所有 episode 的标注结果（instructions 数组，与 episode 顺序一致）
 */
export async function getTaskInstructions(
  taskId: string
): Promise<ApiResponse<{ instructions: string[] }>> {
  return apiGet<{ instructions: string[] }>(`/api/label/tasks/${taskId}/instructions`);
}

/**
 * 获取任务数据集目录下的 instructions.json 内容（用于查看）
 */
export async function getTaskInstructionsFile(
  taskId: string
): Promise<ApiResponse<{ content: string }>> {
  return apiGet<{ content: string }>(`/api/label/tasks/${taskId}/instructions_file`);
}

/**
 * 从数据仓库查询单条标注（用于「下载当前条」）
 */
export async function getAnnotationDownloadOne(
  taskId: string,
  episodeId: string
): Promise<ApiResponse<{ instruction: string }>> {
  const params = new URLSearchParams({ taskId, episodeId });
  return apiGet<{ instruction: string }>(`/api/label/annotations/download_one?${params}`);
}

/**
 * 从数据仓库批量查询当前任务下所有标注（用于「下载整个数据集」）
 */
export async function getAnnotationDownloadBatch(
  taskId: string
): Promise<ApiResponse<{ items: Array<{ episode_id: string; path: string; instruction: string }> }>> {
  return apiGet<{ items: Array<{ episode_id: string; path: string; instruction: string }> }>(
    `/api/label/annotations/download_batch?taskId=${encodeURIComponent(taskId)}`
  );
}

/**
 * 数据资产：从数据仓库查询单条标注（用于「下载当前条」）
 */
export async function getAssetAnnotationDownloadOne(
  assetId: string,
  episodeId: string
): Promise<ApiResponse<{ instruction: string }>> {
  const params = new URLSearchParams({ assetId, episodeId });
  return apiGet<{ instruction: string }>(`/api/data-assets/annotations/download_one?${params}`);
}

/**
 * 数据资产：从数据仓库批量查询该资产下所有标注（用于「下载整个数据集」）
 */
export async function getAssetAnnotationDownloadBatch(
  assetId: string
): Promise<ApiResponse<{ items: Array<{ episode_id: string; path: string; instruction: string }> }>> {
  return apiGet<{ items: Array<{ episode_id: string; path: string; instruction: string }> }>(
    `/api/data-assets/annotations/download_batch?assetId=${encodeURIComponent(assetId)}`
  );
}

export interface BatchAnnotationByTaskRequest {
  taskId: string;
  camera_name?: string;
  fallback_first_camera?: boolean;
  model?: string;
  openai_api_key?: string;
  openai_base_url?: string;
}

export interface BatchAnnotationByTaskResult {
  episode_id: string;
  path?: string;
  output_path?: string;
  camera_used?: string;
  instruction?: string;
  error?: string;
}

/**
 * 按任务批量自动标注
 */
export async function batchGenerateAnnotation(
  taskId: string,
  cameraName?: string,
  options?: { model?: string; openai_api_key?: string; openai_base_url?: string }
): Promise<ApiResponse<{ results: BatchAnnotationByTaskResult[] }>> {
  const body: BatchAnnotationByTaskRequest = {
    taskId,
    camera_name: cameraName,
    fallback_first_camera: true,
  };
  if (options?.model) {
    body.model = options.model;
  }
  if (options?.openai_api_key) {
    body.openai_api_key = options.openai_api_key;
  }
  if (options?.openai_base_url) {
    body.openai_base_url = options.openai_base_url;
  }
  return apiPost<{ results: BatchAnnotationByTaskResult[] }>(
    '/api/label/annotation/batch_by_task',
    body
  );
}
