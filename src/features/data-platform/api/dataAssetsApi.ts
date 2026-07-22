/**
 * 数据资产 API（真实本地 HDF5/MCAP/LeRobot，存 backend/data/assets/assets.db）
 */
import {
  apiGet,
  apiPost,
  apiUpload,
  apiUploadWithProgress,
  apiDelete,
  type ApiResponse,
} from './client';
import {
  fetchAndSaveExportZip,
  type ExportZipDownloadProgress,
} from './exportZipDownload';
import { importDiagLog } from '@/lib/importDiagLog';

const DIRECT_UPLOAD_HTTP_TIMEOUT_MS = 120_000;

function withHttpTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return new Promise((resolve, reject) => {
    timer = setTimeout(() => {
      timer = undefined;
      reject(new Error(`${label} 请求超时（>${Math.round(ms / 1000)}s）`));
    }, ms);
    p.then(
      (v) => {
        if (timer) clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        if (timer) clearTimeout(timer);
        reject(e);
      },
    );
  });
}

// 统一来源语义：导入 / 采集 / 标注 / 转换
export type DataAssetSource = 'import' | 'collect' | 'label' | 'convert';

export interface DataAssetItem {
  id: number;
  /** 数据专属标识 DSxxxx（若后端已分配） */
  dataset_id?: string | null;
  code: string;
  filename: string;
  format: string;
  source: string;
  project_id: string | null;
  project_name: string | null;
  file_path: string;
  /** 对象存储 URI（API 推导，优先用于展示） */
  warehouse_uri?: string | null;
  file_size_bytes: number;
  created_at: string;
  created_at_ms?: number;
  /** 最近一次业务动作时间（导入 / 标注任务 / 转换 / 导出等） */
  updated_at: string;
  updated_at_ms?: number;
  meta: string | null;
  parse_status: string;
  error_msg: string | null;
  /** 关联的标注任务名称（按任务划分展示） */
  label_task_name?: string | null;
  /** 关联的采集任务名称（按任务划分展示） */
  collect_task_name?: string | null;
  /** 关联的转换任务名称（按任务划分展示） */
  conversion_task_name?: string | null;
  /** 标注正文（与 instruction 同步，单条查询时更完整） */
  instruction_text?: string | null;
  /** 采集来源：平台 devices.id，区分不同采集端（列表「操作人」列不再展示） */
  device_id?: string | null;
  /** 操作人账号名（资产创建/导入/登记时写入，历史字段） */
  operator_name?: string | null;
  sync_status?: 'unsynced' | 'syncing' | 'synced' | 'failed' | string;
  sync_error?: string | null;
  /** 列表开启 reconcile_collect_disk 时：相对作业 workspace 的 episode 路径 */
  collect_episode_rel_path?: string | null;
  /** 列表开启磁盘对账时：采集端是否仍存在该 episode 目录 */
  collect_episode_on_device?: boolean | null;
}

/** @deprecated Use getSourceLabel(source, t) for i18n */
export function getSourceDisplay(source: string | null | undefined): string {
  const raw = (source || '').toLowerCase();
  if (raw === 'collect') return '采集';
  if (raw === 'label') return '标注';
  if (raw === 'convert') return '转换';
  if (raw === 'local') return '本地';
  if (raw === 'import' || source === '本地' || !raw) return '导入';
  return '导入';
}

export type SourceLabelT = (path: string) => string;

/**
 * 界面展示的「数据仓库」路径：优先 API 的 warehouse_uri，其次 file_path / meta.storage.minio_path
 */
export function getDataAssetWarehouseDisplayPath(
  asset: Pick<DataAssetItem, 'file_path' | 'meta' | 'warehouse_uri'>
): string {
  const apiWh = (asset.warehouse_uri || '').trim();
  if (apiWh.startsWith('minio://')) return apiWh;
  const fp = (asset.file_path || '').trim();
  if (fp.startsWith('minio://')) return fp;
  const metaStr = asset.meta;
  if (metaStr) {
    try {
      const parsed = JSON.parse(metaStr) as { storage?: { minio_path?: string } };
      const mp = parsed?.storage?.minio_path;
      if (typeof mp === 'string') {
        const m = mp.trim();
        if (m.startsWith('minio://')) return m;
      }
    } catch {
      /* ignore */
    }
  }
  return fp;
}

export function getSourceLabel(source: string | null | undefined, t: SourceLabelT): string {
  const raw = (source || '').toLowerCase();
  if (raw === 'collect') return t('dataPage.sourceCollect');
  if (raw === 'label') return t('dataPage.sourceLabel');
  if (raw === 'convert') return t('dataPage.sourceConvert');
  if (raw === 'local') return t('dataPage.sourceLocal');
  return t('dataPage.sourceImport');
}

/**
 * 是否属于「需要从采集端拉取到平台」的同步链路。
 * 导入、转换、标注等资产已在平台侧，不参与采集端同步。
 */
export function dataAssetRequiresAgentSync(asset: Pick<DataAssetItem, 'source'>): boolean {
  return String(asset.source || '').trim().toLowerCase() === 'collect';
}

/**
 * 统一判断资产是否可执行查看/标注/转换/导出：
 * - 非采集来源：视为已就绪（无需采集端同步）
 * - 采集来源：显式 synced 才视为已同步；unsynced/syncing/failed 或未返回状态视为未同步
 */
export function isDataAssetSynced(asset: Pick<DataAssetItem, 'source' | 'sync_status'>): boolean {
  if (!dataAssetRequiresAgentSync(asset)) return true;
  const status = String(asset.sync_status || '').trim().toLowerCase();
  if (status === 'synced') return true;
  if (status === 'unsynced' || status === 'syncing' || status === 'failed') return false;
  return false;
}

export interface DataAssetListResponse {
  items: DataAssetItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface DataAssetQueryParams {
  keyword?: string;
  project?: string;
  format?: string;
  source?: string;
  task_id?: string;
  task_name?: string;
  /** 创建/入库日期起 YYYY-MM-DD（UTC 日界，含当日） */
  created_from?: string;
  /** 创建/入库日期止 YYYY-MM-DD（UTC 日界，含当日） */
  created_to?: string;
  page?: number;
  page_size?: number;
  /** 为 true 且 Agent 在线时，核对采集端 episode 目录是否在盘上（略增耗时） */
  reconcile_collect_disk?: boolean;
}

/** 与 POST /api/data-assets/import 返回 data.imported[] 一致；minio_path 成功时必有 */
export type DataAssetImportRow = { name: string; id: number; minio_path?: string };

export interface ImportResult {
  imported: DataAssetImportRow[];
  failed: Array<{ name: string; reason: string }>;
}

export async function getDataAssets(
  params: DataAssetQueryParams = {}
): Promise<ApiResponse<DataAssetListResponse>> {
  const q = new URLSearchParams();
  if (params.keyword) q.set('keyword', params.keyword);
  if (params.project) q.set('project', params.project);
  if (params.format) q.set('format', params.format);
  if (params.source) q.set('source', params.source);
  if (params.task_id) q.set('task_id', params.task_id);
  if (params.task_name) q.set('task_name', params.task_name);
  if (params.created_from) q.set('created_from', params.created_from);
  if (params.created_to) q.set('created_to', params.created_to);
  if (params.page) q.set('page', String(params.page));
  if (params.page_size) q.set('page_size', String(params.page_size));
  if (params.reconcile_collect_disk) q.set('reconcile_collect_disk', 'true');
  const query = q.toString();
  return apiGet<DataAssetListResponse>(`/api/data-assets${query ? `?${query}` : ''}`);
}

export async function getDataAssetTaskOptions(params: {
  project?: string;
  format?: string;
  source?: string;
} = {}): Promise<ApiResponse<{ items: Array<{ value: string; label: string }> }>> {
  const q = new URLSearchParams();
  if (params.project) q.set('project', params.project);
  if (params.format) q.set('format', params.format);
  if (params.source) q.set('source', params.source);
  const query = q.toString();
  return apiGet<{ items: Array<{ value: string; label: string }> }>(`/api/data-assets/task-options${query ? `?${query}` : ''}`);
}

export async function getDataAsset(id: number): Promise<ApiResponse<DataAssetItem>> {
  return apiGet<DataAssetItem>(`/api/data-assets/by-id/${id}`);
}

export async function getDatasetCountByProject(
  projectId: string,
  projectName?: string
): Promise<number> {
  const res = await getDataAssets({ project: projectId, page_size: 1 });
  if (res.ok && res.data && res.data.total > 0) return res.data.total;
  if (projectName && projectName.trim()) {
    const res2 = await getDataAssets({ project: projectName.trim(), page_size: 1 });
    if (res2.ok && res2.data) return res2.data.total;
  }
  return 0;
}

/**
 * 数据页标准导入：POST /api/data-assets/import，落盘 + MinIO，meta.storage.minio_path 由后端写入。
 * 单文件无 webkitRelativePath；文件夹选择产生的 File 带 webkitRelativePath 时会作为 multipart 文件名传入以还原目录结构。
 * @param onProgress 上传字节进度 0–100（浏览器能计算总量时）
 */
export type DirectUploadMode = 'single_file' | 'multi_file' | 'directory';

export interface DirectUploadFileItemSpec {
  client_file_id: string;
  relative_path: string;
  size_bytes: number;
  content_type?: string | null;
}

export interface DirectUploadInitItem {
  client_file_id: string;
  relative_path: string;
  object_key: string;
  upload_url: string;
  method: string;
  headers: Record<string, string>;
}

/** POST /api/data-assets/upload-init 返回 data */
export interface DirectUploadInitData {
  upload_session_id: string;
  bucket: string;
  expires_at: string;
  upload_mode: DirectUploadMode | string;
  upload_items: DirectUploadInitItem[];
  root_dir_name?: string;
  object_key?: string;
  upload_url?: string;
  method?: string;
  headers?: Record<string, string>;
}

function newDirectUploadClientFileId(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/** 从 relative_path 取叶子文件名（供单文件 items[0] 与顶层 filename 对齐） */
function leafBasename(relativePath: string): string {
  const r = (relativePath || '').replace(/\\/g, '/').trim();
  const seg = r.split('/').pop()?.trim();
  return seg || r || '';
}

export async function initDirectUpload(params: {
  project_id: string;
  upload_mode?: DirectUploadMode;
  filename?: string;
  size_bytes?: number;
  content_type?: string | null;
  items?: DirectUploadFileItemSpec[];
  root_dir_name?: string | null;
}): Promise<ApiResponse<DirectUploadInitData>> {
  const mode = params.upload_mode ?? 'single_file';
  const body: Record<string, unknown> = {
    upload_mode: mode,
    project_id: params.project_id,
  };

  if (mode === 'single_file') {
    if (params.items?.length === 1) {
      const it = params.items[0];
      body.items = params.items;
      const leaf = leafBasename(it.relative_path || '');
      body.filename = (params.filename && params.filename.trim()) || leaf;
      body.size_bytes = params.size_bytes ?? it.size_bytes;
      const ct = params.content_type?.trim() || it.content_type?.trim();
      if (ct) body.content_type = ct;
    } else if (params.filename != null && params.size_bytes != null) {
      body.filename = params.filename;
      body.size_bytes = params.size_bytes;
      if (params.content_type?.trim()) body.content_type = params.content_type.trim();
      body.items = [
        {
          client_file_id: newDirectUploadClientFileId(),
          relative_path: params.filename,
          size_bytes: params.size_bytes,
          content_type: params.content_type?.trim() || null,
        },
      ];
    }
  } else if (params.items?.length) {
    body.items = params.items;
  }

  if (params.root_dir_name?.trim()) body.root_dir_name = params.root_dir_name.trim();
  const itemsCount =
    mode === 'single_file'
      ? Array.isArray(body.items)
        ? body.items.length
        : body.filename
          ? 1
          : 0
      : Array.isArray(body.items)
        ? body.items.length
        : 0;
  importDiagLog('upload_init_request', {
    upload_mode: mode,
    project_id: params.project_id,
    items_count: itemsCount,
    root_dir_name: params.root_dir_name ?? null,
  });
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const resp = await withHttpTimeout(
    apiPost<DirectUploadInitData>('/api/data-assets/upload-init', body),
    DIRECT_UPLOAD_HTTP_TIMEOUT_MS,
    'upload-init',
  );
  const dt =
    typeof performance !== 'undefined' ? Math.round(performance.now() - t0) : Date.now() - t0;
  importDiagLog('upload_init_response', {
    ok: resp.ok,
    ms: dt,
    upload_session_id: resp.data?.upload_session_id,
    upload_items: resp.data?.upload_items?.length,
    top_level_upload_url: !!resp.data?.upload_url,
    error: resp.error,
  });
  return resp;
}

export interface DirectUploadCompleteMultiData {
  assets: DataAssetItem[];
  failed_items: Array<{ object_key?: string; relative_path?: string; reason?: string }>;
}

export async function completeDirectUpload(params: {
  upload_session_id: string;
  size_bytes?: number;
  manifest?: {
    root_dir_name: string;
    paths: { relative_path: string; size_bytes: number }[];
    total_files: number;
    total_size_bytes: number;
  };
}): Promise<ApiResponse<{ asset?: DataAssetItem; assets?: DataAssetItem[]; failed_items?: DirectUploadCompleteMultiData['failed_items'] }>> {
  const body: Record<string, unknown> = { upload_session_id: params.upload_session_id };
  if (params.size_bytes != null) body.size_bytes = params.size_bytes;
  if (params.manifest) body.manifest = params.manifest;
  importDiagLog('upload_complete_request', {
    upload_session_id: params.upload_session_id,
    has_manifest: !!params.manifest,
    size_bytes: params.size_bytes ?? null,
  });
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const resp = await withHttpTimeout(
    apiPost<{ asset?: DataAssetItem; assets?: DataAssetItem[]; failed_items?: DirectUploadCompleteMultiData['failed_items'] }>(
      '/api/data-assets/upload-complete',
      body,
    ),
    DIRECT_UPLOAD_HTTP_TIMEOUT_MS,
    'upload-complete',
  );
  const dt =
    typeof performance !== 'undefined' ? Math.round(performance.now() - t0) : Date.now() - t0;
  importDiagLog('upload_complete_response', {
    ok: resp.ok,
    ms: dt,
    upload_session_id: params.upload_session_id,
    assets: resp.data?.assets?.length,
    asset_single: !!resp.data?.asset,
    failed_items: resp.data?.failed_items?.length,
    error: resp.error,
  });
  return resp;
}

/** GET /api/data-assets/upload-sessions — 当前用户直传会话，供任务中心刷新恢复 */
export interface UploadSessionListItem {
  upload_session_id: string;
  project_id: string;
  status: string;
  upload_mode: string;
  filename: string;
  expected_count?: number | null;
  size_bytes?: number;
  created_at: string;
  expires_at?: string | null;
  root_dir_name?: string | null;
  asset_name?: string | null;
}

export async function listUploadSessions(
  limit = 50
): Promise<ApiResponse<{ items: UploadSessionListItem[] }>> {
  return apiGet(`/api/data-assets/upload-sessions?limit=${encodeURIComponent(String(limit))}`);
}

/** POST /api/data-assets/upload-sessions/{id}/cancel — 取消直传会话，刷新后不再恢复为进行中 */
export async function cancelUploadSession(
  uploadSessionId: string
): Promise<ApiResponse<{ upload_session_id: string; status: string }>> {
  return apiPost<{ upload_session_id: string; status: string }>(
    `/api/data-assets/upload-sessions/${encodeURIComponent(uploadSessionId)}/cancel`,
    {}
  );
}

/**
 * 浏览器直传 MinIO：PUT 到预签名 URL（勿带平台 Authorization）。
 * @param onProgress 0–100（XHR 进度或按 loaded 估算）
 * @param onByteProgress 当前文件已上传字节 / 该文件总字节（lengthComputable 时最准），任务级聚合在 runner
 */
export function uploadFileToPresignedUrl(
  file: File,
  uploadUrl: string,
  options?: {
    method?: string;
    headers?: Record<string, string>;
    onProgress?: (percent: number) => void;
    /** 与 onProgress 并行：当前文件已上传字节 / 该文件总字节（XHR 可得时更准） */
    onByteProgress?: (loaded: number, total: number) => void;
    signal?: AbortSignal;
  }
): Promise<void> {
  const method = (options?.method || 'PUT').toUpperCase();
  const headers = options?.headers || {};
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const sig = options?.signal;
    const urlHasQuery = (() => {
      try {
        return new URL(uploadUrl).search.length > 0;
      } catch {
        return uploadUrl.includes('?');
      }
    })();
    importDiagLog('presigned_put_start', {
      name: file.name,
      size_bytes: file.size,
      method,
      presigned_url_present: !!uploadUrl.trim(),
      presigned_url_has_query: urlHasQuery,
    });
    if (sig) {
      if (sig.aborted) {
        importDiagLog('presigned_put_aborted_before_send', { name: file.name });
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      sig.addEventListener(
        'abort',
        () => {
          try {
            xhr.abort();
          } catch {
            /* ignore */
          }
          importDiagLog('presigned_put_abort_signal', { name: file.name });
          reject(new DOMException('Aborted', 'AbortError'));
        },
        { once: true }
      );
    }
    xhr.open(method, uploadUrl);
    Object.entries(headers).forEach(([k, v]) => {
      if (v != null && String(v).length > 0) {
        xhr.setRequestHeader(k, String(v));
      }
    });
    xhr.upload.onloadstart = () => {
      try {
        const t = file.size > 0 ? file.size : 1;
        options?.onByteProgress?.(0, t);
        options?.onProgress?.(1);
      } catch {
        /* ignore */
      }
    };
    xhr.upload.onprogress = (ev) => {
      const byteCb = options?.onByteProgress;
      const pctCb = options?.onProgress;
      if (!byteCb && !pctCb) return;
      if (ev.lengthComputable && ev.total > 0) {
        try {
          byteCb?.(ev.loaded, ev.total);
        } catch {
          /* ignore */
        }
        pctCb?.(Math.min(100, Math.round((ev.loaded / ev.total) * 100)));
        return;
      }
      if (file.size > 0 && ev.loaded > 0) {
        try {
          byteCb?.(ev.loaded, file.size);
        } catch {
          /* ignore */
        }
        pctCb?.(Math.min(95, Math.round((ev.loaded / file.size) * 100)));
        return;
      }
      if (ev.loaded > 0) {
        try {
          byteCb?.(ev.loaded, Math.max(ev.loaded, 1));
        } catch {
          /* ignore */
        }
        pctCb?.(50);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        importDiagLog('presigned_put_success', { name: file.name, http_status: xhr.status });
        resolve();
        return;
      }
      importDiagLog('presigned_put_http_error', { name: file.name, http_status: xhr.status });
      reject(new Error(`直传 MinIO 失败：HTTP ${xhr.status}`));
    };
    xhr.onerror = () => {
      importDiagLog('presigned_put_network_error', { name: file.name });
      reject(new Error('直传 MinIO 网络错误'));
    };
    xhr.send(file);
  });
}

export async function importDataAssetFiles(
  files: File[],
  projectId: string,
  projectName?: string,
  options?: { onProgress?: (percent: number) => void; signal?: AbortSignal }
): Promise<ApiResponse<{ imported: ImportResult['imported']; failed: ImportResult['failed'] }>> {
  const formData = new FormData();
  files.forEach((file) => {
    const anyFile = file as File & { webkitRelativePath?: string };
    const rel = anyFile.webkitRelativePath && anyFile.webkitRelativePath.trim();
    if (rel) {
      formData.append('files', file, rel);
    } else {
      formData.append('files', file);
    }
  });
  formData.append('project', projectId);
  if (projectName) formData.append('project_name', projectName);
  if (options?.onProgress != null || options?.signal) {
    return apiUploadWithProgress<{ imported: ImportResult['imported']; failed: ImportResult['failed'] }>(
      '/api/data-assets/import',
      formData,
      options?.onProgress,
      options?.signal
    );
  }
  return apiUpload<{ imported: ImportResult['imported']; failed: ImportResult['failed'] }>(
    '/api/data-assets/import',
    formData
  );
}

/** 按筛选条件导出清单为 CSV（GET，旧逻辑，保留兼容） */
export function exportDataAssets(params: DataAssetQueryParams = {}): void {
  const q = new URLSearchParams();
  if (params.keyword) q.set('keyword', params.keyword);
  if (params.project) q.set('project', params.project);
  if (params.format) q.set('format', params.format);
  const query = q.toString();
  window.open(`/api/data-assets/export${query ? `?${query}` : ''}`, '_blank');
}

/**
 * 按资产 ID 列表导出到指定路径（单条或批量）。不再走浏览器下载。
 */
export type ExportJobStatus =
  | 'validating'
  | 'collecting_files'
  | 'collecting_annotations'
  | 'generating_asset_list'
  | 'packaging'
  | 'writing'
  | 'ready'
  | 'failed'
  | 'cancelled';

export interface ExportJobCreated {
  jobId: string;
  status: ExportJobStatus;
}

export interface ExportJobStatusResponse {
  jobId: string;
  status: ExportJobStatus;
  progress: number;
  currentStep: string;
  fileName: string;
  downloadUrl: string;
  errorMessage: string;
  /** 本地导出时的导出目录名，如 export_20260306_182700 */
  exportDirName: string;
  /** 本地导出时的完整输出路径 */
  fullOutputPath: string;
  completedAssets: number;
  totalAssets: number;
  /** browser_zip：浏览器下载；local_path：写入服务器白名单目录 */
  deliveryMode: string;
}

export interface CreateExportJobParams {
  target: 'local';
  /** 不传或空则服务端生成临时 zip，通过 /export/download 下载 */
  output_path?: string;
}

export async function createExportJob(
  assetIds: number[],
  params: CreateExportJobParams
): Promise<ApiResponse<ExportJobCreated>> {
  if (!assetIds.length) return { ok: false, error: '请至少选择一个资产' };
  const body: Record<string, unknown> = {
    asset_ids: assetIds,
    target: params.target,
  };
  const op = (params.output_path || '').trim();
  if (op) body.output_path = op;
  return apiPost<ExportJobCreated>('/api/data-assets/export/jobs', body);
}

export type DownloadExportZipResult = { ok: boolean; error?: string; cancelled?: boolean };

/** 同一 jobId 并发/连点只走一次 fetch，避免服务端出现多条重复 /export/download */
const _exportZipDownloadInflight = new Map<string, Promise<DownloadExportZipResult>>();

async function downloadExportZipFileOnce(
  jobId: string,
  fallbackFileName?: string,
  onProgress?: (p: ExportZipDownloadProgress) => void
): Promise<DownloadExportZipResult> {
  return fetchAndSaveExportZip(jobId, fallbackFileName, onProgress);
}

/**
 * 用户点击后下载导出 zip：流式读取；优先 File System Access API 直写磁盘，否则 Blob + <a download>。
 * 同一 jobId 在请求未完成前重复调用会共享同一次下载流程（含进度回调合并到同一次 inflight 时仅后者生效）。
 */
export async function downloadExportZipFile(
  jobId: string,
  fallbackFileName?: string,
  onProgress?: (p: ExportZipDownloadProgress) => void
): Promise<DownloadExportZipResult> {
  let inflight = _exportZipDownloadInflight.get(jobId);
  if (inflight) return inflight;

  inflight = (async () => {
    try {
      return await downloadExportZipFileOnce(jobId, fallbackFileName, onProgress);
    } finally {
      _exportZipDownloadInflight.delete(jobId);
    }
  })();

  _exportZipDownloadInflight.set(jobId, inflight);
  return inflight;
}

/** 导出预览：是否有平台标注文件，用于弹窗摘要真实展示 */
export interface ExportPreviewResponse {
  has_annotations: boolean;
}

export async function getExportPreview(
  assetIds: number[]
): Promise<ApiResponse<ExportPreviewResponse>> {
  if (!assetIds.length) return { ok: false, error: '请至少选择一个资产' };
  return apiPost<ExportPreviewResponse>('/api/data-assets/export/preview', {
    asset_ids: assetIds,
  });
}

export async function getExportJobStatus(jobId: string): Promise<ApiResponse<ExportJobStatusResponse>> {
  return apiGet<ExportJobStatusResponse>(`/api/data-assets/export/status?jobId=${encodeURIComponent(jobId)}`);
}

/** 与 status 接口字段一致，另含 createdAtTs（Unix 毫秒）用于排序 */
export interface ExportJobListRow extends ExportJobStatusResponse {
  createdAtTs: number;
}

export async function listExportJobs(limit = 50): Promise<ApiResponse<ExportJobListRow[]>> {
  return apiGet<ExportJobListRow[]>(`/api/data-assets/export/jobs?limit=${encodeURIComponent(String(limit))}`);
}

/** 删除导出任务产物（zip 或目录），并移除任务记录。仅允许删除白名单内路径。 */
export async function deleteExportResult(jobId: string): Promise<ApiResponse<void>> {
  return apiPost<void>('/api/data-assets/export/delete-result', { job_id: jobId });
}

/** @deprecated 已改为导出到指定路径，不再使用浏览器下载 */
export async function downloadExportJob(_jobId: string): Promise<ApiResponse<void>> {
  return { ok: false, error: '当前版本请使用导出到指定路径' };
}

export async function deleteDataAsset(
  id: number,
  opts?: {
    deleteFile?: boolean;
    deleteRemote?: boolean;
    deleteCloud?: boolean;
  },
): Promise<ApiResponse<null>> {
  const params = new URLSearchParams();
  if (opts?.deleteFile) params.set('delete_file', 'true');
  if (opts?.deleteRemote) params.set('delete_remote', 'true');
  if (opts?.deleteCloud === false) params.set('delete_cloud', 'false');
  const q = params.toString() ? `?${params.toString()}` : '';
  return apiDelete<null>(`/api/data-assets/${id}${q}`);
}

/** 批量删除资产（服务端单接口 + 审计 ASSET_BATCH_DELETE） */
export async function deleteDataAssetsBatch(
  assetIds: number[],
  opts?: {
    deleteFile?: boolean;
    deleteRemote?: boolean;
    deleteCloud?: boolean;
  },
): Promise<ApiResponse<{ deleted: number[]; errors: string[]; warnings?: string[] }>> {
  return apiPost<{ deleted: number[]; errors: string[] }>('/api/data-assets/delete-batch', {
    asset_ids: assetIds,
    delete_file: Boolean(opts?.deleteFile),
    delete_remote: Boolean(opts?.deleteRemote),
    delete_cloud: opts?.deleteCloud !== false,
  });
}

export async function syncDataAsset(
  id: number
): Promise<ApiResponse<{ id: number; sync_status: string; message: string }>> {
  return apiPost<{ id: number; sync_status: string; message: string }>(`/api/data-assets/${id}/sync`, {});
}

/** 批量同步：创建异步任务，立即返回 jobId；请轮询 getSyncBatchJobStatus */
export async function createSyncBatchJob(body: {
  asset_ids: number[];
  agent_id?: string | null;
}): Promise<ApiResponse<{ jobId: string; status: string }>> {
  return apiPost<{ jobId: string; status: string }>('/api/data-assets/sync/batch', body);
}

export interface SyncBatchJobStatusItem {
  assetId: number;
  status: string;
  errorMessage: string;
  minioPath: string;
}

export async function getSyncBatchJobStatus(
  jobId: string
): Promise<
  ApiResponse<{
    jobId: string;
    status: string;
    total: number;
    succeeded: number;
    failed: number;
    progress: number;
    currentStep: string;
    errorMessage: string;
    agentId: string | null;
    items: SyncBatchJobStatusItem[];
  }>
> {
  const q = `?jobId=${encodeURIComponent(jobId)}`;
  return apiGet(`/api/data-assets/sync/batch/status${q}`);
}

export interface RegisterAssetRequest {
  project_id: string;
  project_name?: string;
  type: 'file' | 'dir';
  path: string;
}

/**
 * POST /api/data-assets/register：服务器白名单内路径登记（不经过浏览器上传）。
 * 数据页标准导入请用 importDataAssetFiles。
 */
export async function registerAsset(
  body: RegisterAssetRequest
): Promise<ApiResponse<{ id: number; name: string }>> {
  return apiPost<{ id: number; name: string }>('/api/data-assets/register', body);
}
