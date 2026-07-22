/**
 * 后台任务中心：统一任务类型与状态
 */

export type TaskType = 'export' | 'convert' | 'sync' | 'import';

export type TaskStatus =
  | 'queued'
  | 'running'
  | 'paused'
  | 'success'
  | 'failed'
  | 'cancelled';

/**
 * 浏览器直传同一 `upload_session_id` 断点续传：init 成功后的 PUT 清单（仅存任务 meta / localStorage）。
 * 后端 `upload-complete` 对已落库对象会去重，重复 PUT 同 key 后登记不会重复创建资产。
 */
export interface ImportDirectInitSnapshot {
  upload_session_id: string;
  upload_mode: string;
  expires_at?: string;
  root_dir_name?: string | null;
  items: Array<{
    relative_path: string;
    size_bytes: number;
    upload_url: string;
    method: string;
    headers: Record<string, string>;
  }>;
}

export interface BackgroundTaskMeta {
  format?: string;
  count?: number;
  outputPath?: string;
  outputFile?: string;
  /** 导出完成后的完整路径（zip 或目录），用于展示与后端删除依据 */
  fullOutputPath?: string;
  inputFormat?: string;
  outputFormat?: string;
  /** 浏览器内直传型导入，可写入 localStorage 并在刷新后恢复/标记中断 */
  importBrowserDirect?: boolean;
  /** 由 GET /data-assets/upload-sessions 恢复的任务（后端 upload_sessions 为真源） */
  importFromUploadSession?: boolean;
  /** 直传 upload_session_id；单会话导入与任务 id「import:{id}」一致 */
  importUploadSessionId?: string;
  /** 多会话目录树批次可能对应多条后端会话，刷新后列表分别恢复 */
  importSessionIds?: string[];
  /** 直传 init 成功后的履约快照，供「继续导入」复用同一会话（单会话批次） */
  importDirectInitSnapshot?: ImportDirectInitSnapshot;
  /** 数据导入任务 */
  importProjectId?: string;
  importProjectName?: string;
  importModeSummary?: string;
  importAssetNamesPreview?: string;
  importCompletedAssets?: number;
  importFailedAssets?: number;
  /** 直传进行中：任务级已上传/总字节（详情与诊断） */
  importUploadBytesLoaded?: number;
  importUploadBytesTotal?: number;
  /** 批量导入成功项名称（若干条，用于详情） */
  importSuccessAssetNames?: string[];
  /** 批量导入失败项：名称 + 原因（多条） */
  importFailureEntries?: Array<{ name: string; reason: string }>;
  /** 存在失败项但亦有成功项时用于详情提示（主状态仍为 success） */
  importHadPartialFailures?: boolean;
  /** 导出任务（浏览器 zip） */
  exportDeliveryMode?: 'browser_zip' | 'local_path';
  exportCompletedAssets?: number;
  exportTotalAssets?: number;
  exportProjectId?: string;
  exportProjectName?: string;
  exportFormatSummary?: string;
  exportAssetNamesPreview?: string;
  /** 转换任务：各状态计数（与后端 batches 接口一致） */
  convertBatchTotal?: number;
  convertBatchSuccess?: number;
  convertBatchFailed?: number;
  convertBatchRunning?: number;
  convertBatchPending?: number;
  convertBatchProgressPercent?: number;
  convertBatchOverall?: string;
  convertBatchLegacy?: boolean;
}

export interface BackgroundTask {
  id: string;
  type: TaskType;
  title: string;
  status: TaskStatus;
  progress: number;
  currentStep: string;
  createdAt: string;
  updatedAt: string;
  meta?: BackgroundTaskMeta;
  errorMessage?: string;
  /** 导出任务：后端 jobId，用于轮询 */
  exportJobId?: string;
  /** 转换任务：后端 jobId，用于轮询（单条旧任务） */
  convertJobId?: string;
  /** 转换列表对应的后台任务标识，用于轮询与取消（与 convertJobId 互斥时优先使用本字段） */
  convertBatchId?: string;
  /** 同步任务：后端 jobId（sync_batch_xxx），用于轮询/取消 */
  syncJobId?: string;
}

/** 进行中（面板「进行中」tab 与清空已完成用） */
export function isActiveStatus(s: TaskStatus): boolean {
  return s === 'queued' || s === 'running' || s === 'paused';
}

/** 排队中、执行中、已暂停计入右下角 badge（与「进行中」标签一致） */
export function isCountedInBadge(s: TaskStatus): boolean {
  return s === 'queued' || s === 'running' || s === 'paused';
}

export const TASK_STATUS_LABEL_KEY: Record<TaskStatus, string> = {
  queued: 'backgroundTasks.statusQueued',
  running: 'backgroundTasks.statusRunning',
  paused: 'backgroundTasks.statusPaused',
  success: 'backgroundTasks.statusSuccess',
  failed: 'backgroundTasks.statusFailed',
  cancelled: 'backgroundTasks.statusCancelled',
};

export const TASK_TYPE_LABEL_KEY: Record<TaskType, string> = {
  export: 'backgroundTasks.typeExport',
  convert: 'backgroundTasks.typeConvert',
  sync: 'backgroundTasks.typeSync',
  import: 'backgroundTasks.typeImport',
};
