'use client';

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  useRef,
  useEffect,
} from 'react';
import { apiGet } from '@/features/data-platform/api/client';
import {
  createExportJob,
  getExportJobStatus,
  deleteExportResult,
  createSyncBatchJob,
  getSyncBatchJobStatus,
  getDataAsset,
  isDataAssetSynced,
  listExportJobs,
  listUploadSessions,
  cancelUploadSession,
  type UploadSessionListItem,
} from '@/features/data-platform/api/dataAssetsApi';
import { cancelTaskJob } from '@/features/data-platform/api/taskJobsApi';
import type { ExportJobListRow } from '@/features/data-platform/api/dataAssetsApi';
import {
  getJob,
  deleteConversionResult,
  listBatches,
  getBatchDetail,
  deleteConversionBatch,
  cancelConversionBatch,
  type ConversionBatchSummary,
} from '@/lib/conversion/conversionApi';
import type { ConversionJob } from '@/lib/conversion/mockConversion';
import type { BackgroundTask, ImportDirectInitSnapshot, TaskStatus } from './types';
import { isActiveStatus, isCountedInBadge } from './types';
import type { DataImportProgress, DataImportResult } from '@/components/assets/dataAssetImportRunner';
import { importDiagLog } from '@/lib/importDiagLog';

const ASSISTANT_POSITION_KEY = 'epi_assistant_position';
const ASSISTANT_HIDDEN_KEY = 'epi_assistant_hidden';
const ASSISTANT_SIZE = 64;
const ASSISTANT_MARGIN = 24;

export type AssistantPosition = { x: number; y: number };

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function getDefaultAssistantPosition(): AssistantPosition {
  if (typeof window === 'undefined') return { x: 0, y: 0 };
  const x = window.innerWidth - ASSISTANT_MARGIN - ASSISTANT_SIZE;
  const y = window.innerHeight - ASSISTANT_MARGIN - ASSISTANT_SIZE;
  return { x: Math.max(ASSISTANT_MARGIN, x), y: Math.max(ASSISTANT_MARGIN, y) };
}

function clampToViewport(pos: AssistantPosition): AssistantPosition {
  if (typeof window === 'undefined') return pos;
  const min = 8;
  const maxX = Math.max(min, window.innerWidth - ASSISTANT_SIZE - min);
  const maxY = Math.max(min, window.innerHeight - ASSISTANT_SIZE - min);
  return { x: clamp(pos.x, min, maxX), y: clamp(pos.y, min, maxY) };
}

function readStoredPosition(): AssistantPosition | null {
  try {
    if (typeof window === 'undefined') return null;
    const raw = window.localStorage.getItem(ASSISTANT_POSITION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AssistantPosition>;
    if (typeof parsed.x !== 'number' || typeof parsed.y !== 'number') return null;
    return clampToViewport({ x: parsed.x, y: parsed.y });
  } catch {
    return null;
  }
}

function writeStoredPosition(pos: AssistantPosition) {
  try {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(ASSISTANT_POSITION_KEY, JSON.stringify(pos));
  } catch {
    // ignore
  }
}

function readStoredHidden(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(ASSISTANT_HIDDEN_KEY) === 'true';
  } catch {
    return false;
  }
}

function writeStoredHidden(hidden: boolean) {
  try {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(ASSISTANT_HIDDEN_KEY, hidden ? 'true' : 'false');
  } catch {
    // ignore
  }
}

function normalizeFormatLabel(fmt?: string): string {
  const v = (fmt || '').trim();
  if (!v) return '';
  const lower = v.toLowerCase();
  if (lower === 'hdf5' || lower === 'h5') return 'HDF5';
  if (lower === 'mcap') return 'MCAP';
  if (lower === 'lerobot') return 'LeRobot';
  if (lower === 'zip') return 'LeRobot';
  return v.toUpperCase();
}

function inferFormatFromPath(p?: string): string {
  const s = (p || '').trim().toLowerCase();
  if (!s) return '';
  if (s.endsWith('.hdf5') || s.endsWith('.h5')) return 'HDF5';
  if (s.endsWith('.mcap')) return 'MCAP';
  if (s.endsWith('.zip')) return 'LeRobot';
  return '';
}

function stageLabel(stage?: string | null): string {
  const s = (stage || '').trim();
  if (!s) return '';
  const map: Record<string, string> = {
    init: '初始化',
    Parse: '解析',
    Align: '对齐',
    Write: '写入',
    Validate: '校验',
  };
  return map[s] ?? map[s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()] ?? s;
}

const BROWSER_IMPORT_STORAGE_KEY = 'eai_task_center_browser_imports_v1';
const DISMISSED_TASK_IDS_KEY = 'eai_task_center_dismissed_task_ids_v1';

function exportStableTaskId(jobId: string): string {
  return `export:${jobId}`;
}

function convertStableTaskId(jobId: string): string {
  return `convert:${jobId}`;
}

function convertBatchStableTaskId(batchId: string): string {
  return `convertBatch:${batchId}`;
}

function mapOverallToTaskStatus(overall: string): TaskStatus {
  const u = (overall || '').toUpperCase();
  if (u === 'PENDING') return 'queued';
  if (u === 'RUNNING') return 'running';
  if (u === 'SUCCESS') return 'success';
  if (u === 'PARTIAL_SUCCESS') return 'success';
  if (u === 'FAILED') return 'failed';
  if (u === 'CANCELED') return 'cancelled';
  return 'running';
}

function mapBatchSummaryToConvertTask(b: ConversionBatchSummary): BackgroundTask {
  const total = Math.max(1, b.totalCount || 1);
  const ended = (b.successCount || 0) + (b.failedCount || 0);
  const inputFmt = 'MCAP';
  const outputFmt = normalizeFormatLabel(b.targetFormat) || 'HDF5';
  const st = mapOverallToTaskStatus(String(b.overallStatus || ''));
  const progress =
    typeof b.progressPercent === 'number' && Number.isFinite(b.progressPercent)
      ? Math.min(100, Math.max(0, b.progressPercent))
      : st === 'success' || st === 'cancelled'
        ? 100
        : Math.round((ended / total) * 100);
  const sub = `${ended} / ${total}`;
  const stats = `成功 ${b.successCount ?? 0} · 失败 ${b.failedCount ?? 0} · 处理中 ${(b.runningCount ?? 0) + (b.pendingCount ?? 0)}`;
  return {
    id: convertBatchStableTaskId(b.batchId),
    type: 'convert',
    title: `转换 ${inputFmt} → ${outputFmt}（${total}条）`,
    status: st,
    progress,
    currentStep: `${sub} · ${stats}`,
    createdAt: b.createdAt || new Date().toISOString(),
    updatedAt: b.updatedAt || b.createdAt || new Date().toISOString(),
    convertBatchId: b.batchId,
    meta: {
      inputFormat: inputFmt,
      outputFormat: outputFmt,
      count: total,
      convertBatchTotal: total,
      convertBatchSuccess: b.successCount,
      convertBatchFailed: b.failedCount,
      convertBatchRunning: b.runningCount,
      convertBatchPending: b.pendingCount,
      convertBatchProgressPercent: b.progressPercent,
      convertBatchOverall: String(b.overallStatus || ''),
      convertBatchLegacy: !!b.legacySingleFile,
    },
  };
}

function importStableTaskId(sessionId: string): string {
  return `import:${sessionId}`;
}

/** 同一导入任务在 genId 与 import:session 下可能各注册一次，取消/收尾时需按 controller 实例一并移除 */
function purgeImportAbortEntries(map: Map<string, AbortController>, controller: AbortController): void {
  for (const [k, v] of [...map.entries()]) {
    if (v === controller) map.delete(k);
  }
}

/**
 * 仅解析真实 upload_session_id。
 * 禁止把尚未绑定会话的 genId 任务 id 当作 session（否则会调 cancel 报「上传会话不存在」）。
 */
function resolveImportTaskSessionId(task: BackgroundTask): string {
  const fromMeta = (task.meta?.importUploadSessionId || '').trim();
  if (fromMeta) return fromMeta;
  const ids = task.meta?.importSessionIds;
  if (Array.isArray(ids)) {
    for (let i = ids.length - 1; i >= 0; i--) {
      const s = String(ids[i] || '').trim();
      if (s) return s;
    }
  }
  const id = (task.id || '').trim();
  if (id.startsWith('import:')) return id.slice('import:'.length).trim();
  return '';
}

function uploadModeSummaryLabel(mode: string): string {
  const m = (mode || '').trim().toLowerCase();
  if (m === 'single_file') return '单文件直传';
  if (m === 'multi_file') return '多文件直传';
  if (m === 'directory') return '目录直传';
  return (mode || '直传').trim() || '直传';
}

function parseIsoMs(iso?: string | null): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? 0 : t;
}

function resolveImportTaskFromUploadSessionRow(
  row: UploadSessionListItem,
  liveLocalUpload: boolean
): BackgroundTask {
  const task = mapUploadSessionListRowToTask(row);
  if (liveLocalUpload) return task;
  const raw = (row.status || '').trim().toLowerCase();
  /**
   * 浏览器直传：DB 在 PUT 完成前长期为 presigned；mapUploadSessionListRowToTask 只能给占位「执行中 10%」。
   * 若本标签页并无活跃 XHR（含整页刷新后），上传已不可能继续，必须标为已中断，不能依赖「等 14 分钟」才变状态。
   * （expires_at 已过时 map 已映射为 expired→失败，此处若非 live 则可能为失败态。）
   */
  if (
    raw === 'presigned' &&
    task.meta?.importBrowserDirect === true &&
    task.status === 'running'
  ) {
    return {
      ...task,
      status: 'failed',
      progress: 100,
      currentStep: 'backgroundTasks.importBrowserInterrupted',
      errorMessage: 'backgroundTasks.importBrowserInterrupted',
    };
  }
  return task;
}

/**
 * 从 upload_sessions 恢复时后端对 presigned 仅能给占位进度（10%）；若 localStorage 仍有「进行中」的直传记录，
 * 说明本次挂载并无活跃 XHR，直传已无法继续，合并为「已中断」而非假「执行中 10%」。
 */
function mergeBrowserImportHydrateFromStoredApi(api: BackgroundTask, storedImports: BackgroundTask[]): BackgroundTask {
  if (!api.meta?.importBrowserDirect) return api;
  const terminal = api.status === 'success' || api.status === 'failed' || api.status === 'cancelled';
  if (terminal) return api;
  const sidApi = (api.meta?.importUploadSessionId || api.id.replace(/^import:/, '')).trim();
  if (!sidApi) return api;
  const stored = storedImports.find((s) => {
    if (s.type !== 'import' || !s.meta?.importBrowserDirect) return false;
    const sid = (s.meta?.importUploadSessionId || s.id.replace(/^import:/, '')).trim();
    return sid === sidApi;
  });
  if (!stored || !isActiveStatus(stored.status)) return api;
  return {
    ...api,
    status: 'failed',
    progress: 100,
    currentStep: 'backgroundTasks.importBrowserInterrupted',
    errorMessage: 'backgroundTasks.importBrowserInterrupted',
    meta: { ...api.meta, ...stored.meta },
  };
}

function mapUploadSessionListRowToTask(row: UploadSessionListItem): BackgroundTask {
  const sid = (row.upload_session_id || '').trim();
  const now = Date.now();
  const expMs = parseIsoMs(row.expires_at);
  const raw = (row.status || '').trim().toLowerCase();
  let effective = raw;
  if (raw === 'presigned' && expMs > 0 && now > expMs) {
    effective = 'expired';
  }

  let status: TaskStatus;
  let progress: number;
  let errorMessage: string | undefined;
  let currentStep: string;

  if (effective === 'completed') {
    status = 'success';
    progress = 100;
    currentStep = 'backgroundTasks.statusSuccess';
  } else if (effective === 'cancelled' || effective === 'canceled') {
    status = 'cancelled';
    progress = 0;
    currentStep = 'backgroundTasks.statusCancelled';
  } else if (effective === 'failed') {
    status = 'failed';
    progress = 100;
    currentStep = 'backgroundTasks.statusFailed';
    errorMessage = 'backgroundTasks.importUploadSessionFailed';
  } else if (effective === 'expired') {
    status = 'failed';
    progress = 100;
    currentStep = 'backgroundTasks.importSessionExpiredHint';
    errorMessage = 'backgroundTasks.importSessionExpiredHint';
  } else {
    status = 'running';
    progress = 10;
    currentStep = 'backgroundTasks.importSessionBackendProgressPlaceholder';
  }

  const n = typeof row.expected_count === 'number' && row.expected_count > 0 ? row.expected_count : 0;
  const fn = (row.filename || '').trim();
  const root = (row.root_dir_name || '').trim();
  const title =
    n > 0 ? `导入数据（${n} 条）` : root ? `导入数据（${root}）` : fn ? `导入数据（${fn}）` : '导入数据';

  const createdAt = row.created_at || new Date().toISOString();

  return {
    id: importStableTaskId(sid),
    type: 'import',
    title,
    status,
    progress,
    currentStep,
    createdAt,
    updatedAt: createdAt,
    errorMessage,
    meta: {
      importBrowserDirect: true,
      importFromUploadSession: true,
      importUploadSessionId: sid,
      count: n > 0 ? n : 1,
      importProjectId: row.project_id || undefined,
      importModeSummary: uploadModeSummaryLabel(row.upload_mode || ''),
      importAssetNamesPreview: fn || root || undefined,
    },
  };
}

function annotateRefreshedActiveImport(t: BackgroundTask): BackgroundTask {
  if (!isActiveStatus(t.status)) return t;
  /**
   * 仅从 localStorage 恢复的、仍显示为进行中的导入：整页刷新后前端执行体已不存在，
   * 不能再继续直传或标准导入进度，必须收口为失败，避免出现「import: 会话 id / 已暂停 / 0%」长期挂在进行中。
   */
  return {
    ...t,
    status: 'failed',
    progress: 100,
    currentStep: 'backgroundTasks.importBrowserInterrupted',
    errorMessage: 'backgroundTasks.importBrowserInterrupted',
    updatedAt: new Date().toISOString(),
  };
}

function isGenericImportPendingStep(step?: string): boolean {
  const s = (step || '').trim();
  return (
    s === 'backgroundTasks.importSessionPendingRefreshHint' ||
    s === 'backgroundTasks.importSessionBackendProgressPlaceholder'
  );
}

function mapExportListRowToTask(row: ExportJobListRow): BackgroundTask {
  const jobId = (row.jobId || '').trim();
  const delivery = (row.deliveryMode || '').trim();
  const browserZip =
    delivery === 'browser_zip' ||
    (!!(row.downloadUrl || '').trim() && !(row.fullOutputPath || '').trim());
  let status: TaskStatus = 'running';
  if (row.status === 'ready') status = 'success';
  else if (row.status === 'failed') status = 'failed';
  else if (row.status === 'cancelled') status = 'cancelled';
  const ts = typeof row.createdAtTs === 'number' && row.createdAtTs > 0 ? row.createdAtTs : Date.now();
  const createdAt = new Date(ts).toISOString();
  let currentStep = row.currentStep || '';
  if (status === 'success' && browserZip) {
    currentStep = 'backgroundTasks.exportReadyClickDownload';
  }
  const total = typeof row.totalAssets === 'number' ? row.totalAssets : 0;
  const title = total > 0 ? `导出数据（${total} 条）` : '导出数据';
  const progress =
    status === 'success' ? 100 : typeof row.progress === 'number' ? row.progress : 5;
  return {
    id: exportStableTaskId(jobId),
    type: 'export',
    title,
    status,
    progress,
    currentStep,
    createdAt,
    updatedAt: createdAt,
    exportJobId: jobId,
    errorMessage: row.errorMessage || undefined,
    meta: {
      count: total > 0 ? total : 1,
      exportDeliveryMode: browserZip ? 'browser_zip' : 'local_path',
      exportTotalAssets: total > 0 ? total : undefined,
      exportCompletedAssets: typeof row.completedAssets === 'number' ? row.completedAssets : undefined,
      outputFile: row.fileName || undefined,
      fullOutputPath: row.fullOutputPath || undefined,
      outputPath: row.fullOutputPath ? row.fullOutputPath.replace(/\/[^/]+$/, '') : undefined,
    },
  };
}

function mapConversionJobToTask(job: ConversionJob): BackgroundTask {
  const now = new Date().toISOString();
  const status = toTaskStatus((job.status || '').toLowerCase());
  const inputFormat =
    normalizeFormatLabel((job as any).fileFormat) ||
    inferFormatFromPath((job as any).fileName) ||
    inferFormatFromPath((job as any).assetName) ||
    '—';
  const outputFormat =
    normalizeFormatLabel((job as any).outputFormat) ||
    inferFormatFromPath((job as any).outputFileName) ||
    inferFormatFromPath((job as any).fileName) ||
    '—';
  const currentStep = job.currentStage
    ? `阶段：${stageLabel(job.currentStage)}`
    : status === 'queued'
      ? '排队中'
      : status === 'success'
        ? '已完成'
        : status === 'failed'
          ? '失败'
          : status === 'cancelled'
            ? '已取消'
            : '执行中';
  return {
    id: convertStableTaskId(job.jobId),
    type: 'convert',
    title: `转换 ${inputFormat} → ${outputFormat}`,
    status,
    progress: typeof job.progressPercent === 'number' ? job.progressPercent : status === 'success' ? 100 : 0,
    currentStep,
    createdAt: job.createdAt || now,
    updatedAt: job.updatedAt || job.createdAt || now,
    meta: {
      inputFormat,
      outputFormat,
      count: 1,
      outputPath: job.outputPath,
      outputFile: job.outputFileName || job.fileName,
    },
    convertJobId: job.jobId,
    errorMessage: job.errorMessage,
  };
}

function readStoredBrowserImportTasks(): BackgroundTask[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(BROWSER_IMPORT_STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return [];
    return arr.filter(
      (t): t is BackgroundTask =>
        t != null &&
        typeof t === 'object' &&
        (t as BackgroundTask).type === 'import' &&
        !!(t as BackgroundTask).meta?.importBrowserDirect
    );
  } catch {
    return [];
  }
}

function readDismissedTaskIds(): Set<string> {
  if (typeof window === 'undefined') return new Set();
  try {
    const raw = localStorage.getItem(DISMISSED_TASK_IDS_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return new Set();
    return new Set(arr.filter((x): x is string => typeof x === 'string' && x.trim().length > 0));
  } catch {
    return new Set();
  }
}

function writeDismissedTaskIds(ids: Set<string>) {
  if (typeof window === 'undefined') return;
  try {
    if (ids.size === 0) {
      localStorage.removeItem(DISMISSED_TASK_IDS_KEY);
      return;
    }
    localStorage.setItem(DISMISSED_TASK_IDS_KEY, JSON.stringify(Array.from(ids)));
  } catch {
    // ignore
  }
}

function addDismissedTaskIds(ids: string[]) {
  if (ids.length === 0) return;
  const next = readDismissedTaskIds();
  for (const id of ids) {
    const v = (id || '').trim();
    if (v) next.add(v);
  }
  writeDismissedTaskIds(next);
}

function isTaskActiveInPanel(task: BackgroundTask): boolean {
  return task.status === 'queued' || task.status === 'running' || task.status === 'paused';
}

type TaskPanelGroup = 'active' | 'done' | 'failed';

function taskPanelGroup(task: BackgroundTask): TaskPanelGroup {
  if (task.status === 'failed' || task.status === 'cancelled') return 'failed';
  if (isTaskActiveInPanel(task)) return 'active';
  return 'done';
}

function toTaskStatus(s: string): TaskStatus {
  const map: Record<string, TaskStatus> = {
    queued: 'queued',
    running: 'running',
    paused: 'paused',
    succeeded: 'success',
    success: 'success',
    failed: 'failed',
    canceled: 'cancelled',
    cancelled: 'cancelled',
  };
  return map[s] ?? 'running';
}

interface TaskCenterContextValue {
  tasks: BackgroundTask[];
  activeCount: number;
  panelOpen: boolean;
  setPanelOpen: (open: boolean) => void;
  assistantHidden: boolean;
  assistantVisible: boolean;
  assistantPosition: AssistantPosition;
  setAssistantPosition: (pos: AssistantPosition) => void;
  persistAssistantPosition: (pos: AssistantPosition) => void;
  hideAssistant: () => void;
  /** 仅用于“启动后台任务”时恢复显示（导出/转换统一调用）。 */
  showAssistantForBackgroundTask: (opts?: { openPanel?: boolean }) => void;
  addExportTask: (params: {
    assetIds: number[];
    exportCount: number;
    formatLabel: string;
    formatSummary?: string;
    projectId?: string;
    projectName?: string;
    assetNamesPreview?: string;
  }) => Promise<{ ok: boolean; error?: string }>;
  addConvertTask: (job: ConversionJob) => void;
  addConvertBatchTask: (batch: ConversionBatchSummary) => void;
  addSyncTask: (params: { assetId: number; filename?: string }) => Promise<{ ok: boolean; error?: string }>;
  addBatchSyncTask: (params: { count: number; title?: string }) => string;
  /** 数据导入：直传进度由前端上报；刷新后由 upload_sessions + upload-sessions 接口恢复 */
  runDataImportJob: (params: {
    title: string;
    totalUnits: number;
    projectId: string;
    projectName: string;
    modeSummary: string;
    assetNamesPreview: string;
    onImportFinished?: (message?: string) => void;
    run: (
      onProgress: (p: DataImportProgress) => void,
      opts?: {
        signal?: AbortSignal;
        onUploadSessionReady?: (uploadSessionId: string) => void;
        onDirectInitSnapshot?: (snap: ImportDirectInitSnapshot) => void;
      }
    ) => Promise<DataImportResult>;
  }) => void;
  updateTask: (id: string, patch: Partial<BackgroundTask>) => void;
  removeTask: (id: string) => void;
  /** 仅从当前面板移除分组内任务，不调后端删除。 */
  clearTaskGroup: (group: TaskPanelGroup) => void;
  /** 真实删除任务产物（导出 zip/目录 或 转换结果）并移除任务记录。返回 { ok, error }。 */
  deleteTaskResult: (task: BackgroundTask) => Promise<{ ok: boolean; error?: string }>;
  /** 进行中任务：调用协作式取消（export/convert TaskJob、import Abort、sync 仅移除展示） */
  cancelBackgroundTask: (
    task: BackgroundTask
  ) => Promise<{ ok: boolean; error?: string; displayOnly?: boolean }>;
}

const TaskCenterContext = createContext<TaskCenterContextValue | null>(null);

function genId(): string {
  return `task_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

const TASK_CENTER_POLL_INTERVAL_MS = 5000;

function hasActiveImportExportPollTargets(list: BackgroundTask[]): boolean {
  const exportConvert = list.some(
    (t) =>
      (t.status === 'queued' || t.status === 'running') &&
      (t.exportJobId || t.convertJobId || t.convertBatchId)
  );
  const importTasks = list.some(
    (t) => t.type === 'import' && isActiveStatus(t.status) && t.id.startsWith('import:')
  );
  return exportConvert || importTasks;
}

export function TaskCenterProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<BackgroundTask[]>([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const [assistantHidden, setAssistantHidden] = useState(false);
  const [assistantPosition, setAssistantPosition] = useState<AssistantPosition>({ x: 0, y: 0 });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollActiveTasksRef = useRef<(() => Promise<void>) | null>(null);
  const tasksRef = useRef<BackgroundTask[]>([]);
  tasksRef.current = tasks;
  const importAbortRef = useRef<Map<string, AbortController>>(new Map());
  const syncInFlightRef = useRef<Set<number>>(new Set());
  /** 首帧 tasks=[] 时勿写回 localStorage，避免 Strict Mode / 异步恢复前误删 eai_task_center_browser_imports_v1 */
  const browserImportHydrateDoneRef = useRef(false);

  // 初始化：恢复导出 / 转换 / 直传导入会话（后端）+ 合并本地暂存的导入（localStorage）
  useEffect(() => {
    const hidden = readStoredHidden();
    setAssistantHidden(hidden);
    const storedPos = readStoredPosition();
    setAssistantPosition(storedPos ?? clampToViewport(getDefaultAssistantPosition()));
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      let importApiRestored: BackgroundTask[] = [];
      let exportRestored: BackgroundTask[] = [];
      let convertRestored: BackgroundTask[] = [];
      try {
        const res = await listExportJobs(50);
        if (!cancelled && res.ok && Array.isArray(res.data)) {
          exportRestored = res.data
            .filter((row) => (row.jobId || '').trim())
            .map((row) => mapExportListRowToTask(row));
        }
      } catch {
        /* 未登录或网络失败时跳过导出恢复 */
      }
      try {
        const batches = await listBatches();
        if (!cancelled && Array.isArray(batches)) {
          // 恢复近期批量（含 PARTIAL_SUCCESS/FAILED/CANCELED），避免刷新后只剩「执行中」占位或任务从面板消失
          convertRestored = batches
            .filter((b) => (b.batchId || '').trim())
            .slice(0, 50)
            .map((b) => mapBatchSummaryToConvertTask(b));
        }
      } catch {
        /* 未登录或网络失败时跳过转换恢复 */
      }
      const rawStoredImports = readStoredBrowserImportTasks();
      try {
        const ures = await listUploadSessions(80);
        if (!cancelled && ures.ok && Array.isArray(ures.data?.items)) {
          importApiRestored = ures.data!.items
            .filter((row) => (row.upload_session_id || '').trim())
            .map((row) =>
              mergeBrowserImportHydrateFromStoredApi(
                resolveImportTaskFromUploadSessionRow(
                  row,
                  importAbortRef.current.has(importStableTaskId((row.upload_session_id || '').trim()))
                ),
                rawStoredImports
              )
            );
        }
      } catch {
        /* 未登录或网络失败时跳过导入会话恢复 */
      }
      const importApiIds = new Set(importApiRestored.map((t) => t.id));
      const importApiSessionIds = new Set(
        importApiRestored.map((t) => (t.meta?.importUploadSessionId || '').trim()).filter(Boolean)
      );
      const importFromStorage = readStoredBrowserImportTasks()
        .map(annotateRefreshedActiveImport)
        .filter((t) => {
          if (importApiIds.has(t.id)) return false;
          const sid = (t.meta?.importUploadSessionId || '').trim();
          if (sid && importApiSessionIds.has(sid)) return false;
          return true;
        });
      if (cancelled) return;
      browserImportHydrateDoneRef.current = true;
      const dismissedIds = readDismissedTaskIds();
      setTasks((prev) => {
        const byId = new Map<string, BackgroundTask>();
        for (const t of prev) {
          byId.set(t.id, { ...t });
        }
        for (const t of exportRestored) {
          byId.set(t.id, t);
        }
        for (const t of convertRestored) {
          byId.set(t.id, t);
        }
        for (const t of importApiRestored) {
          byId.set(t.id, t);
        }
        for (const t of importFromStorage) {
          if (!byId.has(t.id)) {
            byId.set(t.id, t);
          }
        }
        return Array.from(byId.values())
          .filter((t) => !dismissedIds.has(t.id))
          .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
      });
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!browserImportHydrateDoneRef.current) return;
    try {
      const toStore = tasks.filter((t) => t.type === 'import' && t.meta?.importBrowserDirect === true);
      if (toStore.length === 0) {
        localStorage.removeItem(BROWSER_IMPORT_STORAGE_KEY);
      } else {
        localStorage.setItem(BROWSER_IMPORT_STORAGE_KEY, JSON.stringify(toStore));
      }
    } catch {
      /* ignore */
    }
  }, [tasks]);

  // 视口变化时做边界约束（避免拖出去后缩放窗口导致不可见）
  useEffect(() => {
    const onResize = () => setAssistantPosition((p) => clampToViewport(p));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const activeCount = useMemo(
    () => tasks.filter((t) => isCountedInBadge(t.status)).length,
    [tasks]
  );

  const assistantVisible = !assistantHidden;

  const persistAssistantPosition = useCallback((pos: AssistantPosition) => {
    const clamped = clampToViewport(pos);
    setAssistantPosition(clamped);
    writeStoredPosition(clamped);
  }, []);

  const hideAssistant = useCallback(() => {
    setAssistantHidden(true);
    writeStoredHidden(true);
    setPanelOpen(false);
  }, []);

  const showAssistantForBackgroundTask = useCallback((opts?: { openPanel?: boolean }) => {
    // 只有“启动后台任务”时才允许自动恢复
    setAssistantHidden(false);
    writeStoredHidden(false);
    if (opts?.openPanel) setPanelOpen(true);
  }, []);

  const updateTask = useCallback((id: string, patch: Partial<BackgroundTask>) => {
    setTasks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const { meta: patchMeta, ...rest } = patch;
        const next: BackgroundTask = { ...t, ...rest, updatedAt: new Date().toISOString() };
        if (patchMeta !== undefined) {
          next.meta = { ...t.meta, ...patchMeta };
        }
        return next;
      })
    );
  }, []);

  const removeTask = useCallback((id: string) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const clearTaskGroup = useCallback((group: TaskPanelGroup) => {
    setTasks((prev) => {
      const removedIds = prev.filter((t) => taskPanelGroup(t) === group).map((t) => t.id);
      addDismissedTaskIds(removedIds);
      return prev.filter((t) => taskPanelGroup(t) !== group);
    });
  }, []);

  const deleteTaskResult = useCallback(async (task: BackgroundTask): Promise<{ ok: boolean; error?: string }> => {
    if (task.type === 'export') {
      if (!task.exportJobId) return { ok: false, error: 'feedback.requestFailed' };
      const res = await deleteExportResult(task.exportJobId);
      return { ok: !!res.ok, error: res.error };
    }
    if (task.type === 'convert') {
      if (task.convertBatchId) {
        try {
          await deleteConversionBatch(task.convertBatchId);
          return { ok: true };
        } catch (e) {
          return { ok: false, error: e instanceof Error ? e.message : '删除失败' };
        }
      }
      if (!task.convertJobId) return { ok: false, error: 'feedback.requestFailed' };
      return deleteConversionResult(task.convertJobId);
    }
    if (task.type === 'import' || task.type === 'sync') {
      return { ok: true };
    }
    return { ok: false, error: 'feedback.requestFailed' };
  }, []);

  const cancelBackgroundTask = useCallback(
    async (task: BackgroundTask): Promise<{ ok: boolean; error?: string; displayOnly?: boolean }> => {
      if (task.type === 'export') {
        if (!task.exportJobId) return { ok: false, error: 'feedback.requestFailed' };
        const res = await cancelTaskJob(task.exportJobId);
        if (!res.ok) return { ok: false, error: res.error || '取消失败' };
        updateTask(task.id, {
          status: 'cancelled',
          progress: 0,
          currentStep: 'backgroundTasks.statusCancelled',
          errorMessage: undefined,
        });
        return { ok: true };
      }
      if (task.type === 'convert') {
        if (task.convertBatchId) {
          try {
            await cancelConversionBatch(task.convertBatchId);
          } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : '取消失败' };
          }
          updateTask(task.id, {
            status: 'cancelled',
            progress: 100,
            currentStep: 'backgroundTasks.statusCancelled',
            errorMessage: undefined,
          });
          return { ok: true };
        }
        if (!task.convertJobId) return { ok: false, error: 'feedback.requestFailed' };
        const res = await cancelTaskJob(task.convertJobId);
        if (!res.ok) return { ok: false, error: res.error || '取消失败' };
        updateTask(task.id, {
          status: 'cancelled',
          progress: 0,
          currentStep: 'backgroundTasks.statusCancelled',
          errorMessage: undefined,
        });
        return { ok: true };
      }
      if (task.type === 'import') {
        if (task.status === 'success' || task.status === 'failed' || task.status === 'cancelled') {
          return { ok: true };
        }
        const sid = resolveImportTaskSessionId(task);
        const m = importAbortRef.current;
        let ac = m.get(task.id);
        if (!ac && sid) ac = m.get(importStableTaskId(sid));
        importDiagLog('import_cancel_request', {
          taskId: task.id,
          upload_session_id: sid || null,
          hasLocalAbortController: !!ac,
        });
        if (ac) {
          try {
            ac.abort();
          } catch {
            /* ignore */
          }
          purgeImportAbortEntries(m, ac);
        }
        const clearImportAbortKeysById = () => {
          m.delete(task.id);
          if (sid) m.delete(importStableTaskId(sid));
        };
        if (sid) {
          const res = await cancelUploadSession(sid);
          if (!res.ok) {
            const msg = (res.error || '').trim();
            const benign =
              msg.includes('上传会话不存在') ||
              msg.includes('上传会话已完成') ||
              msg.includes('不能取消') ||
              msg.includes('已过期');
            if (!benign) {
              if (!ac) clearImportAbortKeysById();
              updateTask(task.id, {
                status: 'failed',
                progress: 100,
                currentStep: msg || '取消失败',
                errorMessage: msg || '取消失败',
              });
              return { ok: false, error: msg || '取消失败' };
            }
            importDiagLog('import_cancel_backend_benign', { taskId: task.id, upload_session_id: sid, error: msg });
          }
        }
        if (!ac) clearImportAbortKeysById();
        updateTask(task.id, {
          status: 'cancelled',
          progress: 0,
          currentStep: 'backgroundTasks.importCancelledByUser',
          errorMessage: 'backgroundTasks.importCancelledByUser',
        });
        return { ok: true };
      }
      if (task.type === 'sync') {
        const jid = (task.syncJobId || '').trim();
        if (!jid) {
          removeTask(task.id);
          return { ok: true, displayOnly: true };
        }
        const res = await cancelTaskJob(jid);
        if (!res.ok) return { ok: false, error: res.error || '取消失败' };
        updateTask(task.id, {
          status: 'cancelled',
          progress: 100,
          currentStep: 'backgroundTasks.statusCancelled',
          errorMessage: undefined,
        });
        return { ok: true };
      }
      return { ok: false, error: 'feedback.requestFailed' };
    },
    [removeTask, updateTask]
  );

  const addExportTask = useCallback(
    async (params: {
      assetIds: number[];
      exportCount: number;
      formatLabel: string;
      formatSummary?: string;
      projectId?: string;
      projectName?: string;
      assetNamesPreview?: string;
    }): Promise<{ ok: boolean; error?: string }> => {
      showAssistantForBackgroundTask({ openPanel: true });
      const res = await createExportJob(params.assetIds, { target: 'local' });
      if (!res.ok) return { ok: false, error: res.error || 'feedback.requestFailed' };
      const jobId = res.data!.jobId;
      const now = new Date().toISOString();
      const fmtSummary = params.formatSummary || params.formatLabel;
      const task: BackgroundTask = {
        id: exportStableTaskId(jobId),
        type: 'export',
        title: `导出数据（${params.exportCount} 条）`,
        status: 'running',
        progress: 5,
        currentStep: '0 / ' + params.exportCount + ' · 校验导出项',
        createdAt: now,
        updatedAt: now,
        meta: {
          format: params.formatLabel,
          count: params.exportCount,
          exportDeliveryMode: 'browser_zip',
          exportFormatSummary: fmtSummary,
          exportProjectId: params.projectId,
          exportProjectName: params.projectName,
          exportAssetNamesPreview: params.assetNamesPreview,
          exportCompletedAssets: 0,
          exportTotalAssets: params.exportCount,
        },
        exportJobId: jobId,
      };
      setTasks((prev) => [task, ...prev]);
      setPanelOpen(true);
      return { ok: true };
    },
    [showAssistantForBackgroundTask]
  );

  const addConvertTask = useCallback((job: ConversionJob) => {
    showAssistantForBackgroundTask({ openPanel: true });
    const task = mapConversionJobToTask(job);
    setTasks((prev) => {
      const filtered = prev.filter((t) => t.id !== task.id);
      return [task, ...filtered];
    });
    setPanelOpen(true);
  }, [showAssistantForBackgroundTask]);

  const addConvertBatchTask = useCallback(
    (batch: ConversionBatchSummary) => {
      showAssistantForBackgroundTask({ openPanel: true });
      const task = mapBatchSummaryToConvertTask(batch);
      setTasks((prev) => {
        const filtered = prev.filter((t) => t.id !== task.id);
        return [task, ...filtered];
      });
      setPanelOpen(true);
    },
    [showAssistantForBackgroundTask]
  );

  const addSyncTask = useCallback(
    async (params: { assetId: number; filename?: string }): Promise<{ ok: boolean; error?: string }> => {
      const inflight = syncInFlightRef.current;
      if (inflight.has(params.assetId)) {
        return { ok: false, error: '该数据正在同步中，请勿重复发起同步' };
      }
      inflight.add(params.assetId);
      try {
        try {
          const st = await getDataAsset(params.assetId);
          if (st.ok && st.data) {
            const status = String(st.data.sync_status || '').trim().toLowerCase();
            if (status === 'syncing') {
              return { ok: false, error: '该数据正在同步中，请勿重复发起同步' };
            }
          }
        } catch {
          /* ignore */
        }
      showAssistantForBackgroundTask({ openPanel: true });
      const now = new Date().toISOString();
      const taskId = genId();
      const title = params.filename ? `同步 ${params.filename}` : '同步数据资产';
      const initialTask: BackgroundTask = {
        id: taskId,
        type: 'sync',
        title,
        status: 'running',
        progress: 10,
        currentStep: '正在发起同步请求...',
        createdAt: now,
        updatedAt: now,
        meta: {
          count: 1,
        },
      };
      setTasks((prev) => [initialTask, ...prev]);
      setPanelOpen(true);

      const createRes = await createSyncBatchJob({ asset_ids: [params.assetId] });
      if (!createRes.ok || !createRes.data?.jobId) {
        const err = createRes.error || '发起同步失败';
        updateTask(taskId, {
          status: 'failed',
          progress: 100,
          currentStep: err,
          errorMessage: err,
        });
        return { ok: false, error: err };
      }

      const jobId = createRes.data.jobId;
      updateTask(taskId, {
        syncJobId: jobId,
        status: 'running',
        progress: 5,
        currentStep: '已创建同步任务，排队中',
      });

      const startedAt = Date.now();
      const pollIntervalMs = 1200;
      const maxWaitMs = 30 * 60 * 1000;

      const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

      while (Date.now() - startedAt < maxWaitMs) {
        await sleep(pollIntervalMs);
        const local = tasksRef.current.find((t) => t.id === taskId);
        if (local && String(local.status) === 'cancelled') {
          return { ok: false, error: '已取消' };
        }
        const stRes = await getSyncBatchJobStatus(jobId);
        if (!stRes.ok || !stRes.data) {
          const err = stRes.error || '同步状态查询失败';
          updateTask(taskId, { progress: 99, currentStep: err });
          continue;
        }
        const data = stRes.data;
        const status = String(data.status || '').trim().toLowerCase();
        const step = String(data.currentStep || '').trim();
        const prog = typeof data.progress === 'number' ? data.progress : 0;
        updateTask(taskId, { progress: Math.min(99, Math.max(5, Math.round(prog))), currentStep: step || `进度 ${Math.round(prog)}%` });

        if (status === 'succeeded') {
          updateTask(taskId, { status: 'success', progress: 100, currentStep: '同步完成' });
          return { ok: true };
        }
        if (status === 'failed') {
          const err = String(data.errorMessage || '同步失败').trim() || '同步失败';
          updateTask(taskId, { status: 'failed', progress: 100, currentStep: err, errorMessage: err });
          return { ok: false, error: err };
        }
        if (status === 'canceled' || status === 'cancelled') {
          updateTask(taskId, { status: 'cancelled', progress: 100, currentStep: 'backgroundTasks.statusCancelled', errorMessage: undefined });
          return { ok: false, error: '已取消' };
        }
      }

      updateTask(taskId, { status: 'failed', progress: 100, currentStep: '同步超时', errorMessage: '同步超时' });
      return { ok: false, error: '同步超时' };
      } finally {
        inflight.delete(params.assetId);
      }
    },
    [showAssistantForBackgroundTask, updateTask]
  );

  const addBatchSyncTask = useCallback(
    (params: { count: number; title?: string }): string => {
      showAssistantForBackgroundTask({ openPanel: true });
      const now = new Date().toISOString();
      const taskId = genId();
      const task: BackgroundTask = {
        id: taskId,
        type: 'sync',
        title: params.title || `批量同步（${Math.max(1, params.count)} 条）`,
        status: 'running',
        progress: 0,
        currentStep: '排队中',
        createdAt: now,
        updatedAt: now,
        meta: {
          count: Math.max(1, params.count),
        },
      };
      setTasks((prev) => [task, ...prev]);
      setPanelOpen(true);
      return taskId;
    },
    [showAssistantForBackgroundTask]
  );

  const finalizeDataImportTask = useCallback(
    (
      taskId: string,
      totalUnits: number,
      result: DataImportResult,
      onImportFinished?: (message?: string) => void
    ) => {
      const { successCount, failedCount, errorMessage, summaryMessage, successAssetNames, failureEntries } = result;

      const importDetailMeta = {
        importCompletedAssets: successCount,
        importFailedAssets: failedCount,
        count: totalUnits,
        ...(successAssetNames?.length ? { importSuccessAssetNames: successAssetNames.slice(0, 50) } : {}),
        ...(failureEntries?.length ? { importFailureEntries: failureEntries.slice(0, 50) } : {}),
        importDirectInitSnapshot: undefined,
        importUploadBytesLoaded: undefined,
        importUploadBytesTotal: undefined,
      };

      if (successCount === 0 && failedCount === 0) {
        if (errorMessage === '用户已取消导入') {
          updateTask(taskId, {
            status: 'cancelled',
            progress: 0,
            currentStep: 'backgroundTasks.importCancelledByUser',
            errorMessage: 'backgroundTasks.importCancelledByUser',
            meta: {
              importDirectInitSnapshot: undefined,
              importUploadBytesLoaded: undefined,
              importUploadBytesTotal: undefined,
            },
          });
          return;
        }
        updateTask(taskId, {
          status: 'failed',
          progress: 100,
          currentStep: errorMessage || '导入失败',
          errorMessage: errorMessage || '导入失败',
          meta: {
            importDirectInitSnapshot: undefined,
            importUploadBytesLoaded: undefined,
            importUploadBytesTotal: undefined,
          },
        });
        return;
      }

      if (successCount === 0 && failedCount > 0) {
        updateTask(taskId, {
          status: 'failed',
          progress: 100,
          currentStep: errorMessage || `全部失败（${failedCount} 条）`,
          errorMessage: errorMessage || `全部失败（${failedCount} 条）`,
          meta: {
            ...importDetailMeta,
            importHadPartialFailures: false,
          },
        });
        return;
      }

      const partial = failedCount > 0;
      const stepText =
        summaryMessage ||
        (partial ? `成功 ${successCount} 条，失败 ${failedCount} 条` : `成功 ${successCount} 条`);
      updateTask(taskId, {
        status: 'success',
        progress: 100,
        currentStep: stepText,
        errorMessage: undefined,
        meta: {
          ...importDetailMeta,
          importHadPartialFailures: partial,
        },
      });
      onImportFinished?.(summaryMessage || stepText);
    },
    [updateTask]
  );

  const runDataImportJob = useCallback(
    (params: {
      title: string;
      totalUnits: number;
      projectId: string;
      projectName: string;
      modeSummary: string;
      assetNamesPreview: string;
      onImportFinished?: (message?: string) => void;
      run: (
        onProgress: (p: DataImportProgress) => void,
        opts?: {
          signal?: AbortSignal;
          onUploadSessionReady?: (uploadSessionId: string) => void;
          onDirectInitSnapshot?: (snap: ImportDirectInitSnapshot) => void;
        }
      ) => Promise<DataImportResult>;
    }) => {
      const {
        title,
        totalUnits,
        projectId,
        projectName,
        modeSummary,
        assetNamesPreview,
        onImportFinished,
        run,
      } = params;
      showAssistantForBackgroundTask({ openPanel: true });
      let currentTaskId = genId();
      const now = new Date().toISOString();
      const initialTask: BackgroundTask = {
        id: currentTaskId,
        type: 'import',
        title,
        status: 'running',
        progress: 0,
        currentStep: totalUnits > 0 ? `0 / ${totalUnits} · 排队中` : '排队中',
        createdAt: now,
        updatedAt: now,
        meta: {
          importBrowserDirect: true,
          count: totalUnits,
          importProjectId: projectId,
          importProjectName: projectName,
          importModeSummary: modeSummary,
          importAssetNamesPreview: assetNamesPreview,
          importCompletedAssets: 0,
          importFailedAssets: 0,
        },
      };
      setTasks((prev) => [initialTask, ...prev]);
      setPanelOpen(true);
      importDiagLog('import_task_created', {
        taskId: currentTaskId,
        totalUnits,
        modeSummary,
        assetNamesPreview: (assetNamesPreview || '').slice(0, 200),
      });

      const ac = new AbortController();
      importAbortRef.current.set(currentTaskId, ac);
      const currentIdRef = { current: currentTaskId };

      const applyProgress = (p: DataImportProgress) => {
        const processed = p.completedUnits + p.failedUnits;
        const phase = (p.phase || '').trim();
        const total = p.totalUnits;
        const bl = p.uploadBytesLoaded;
        const bt = p.uploadBytesTotal;
        let progress = 0;
        if (
          bt != null &&
          bt > 0 &&
          bl != null &&
          Number.isFinite(bl) &&
          Number.isFinite(bt)
        ) {
          progress = Math.floor((Math.min(Math.max(0, bl), bt) / bt) * 100);
        } else if (total > 0 && processed > 0) {
          progress = Math.round((processed / total) * 100);
        } else if (p.uploadHintPercent != null && Number.isFinite(p.uploadHintPercent)) {
          /** 标准 multipart 等仅有百分比、无总字节时 */
          progress = Math.floor(Math.min(100, Math.max(0, p.uploadHintPercent)));
        } else {
          progress = 0;
        }
        progress = Math.min(99, Math.max(0, progress));
        const label = p.currentLabel || '';
        const denom = Math.max(total, 1);
        const fileSeg =
          p.uploadFileCount != null &&
          p.uploadFileCount > 1 &&
          p.uploadFileIndex != null &&
          p.uploadFileIndex >= 0
            ? `文件 ${p.uploadFileIndex + 1}/${p.uploadFileCount}`
            : '';
        const parts: string[] = [`${processed} / ${denom}`];
        if (fileSeg) parts.push(fileSeg);
        if (label) parts.push(label);
        if (phase) parts.push(phase);
        const currentStep = parts.join(' · ');
        updateTask(currentIdRef.current, {
          progress,
          currentStep,
          meta: {
            importCompletedAssets: p.completedUnits,
            importFailedAssets: p.failedUnits,
            count: total > 0 ? total : Math.max(1, processed),
            ...(bt != null && bl != null ? { importUploadBytesLoaded: bl, importUploadBytesTotal: bt } : {}),
          },
        });
      };

      void (async () => {
        let result: DataImportResult | undefined;
        const clearImportAbortForCurrent = () => {
          purgeImportAbortEntries(importAbortRef.current, ac);
        };
        try {
          result = await run(applyProgress, {
            signal: ac.signal,
            onDirectInitSnapshot: (snap) => {
              importDiagLog('import_direct_init_snapshot', {
                taskId: currentIdRef.current,
                upload_session_id: snap.upload_session_id,
                itemCount: snap.items?.length ?? 0,
                upload_mode: snap.upload_mode,
              });
              updateTask(currentIdRef.current, { meta: { importDirectInitSnapshot: snap } });
            },
            onUploadSessionReady: (uploadSessionId) => {
              const sid = (uploadSessionId || '').trim();
              if (!sid) return;
              const stable = importStableTaskId(sid);
              if (currentIdRef.current === stable) {
                importDiagLog('import_upload_session_ready', { taskId: currentIdRef.current, upload_session_id: sid });
                updateTask(currentIdRef.current, {
                  meta: {
                    importUploadSessionId: sid,
                    importSessionIds: [sid],
                  },
                });
                return;
              }
              /**
               * 必须用「迁移前 id」闭包捕获：React 19 会把 setState updater 延后执行，
               * 若在这里读 currentIdRef.current，执行时往往已是 stable，导致 t.id 永远对不上、迁移静默失败，
               * 后续 updateTask(import:…) 全部打空，多文件会长期停在「准备上传」且进度 0%。
               */
              const fromTaskId = currentIdRef.current;
              importDiagLog('import_task_id_stable', { from: fromTaskId, to: stable, upload_session_id: sid });
              setTasks((prev) =>
                prev.map((t) =>
                  t.id === fromTaskId
                    ? {
                        ...t,
                        id: stable,
                        meta: {
                          ...t.meta,
                          importUploadSessionId: sid,
                          importSessionIds: [...(t.meta?.importSessionIds ?? []), sid],
                        },
                        updatedAt: new Date().toISOString(),
                      }
                    : t
                )
              );
              /** 保留 genId 键 + 增加 import: 键，避免仅用 stable id 查找时拿不到同一 AbortController，多文件取消无法 abort 后续 PUT */
              importAbortRef.current.set(stable, ac);
              currentIdRef.current = stable;
              currentTaskId = stable;
            },
          });
        } catch (e) {
          const aborted =
            (e instanceof DOMException && e.name === 'AbortError') ||
            (e instanceof Error && e.name === 'AbortError');
          if (aborted) {
            updateTask(currentIdRef.current, {
              status: 'cancelled',
              progress: 0,
              currentStep: 'backgroundTasks.importCancelledByUser',
              errorMessage: 'backgroundTasks.importCancelledByUser',
            });
            clearImportAbortForCurrent();
            return;
          }
          const msg = e instanceof Error ? e.message : '导入异常';
          updateTask(currentIdRef.current, {
            status: 'failed',
            progress: 100,
            currentStep: msg,
            errorMessage: msg,
          });
          clearImportAbortForCurrent();
          return;
        }

        if (result) {
          finalizeDataImportTask(currentIdRef.current, totalUnits, result, onImportFinished);
        }
        clearImportAbortForCurrent();
      })();
    },
    [finalizeDataImportTask, showAssistantForBackgroundTask, updateTask]
  );

  useEffect(() => {
    pollActiveTasksRef.current = async () => {
      const list = tasksRef.current;
      const pollTargets = list.filter(
        (t) =>
          (t.status === 'queued' || t.status === 'running') &&
          (t.exportJobId || t.convertJobId || t.convertBatchId)
      );
      const importPollTargets = list.filter(
        (t) => t.type === 'import' && isActiveStatus(t.status) && t.id.startsWith('import:')
      );
      if (pollTargets.length === 0 && importPollTargets.length === 0) return;
      const updates = new Map<string, Partial<BackgroundTask>>();
      await Promise.all(
        pollTargets.map(async (t) => {
          if (t.exportJobId) {
            const res = await getExportJobStatus(t.exportJobId);
            if (!res.ok || !res.data) return;
            const d = res.data;
            const delivery = (d.deliveryMode || '').trim();
            const browserZip =
              delivery === 'browser_zip' || (!!(d.downloadUrl || '').trim() && !(d.fullOutputPath || '').trim());
            if (d.status === 'ready') {
              updates.set(t.id, {
                status: 'success',
                progress: 100,
                currentStep: browserZip
                  ? 'backgroundTasks.exportReadyClickDownload'
                  : '导出内容已保存到指定路径',
                meta: {
                  ...t.meta,
                  outputFile: d.fileName || t.meta?.outputFile,
                  outputPath: d.fullOutputPath ? d.fullOutputPath.replace(/\/[^/]+$/, '') : t.meta?.outputPath,
                  fullOutputPath: d.fullOutputPath || t.meta?.fullOutputPath,
                  exportDeliveryMode: browserZip ? 'browser_zip' : 'local_path',
                  exportCompletedAssets:
                    typeof d.completedAssets === 'number' ? d.completedAssets : t.meta?.exportTotalAssets ?? t.meta?.count,
                  exportTotalAssets: typeof d.totalAssets === 'number' ? d.totalAssets : t.meta?.exportTotalAssets,
                },
              });
            } else if (d.status === 'failed') {
              updates.set(t.id, {
                status: 'failed',
                errorMessage: d.errorMessage || 'backgroundTasks.statusFailed',
              });
            } else if (d.status === 'cancelled') {
              updates.set(t.id, {
                status: 'cancelled',
                progress: 0,
                currentStep: d.errorMessage || 'backgroundTasks.statusCancelled',
                errorMessage: d.errorMessage || undefined,
              });
            } else {
              const progress = typeof d.progress === 'number' && d.progress >= 0 && d.progress <= 100 ? d.progress : t.progress;
              const baseStep = d.currentStep || t.currentStep || '';
              const runningLong =
                Date.now() - new Date(t.createdAt).getTime() > 3 * 60 * 1000 && progress <= 8;
              const currentStep =
                runningLong && !baseStep.includes('等待较久')
                  ? `${baseStep}（等待较久，若仍无进度请查看服务日志 [Export]）`
                  : baseStep;
              updates.set(t.id, {
                progress,
                currentStep,
                meta: {
                  ...t.meta,
                  exportCompletedAssets:
                    typeof d.completedAssets === 'number' ? d.completedAssets : t.meta?.exportCompletedAssets,
                  exportTotalAssets: typeof d.totalAssets === 'number' ? d.totalAssets : t.meta?.exportTotalAssets,
                },
              });
            }
            return;
          }
          if (t.convertBatchId) {
            try {
              const detail = await getBatchDetail(t.convertBatchId);
              const b = detail.batch;
              const st = mapOverallToTaskStatus(String(b.overallStatus || ''));
              const total = Math.max(1, b.totalCount || 1);
              const ended = (b.successCount || 0) + (b.failedCount || 0);
              const sub = `${ended} / ${total}`;
              const stats = `成功 ${b.successCount ?? 0} · 失败 ${b.failedCount ?? 0} · 处理中 ${(b.runningCount ?? 0) + (b.pendingCount ?? 0)}`;
              const progress =
                typeof b.progressPercent === 'number' && Number.isFinite(b.progressPercent)
                  ? Math.min(100, Math.max(0, b.progressPercent))
                  : st === 'success' || st === 'cancelled'
                    ? 100
                    : Math.round((ended / total) * 100);
              const inputFmt = 'MCAP';
              const outputFmt = normalizeFormatLabel(b.targetFormat) || t.meta?.outputFormat || 'HDF5';
              updates.set(t.id, {
                status: st,
                progress,
                title: `转换 ${inputFmt} → ${outputFmt}（${total}条）`,
                currentStep: `${sub} · ${stats}`,
                errorMessage: undefined,
                meta: {
                  ...t.meta,
                  inputFormat: inputFmt,
                  outputFormat: outputFmt,
                  count: total,
                  convertBatchTotal: total,
                  convertBatchSuccess: b.successCount,
                  convertBatchFailed: b.failedCount,
                  convertBatchRunning: b.runningCount,
                  convertBatchPending: b.pendingCount,
                  convertBatchProgressPercent: b.progressPercent,
                  convertBatchOverall: String(b.overallStatus || ''),
                  convertBatchLegacy: !!b.legacySingleFile,
                },
              });
            } catch {
              // ignore
            }
            return;
          }
          if (t.convertJobId) {
            try {
              const job = await getJob(t.convertJobId);
              const st = toTaskStatus(job.status);
              const inputFormat =
                normalizeFormatLabel((job as any).fileFormat) ||
                inferFormatFromPath((job as any).fileName) ||
                inferFormatFromPath((job as any).assetName) ||
                t.meta?.inputFormat ||
                '—';
              const outputFormat =
                normalizeFormatLabel((job as any).outputFormat) ||
                inferFormatFromPath((job as any).outputFileName) ||
                inferFormatFromPath((job as any).fileName) ||
                t.meta?.outputFormat ||
                '—';
              const title = `转换 ${inputFormat} → ${outputFormat}`;
              const currentStep = job.currentStage
                ? `阶段：${stageLabel(job.currentStage)}`
                : st === 'queued'
                  ? '排队中'
                  : st === 'success'
                    ? '已完成'
                    : st === 'failed'
                      ? '失败'
                      : t.currentStep;
              if (st === 'success' || st === 'failed') {
                updates.set(t.id, {
                  status: st,
                  progress: st === 'success' ? 100 : t.progress,
                  title,
                  currentStep,
                  errorMessage: job.errorMessage,
                  meta: {
                    ...t.meta,
                    inputFormat,
                    outputFormat,
                    outputFile: job.outputFileName || job.fileName,
                    outputPath: job.outputPath,
                  },
                });
              } else {
                updates.set(t.id, {
                  status: st,
                  progress: job.progressPercent ?? t.progress,
                  title,
                  currentStep,
                  meta: {
                    ...t.meta,
                    inputFormat,
                    outputFormat,
                  },
                });
              }
            } catch {
              // ignore
            }
          }
        })
      );
      if (importPollTargets.length > 0) {
        try {
          const ures = await listUploadSessions(80);
          if (ures.ok && Array.isArray(ures.data?.items)) {
            const bySid = new Map(
              ures.data!.items.map((row) => [(row.upload_session_id || '').trim(), row] as const)
            );
            for (const t of importPollTargets) {
              const sid = t.id.slice('import:'.length);
              const row = bySid.get(sid);
              if (!row) continue;
              const latest = tasksRef.current.find((x) => x.id === t.id) ?? t;
              if (
                latest.type === 'import' &&
                (latest.status === 'success' ||
                  latest.status === 'failed' ||
                  latest.status === 'cancelled')
              ) {
                continue;
              }
              const isLiveLocalUploading = importAbortRef.current.has(t.id);
              const next = resolveImportTaskFromUploadSessionRow(row, isLiveLocalUploading);
              if (
                isLiveLocalUploading &&
                next.status === 'running' &&
                isGenericImportPendingStep(next.currentStep)
              ) {
                // 本地上传仍在进行：后端 upload_sessions 的“处理中(10%)”仅是占位状态，不覆盖实时上传文案/进度
                continue;
              }
              const mergedProgress =
                next.status === 'success' || next.status === 'failed' || next.status === 'cancelled'
                  ? next.progress
                  : Math.max(latest.progress, next.progress);
              const mergedStep =
                isGenericImportPendingStep(next.currentStep) && latest.progress > next.progress
                  ? latest.currentStep
                  : next.currentStep;
              const mergedError =
                isGenericImportPendingStep(next.currentStep) && latest.progress > next.progress
                  ? latest.errorMessage
                  : next.errorMessage;
              if (
                latest.status !== next.status ||
                latest.progress !== mergedProgress ||
                latest.currentStep !== mergedStep ||
                (latest.errorMessage || '') !== (mergedError || '')
              ) {
                updates.set(t.id, {
                  status: next.status,
                  progress: mergedProgress,
                  currentStep: mergedStep,
                  title: next.title,
                  errorMessage: mergedError,
                  meta: { ...latest.meta, ...next.meta },
                });
              }
            }
          }
        } catch {
          /* ignore */
        }
      }
      if (updates.size > 0) {
        setTasks((prev) =>
          prev.map((task) => {
            if (!updates.has(task.id)) return task;
            const patch = updates.get(task.id)!;
            if (task.type === 'import') {
              /** 本地 finalize 已成功后，异步 poll 仍可能拿到滞后 presigned，禁止覆盖终态 */
              if (task.status === 'success') return task;
              const wasTerminal =
                task.status === 'failed' || task.status === 'cancelled';
              const patchDegradesTerminal =
                wasTerminal &&
                (patch.status === 'running' || patch.status === 'queued' || patch.status === 'paused');
              if (patchDegradesTerminal) return task;
            }
            return { ...task, ...patch, updatedAt: new Date().toISOString() };
          })
        );
      }
    };
  });

  useEffect(() => {
    const active = hasActiveImportExportPollTargets(tasks);
    if (!active) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current) return;

    void pollActiveTasksRef.current?.();
    pollRef.current = setInterval(() => {
      void pollActiveTasksRef.current?.();
    }, TASK_CENTER_POLL_INTERVAL_MS);
  }, [tasks]);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  /** 批量直传期间轻量保活：触发已鉴权请求，便于在 AT 将过期时走 401→refresh 链（与 JWT 主动续期互补） */
  useEffect(() => {
    const needKeepAlive = tasks.some(
      (t) =>
        t.type === 'import' &&
        t.meta?.importBrowserDirect === true &&
        (t.status === 'running' || t.status === 'queued')
    );
    if (!needKeepAlive) return;
    const tick = () => {
      void apiGet<unknown>('/api/auth/me');
    };
    const h = window.setInterval(tick, 4 * 60 * 1000);
    tick();
    return () => clearInterval(h);
  }, [tasks]);

  const value = useMemo(
    () => ({
      tasks,
      activeCount,
      panelOpen,
      setPanelOpen,
      assistantHidden,
      assistantVisible,
      assistantPosition,
      setAssistantPosition: (pos: AssistantPosition) => setAssistantPosition(clampToViewport(pos)),
      persistAssistantPosition,
      hideAssistant,
      showAssistantForBackgroundTask,
      addExportTask,
      addConvertTask,
      addConvertBatchTask,
      addSyncTask,
      addBatchSyncTask,
      runDataImportJob,
      updateTask,
      removeTask,
      clearTaskGroup,
      deleteTaskResult,
      cancelBackgroundTask,
    }),
    [
      tasks,
      activeCount,
      panelOpen,
      assistantHidden,
      assistantVisible,
      assistantPosition,
      persistAssistantPosition,
      hideAssistant,
      showAssistantForBackgroundTask,
      addExportTask,
      addConvertTask,
      addConvertBatchTask,
      addSyncTask,
      addBatchSyncTask,
      runDataImportJob,
      updateTask,
      removeTask,
      clearTaskGroup,
      deleteTaskResult,
      cancelBackgroundTask,
    ]
  );

  return (
    <TaskCenterContext.Provider value={value}>
      {children}
    </TaskCenterContext.Provider>
  );
}

export function useTaskCenter(): TaskCenterContextValue {
  const ctx = useContext(TaskCenterContext);
  if (!ctx) throw new Error('useTaskCenter must be used within TaskCenterProvider');
  return ctx;
}
