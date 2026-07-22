/**
 * 数据资产导入后台执行体：与 ImportDataDialog 内原逻辑一致，通过 onProgress 按「资产单元」汇报。
 */

import {
  importDataAssetFiles,
  initDirectUpload,
  completeDirectUpload,
  uploadFileToPresignedUrl,
  type DirectUploadFileItemSpec,
  type DirectUploadInitData,
  type DirectUploadMode,
} from '@/features/data-platform/api/dataAssetsApi';
import { recordProjectActivityAndTouch } from '@/lib/projects/projectService';
import {
  planFolderTreeImport,
  buildDirectoryUploadItems,
  resolveFilesForDirectUploadItems,
  fileKey,
  type FolderImportJob,
  type ImportAssetRow,
} from '@/components/assets/folderImportPlanner';
import type { ImportDirectInitSnapshot } from '@/components/task-center/types';
import { importDiagLog } from '@/lib/importDiagLog';
import { DirectImportWatchdog, raceWithAbort } from '@/lib/importDirectWatchdog';

const ENABLE_DIRECT_MULTI = process.env.NEXT_PUBLIC_DIRECT_UPLOAD_MULTI !== '0';

const RESOLVE_FILES_TIMEOUT_MS = 60_000;

function withLocalTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return new Promise((resolve, reject) => {
    timer = setTimeout(() => {
      timer = undefined;
      reject(new Error(`${label} 超时（>${Math.round(ms / 1000)}s），可能卡在读本地文件大小/切片探测）`));
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

/** 将 upload-init 结果写入任务 meta，供同一会话「继续导入」复用 PUT URL（单会话树批次才启用）。 */
function emitImportDirectInitSnapshot(
  cb: ((s: ImportDirectInitSnapshot) => void) | undefined,
  d: DirectUploadInitData,
  sizes: number[],
  opts?: { fallbackRelativePath?: string }
) {
  if (!cb) return;
  const rawItems = d.upload_items || [];
  let items: ImportDirectInitSnapshot['items'] = [];
  if (rawItems.length > 0) {
    items = rawItems.map((it, idx) => ({
      relative_path: (it.relative_path || '').replace(/\\/g, '/'),
      size_bytes: Number(sizes[idx] ?? 0),
      upload_url: it.upload_url,
      method: (it.method || 'PUT').trim(),
      headers: (it.headers || {}) as Record<string, string>,
    }));
  } else if (d.upload_url && sizes.length === 1) {
    const rel = (opts?.fallbackRelativePath || 'file').replace(/\\/g, '/');
    items = [
      {
        relative_path: rel,
        size_bytes: Number(sizes[0] ?? 0),
        upload_url: d.upload_url,
        method: (d.method || 'PUT').trim(),
        headers: (d.headers || {}) as Record<string, string>,
      },
    ];
  }
  if (items.length === 0 || items.some((x) => !x.upload_url?.trim())) return;
  cb({
    upload_session_id: d.upload_session_id,
    upload_mode: String(d.upload_mode || ''),
    expires_at: d.expires_at,
    root_dir_name: d.root_dir_name ?? null,
    items,
  });
}

export type DataImportProgress = {
  completedUnits: number;
  failedUnits: number;
  totalUnits: number;
  currentLabel?: string;
  phase?: string;
  /** 标准导入 multipart 等仅有百分比、无总字节时的回退 */
  uploadHintPercent?: number;
  /** 直传（单/多文件聚合）：已上传字节 / 本会话总字节，用于任务中心真实进度条 */
  uploadBytesLoaded?: number;
  uploadBytesTotal?: number;
  /** 多文件直传：当前正在上传的文件序号（0-based）与文件数，用于文案「文件 k/n」（与字节进度语义分离） */
  uploadFileIndex?: number;
  uploadFileCount?: number;
};

export type ImportFailureEntry = { name: string; reason: string };

export type DataImportResult = {
  successCount: number;
  failedCount: number;
  errorMessage?: string;
  summaryMessage?: string;
  putStarted: boolean;
  /** 成功导入的资产名（用于任务详情，截断由调用方控制） */
  successAssetNames?: string[];
  /** 失败项明细（可多条） */
  failureEntries?: ImportFailureEntry[];
};

function importUnitsForJob(job: FolderImportJob): number {
  if (job.kind === 'multi_file') return job.files.length;
  return 1;
}

function leafNameFromRelPath(rel: string | undefined): string {
  const r = (rel || '').replace(/\\/g, '/').trim();
  if (!r) return '—';
  const seg = r.split('/').pop()?.trim();
  return seg || r;
}

function pushJobWideFailures(entries: ImportFailureEntry[], job: FolderImportJob, reason: string): void {
  if (job.kind === 'multi_file') {
    for (const f of job.files) entries.push({ name: f.name || '—', reason });
    return;
  }
  if (job.kind === 'directory') {
    entries.push({ name: job.rootDirName || '目录', reason });
    return;
  }
  entries.push({ name: job.file.name || '—', reason });
}

function mergeFailedItemsIntoEntries(
  items: Array<{ relative_path?: string; object_key?: string; reason?: string }>,
  entries: ImportFailureEntry[],
): void {
  for (const it of items) {
    const name = leafNameFromRelPath(it.relative_path) || (it.object_key || '').slice(-48) || '—';
    const reason = (it.reason || '登记失败').slice(0, 220);
    if (entries.some((e) => e.name === name && e.reason === reason)) continue;
    entries.push({ name, reason });
  }
}

function isTreeUploadFile(f: File): boolean {
  const rel = ((f as File & { webkitRelativePath?: string }).webkitRelativePath || '').replace(/\\/g, '/');
  return rel.includes('/');
}

function shouldUseDirectMinioUpload(files: File[]): boolean {
  return files.length === 1 && !isTreeUploadFile(files[0]);
}

function shouldUseDirectMultiBatch(files: File[]): boolean {
  if (files.length === 0) return false;
  if (files.some(isTreeUploadFile)) return true;
  return files.length >= 2;
}

function newClientFileId(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function totalBytesForFolderJobs(jobs: FolderImportJob[]): number {
  let s = 0;
  for (const j of jobs) {
    if (j.kind === 'directory' || j.kind === 'multi_file') {
      s += j.files.reduce((a, f) => a + f.size, 0);
    } else {
      s += j.file.size;
    }
  }
  return s;
}

const SKIP_NO_READABLE_SIZE =
  '无法参与直传（无有效可读大小，或已作为目录占位项跳过）。其余文件将照常处理。';

/**
 * 文件夹树导入任务中的单文件直传（与循环内原 single_file 分支一致，供 multi_file 在过滤后仅剩 1 个有效文件时复用）。
 */
async function runTreeJobSingleFileDirectUpload(params: {
  projectId: string;
  file: File;
  planJobs: FolderImportJob[];
  job: FolderImportJob;
  wd: DirectImportWatchdog;
  fireUploadSession?: (uploadSessionId: string) => void;
  onDirectInitSnapshot?: (snap: ImportDirectInitSnapshot) => void;
  tick: (partial: Partial<DataImportProgress>) => void;
  completed: number;
  failed: number;
  putStarted: boolean;
}): Promise<{ importedId: string; assetName: string; putStarted: boolean }> {
  const { projectId, file, planJobs, job, wd, fireUploadSession, onDirectInitSnapshot, tick, completed, failed, putStarted } =
    params;
  const opSignal = wd.signal;
  const label = jobLabel(job);
  const init = await raceWithAbort(
    initDirectUpload({
      project_id: projectId,
      upload_mode: 'single_file',
      filename: file.name,
      size_bytes: file.size,
      content_type: file.type || null,
    }),
    opSignal,
  );
  if (!init.ok || !init.data) {
    throw new Error(init.error || '申请上传失败');
  }
  const d = init.data;
  fireUploadSession?.(d.upload_session_id);
  emitImportDirectInitSnapshot(onDirectInitSnapshot, d, [file.size]);
  const item0 = d.upload_items?.[0];
  const putUrl = item0?.upload_url ?? d.upload_url;
  const putMethod = item0?.method ?? d.method ?? 'PUT';
  const putHeaders = item0?.headers ?? d.headers ?? {};
  if (!putUrl) throw new Error('缺少预签名地址');
  tick({ phase: '上传中', currentLabel: label });
  let nextPutStarted = putStarted;
  nextPutStarted = true;
  const totalAll = totalBytesForFolderJobs(planJobs);
  let aggregatedDone = 0;
  for (const prev of planJobs) {
    if (prev === job) break;
    if (prev.kind === 'single_file') aggregatedDone += prev.file.size;
    else aggregatedDone += prev.files.reduce((s, x) => s + x.size, 0);
  }
  await uploadFileToPresignedUrl(file, putUrl, {
    method: putMethod,
    headers: putHeaders,
    signal: opSignal,
    onByteProgress: (loaded, totForFile) => {
      const t = totForFile > 0 ? totForFile : file.size;
      const clamped = Math.min(Math.max(0, loaded), t);
      const denom = totalAll > 0 ? totalAll : file.size;
      tick({
        uploadBytesLoaded: aggregatedDone + clamped,
        uploadBytesTotal: denom,
        phase: '上传中',
        currentLabel: label,
      });
    },
  });
  tick({
    phase: '写入登记',
    currentLabel: label,
    uploadBytesLoaded: aggregatedDone + file.size,
    uploadBytesTotal: totalAll > 0 ? totalAll : file.size,
  });
  let done;
  try {
    done = await raceWithAbort(
      completeDirectUpload({
        upload_session_id: d.upload_session_id,
        size_bytes: file.size,
      }),
      opSignal,
    );
  } finally {
    wd.markCompletePhaseEnded();
  }
  if (!done.ok || !done.data?.asset) {
    throw new Error(done.error || '登记失败');
  }
  const asset = done.data.asset;
  tick({ completedUnits: completed + 1, failedUnits: failed, currentLabel: label, phase: '' });
  return {
    importedId: String(asset.id),
    assetName: String(asset.filename || file.name),
    putStarted: nextPutStarted,
  };
}

function jobLabel(job: FolderImportJob): string {
  if (job.kind === 'directory') return job.rootDirName;
  if (job.kind === 'single_file') return job.file.name;
  if (job.files.length === 1) return job.files[0].name;
  return `${job.files.length} 个文件`;
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
}

function isAbortError(e: unknown): boolean {
  return (
    (e instanceof DOMException && e.name === 'AbortError') ||
    (e instanceof Error && (e.name === 'AbortError' || e.message === '已取消'))
  );
}

/** 用户点击取消导致的 Abort；watchdog 触发的 merged abort 不应走此分支 */
function isUserCancelledAbort(e: unknown, userSignal?: AbortSignal): boolean {
  return isAbortError(e) && !!userSignal?.aborted;
}

export function resolveImportModeSummary(files: File[], treeBatch: boolean): string {
  if (shouldUseDirectMinioUpload(files)) return 'single_file';
  if (ENABLE_DIRECT_MULTI && treeBatch && shouldUseDirectMultiBatch(files)) return 'directory_tree';
  if (ENABLE_DIRECT_MULTI && !treeBatch && shouldUseDirectMultiBatch(files)) return 'multi_file';
  return 'fallback';
}

export function formatImportModeSummaryLabel(key: string): string {
  const map: Record<string, string> = {
    single_file: '单文件直传',
    multi_file: '多文件直传',
    directory_tree: '目录直传',
    fallback: '标准导入（/import）',
  };
  return map[key] ?? key;
}

export function buildImportTaskTitle(rows: ImportAssetRow[]): string {
  const n = rows.length;
  if (n === 0) return '导入数据';
  const allLr = rows.every((r) => r.assetType === 'LeRobot');
  if (allLr) return `导入 LeRobot 数据集（${n} 条）`;
  if (n === 1) return `导入数据（1 条）`;
  return `导入数据（${n} 条）`;
}

export async function executeDataAssetImportBackground(params: {
  files: File[];
  projectId: string;
  projectName: string;
  treeBatch: boolean;
  totalUnits: number;
  onProgress: (p: DataImportProgress) => void;
  /** 取消时中断 XHR 直传与标准导入上传 */
  signal?: AbortSignal;
  /**
   * 直传 upload-init 成功后回调（每个 upload_session_id 一次）。
   * 多 job 的文件夹树批次（多个独立会话）不在此回调，以免任务中心 id 错乱；刷新后由 upload-sessions 列表恢复多条。
   */
  onUploadSessionReady?: (uploadSessionId: string) => void;
  /** upload-init 成功后写入任务快照，供同 upload_session 断点续传（不含多会话树批次） */
  onDirectInitSnapshot?: (snap: ImportDirectInitSnapshot) => void;
}): Promise<DataImportResult> {
  const {
    files,
    projectId,
    projectName,
    totalUnits,
    onProgress,
    treeBatch,
    signal,
    onUploadSessionReady,
    onDirectInitSnapshot,
  } = params;
  let completed = 0;
  let failed = 0;
  let putStarted = false;
  let activeWatchdog: DirectImportWatchdog | null = null;

  importDiagLog('execute_import_start', {
    fileCount: files.length,
    treeBatch,
    totalUnits,
    mode: resolveImportModeSummary(files, treeBatch),
    projectId: projectId,
  });

  let uploadByteLoaded: number | undefined;
  let uploadByteTotal: number | undefined;
  let uploadFileIdx: number | undefined;
  let uploadFileCnt: number | undefined;
  /** 与 uploadBytes* 一样在 tick 间 carry，避免 partial 未带 phase/label 时把 undefined 传给任务中心 */
  let carriedPhase: string | undefined;
  let carriedLabel: string | undefined;

  const tick = (partial: Partial<DataImportProgress>) => {
    if (partial.phase === '排队中') {
      uploadByteLoaded = 0;
      uploadByteTotal = undefined;
      uploadFileIdx = undefined;
      uploadFileCnt = undefined;
      carriedPhase = '排队中';
      carriedLabel = undefined;
    } else {
      if (partial.phase !== undefined) carriedPhase = partial.phase;
      if (partial.currentLabel !== undefined) carriedLabel = partial.currentLabel;
    }
    if (partial.uploadBytesLoaded !== undefined) uploadByteLoaded = partial.uploadBytesLoaded;
    if (partial.uploadBytesTotal !== undefined) uploadByteTotal = partial.uploadBytesTotal;
    if (partial.uploadFileIndex !== undefined) uploadFileIdx = partial.uploadFileIndex;
    if (partial.uploadFileCount !== undefined) uploadFileCnt = partial.uploadFileCount;

    let hint = partial.uploadHintPercent;
    if (
      hint == null &&
      uploadByteTotal != null &&
      uploadByteTotal > 0 &&
      uploadByteLoaded != null
    ) {
      hint = Math.min(100, Math.round((uploadByteLoaded / uploadByteTotal) * 100));
    }

    const merged: DataImportProgress = {
      completedUnits: partial.completedUnits ?? completed,
      failedUnits: partial.failedUnits ?? failed,
      totalUnits: partial.totalUnits ?? totalUnits,
      currentLabel: partial.currentLabel !== undefined ? partial.currentLabel : carriedLabel,
      phase: partial.phase !== undefined ? partial.phase : carriedPhase,
      uploadHintPercent: hint,
      uploadBytesLoaded: uploadByteLoaded,
      uploadBytesTotal: uploadByteTotal,
      uploadFileIndex: uploadFileIdx,
      uploadFileCount: uploadFileCnt,
    };
    activeWatchdog?.noteProgress(merged);
    onProgress(merged);
  };

  tick({ completedUnits: 0, failedUnits: 0, phase: '排队中' });
  throwIfAborted(signal);

  if (shouldUseDirectMinioUpload(files)) {
    const f = files[0];
    activeWatchdog = new DirectImportWatchdog(signal);
    const wd = activeWatchdog;
    const opSignal = wd.signal;
    try {
      tick({
        phase: '准备上传',
        currentLabel: f.name,
        uploadBytesLoaded: 0,
        uploadBytesTotal: f.size > 0 ? f.size : undefined,
      });
      const init = await raceWithAbort(
        initDirectUpload({
          project_id: projectId,
          upload_mode: 'single_file',
          filename: f.name,
          size_bytes: f.size,
          content_type: f.type || null,
        }),
        opSignal,
      );
      if (!init.ok || !init.data) {
        return { successCount: 0, failedCount: 0, errorMessage: init.error || '申请上传失败', putStarted: false };
      }
      const d = init.data;
      onUploadSessionReady?.(d.upload_session_id);
      emitImportDirectInitSnapshot(onDirectInitSnapshot, d, [f.size], { fallbackRelativePath: f.name });
      const item0 = d.upload_items?.[0];
      const putUrl = item0?.upload_url ?? d.upload_url;
      const putMethod = item0?.method ?? d.method ?? 'PUT';
      const putHeaders = item0?.headers ?? d.headers ?? {};
      if (!putUrl) {
        return { successCount: 0, failedCount: 0, errorMessage: '缺少预签名地址', putStarted: false };
      }
      tick({ phase: '上传中', currentLabel: f.name });
      putStarted = true;
      await uploadFileToPresignedUrl(f, putUrl, {
        method: putMethod,
        headers: putHeaders,
        signal: opSignal,
        onByteProgress: (loaded, totForFile) => {
          const t = totForFile > 0 ? totForFile : f.size;
          tick({
            uploadBytesLoaded: Math.min(Math.max(0, loaded), t),
            uploadBytesTotal: t,
            phase: '上传中',
            currentLabel: f.name,
          });
        },
      });
      tick({
        phase: '写入登记',
        currentLabel: f.name,
        uploadBytesLoaded: f.size,
        uploadBytesTotal: f.size,
      });
      let done;
      try {
        done = await raceWithAbort(
          completeDirectUpload({
            upload_session_id: d.upload_session_id,
            size_bytes: f.size,
          }),
          opSignal,
        );
      } finally {
        wd.markCompletePhaseEnded();
      }
      if (!done.ok || !done.data?.asset) {
        failed = 1;
        tick({ completedUnits: 0, failedUnits: 1, phase: '' });
        return {
          successCount: 0,
          failedCount: 1,
          errorMessage: done.error || '登记失败',
          putStarted: true,
        };
      }
      const asset = done.data.asset;
      if (projectId) {
        recordProjectActivityAndTouch(projectId, 'DATA_IMPORTED', `直传导入数据 1 条`, '当前用户', String(asset.id));
      }
      completed = 1;
      tick({ completedUnits: 1, failedUnits: 0, phase: '' });
      return {
        successCount: 1,
        failedCount: 0,
        summaryMessage: `直传成功：${asset.filename}`,
        putStarted: true,
      };
    } catch (e) {
      if (isAbortError(e) && !isUserCancelledAbort(e, signal)) {
        const r = wd.consumeLastFireReason();
        const msg = r ? `直传导入 watchdog：${r}` : '直传导入 watchdog：操作超时';
        importDiagLog('import_watchdog_task_outcome', { outcome: 'failed', message: msg });
        failed = 1;
        tick({ completedUnits: 0, failedUnits: 1, phase: '' });
        return {
          successCount: 0,
          failedCount: 1,
          errorMessage: msg,
          putStarted,
        };
      }
      if (isUserCancelledAbort(e, signal)) {
        tick({ completedUnits: 0, failedUnits: 0, phase: '' });
        return { successCount: 0, failedCount: 0, errorMessage: '用户已取消导入', putStarted };
      }
      const msg = e instanceof Error ? e.message : '直传异常';
      failed = 1;
      tick({ completedUnits: 0, failedUnits: 1, phase: '' });
      return {
        successCount: 0,
        failedCount: 1,
        errorMessage: msg,
        putStarted,
      };
    } finally {
      wd.dispose();
      activeWatchdog = null;
    }
  }

  if (ENABLE_DIRECT_MULTI && treeBatch && shouldUseDirectMultiBatch(files)) {
    const plan = planFolderTreeImport(files);
    if (!plan.ok) {
      return { successCount: 0, failedCount: 0, errorMessage: plan.message, putStarted: false };
    }
    const fireUploadSession =
      plan.jobs.length > 1 ? undefined : onUploadSessionReady;
    const treeResumeSnap = plan.jobs.length <= 1 ? onDirectInitSnapshot : undefined;

    const importedIds: string[] = [];
    const failureEntries: ImportFailureEntry[] = [];
    const successAssetNames: string[] = [];

    activeWatchdog = new DirectImportWatchdog(signal);
    const wd = activeWatchdog;
    const opSignal = wd.signal;
    try {
    for (const job of plan.jobs) {
      throwIfAborted(opSignal);
      const label = jobLabel(job);
      const c0 = completed;
      const f0 = failed;
      const totalAllTree = totalBytesForFolderJobs(plan.jobs);
      let aggBytesBeforeJob = 0;
      for (const prev of plan.jobs) {
        if (prev === job) break;
        if (prev.kind === 'single_file') aggBytesBeforeJob += prev.file.size;
        else aggBytesBeforeJob += prev.files.reduce((s, x) => s + x.size, 0);
      }
      tick({
        phase: '准备上传',
        currentLabel: label,
        uploadBytesLoaded: totalAllTree > 0 ? aggBytesBeforeJob : 0,
        uploadBytesTotal: totalAllTree > 0 ? totalAllTree : undefined,
      });

      try {
        if (job.kind === 'single_file') {
          const f = job.file;
          const one = await runTreeJobSingleFileDirectUpload({
            projectId,
            file: f,
            planJobs: plan.jobs,
            job,
            wd,
            fireUploadSession,
            onDirectInitSnapshot: treeResumeSnap,
            tick,
            completed,
            failed,
            putStarted,
          });
          putStarted = one.putStarted;
          importedIds.push(one.importedId);
          successAssetNames.push(one.assetName);
          completed += 1;
          continue;
        }

        if (job.kind === 'multi_file') {
          importDiagLog('tree_resolve_files_begin', { jobLabel: label, inputCount: job.files.length });
          const uploadFiles = await withLocalTimeout(
            resolveFilesForDirectUploadItems(job.files),
            RESOLVE_FILES_TIMEOUT_MS,
            'resolveFilesForDirectUploadItems',
          );
          importDiagLog('tree_resolve_files_done', { jobLabel: label, readyCount: uploadFiles.length });
          const resolvedKeySet = new Set(uploadFiles.map((f) => fileKey(f)));
          for (const f of job.files) {
            if (resolvedKeySet.has(fileKey(f))) continue;
            if (failureEntries.some((e) => e.name === (f.name || '—') && e.reason === SKIP_NO_READABLE_SIZE))
              continue;
            failureEntries.push({ name: f.name || '—', reason: SKIP_NO_READABLE_SIZE });
          }

          if (uploadFiles.length === 0) {
            failed += job.files.length;
            tick({ completedUnits: completed, failedUnits: failed, currentLabel: label, phase: '' });
            continue;
          }

          if (uploadFiles.length === 1) {
            failed += Math.max(0, job.files.length - 1);
            const f = uploadFiles[0];
            const one = await runTreeJobSingleFileDirectUpload({
              projectId,
              file: f,
              planJobs: plan.jobs,
              job,
              wd,
              fireUploadSession,
              onDirectInitSnapshot: treeResumeSnap,
              tick,
              completed,
              failed,
              putStarted,
            });
            putStarted = one.putStarted;
            importedIds.push(one.importedId);
            successAssetNames.push(one.assetName);
            completed += 1;
            continue;
          }
          const items: DirectUploadFileItemSpec[] = uploadFiles.map((f) => ({
            client_file_id: newClientFileId(),
            relative_path: (
              (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
            ).replace(/\\/g, '/'),
            size_bytes: f.size,
            content_type: f.type || null,
          }));
          const init = await raceWithAbort(
            initDirectUpload({
              project_id: projectId,
              upload_mode: 'multi_file',
              items,
            }),
            opSignal,
          );
          if (!init.ok || !init.data?.upload_items?.length) {
            throw new Error(init.error || '申请上传失败');
          }
          const d = init.data;
          if (d.upload_items.length !== uploadFiles.length) {
            throw new Error(
              `upload-init 返回 ${d.upload_items.length} 个预签名项，与本地可上传文件数 ${uploadFiles.length} 不一致`,
            );
          }
          fireUploadSession?.(d.upload_session_id);
          emitImportDirectInitSnapshot(treeResumeSnap, d, uploadFiles.map((x) => x.size));
          const jobTotal = uploadFiles.reduce((s, f) => s + f.size, 0);
          let jobDone = 0;
          const totalAll = totalBytesForFolderJobs(plan.jobs);
          let aggregatedDone = 0;
          for (const prev of plan.jobs) {
            if (prev === job) break;
            if (prev.kind === 'single_file') aggregatedDone += prev.file.size;
            else aggregatedDone += prev.files.reduce((s, x) => s + x.size, 0);
          }
          tick({ phase: '上传中', currentLabel: label });
          putStarted = true;
          const uploadErrs: ImportFailureEntry[] = [];
          for (let i = 0; i < d.upload_items.length; i++) {
            throwIfAborted(opSignal);
            const it = d.upload_items[i];
            const file = uploadFiles[i];
            try {
              if (!file) {
                throw new Error('本地文件队列与 upload-init 返回项不匹配（缺少对应 File）');
              }
              importDiagLog('tree_multi_put_item', {
                index: i,
                relative_path: it.relative_path,
                object_key: it.object_key,
                size_bytes: file.size,
                has_presigned_url: !!it.upload_url?.trim(),
              });
              if (!it.upload_url?.trim()) throw new Error('缺少预签名地址');
              const fileCount = d.upload_items.length;
              await uploadFileToPresignedUrl(file, it.upload_url, {
                method: it.method,
                headers: it.headers,
                signal: opSignal,
                onByteProgress: (loaded, totForFile) => {
                  const t = totForFile > 0 ? totForFile : file.size;
                  const clamped = Math.min(Math.max(0, loaded), t);
                  const denom = totalAll > 0 ? totalAll : aggregatedDone + jobTotal;
                  tick({
                    uploadBytesLoaded: aggregatedDone + jobDone + clamped,
                    uploadBytesTotal: denom,
                    uploadFileIndex: i,
                    uploadFileCount: fileCount,
                    phase: '上传中',
                    currentLabel: label,
                  });
                },
              });
              jobDone += file.size;
            } catch (up) {
              if (isAbortError(up)) throw up;
              uploadErrs.push({
                name: file?.name || leafNameFromRelPath(it.relative_path) || '—',
                reason: up instanceof Error ? up.message : '上传失败',
              });
            }
            throwIfAborted(opSignal);
          }
          throwIfAborted(opSignal);
          tick({
            phase: '写入登记',
            currentLabel: label,
            uploadBytesLoaded: aggregatedDone + jobTotal,
            uploadBytesTotal: totalAll > 0 ? totalAll : aggregatedDone + jobTotal,
          });
          let doneRes;
          try {
            doneRes = await raceWithAbort(
              completeDirectUpload({ upload_session_id: d.upload_session_id }),
              opSignal,
            );
          } finally {
            wd.markCompletePhaseEnded();
          }
          if (!doneRes.ok || !doneRes.data) {
            throw new Error(doneRes.error || '多文件登记失败');
          }
          const assets = doneRes.data.assets ?? [];
          const failedItems = doneRes.data.failed_items ?? [];
          const namesFromServer = new Set(
            failedItems.map((fi) => leafNameFromRelPath(fi.relative_path)),
          );
          mergeFailedItemsIntoEntries(failedItems, failureEntries);
          for (const u of uploadErrs) {
            if (!namesFromServer.has(u.name)) failureEntries.push(u);
          }
          for (const a of assets) {
            importedIds.push(String(a.id));
            successAssetNames.push(String(a.filename || ''));
          }
          completed += assets.length;
          failed += uploadFiles.length - assets.length;
          tick({ completedUnits: completed, failedUnits: failed, currentLabel: label, phase: '' });
          continue;
        }

        importDiagLog('tree_directory_resolve_begin', { jobLabel: label, inputCount: job.files.length });
        const uploadFiles = await withLocalTimeout(
          resolveFilesForDirectUploadItems(job.files),
          RESOLVE_FILES_TIMEOUT_MS,
          'resolveFilesForDirectUploadItems',
        );
        importDiagLog('tree_directory_resolve_done', { jobLabel: label, readyCount: uploadFiles.length });
        if (uploadFiles.length === 0) {
          throw new Error('目录内没有可读大小的文件（可能均为 0 字节目录占位项）');
        }
        const pathRows = buildDirectoryUploadItems(uploadFiles, job.rootDirName);
        const items: DirectUploadFileItemSpec[] = uploadFiles.map((f, idx) => ({
          client_file_id: newClientFileId(),
          relative_path: pathRows[idx].relative_path,
          size_bytes: f.size,
          content_type: f.type || null,
        }));
        const init = await raceWithAbort(
          initDirectUpload({
            project_id: projectId,
            upload_mode: 'directory',
            items,
            root_dir_name: job.rootDirName,
          }),
          opSignal,
        );
        if (!init.ok || !init.data?.upload_items?.length) {
          throw new Error(init.error || '申请上传失败');
        }
        const d = init.data;
        if (d.upload_items.length !== uploadFiles.length) {
          throw new Error(
            `upload-init 返回 ${d.upload_items.length} 个预签名项，与本地可上传文件数 ${uploadFiles.length} 不一致`,
          );
        }
        fireUploadSession?.(d.upload_session_id);
        emitImportDirectInitSnapshot(treeResumeSnap, d, uploadFiles.map((x) => x.size));
        const jobTotal = uploadFiles.reduce((s, f) => s + f.size, 0);
        let jobDone = 0;
        const totalAll = totalBytesForFolderJobs(plan.jobs);
        let aggregatedDone = 0;
        for (const prev of plan.jobs) {
          if (prev === job) break;
          if (prev.kind === 'single_file') aggregatedDone += prev.file.size;
          else aggregatedDone += prev.files.reduce((s, x) => s + x.size, 0);
        }
        tick({ phase: '上传中', currentLabel: label });
        putStarted = true;
        for (let i = 0; i < d.upload_items.length; i++) {
          throwIfAborted(opSignal);
          const it = d.upload_items[i];
          const file = uploadFiles[i];
          if (!file) {
            throw new Error('本地文件队列与 upload-init 返回项不匹配（缺少对应 File）');
          }
          importDiagLog('directory_put_item', {
            index: i,
            relative_path: it.relative_path,
            object_key: it.object_key,
            size_bytes: file.size,
            has_presigned_url: !!it.upload_url?.trim(),
          });
          if (!it.upload_url?.trim()) throw new Error('缺少预签名地址');
          const dirFileCount = d.upload_items.length;
          await uploadFileToPresignedUrl(file, it.upload_url, {
            method: it.method,
            headers: it.headers,
            signal: opSignal,
            onByteProgress: (loaded, totForFile) => {
              const t = totForFile > 0 ? totForFile : file.size;
              const clamped = Math.min(Math.max(0, loaded), t);
              const denom = totalAll > 0 ? totalAll : aggregatedDone + jobTotal;
              tick({
                uploadBytesLoaded: aggregatedDone + jobDone + clamped,
                uploadBytesTotal: denom,
                uploadFileIndex: i,
                uploadFileCount: dirFileCount,
                phase: '上传中',
                currentLabel: label,
              });
            },
          });
          jobDone += file.size;
          throwIfAborted(opSignal);
        }
        throwIfAborted(opSignal);
        tick({
          phase: '写入登记',
          currentLabel: label,
          uploadBytesLoaded: aggregatedDone + jobTotal,
          uploadBytesTotal: totalAll > 0 ? totalAll : aggregatedDone + jobTotal,
        });
        const rn = (d.root_dir_name || job.rootDirName || '').trim();
        const manifest = {
          root_dir_name: rn,
          paths: d.upload_items.map((it, idx) => ({
            relative_path: it.relative_path,
            size_bytes: uploadFiles[idx].size,
          })),
          total_files: d.upload_items.length,
          total_size_bytes: jobTotal,
        };
        let doneRes;
        try {
          doneRes = await raceWithAbort(
            completeDirectUpload({
              upload_session_id: d.upload_session_id,
              manifest,
            }),
            opSignal,
          );
        } finally {
          wd.markCompletePhaseEnded();
        }
        if (!doneRes.ok || !doneRes.data?.asset) {
          throw new Error(doneRes.error || '目录登记失败');
        }
        const dirAsset = doneRes.data.asset;
        importedIds.push(String(dirAsset.id));
        successAssetNames.push(String(dirAsset.filename || job.rootDirName));
        completed += 1;
        tick({ completedUnits: completed, failedUnits: failed, currentLabel: label, phase: '' });
      } catch (e) {
        if (isAbortError(e) && !isUserCancelledAbort(e, signal)) {
          const r = wd.consumeLastFireReason();
          const msg = r ? `直传导入 watchdog：${r}` : '直传导入 watchdog：操作超时';
          importDiagLog('import_watchdog_task_outcome', { outcome: 'failed', message: msg });
          return {
            successCount: completed,
            failedCount: failed,
            errorMessage: msg,
            putStarted,
            successAssetNames: successAssetNames.slice(0, 80),
            failureEntries: failureEntries.slice(0, 80),
          };
        }
        if (isUserCancelledAbort(e, signal)) {
          return {
            successCount: completed,
            failedCount: failed,
            errorMessage: '用户已取消导入',
            putStarted,
            successAssetNames: successAssetNames.slice(0, 80),
            failureEntries: failureEntries.slice(0, 80),
          };
        }
        const raw = e instanceof Error ? e.message : '导入失败';
        const expectedUnits = importUnitsForJob(job);
        const processedInJob = completed - c0 + (failed - f0);
        const remaining = Math.max(0, expectedUnits - processedInJob);
        const detail =
          putStarted && remaining === expectedUnits
            ? `${raw}（已向存储写入部分数据，未完成本项登记；为避免重复导入未自动改走其他方式）`
            : raw;
        if (remaining > 0) {
          failed += remaining;
          if (remaining === expectedUnits) {
            pushJobWideFailures(failureEntries, job, detail);
          } else {
            failureEntries.push({ name: label, reason: detail });
          }
        }
        tick({ completedUnits: completed, failedUnits: failed, currentLabel: label, phase: '' });
      }
    }

    if (projectId && importedIds.length) {
      recordProjectActivityAndTouch(
        projectId,
        'DATA_IMPORTED',
        `文件夹识别导入 ${importedIds.length} 条`,
        '当前用户',
        importedIds.join(','),
      );
    }
    const countLine = `共 ${totalUnits} 项，成功 ${completed}，失败 ${failed}`;
    const summaryMessage = plan.summary
      ? `${plan.summary}；${countLine}`
      : `导入结束：${countLine}`;
    return {
      successCount: completed,
      failedCount: failed,
      summaryMessage,
      putStarted: true,
      successAssetNames: successAssetNames.slice(0, 80),
      failureEntries: failureEntries.slice(0, 80),
    };
    } finally {
      wd.dispose();
      activeWatchdog = null;
    }
  }

  if (ENABLE_DIRECT_MULTI && shouldUseDirectMultiBatch(files) && !treeBatch) {
    const failureEntries: ImportFailureEntry[] = [];
    const successAssetNames: string[] = [];
    let filesForFlat: File[] = files;
    activeWatchdog = new DirectImportWatchdog(signal);
    const wd = activeWatchdog;
    const opSignal = wd.signal;
    try {
      importDiagLog('flat_multi_resolve_files_begin', { inputFiles: files.length });
      const filesReady = await withLocalTimeout(
        resolveFilesForDirectUploadItems(files),
        RESOLVE_FILES_TIMEOUT_MS,
        'resolveFilesForDirectUploadItems',
      );
      importDiagLog('flat_multi_resolve_files_done', { filesReady: filesReady.length });
      filesForFlat = filesReady;
      if (filesReady.length === 0) {
        return {
          successCount: 0,
          failedCount: files.length,
          errorMessage:
            '没有可上传的文件：所选条目中没有任何具备有效可读大小的文件（例如均为目录占位项）。',
          putStarted: false,
        };
      }
      if (filesReady.length === 1) {
        const resolvedKeySet = new Set(filesReady.map((f) => fileKey(f)));
        for (const rf of files) {
          if (resolvedKeySet.has(fileKey(rf))) continue;
          if (failureEntries.some((e) => e.name === (rf.name || '—') && e.reason === SKIP_NO_READABLE_SIZE))
            continue;
          failureEntries.push({ name: rf.name || '—', reason: SKIP_NO_READABLE_SIZE });
        }
        const f = filesReady[0];
        tick({
          phase: '准备上传',
          currentLabel: f.name,
          uploadBytesLoaded: 0,
          uploadBytesTotal: f.size > 0 ? f.size : undefined,
        });
        throwIfAborted(opSignal);
        const initOne = await raceWithAbort(
          initDirectUpload({
            project_id: projectId,
            upload_mode: 'single_file',
            filename: f.name,
            size_bytes: f.size,
            content_type: f.type || null,
          }),
          opSignal,
        );
        if (!initOne.ok || !initOne.data) {
          return {
            successCount: 0,
            failedCount: files.length,
            errorMessage: initOne.error || '申请上传失败',
            putStarted: false,
            failureEntries: failureEntries.slice(0, 80),
          };
        }
        const d0 = initOne.data;
        onUploadSessionReady?.(d0.upload_session_id);
        emitImportDirectInitSnapshot(onDirectInitSnapshot, d0, [f.size], { fallbackRelativePath: f.name });
        const item0 = d0.upload_items?.[0];
        const putUrl0 = item0?.upload_url ?? d0.upload_url;
        const putMethod0 = item0?.method ?? d0.method ?? 'PUT';
        const putHeaders0 = item0?.headers ?? d0.headers ?? {};
        if (!putUrl0) {
          return {
            successCount: 0,
            failedCount: files.length,
            errorMessage: '缺少预签名地址',
            putStarted: false,
            failureEntries: failureEntries.slice(0, 80),
          };
        }
        tick({ phase: '上传中', currentLabel: f.name });
        putStarted = true;
        await uploadFileToPresignedUrl(f, putUrl0, {
          method: putMethod0,
          headers: putHeaders0,
          signal: opSignal,
          onByteProgress: (loaded, totForFile) => {
            const t = totForFile > 0 ? totForFile : f.size;
            tick({
              uploadBytesLoaded: Math.min(Math.max(0, loaded), t),
              uploadBytesTotal: t,
              phase: '上传中',
              currentLabel: f.name,
            });
          },
        });
        tick({
          phase: '写入登记',
          currentLabel: f.name,
          uploadBytesLoaded: f.size,
          uploadBytesTotal: f.size,
        });
        let done0;
        try {
          done0 = await raceWithAbort(
            completeDirectUpload({
              upload_session_id: d0.upload_session_id,
              size_bytes: f.size,
            }),
            opSignal,
          );
        } finally {
          wd.markCompletePhaseEnded();
        }
        if (!done0.ok || !done0.data?.asset) {
          failed = files.length;
          tick({ completedUnits: 0, failedUnits: failed, phase: '' });
          return {
            successCount: 0,
            failedCount: failed,
            errorMessage: done0.error || '登记失败',
            putStarted: true,
            failureEntries: failureEntries.slice(0, 80),
          };
        }
        const asset0 = done0.data.asset;
        if (projectId) {
          recordProjectActivityAndTouch(
            projectId,
            'DATA_IMPORTED',
            `直传导入数据 1 条`,
            '当前用户',
            String(asset0.id),
          );
        }
        const skipped = Math.max(0, files.length - 1);
        tick({ completedUnits: 1, failedUnits: skipped, phase: '' });
        return {
          successCount: 1,
          failedCount: skipped,
          summaryMessage:
            skipped > 0
              ? `已用单文件直传完成 1 个有效文件；另有 ${skipped} 个条目无有效大小已跳过。`
              : `直传成功：${asset0.filename}`,
          putStarted: true,
          successAssetNames: [String(asset0.filename || f.name)],
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      const mode: DirectUploadMode = 'multi_file';
      const totalBytesAll = filesReady.reduce((s, f) => s + f.size, 0);
      const items: DirectUploadFileItemSpec[] = filesReady.map((f) => ({
        client_file_id: newClientFileId(),
        relative_path: (
          (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
        ).replace(/\\/g, '/'),
        size_bytes: f.size,
        content_type: f.type || null,
      }));
      tick({
        phase: '准备上传',
        currentLabel: `${filesReady.length} 个文件`,
        uploadBytesLoaded: 0,
        uploadBytesTotal: totalBytesAll > 0 ? totalBytesAll : undefined,
      });
      throwIfAborted(opSignal);
      const init = await raceWithAbort(
        initDirectUpload({
          project_id: projectId,
          upload_mode: mode,
          items,
        }),
        opSignal,
      );
      if (!init.ok || !init.data?.upload_items?.length) {
        const reason = init.error || '申请上传失败';
        for (const f of filesReady) failureEntries.push({ name: f.name || '—', reason });
        return {
          successCount: 0,
          failedCount: filesReady.length,
          errorMessage: reason,
          putStarted: false,
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      const d = init.data;
      if (d.upload_items.length !== filesReady.length) {
        const reason = `upload-init 返回 ${d.upload_items.length} 个预签名项，与本地可上传文件数 ${filesReady.length} 不一致`;
        importDiagLog('flat_multi_upload_items_mismatch', {
          upload_session_id: d.upload_session_id,
          server_items: d.upload_items.length,
          local_files: filesReady.length,
        });
        for (const f of filesReady) failureEntries.push({ name: f.name || '—', reason });
        return {
          successCount: 0,
          failedCount: filesReady.length,
          errorMessage: reason,
          putStarted: false,
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      onUploadSessionReady?.(d.upload_session_id);
      emitImportDirectInitSnapshot(onDirectInitSnapshot, d, filesReady.map((x) => x.size));
      const totalBytes = totalBytesAll;
      let doneBytes = 0;
      tick({ phase: '上传中', currentLabel: `${filesReady.length} 个文件` });
      putStarted = true;
      const uploadErrs: ImportFailureEntry[] = [];
      for (let i = 0; i < d.upload_items.length; i++) {
        throwIfAborted(opSignal);
        const it = d.upload_items[i];
        const file = filesReady[i];
        try {
          if (!file) {
            throw new Error('本地文件队列与 upload-init 返回项不匹配（缺少对应 File）');
          }
          importDiagLog('flat_multi_put_item', {
            index: i,
            relative_path: it.relative_path,
            object_key: it.object_key,
            size_bytes: file.size,
            has_presigned_url: !!it.upload_url?.trim(),
          });
          if (!it.upload_url?.trim()) throw new Error('缺少预签名地址');
          const flatFileCount = d.upload_items.length;
          await uploadFileToPresignedUrl(file, it.upload_url, {
            method: it.method,
            headers: it.headers,
            signal: opSignal,
            onByteProgress: (loaded, totForFile) => {
              const t = totForFile > 0 ? totForFile : file.size;
              const clamped = Math.min(Math.max(0, loaded), t);
              tick({
                uploadBytesLoaded: doneBytes + clamped,
                uploadBytesTotal: totalBytes,
                uploadFileIndex: i,
                uploadFileCount: flatFileCount,
                phase: '上传中',
                currentLabel: file.name,
              });
            },
          });
          doneBytes += file.size;
        } catch (up) {
          if (isAbortError(up)) throw up;
          uploadErrs.push({
            name: file.name || '—',
            reason: up instanceof Error ? up.message : '上传失败',
          });
        }
        throwIfAborted(opSignal);
      }
      throwIfAborted(opSignal);
      tick({
        phase: '写入登记',
        currentLabel: `${filesReady.length} 个文件`,
        uploadBytesLoaded: totalBytes,
        uploadBytesTotal: totalBytes,
      });
      let doneRes;
      try {
        doneRes = await raceWithAbort(
          completeDirectUpload({ upload_session_id: d.upload_session_id }),
          opSignal,
        );
      } finally {
        wd.markCompletePhaseEnded();
      }
      if (!doneRes.ok || !doneRes.data) {
        const base = doneRes.error || '多文件登记失败';
        const err = putStarted
          ? `直传已向 MinIO 上传数据，未完成登记，已禁止自动回退到标准导入（避免重复资产）。原因：${base}`
          : base;
        for (const f of filesReady) failureEntries.push({ name: f.name || '—', reason: err });
        return {
          successCount: 0,
          failedCount: filesReady.length,
          errorMessage: err,
          putStarted,
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      const assets = doneRes.data.assets ?? [];
      const failedItems = doneRes.data.failed_items ?? [];
      const namesFromServer = new Set(
        failedItems.map((fi) => leafNameFromRelPath(fi.relative_path)),
      );
      mergeFailedItemsIntoEntries(failedItems, failureEntries);
      for (const u of uploadErrs) {
        if (!namesFromServer.has(u.name)) failureEntries.push(u);
      }
      for (const a of assets) {
        successAssetNames.push(String(a.filename || ''));
      }
      completed = assets.length;
      failed = filesReady.length - assets.length;
      tick({ completedUnits: completed, failedUnits: failed, phase: '' });
      if (projectId && assets.length > 0) {
        recordProjectActivityAndTouch(
          projectId,
          'DATA_IMPORTED',
          `多文件直传 ${assets.length} 条`,
          '当前用户',
          assets.map((x) => x.id).join(','),
        );
      }
      const countLine = `共 ${totalUnits} 项，成功 ${completed}，失败 ${failed}`;
      const summaryMessage =
        failed > 0 ? `多文件直传：${countLine}` : `多文件直传成功 ${completed} 条`;
      return {
        successCount: completed,
        failedCount: failed,
        summaryMessage,
        putStarted: true,
        successAssetNames: successAssetNames.slice(0, 80),
        failureEntries: failureEntries.slice(0, 80),
      };
    } catch (e) {
      if (isAbortError(e) && !isUserCancelledAbort(e, signal)) {
        const r = wd.consumeLastFireReason();
        const msg = r ? `直传导入 watchdog：${r}` : '直传导入 watchdog：操作超时';
        importDiagLog('import_watchdog_task_outcome', { outcome: 'failed', message: msg });
        return {
          successCount: completed,
          failedCount: failed,
          errorMessage: msg,
          putStarted,
          successAssetNames: successAssetNames.slice(0, 80),
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      if (isUserCancelledAbort(e, signal)) {
        return {
          successCount: completed,
          failedCount: failed,
          errorMessage: '用户已取消导入',
          putStarted,
          successAssetNames: successAssetNames.slice(0, 80),
          failureEntries: failureEntries.slice(0, 80),
        };
      }
      const base = e instanceof Error ? e.message : '导入失败';
      const err = putStarted
        ? `直传已向 MinIO 上传数据，未完成登记，已禁止自动回退到标准导入（避免重复资产）。原因：${base}`
        : base;
      for (const f of filesForFlat) {
        if (!failureEntries.some((x) => x.name === f.name)) {
          failureEntries.push({ name: f.name || '—', reason: err });
        }
      }
      return {
        successCount: completed,
        failedCount: Math.max(failed, filesForFlat.length - completed),
        errorMessage: err,
        putStarted,
        successAssetNames: successAssetNames.slice(0, 80),
        failureEntries: failureEntries.slice(0, 80),
      };
    } finally {
      wd.dispose();
      activeWatchdog = null;
    }
  }

  try {
    throwIfAborted(signal);
    const res = await importDataAssetFiles(files, projectId, projectName, {
      onProgress: (p) =>
        tick({
          uploadHintPercent: p,
          phase: '上传中',
          currentLabel: '标准导入',
        }),
      signal,
    });

    if (!res.ok || !res.data) {
      return { successCount: 0, failedCount: 0, errorMessage: res.error || '导入请求失败', putStarted: false };
    }

    const { imported, failed: failedList } = res.data;
    completed = imported.length;
    failed = failedList.length;
    tick({ completedUnits: completed, failedUnits: failed, phase: '' });

    if (completed > 0 && projectId) {
      recordProjectActivityAndTouch(
        projectId,
        'DATA_IMPORTED',
        `导入数据 ${completed} 条`,
        '当前用户',
        imported.map((i) => i.id).join(','),
      );
    }

    const stdFailures: ImportFailureEntry[] = failedList.map((f) => ({
      name: f.name || '—',
      reason: (f.reason || '失败').slice(0, 220),
    }));
    const stdSuccessNames = imported.map((i) => i.name || String(i.id));

    if (completed === 0 && failed > 0) {
      return {
        successCount: 0,
        failedCount: failed,
        errorMessage: `全部失败（共 ${failed} 项）：${failedList.map((f) => `${f.name}: ${f.reason}`).join('；')}`,
        putStarted: false,
        failureEntries: stdFailures.slice(0, 80),
      };
    }

    if (completed > 0 && failed > 0) {
      const head = failedList
        .slice(0, 4)
        .map((f) => `${f.name}: ${f.reason}`)
        .join('；');
      const more = failedList.length > 4 ? ` … 等 ${failedList.length} 项失败` : '';
      return {
        successCount: completed,
        failedCount: failed,
        summaryMessage: `共 ${totalUnits} 项，成功 ${completed}，失败 ${failed}。${head}${more}`,
        putStarted: false,
        successAssetNames: stdSuccessNames.slice(0, 80),
        failureEntries: stdFailures.slice(0, 80),
      };
    }

    if (completed > 0) {
      const names = imported.map((i) => i.name).slice(0, 3);
      const tail =
        completed <= 3 ? names.join('、') : `${names.join('、')} 等 ${completed} 条`;
      return {
        successCount: completed,
        failedCount: 0,
        summaryMessage: `成功导入 ${completed} 条：${tail}`,
        putStarted: false,
        successAssetNames: stdSuccessNames.slice(0, 80),
      };
    }

    return { successCount: 0, failedCount: 0, errorMessage: '未导入任何文件', putStarted: false };
  } catch (e) {
    if (isAbortError(e)) {
      return { successCount: 0, failedCount: 0, errorMessage: '用户已取消导入', putStarted: false };
    }
    return {
      successCount: 0,
      failedCount: 0,
      errorMessage: e instanceof Error ? e.message : '导入异常',
      putStarted: false,
    };
  }
}
