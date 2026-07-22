/**
 * 同一 upload_session 下的直传「继续导入」：用户重新选择文件后，按路径+大小与会话项匹配，仅向未完成 PUT 的项补传并 upload-complete。
 */

import type { ImportDirectInitSnapshot } from '@/components/task-center/types';
import {
  completeDirectUpload,
  uploadFileToPresignedUrl,
} from '@/features/data-platform/api/dataAssetsApi';
import type { DataImportProgress, DataImportResult } from '@/components/assets/dataAssetImportRunner';

function normRel(p: string): string {
  return (p || '').replace(/\\/g, '/').replace(/^\/+/, '').trim();
}

function fileRel(f: File): string {
  const w = (f as File & { webkitRelativePath?: string }).webkitRelativePath || '';
  return normRel(w || f.name);
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
}

function isAbortError(e: unknown): boolean {
  return (
    (e instanceof DOMException && e.name === 'AbortError') ||
    (e instanceof Error && (e.name === 'AbortError' || e.message === '已取消'))
  );
}

export function snapshotLikelyExpired(snap: ImportDirectInitSnapshot, skewMs = 90_000): boolean {
  const raw = (snap.expires_at || '').trim();
  if (!raw) return false;
  const t = Date.parse(raw);
  if (Number.isNaN(t)) return false;
  return Date.now() > t - skewMs;
}

/**
 * 将用户选择的文件与会话 items 严格一一对应（路径规范化 + 字节大小一致）。
 */
export function matchResumeFilesToSnapshot(
  snapshot: ImportDirectInitSnapshot,
  files: File[]
): { ok: true; ordered: File[] } | { ok: false; error: string } {
  const rows = snapshot.items;
  if (rows.length === 0) return { ok: false, error: '会话缺少上传项' };
  const used = new Set<File>();
  const ordered: File[] = [];
  for (const row of rows) {
    const want = normRel(row.relative_path);
    const wantSize = Number(row.size_bytes);
    let hit: File | undefined;
    for (const f of files) {
      if (used.has(f)) continue;
      if (f.size !== wantSize) continue;
      const rel = fileRel(f);
      if (rel === want || rel.endsWith(`/${want}`) || want.endsWith(`/${rel}`)) {
        hit = f;
        break;
      }
    }
    if (!hit) {
      return {
        ok: false,
        error: `未找到与会话匹配的文件：${row.relative_path}（${wantSize} 字节）。请重新选择与原任务一致的文件或目录。`,
      };
    }
    used.add(hit);
    ordered.push(hit);
  }
  return { ok: true, ordered };
}

export async function continueDirectImportFromSnapshot(params: {
  snapshot: ImportDirectInitSnapshot;
  files: File[];
  signal?: AbortSignal;
  onProgress: (p: DataImportProgress) => void;
  totalUnits: number;
}): Promise<DataImportResult> {
  const { snapshot, files, signal, onProgress, totalUnits } = params;
  const mode = (snapshot.upload_mode || '').trim().toLowerCase();
  const tick = (partial: Partial<DataImportProgress>) => {
    onProgress({
      completedUnits: partial.completedUnits ?? 0,
      failedUnits: partial.failedUnits ?? 0,
      totalUnits: partial.totalUnits ?? totalUnits,
      currentLabel: partial.currentLabel,
      phase: partial.phase,
      uploadHintPercent: partial.uploadHintPercent,
    });
  };

  const matched = matchResumeFilesToSnapshot(snapshot, files);
  if (!matched.ok) {
    return { successCount: 0, failedCount: 0, errorMessage: matched.error, putStarted: false };
  }
  const ordered = matched.ordered;
  let putStarted = false;

  try {
    if (mode === 'single_file') {
      if (snapshot.items.length !== 1) {
        return { successCount: 0, failedCount: 0, errorMessage: '单文件会话项数异常', putStarted: false };
      }
      const row = snapshot.items[0];
      const file = ordered[0];
      throwIfAborted(signal);
      tick({ phase: '上传中', currentLabel: file.name, completedUnits: 0, failedUnits: 0, totalUnits });
      putStarted = true;
      await uploadFileToPresignedUrl(file, row.upload_url, {
        method: row.method,
        headers: row.headers,
        signal,
        onProgress: (pct) => tick({ uploadHintPercent: pct, phase: '上传中', currentLabel: file.name }),
      });
      tick({ phase: '写入登记', currentLabel: file.name });
      const done = await completeDirectUpload({
        upload_session_id: snapshot.upload_session_id,
        size_bytes: file.size,
      });
      if (!done.ok || !done.data?.asset) {
        return {
          successCount: 0,
          failedCount: 1,
          errorMessage: done.error || '登记失败',
          putStarted: true,
        };
      }
      tick({ completedUnits: 1, failedUnits: 0, phase: '' });
      return {
        successCount: 1,
        failedCount: 0,
        summaryMessage: `直传成功：${done.data.asset.filename}`,
        putStarted: true,
        successAssetNames: [String(done.data.asset.filename || file.name)],
      };
    }

    if (mode === 'multi_file') {
      const totalBytes = ordered.reduce((s, f) => s + f.size, 0);
      let doneBytes = 0;
      tick({ phase: '上传中', currentLabel: `${ordered.length} 个文件`, completedUnits: 0, failedUnits: 0 });
      putStarted = true;
      const uploadErrs: Array<{ name: string; reason: string }> = [];
      for (let i = 0; i < snapshot.items.length; i++) {
        throwIfAborted(signal);
        const row = snapshot.items[i];
        const file = ordered[i];
        try {
          if (!row.upload_url?.trim()) throw new Error('缺少预签名地址');
          await uploadFileToPresignedUrl(file, row.upload_url, {
            method: row.method,
            headers: row.headers,
            signal,
            onProgress: (pct) => {
              const cur = doneBytes + (file.size * pct) / 100;
              tick({
                uploadHintPercent: totalBytes ? Math.min(99, Math.round((cur / totalBytes) * 100)) : pct,
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
      }
      tick({ phase: '写入登记', currentLabel: `${ordered.length} 个文件` });
      const doneRes = await completeDirectUpload({ upload_session_id: snapshot.upload_session_id });
      if (!doneRes.ok || !doneRes.data) {
        return {
          successCount: 0,
          failedCount: ordered.length,
          errorMessage: doneRes.error || '多文件登记失败',
          putStarted: true,
          failureEntries: uploadErrs.slice(0, 80),
        };
      }
      const assets = doneRes.data.assets ?? [];
      const failedItems = doneRes.data.failed_items ?? [];
      const failureEntries = [...uploadErrs];
      for (const fi of failedItems) {
        const name = normRel(String(fi.relative_path || '')).split('/').pop() || '—';
        failureEntries.push({ name, reason: String(fi.reason || '登记失败').slice(0, 220) });
      }
      const successAssetNames = assets.map((a) => String(a.filename || ''));
      const completed = assets.length;
      const failed = Math.max(0, ordered.length - completed);
      tick({ completedUnits: completed, failedUnits: failed, phase: '' });
      return {
        successCount: completed,
        failedCount: failed,
        summaryMessage: `续传完成：成功 ${completed}，失败 ${failed}`,
        putStarted: true,
        successAssetNames: successAssetNames.slice(0, 80),
        failureEntries: failureEntries.slice(0, 80),
      };
    }

    if (mode === 'directory') {
      const jobTotal = ordered.reduce((s, f) => s + f.size, 0);
      let jobDone = 0;
      const rn = (snapshot.root_dir_name || '').trim();
      tick({ phase: '上传中', currentLabel: rn || '目录', completedUnits: 0, failedUnits: 0 });
      putStarted = true;
      for (let i = 0; i < snapshot.items.length; i++) {
        throwIfAborted(signal);
        const row = snapshot.items[i];
        const file = ordered[i];
        if (!row.upload_url?.trim()) throw new Error('缺少预签名地址');
        await uploadFileToPresignedUrl(file, row.upload_url, {
          method: row.method,
          headers: row.headers,
          signal,
          onProgress: (pct) => {
            const cur = jobDone + (file.size * pct) / 100;
            tick({
              uploadHintPercent: jobTotal ? Math.min(99, Math.round((cur / jobTotal) * 100)) : pct,
              phase: '上传中',
              currentLabel: rn || file.name,
            });
          },
        });
        jobDone += file.size;
      }
      tick({ phase: '写入登记', currentLabel: rn || '目录' });
      const manifest = {
        root_dir_name: rn,
        paths: snapshot.items.map((it, idx) => ({
          relative_path: normRel(it.relative_path),
          size_bytes: ordered[idx].size,
        })),
        total_files: snapshot.items.length,
        total_size_bytes: jobTotal,
      };
      const doneRes = await completeDirectUpload({
        upload_session_id: snapshot.upload_session_id,
        manifest,
      });
      if (!doneRes.ok || !doneRes.data?.asset) {
        return {
          successCount: 0,
          failedCount: 1,
          errorMessage: doneRes.error || '目录登记失败',
          putStarted: true,
        };
      }
      const dirAsset = doneRes.data.asset;
      tick({ completedUnits: 1, failedUnits: 0, phase: '' });
      return {
        successCount: 1,
        failedCount: 0,
        summaryMessage: `目录续传成功：${dirAsset.filename}`,
        putStarted: true,
        successAssetNames: [String(dirAsset.filename || rn)],
      };
    }

    return {
      successCount: 0,
      failedCount: 0,
      errorMessage: `暂不支持的续传模式：${mode}`,
      putStarted: false,
    };
  } catch (e) {
    if (isAbortError(e)) {
      return { successCount: 0, failedCount: 0, errorMessage: '用户已取消导入', putStarted };
    }
    return {
      successCount: 0,
      failedCount: 0,
      errorMessage: e instanceof Error ? e.message : '续传异常',
      putStarted,
    };
  }
}
