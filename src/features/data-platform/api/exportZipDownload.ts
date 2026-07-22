/**
 * 导出 zip：流式拉取 + 可选进度回调。
 * 优先 File System Access API 直写磁盘（降低大文件内存峰值）；否则回退为内存 Blob + <a download>。
 */

import { getAccessToken } from '@/lib/auth/session';
import { useAuthStore } from '@/store/authStore';

function getAuthToken(): string | null {
  try {
    const { accessToken } = useAuthStore.getState();
    if (accessToken) return accessToken;
  } catch {
    /* SSR / 非 React 树 */
  }
  if (typeof window === 'undefined') return null;
  return getAccessToken();
}

function exportDownloadPath(jobId: string): string {
  return `/api/data-assets/export/download?jobId=${encodeURIComponent(jobId)}`;
}

function parseContentLength(header: string | null): number | null {
  if (!header) return null;
  const n = parseInt(header, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function parseFilenameFromDisposition(disposition: string | null): string | undefined {
  if (!disposition) return undefined;
  const match = /filename="?([^";\n]+)"?/.exec(disposition);
  return match ? match[1].trim() : undefined;
}

async function parseHttpError(response: Response): Promise<string> {
  const text = await response.text();
  let err = `HTTP ${response.status}: ${response.statusText}`;
  try {
    const data = text ? JSON.parse(text) : null;
    if (data?.detail) err = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    else if (data?.error) err = data.error;
  } catch {
    if (text) err = text.slice(0, 200);
  }
  return err;
}

async function pumpStreamToWritable(
  body: ReadableStream<Uint8Array>,
  writable: FileSystemWritableFileStream,
  total: number | null,
  onProgress?: (p: { loaded: number; total: number | null }) => void
): Promise<number> {
  const reader = body.getReader();
  let loaded = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        loaded += value.byteLength;
        onProgress?.({ loaded, total });
        await writable.write(value as Parameters<FileSystemWritableFileStream['write']>[0]);
      }
    }
    await writable.close();
    return loaded;
  } catch (e) {
    await writable.abort().catch(() => {});
    throw e;
  } finally {
    reader.releaseLock();
  }
}

async function readStreamToBlob(
  body: ReadableStream<Uint8Array>,
  total: number | null,
  onProgress?: (p: { loaded: number; total: number | null }) => void
): Promise<{ blob: Blob; loaded: number }> {
  const reader = body.getReader();
  const chunks: BlobPart[] = [];
  let loaded = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value as BlobPart);
        loaded += value.byteLength;
        onProgress?.({ loaded, total });
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { blob: new Blob(chunks, { type: 'application/zip' }), loaded };
}

function saveBlobWithAnchor(blob: Blob, name: string): void {
  const u = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = u;
  a.download = name;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(u);
}

export type ExportZipDownloadProgress = { loaded: number; total: number | null };

export type ExportZipDownloadResult = { ok: boolean; error?: string; cancelled?: boolean };

type SavePicker = NonNullable<
  (Window & { showSaveFilePicker?: (options?: unknown) => Promise<FileSystemFileHandle> })['showSaveFilePicker']
>;

async function fetchExportZipResponse(
  jobId: string,
  headers: HeadersInit
): Promise<
  | { ok: true; response: Response; fileName: string; total: number | null }
  | { ok: false; error: string }
> {
  const path = exportDownloadPath(jobId);
  let response: Response;
  try {
    response = await fetch(path, { method: 'GET', headers, credentials: 'include' });
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const disposition = response.headers.get('Content-Disposition');
  const fileName = parseFilenameFromDisposition(disposition) || 'export.zip';
  if (!response.ok) {
    const err = await parseHttpError(response);
    return { ok: false, error: err };
  }
  const total = parseContentLength(response.headers.get('Content-Length'));
  return { ok: true, response, fileName, total };
}

/**
 * GET /api/data-assets/export/download 并保存为 zip。
 */
export async function fetchAndSaveExportZip(
  jobId: string,
  fallbackFileName: string | undefined,
  onProgress?: (p: ExportZipDownloadProgress) => void
): Promise<ExportZipDownloadResult> {
  if (typeof window === 'undefined') {
    return { ok: false, error: '仅浏览器端可下载' };
  }

  try {
    const token = getAuthToken();
    const headers: HeadersInit = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const first = await fetchExportZipResponse(jobId, headers);
    if (!first.ok) return { ok: false, error: first.error };

    let { response, fileName, total } = first;
    fileName = fileName || fallbackFileName || 'export.zip';

    onProgress?.({ loaded: 0, total });

    if (!response.body) {
      const blob = await response.blob();
      if (blob.size === 0) return { ok: false, error: '下载内容为空' };
      onProgress?.({ loaded: blob.size, total: total ?? blob.size });
      saveBlobWithAnchor(blob, fileName);
      return { ok: true };
    }

    const w = window as Window & { showSaveFilePicker?: SavePicker };
    if (typeof w.showSaveFilePicker === 'function') {
      try {
        const handle = await w.showSaveFilePicker({
          suggestedName: fileName,
          types: [{ description: 'ZIP', accept: { 'application/zip': ['.zip'] } }],
        });
        const writable = await handle.createWritable();
        const loaded = await pumpStreamToWritable(response.body, writable, total, onProgress);
        if (loaded === 0) return { ok: false, error: '下载内容为空' };
        return { ok: true };
      } catch (e) {
        const err = e as { name?: string };
        if (err?.name === 'AbortError') {
          await response.body.cancel().catch(() => {});
          return { ok: false, cancelled: true };
        }
        await response.body.cancel().catch(() => {});
        const second = await fetchExportZipResponse(jobId, headers);
        if (!second.ok) return { ok: false, error: second.error };
        response = second.response;
        fileName = second.fileName || fallbackFileName || 'export.zip';
        total = second.total;
        onProgress?.({ loaded: 0, total });
        if (!response.body) {
          const blob = await response.blob();
          if (blob.size === 0) return { ok: false, error: '下载内容为空' };
          onProgress?.({ loaded: blob.size, total: total ?? blob.size });
          saveBlobWithAnchor(blob, fileName);
          return { ok: true };
        }
        const { blob, loaded } = await readStreamToBlob(response.body, total, onProgress);
        if (loaded === 0) return { ok: false, error: '下载内容为空' };
        saveBlobWithAnchor(blob, fileName);
        return { ok: true };
      }
    }

    const { blob, loaded } = await readStreamToBlob(response.body, total, onProgress);
    if (loaded === 0) return { ok: false, error: '下载内容为空' };
    saveBlobWithAnchor(blob, fileName);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
