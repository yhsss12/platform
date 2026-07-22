'use client';

import { apiDelete, apiGet, apiPost } from '@/lib/api/authClient';
import { getAccessToken, getSessionId, initSessionId } from '@/lib/auth/session';

const API_BASE =
  typeof window !== 'undefined' ? '/api' : process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export interface AssetPipelineFileInfo {
  path: string;
  sizeBytes?: number | null;
  exists?: boolean;
}

export interface AssetPipelineJobStatus {
  jobId: string;
  name?: string | null;
  status: string;
  phase?: string;
  progress?: number;
  message?: string | null;
  error?: string | null;
  updatedAt?: string | null;
  inputImage?: string | null;
  targetEngine?: string | null;
  assetType?: string | null;
  commandSummary?: string | null;
  segmentation?: Record<string, unknown> | null;
  reconstruction?: Record<string, unknown> | null;
  mujocoExport?: MujocoExportInfo | null;
  mujocoVisualization?: MujocoVisualizationInfo | null;
  files?: AssetPipelineFileInfo[];
  extra?: Record<string, unknown>;
}

export interface AssetSegmentPayload {
  prompt?: string | null;
  positiveBoxes?: number[][];
  negativeBoxes?: number[][];
  confidenceThreshold?: number;
  textOnly?: boolean;
}

export interface AssetReconstructPayload {
  cutoutIndex: number;
  seed?: number;
  prepareOnly?: boolean;
}

export interface AssetJobCreateResponse {
  jobId: string;
  status: string;
  inputImage?: string | null;
  name?: string | null;
}

export interface AssetJobListResponse {
  jobs: AssetPipelineJobStatus[];
  total: number;
}

export interface MujocoValidationResult {
  ok?: boolean;
  skipped?: boolean;
  error?: string | null;
  nbody?: number;
  ngeom?: number;
  nmesh?: number;
  nq?: number;
  nv?: number;
}

export interface MujocoExportInfo {
  status?: string;
  outputDir?: string;
  modelPreviewXml?: string;
  modelXml?: string;
  packageZip?: string;
  visualMesh?: string;
  collisionMesh?: string;
  metadataPath?: string;
  scaleLongest?: number;
  mass?: number;
  collision?: string;
  validation?: {
    preview?: MujocoValidationResult;
    physics?: MujocoValidationResult;
  };
  viewerCommands?: {
    preview?: string;
    physics?: string;
  };
  error?: string | null;
  warnings?: string[];
}

export interface MujocoExportDownload {
  key: string;
  relPath: string;
  label: string;
  filename: string;
}

export interface MujocoVisualizationInfo {
  status?: string;
  previewImage?: string;
  xml?: string;
  renderer?: string;
  glBackend?: string;
  width?: number;
  height?: number;
  nbody?: number;
  ngeom?: number;
  nmesh?: number;
  error?: string | null;
}

async function parseApiError(response: Response, fallback: string): Promise<never> {
  const errorData = await response.json().catch(() => ({ detail: response.statusText }));
  const detail = (errorData as { detail?: unknown }).detail;
  if (typeof detail === 'string') {
    throw new Error(detail);
  }
  if (Array.isArray(detail)) {
    throw new Error(detail.map((item) => JSON.stringify(item)).join('; '));
  }
  throw new Error(fallback);
}

export async function createAssetPipelineJob(
  name: string,
  image: File
): Promise<AssetJobCreateResponse> {
  const form = new FormData();
  form.append('name', name);
  form.append('image', image);

  const token = getAccessToken();
  const sessionId = initSessionId();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (sessionId) headers['X-Session-Id'] = sessionId;

  const response = await fetch(`${API_BASE}/workspace/asset-pipeline/jobs`, {
    method: 'POST',
    headers,
    body: form,
    credentials: 'omit',
  });

  if (!response.ok) {
    await parseApiError(response, '创建重建任务失败');
  }

  return response.json() as Promise<AssetJobCreateResponse>;
}

export async function startAssetSegmentation(
  jobId: string,
  payload: AssetSegmentPayload
): Promise<AssetPipelineJobStatus> {
  return apiPost<AssetPipelineJobStatus>(
    `/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}/segment`,
    payload
  );
}

export async function startAssetReconstruction(
  jobId: string,
  payload: AssetReconstructPayload
): Promise<AssetPipelineJobStatus> {
  return apiPost<AssetPipelineJobStatus>(
    `/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}/reconstruct`,
    payload
  );
}

export interface AssetRenderMujocoPayload {
  xmlKind?: 'preview' | 'physics';
  width?: number;
  height?: number;
}

export async function renderMujocoPreview(
  jobId: string,
  payload: AssetRenderMujocoPayload = {}
): Promise<AssetPipelineJobStatus> {
  return apiPost<AssetPipelineJobStatus>(
    `/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}/render-mujoco`,
    payload
  );
}

export async function getAssetPipelineJob(jobId: string): Promise<AssetPipelineJobStatus> {
  return apiGet<AssetPipelineJobStatus>(
    `/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}`
  );
}

export interface AssetJobDeleteResponse {
  ok: boolean;
  jobId: string;
  deleted: boolean;
}

export async function listAssetPipelineJobs(limit = 50): Promise<AssetJobListResponse> {
  return apiGet<AssetJobListResponse>(`/workspace/asset-pipeline/jobs?limit=${limit}`);
}

export async function deleteAssetPipelineJob(jobId: string): Promise<AssetJobDeleteResponse> {
  try {
    return await apiDelete<AssetJobDeleteResponse>(
      `/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}`
    );
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('job not found') || error.message.includes('404')) {
        throw new Error('任务不存在或已被删除');
      }
      if (error.message.includes('任务运行中') || error.message.includes('409')) {
        throw new Error('任务运行中，暂不能删除');
      }
      if (error.message.includes('Authentication failed') || error.message.includes('401')) {
        throw new Error('未授权或登录已过期，请重新登录');
      }
      if (error.message.includes('403') || error.message.includes('Forbidden')) {
        throw new Error('没有权限删除该任务');
      }
      throw error;
    }
    throw new Error('删除任务失败');
  }
}

export function getAssetPipelineFileUrl(jobId: string, relPath: string): string {
  const cleanRelPath = relPath.replace(/^\/+/, '');
  const encodedPath = cleanRelPath.split('/').map((part) => encodeURIComponent(part)).join('/');
  return `${API_BASE}/workspace/asset-pipeline/jobs/${encodeURIComponent(jobId)}/files/${encodedPath}`;
}

export class AssetPipelineFileError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'AssetPipelineFileError';
    this.status = status;
  }
}

export function formatAssetPipelineFileError(error: unknown, action = '下载'): string {
  if (error instanceof AssetPipelineFileError) {
    if (error.status === 401 || error.status === 403) {
      return `${action}失败：未授权或登录已过期，请重新登录后再试`;
    }
    if (error.status === 404) {
      return `${action}失败：文件不存在`;
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return `${action}失败`;
}

export function resolveGsPlyRelPath(job: AssetPipelineJobStatus | null | undefined): string | null {
  const fromRecon = job?.reconstruction?.gsPlyPath ?? job?.reconstruction?.splatPlyPath;
  if (typeof fromRecon === 'string' && fromRecon.trim()) {
    return fromRecon.replace(/^\/+/, '');
  }
  if (jobHasFile(job, 'sam3d/splat.ply')) return 'sam3d/splat.ply';
  if (jobHasFile(job, 'sam3d/gs.ply')) return 'sam3d/gs.ply';
  return null;
}

export interface Sam3dExportDownload {
  key: string;
  relPath: string;
  label: string;
  filename: string;
}

const SAM3D_EXPORT_CANDIDATES: Array<{
  key: string;
  reconKey?: string;
  relPaths: string[];
  label: string;
  filename: string;
}> = [
  {
    key: 'splat_ply',
    reconKey: 'splatPlyPath',
    relPaths: ['sam3d/splat.ply', 'sam3d/gs.ply'],
    label: 'Gaussian Splat (splat.ply)',
    filename: 'splat.ply',
  },
  {
    key: 'gaussian_ply',
    reconKey: 'gaussianPlyPath',
    relPaths: ['sam3d/gaussian.ply'],
    label: 'Gaussian (gaussian.ply)',
    filename: 'gaussian.ply',
  },
  {
    key: 'glb',
    reconKey: 'glbPath',
    relPaths: ['sam3d/object.glb'],
    label: 'GLB (object.glb)',
    filename: 'object.glb',
  },
  {
    key: 'mesh_obj',
    reconKey: 'meshObjPath',
    relPaths: ['sam3d/mesh.obj'],
    label: 'Mesh OBJ',
    filename: 'mesh.obj',
  },
  {
    key: 'mesh_stl',
    reconKey: 'meshStlPath',
    relPaths: ['sam3d/mesh.stl'],
    label: 'Mesh STL',
    filename: 'mesh.stl',
  },
  {
    key: 'mesh_ply',
    reconKey: 'meshPlyPath',
    relPaths: ['sam3d/mesh.ply'],
    label: 'Triangle Mesh PLY',
    filename: 'mesh.ply',
  },
];

export function resolveSam3dExportDownloads(
  job: AssetPipelineJobStatus | null | undefined
): Sam3dExportDownload[] {
  const downloads: Sam3dExportDownload[] = [];
  const recon = job?.reconstruction ?? {};

  for (const candidate of SAM3D_EXPORT_CANDIDATES) {
    const fromRecon = candidate.reconKey ? recon[candidate.reconKey] : null;
    if (typeof fromRecon === 'string' && fromRecon.trim() && jobHasFile(job, fromRecon)) {
      downloads.push({
        key: candidate.key,
        relPath: fromRecon.replace(/^\/+/, ''),
        label: candidate.label,
        filename: candidate.filename,
      });
      continue;
    }
    const relPath = candidate.relPaths.find((path) => jobHasFile(job, path));
    if (relPath) {
      downloads.push({
        key: candidate.key,
        relPath,
        label: candidate.label,
        filename: candidate.filename,
      });
    }
  }

  return downloads;
}

const MUJOCO_EXPORT_DOWNLOADS: Array<{
  key: string;
  exportKey: keyof MujocoExportInfo;
  fallbackPath: string;
  label: string;
  filename: string;
}> = [
  {
    key: 'package_zip',
    exportKey: 'packageZip',
    fallbackPath: 'exports/mujoco/mujoco_package.zip',
    label: 'MuJoCo 资产包',
    filename: 'mujoco_package.zip',
  },
  {
    key: 'model_preview',
    exportKey: 'modelPreviewXml',
    fallbackPath: 'exports/mujoco/model_preview.xml',
    label: 'model_preview.xml',
    filename: 'model_preview.xml',
  },
  {
    key: 'model_physics',
    exportKey: 'modelXml',
    fallbackPath: 'exports/mujoco/model.xml',
    label: 'model.xml',
    filename: 'model.xml',
  },
  {
    key: 'metadata',
    exportKey: 'metadataPath',
    fallbackPath: 'exports/mujoco/metadata.json',
    label: 'metadata.json',
    filename: 'metadata.json',
  },
  {
    key: 'visual_mesh',
    exportKey: 'visualMesh',
    fallbackPath: 'exports/mujoco/meshes/visual.obj',
    label: 'visual.obj',
    filename: 'visual.obj',
  },
  {
    key: 'collision_mesh',
    exportKey: 'collisionMesh',
    fallbackPath: 'exports/mujoco/meshes/collision.obj',
    label: 'collision.obj',
    filename: 'collision.obj',
  },
];

export function resolveMujocoExport(
  job: AssetPipelineJobStatus | null | undefined
): MujocoExportInfo | null {
  const raw = job?.mujocoExport;
  if (!raw || typeof raw !== 'object') return null;
  return raw;
}

export function resolveMujocoVisualization(
  job: AssetPipelineJobStatus | null | undefined
): MujocoVisualizationInfo | null {
  const raw = job?.mujocoVisualization;
  if (!raw || typeof raw !== 'object') return null;
  return raw;
}

export function resolveMujocoPreviewRelPath(
  job: AssetPipelineJobStatus | null | undefined
): string | null {
  const fromVis = job?.mujocoVisualization?.previewImage;
  if (typeof fromVis === 'string' && fromVis.trim() && jobHasFile(job, fromVis)) {
    return fromVis.replace(/^\/+/, '');
  }
  if (jobHasFile(job, 'exports/mujoco/preview.png')) {
    return 'exports/mujoco/preview.png';
  }
  return null;
}

export function resolveMujocoExportDownloads(
  job: AssetPipelineJobStatus | null | undefined
): MujocoExportDownload[] {
  const exportInfo = resolveMujocoExport(job);
  const downloads: MujocoExportDownload[] = [];

  for (const candidate of MUJOCO_EXPORT_DOWNLOADS) {
    const fromExport = exportInfo?.[candidate.exportKey];
    if (typeof fromExport === 'string' && fromExport.trim() && jobHasFile(job, fromExport)) {
      downloads.push({
        key: candidate.key,
        relPath: fromExport.replace(/^\/+/, ''),
        label: candidate.label,
        filename: candidate.filename,
      });
      continue;
    }
    if (jobHasFile(job, candidate.fallbackPath)) {
      downloads.push({
        key: candidate.key,
        relPath: candidate.fallbackPath,
        label: candidate.label,
        filename: candidate.filename,
      });
    }
  }

  return downloads;
}

export type AssetExportGroup = 'sam3d' | 'mujoco' | 'logs';

export interface AssetExportFileItem {
  key: string;
  group: AssetExportGroup;
  relPath: string;
  label: string;
  filename: string;
  tag?: string;
}

const LOG_EXPORT_CANDIDATES: Array<{
  key: string;
  relPaths: string[];
  label: string;
  filename: string;
  tag?: string;
}> = [
  { key: 'reconstruct_log', relPaths: ['logs/reconstruct.log'], label: 'reconstruct.log', filename: 'reconstruct.log', tag: 'log' },
  { key: 'segment_log', relPaths: ['logs/segment.log'], label: 'segment.log', filename: 'segment.log', tag: 'log' },
  { key: 'manifest', relPaths: ['sam3/manifest.json'], label: 'manifest.json', filename: 'manifest.json', tag: 'json' },
];

export function resolveObjectGlbRelPath(job: AssetPipelineJobStatus | null | undefined): string | null {
  const fromRecon = job?.reconstruction?.glbPath;
  if (typeof fromRecon === 'string' && fromRecon.trim() && jobHasFile(job, fromRecon)) {
    return fromRecon.replace(/^\/+/, '');
  }
  if (jobHasFile(job, 'sam3d/object.glb')) return 'sam3d/object.glb';
  return null;
}

export function resolveMujocoVisualObjRelPath(job: AssetPipelineJobStatus | null | undefined): string | null {
  const fromExport = job?.mujocoExport?.visualMesh;
  if (typeof fromExport === 'string' && fromExport.trim() && jobHasFile(job, fromExport)) {
    return fromExport.replace(/^\/+/, '');
  }
  if (jobHasFile(job, 'exports/mujoco/meshes/visual.obj')) return 'exports/mujoco/meshes/visual.obj';
  return null;
}

export function resolveMujocoCollisionObjRelPath(job: AssetPipelineJobStatus | null | undefined): string | null {
  const fromExport = job?.mujocoExport?.collisionMesh;
  if (typeof fromExport === 'string' && fromExport.trim() && jobHasFile(job, fromExport)) {
    return fromExport.replace(/^\/+/, '');
  }
  if (jobHasFile(job, 'exports/mujoco/meshes/collision.obj')) return 'exports/mujoco/meshes/collision.obj';
  return null;
}

export function resolveAssetExportFiles(job: AssetPipelineJobStatus | null | undefined): AssetExportFileItem[] {
  const items: AssetExportFileItem[] = [];
  const seen = new Set<string>();

  const push = (item: AssetExportFileItem) => {
    if (seen.has(item.relPath)) return;
    if (!jobHasFile(job, item.relPath)) return;
    seen.add(item.relPath);
    items.push(item);
  };

  for (const download of resolveSam3dExportDownloads(job)) {
    push({
      key: download.key,
      group: 'sam3d',
      relPath: download.relPath,
      label: download.label,
      filename: download.filename,
      tag: 'SAM3D',
    });
  }

  for (const download of resolveMujocoExportDownloads(job)) {
    push({
      key: download.key,
      group: 'mujoco',
      relPath: download.relPath,
      label: download.label,
      filename: download.filename,
      tag: 'MuJoCo',
    });
  }

  for (const candidate of LOG_EXPORT_CANDIDATES) {
    const relPath = candidate.relPaths.find((path) => jobHasFile(job, path));
    if (relPath) {
      push({
        key: candidate.key,
        group: 'logs',
        relPath,
        label: candidate.label,
        filename: candidate.filename,
        tag: candidate.tag,
      });
    }
  }

  return items;
}

export function formatMujocoValidationStatus(result: MujocoValidationResult | null | undefined): string {
  if (!result) return '未校验';
  if (result.skipped) return '未校验';
  if (result.ok) {
    const parts = [
      result.nbody != null ? `nbody=${result.nbody}` : null,
      result.ngeom != null ? `ngeom=${result.ngeom}` : null,
      result.nmesh != null ? `nmesh=${result.nmesh}` : null,
    ].filter(Boolean);
    return parts.length ? `通过（${parts.join(', ')}）` : '通过';
  }
  return result.error ? `失败：${result.error}` : '失败';
}

export function formatMujocoVisualizationStatus(status: string | null | undefined): string {
  switch (status) {
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    case 'pending':
      return '待生成';
    default:
      return status ? status : '未知';
  }
}

export function formatMujocoExportStatus(status: string | null | undefined): string {
  switch (status) {
    case 'completed':
      return '已完成';
    case 'partial':
      return '部分完成（校验未通过）';
    case 'failed':
      return '失败';
    case 'pending':
      return '待生成';
    case 'skipped':
      return '已跳过';
    default:
      return status ? status : '未知';
  }
}

export function resolveReconstructLogRelPath(job: AssetPipelineJobStatus | null | undefined): string | null {
  if (jobHasFile(job, 'logs/reconstruct.log')) return 'logs/reconstruct.log';
  return null;
}

export function resolveInputImageRelPath(job: AssetPipelineJobStatus | null | undefined): string {
  const files = job?.files || [];
  const fromFiles = files.find((file) => file.path === 'input/image.png' && file.exists !== false);
  if (fromFiles?.path) return fromFiles.path;
  if (job?.inputImage) return job.inputImage.replace(/^\/+/, '');
  return 'input/image.png';
}

export async function fetchAssetPipelineFileBlob(jobId: string, relPath: string): Promise<Blob> {
  const token = getAccessToken();
  const sessionId = initSessionId();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (sessionId) headers['X-Session-Id'] = sessionId;

  const url = getAssetPipelineFileUrl(jobId, relPath);
  const response = await fetch(url, {
    method: 'GET',
    headers,
    credentials: 'omit',
  });

  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: unknown; error?: unknown };
      if (typeof payload.detail === 'string') message = payload.detail;
      else if (typeof payload.error === 'string') message = payload.error;
    } catch {
      // ignore non-json error body
    }
    throw new AssetPipelineFileError(`加载文件失败 (${response.status}): ${message}`, response.status);
  }

  const blob = await response.blob();
  if (blob.size === 0) {
    throw new AssetPipelineFileError('文件为空', response.status);
  }
  return blob;
}

export async function fetchAssetPipelineFileText(jobId: string, relPath: string): Promise<string> {
  const blob = await fetchAssetPipelineFileBlob(jobId, relPath);
  return blob.text();
}

export async function downloadAssetPipelineFile(
  jobId: string,
  relPath: string,
  filename?: string
): Promise<void> {
  const blob = await fetchAssetPipelineFileBlob(jobId, relPath);
  const objectUrl = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = objectUrl;
    anchor.download = filename ?? relPath.split('/').pop() ?? 'download';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }
}

export function jobHasFile(job: AssetPipelineJobStatus | null | undefined, relPath: string): boolean {
  return Boolean(job?.files?.some((file) => file.path === relPath && file.exists !== false));
}

export const PIPELINE_TERMINAL_STATUSES = new Set([
  'segmented',
  'reconstructed',
  'failed',
]);

export const PIPELINE_STATUS_LABELS: Record<string, string> = {
  created: '已创建',
  segmenting: '分割中',
  segmented: '已分割',
  reconstructing: '重建中',
  reconstructed: '已重建',
  failed: '失败',
  unknown: '未知',
};
