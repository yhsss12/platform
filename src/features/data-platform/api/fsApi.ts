/**
 * 文件系统相关 API（平台内置资源浏览器）
 */
import { apiGet, type ApiResponse } from './client';

export interface FsListItem {
  name: string;
  type: 'dir' | 'file';
  size?: number;
  mtime?: string;
}

export interface FsListResponse {
  path: string;
  items: FsListItem[];
}

export interface FsInspectResponse {
  isLeRobot: boolean;
  reason: string;
  metaHint: { hasMeta: boolean; hasData: boolean; hasVideos: boolean };
}

export async function fsList(path: string = ''): Promise<ApiResponse<FsListResponse>> {
  const q = path ? `?path=${encodeURIComponent(path)}` : '';
  return apiGet<FsListResponse>(`/api/fs/list${q}`);
}

export async function agentFsList(params?: {
  path?: string;
  deviceId?: string;
  agentId?: string;
}): Promise<ApiResponse<FsListResponse>> {
  const sp = new URLSearchParams();
  const p = (params?.path || '').trim();
  if (p) sp.set('path', p);
  if (params?.deviceId) sp.set('device_id', String(params.deviceId));
  if (params?.agentId) sp.set('agent_id', String(params.agentId));
  const q = sp.toString();
  return apiGet<FsListResponse>(`/api/fs/agent-list${q ? `?${q}` : ''}`);
}

export async function fsInspect(path: string): Promise<ApiResponse<FsInspectResponse>> {
  return apiGet<FsInspectResponse>(`/api/fs/inspect?path=${encodeURIComponent(path)}`);
}

export interface ListDirsResponse {
  base: string;
  dirs: string[];
}

export interface ListHdf5Response {
  dir: string;
  files: string[];
}

/**
 * 列出指定目录下的所有子目录
 * @param base 基础路径，可选，默认为 ROOT_DIR
 */
export async function listDirs(base?: string): Promise<{ ok: boolean; data?: ListDirsResponse; error?: string }> {
  try {
    // 构建查询参数
    const queryParams = base ? `?base=${encodeURIComponent(base)}` : '';
    const endpoint = `/api/fs/list-dirs${queryParams}`;
    
    const response = await apiGet<ListDirsResponse>(endpoint);
    
    if (response.ok && response.data) {
      return {
        ok: true,
        data: response.data,
      };
    } else {
      return {
        ok: false,
        error: response.error || '未知错误',
      };
    }
  } catch (error: any) {
    return {
      ok: false,
      error: error.message || '请求失败',
    };
  }
}

/**
 * 通过采集端 Agent 列出指定目录下的所有子目录
 * @param deviceId 设备 ID，用于解析 Agent
 * @param base 基础路径，可选，由 Agent 自己解析默认根目录
 */
export async function listAgentDirs(
  deviceId?: string,
  base?: string
): Promise<{ ok: boolean; data?: ListDirsResponse; error?: string }> {
  try {
    const params = new URLSearchParams();
    if (deviceId) params.set('device_id', deviceId);
    if (base) params.set('base', base);
    const endpoint = `/api/fs/agent-list-dirs${params.toString() ? `?${params.toString()}` : ''}`;

    const response = await apiGet<ListDirsResponse>(endpoint);

    if (response.ok && response.data) {
      return {
        ok: true,
        data: response.data,
      };
    } else {
      return {
        ok: false,
        error: response.error || '未知错误',
      };
    }
  } catch (error: any) {
    return {
      ok: false,
      error: error.message || '请求失败',
    };
  }
}

/**
 * 列出指定目录下的所有 .hdf5 文件
 * @param dir 目录路径
 */
export async function listHdf5(dir: string): Promise<{ ok: boolean; data?: ListHdf5Response; error?: string }> {
  try {
    const endpoint = `/api/fs/list-hdf5?dir=${encodeURIComponent(dir)}`;
    const response = await apiGet<ListHdf5Response>(endpoint);
    
    if (response.ok && response.data) {
      return {
        ok: true,
        data: response.data,
      };
    } else {
      return {
        ok: false,
        error: response.error || '未知错误',
      };
    }
  } catch (error: any) {
    return {
      ok: false,
      error: error.message || '请求失败',
    };
  }
}
