/**
 * FastAPI 后端 API 客户端
 * 
 * 注意：所有请求使用相对路径 /api/*，由 Next.js rewrites 代理到后端
 * 这样避免跨域问题和 Mixed Content 问题
 */

import { useAuthStore } from '@/store/authStore';
import {
  ACCESS_TOKEN_KEY,
  getAccessToken,
  getSessionId,
  initSessionId,
  setTokens,
} from '@/lib/auth/session';
import {
  authDebugImportActiveHint,
  refreshToken as refreshTokenByAuthClient,
} from '@/lib/api/authClient';
import { clearAuthState } from '@/lib/auth/session';
import { resolveNetworkConnectionMessage } from '@/lib/errors/networkError';

// 仅在服务端需要直接请求后端时使用（客户端应使用相对路径）
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

function tokenMask(token: string | null | undefined): string | null {
  if (!token) return null;
  if (token.length < 14) return token;
  return `${token.slice(0, 8)}...${token.slice(-6)}`;
}

export interface ApiResponse<T> {
  ok: boolean;
  data?: T;
  error?: string;
  /** 后端成功响应时的提示（如删除采集数据时采集端路径已不存在） */
  warning?: string;
}

/**
 * 获取认证 Token（优先从 store 获取）
 */
function getAuthToken(): string | null {
  try {
    const { accessToken } = useAuthStore.getState();
    if (accessToken) return accessToken;
  } catch (e) {
    // 忽略错误（如在服务端渲染时）
  }

  if (typeof window === 'undefined') return null;
  return getAccessToken();
}

/**
 * 设置认证 Token
 */
export function setAuthToken(token: string): void {
  if (typeof window === 'undefined') return;
  const trimmed = (token || '').trim();
  if (!trimmed) {
    setTokens({ access_token: '', refresh_token: '' });
    useAuthStore.getState().setAccessToken(null);
    return;
  }
  // 仅更新 access_token，避免误删 sessionStorage 中的 refresh_token（401 续期后常见）
  window.sessionStorage.setItem(ACCESS_TOKEN_KEY, trimmed);
  useAuthStore.getState().setAccessToken(trimmed);
}

/**
 * 清除认证 Token
 */
export function clearAuthToken(): void {
  if (typeof window === 'undefined') return;
  setTokens({ access_token: '', refresh_token: '' });
  useAuthStore.getState().setAccessToken(null);
}

/** 用 refresh cookie 换新 access token，失败返回 null（仅浏览器端）。可供 labelApi 等模块在 401 时调用并重试。 */
export async function refreshAccessToken(): Promise<string | null> {
  if (typeof window === 'undefined') return null;
  try {
    console.info('[AUTH-TRACE][DATA_CLIENT] refresh delegated -> authClient');
    const data = await refreshTokenByAuthClient();
    console.info('[AUTH-TRACE][DATA_CLIENT] refresh result', {
      token: tokenMask(data?.access_token ?? null),
    });
    return data?.access_token ?? null;
  } catch {
    console.info('[AUTH-TRACE][DATA_CLIENT] refresh failed');
    return null;
  }
}

/**
 * 通用 API 请求函数
 * 收到 401 时：浏览器端先尝试 refresh cookie 换新 access token 并重试一次（避免 token 刚过期时误清态）；
 * 刷新失败或重试仍 401 再清理认证态（与平台 layout 的「无 user → 登录页」衔接）。
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  retriedAfterRefresh = false
): Promise<ApiResponse<T>> {
  const token = getAuthToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  // Step 4：注入 X-Session-Id（标签页隔离）
  const sid = (typeof window !== 'undefined' ? (getSessionId() || initSessionId()) : null) || null;
  if (sid) {
    headers['X-Session-Id'] = sid;
  }

  try {
    // 客户端使用相对路径，由 Next.js rewrites 代理到后端
    const isServer = typeof window === 'undefined';
    let url = isServer ? `${API_BASE_URL}${endpoint}` : endpoint;
    console.info('[AUTH-TRACE][DATA_CLIENT] request', {
      endpoint,
      method: options.method || 'GET',
      token: tokenMask(token),
      credentials: 'omit',
    });
    const response = await fetch(url, {
      ...options,
      headers,
      // 改造说明：不再依赖 cookie
      credentials: 'omit',
    });

    if (response.status === 401 && !endpoint.includes('/auth/refresh')) {
      if (
        typeof window !== 'undefined' &&
        !retriedAfterRefresh &&
        !endpoint.includes('/auth/logout')
      ) {
        const fresh = await refreshAccessToken();
        if (fresh) {
          setAuthToken(fresh);
          console.info('[AUTH-TRACE][DATA_CLIENT] 401 -> refresh ok, retry once', { endpoint });
          return apiRequest<T>(endpoint, options, true);
        }
      }
      console.info('[AUTH-TRACE][DATA_CLIENT] 401 detected', {
        endpoint,
        status: response.status,
        strategy: retriedAfterRefresh ? 'after-refresh-retry' : 'refresh-failed-or-no-browser',
        ts: new Date().toISOString(),
        pathname: typeof window !== 'undefined' ? window.location.pathname : '',
        importHint: authDebugImportActiveHint(),
      });
      if (typeof window !== 'undefined') {
        console.info('[AUTH-TRACE][DATA_CLIENT] 401 -> clearAuthState (no redirect here; layout uses !user)', {
          endpoint,
          ts: new Date().toISOString(),
          pathname: window.location.pathname,
          importHint: authDebugImportActiveHint(),
        });
        clearAuthState({ preserveRememberedUsername: true });
        useAuthStore.setState({ isHydrated: true });
      }
    }

    // 尝试解析 JSON
    let data: any;
    const rawText = await response.text();
    try {
      data = rawText ? JSON.parse(rawText) : null;
    } catch (parseError) {
      // JSON 解析失败，返回错误（附带响应片段便于调试）
      const snippet = rawText ? rawText.substring(0, 300).replace(/\n/g, ' ') : '(空响应)';
      // Next.js rewrite 默认 30s 代理超时，长耗时接口会先返回纯文本 500，后端仍可能继续执行成功
      const proxyTimeoutHint =
        response.status === 500 && /internal server error/i.test(snippet) && (rawText?.length ?? 0) < 120
          ? ' 提示：长请求可能被 Next 代理提前断开，请检查 next.config.js 的 experimental.proxyTimeout。'
          : '';
      return {
        ok: false,
        error: `响应格式错误: ${response.status} ${response.statusText}。响应内容: ${snippet}${proxyTimeoutHint}`,
      };
    }

    if (response.status === 403 && typeof window !== 'undefined') {
      const detail = typeof data?.detail === 'string' ? data.detail : '';
      if (detail === 'User disabled' || detail === 'Team disabled') {
        clearAuthState({ preserveRememberedUsername: true });
        useAuthStore.setState({ isHydrated: true });
        window.location.href = '/login';
        return {
          ok: false,
          error: detail === 'Team disabled' ? '所属团队已停用' : '账号已禁用',
        };
      }
    }

    if (!response.ok) {
      const errorMessage = data.error 
        ? (data.detail ? `${data.error}: ${typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)}` : data.error)
        : (data.detail || `HTTP ${response.status}: ${response.statusText}`);

      return {
        ok: false,
        error: resolveNetworkConnectionMessage(
          typeof errorMessage === 'string' ? errorMessage : String(errorMessage)
        ),
      };
    }

    return data as ApiResponse<T>;
  } catch (error) {
    const raw = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: resolveNetworkConnectionMessage(raw),
    };
  }
}

/**
 * GET 请求
 */
export async function apiGet<T>(endpoint: string): Promise<ApiResponse<T>> {
  return apiRequest<T>(endpoint, { 
    method: 'GET',
    cache: 'no-store' // 禁用缓存，确保获取最新数据
  });
}

/**
 * POST 请求
 */
export async function apiPost<T>(
  endpoint: string,
  body?: any
): Promise<ApiResponse<T>> {
  return apiRequest<T>(endpoint, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * PATCH 请求
 */
export async function apiPatch<T>(
  endpoint: string,
  body?: any
): Promise<ApiResponse<T>> {
  return apiRequest<T>(endpoint, {
    method: 'PATCH',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * PUT 请求
 */
export async function apiPut<T>(
  endpoint: string,
  body?: any
): Promise<ApiResponse<T>> {
  return apiRequest<T>(endpoint, {
    method: 'PUT',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * DELETE 请求
 */
export async function apiDelete<T>(endpoint: string): Promise<ApiResponse<T>> {
  return apiRequest<T>(endpoint, { method: 'DELETE' });
}

/**
 * POST 请求并返回 Blob（用于导出 zip 等文件下载）
 * 成功时返回 { ok: true, blob }；失败时返回 { ok: false, error }。
 */
export async function apiPostBlob(
  endpoint: string,
  body: unknown
): Promise<{ ok: true; blob: Blob; filename?: string } | { ok: false; error: string }> {
  const token = getAuthToken();
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const sid = (typeof window !== 'undefined' ? (getSessionId() || initSessionId()) : null) || null;
  if (sid) headers['X-Session-Id'] = sid;
  try {
    const isServer = typeof window === 'undefined';
    const url = isServer ? `${API_BASE_URL}${endpoint}` : endpoint;
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      credentials: 'omit',
    });
    const disposition = response.headers.get('Content-Disposition');
    let filename: string | undefined;
    if (disposition) {
      const match = /filename="?([^";\n]+)"?/.exec(disposition);
      if (match) filename = match[1].trim();
    }
    if (!response.ok) {
      const text = await response.text();
      let err = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const data = text ? JSON.parse(text) : null;
        if (data?.detail) err = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        else if (data?.error) err = data.error;
      } catch {
        if (text) err = text.slice(0, 200);
      }
      return { ok: false, error: err };
    }
    const blob = await response.blob();
    return { ok: true, blob, filename };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * GET 请求并返回 Blob（用于下载导出文件）
 */
export async function apiGetBlob(
  endpoint: string
): Promise<{ ok: true; blob: Blob; filename?: string } | { ok: false; error: string }> {
  const token = getAuthToken();
  const headers: HeadersInit = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const sid = (typeof window !== 'undefined' ? (getSessionId() || initSessionId()) : null) || null;
  if (sid) headers['X-Session-Id'] = sid;
  try {
    const isServer = typeof window === 'undefined';
    let url = isServer ? `${API_BASE_URL}${endpoint}` : endpoint;
    const response = await fetch(url, { method: 'GET', headers, credentials: 'omit' });
    const disposition = response.headers.get('Content-Disposition');
    let filename: string | undefined;
    if (disposition) {
      const match = /filename="?([^";\n]+)"?/.exec(disposition);
      if (match) filename = match[1].trim();
    }
    if (!response.ok) {
      const text = await response.text();
      let err = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const data = text ? JSON.parse(text) : null;
        if (data?.detail) err = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        else if (data?.error) err = data.error;
      } catch {
        if (text) err = text.slice(0, 200);
      }
      return { ok: false, error: err };
    }
    const blob = await response.blob();
    return { ok: true, blob, filename };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * 上传文件（multipart/form-data）
 */
export async function apiUpload<T>(
  endpoint: string,
  formData: FormData
): Promise<ApiResponse<T>> {
  const token = getAuthToken();
  const headers: HeadersInit = {};
  
  // 注意：不要设置 Content-Type，让浏览器自动设置（包含 boundary）
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const sid = (typeof window !== 'undefined' ? (getSessionId() || initSessionId()) : null) || null;
  if (sid) headers['X-Session-Id'] = sid;

  try {
    const isServer = typeof window === 'undefined';
    const url = isServer ? `${API_BASE_URL}${endpoint}` : endpoint;
    
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
      credentials: 'omit',
    });

    // 尝试解析 JSON
    let data: any;
    try {
      const text = await response.text();
      if (!text) {
        return {
          ok: false,
          error: `响应格式错误: 服务器返回空响应 (${response.status} ${response.statusText})`,
        };
      }
      try {
        data = JSON.parse(text);
      } catch (parseError) {
        // 如果不是 JSON，返回原始文本（可能是 HTML 错误页面）
        return {
          ok: false,
          error: `响应格式错误: 服务器返回的不是 JSON 格式 (${response.status} ${response.statusText})。响应内容: ${text.substring(0, 200)}`,
        };
      }
    } catch (error) {
      return {
        ok: false,
        error: `读取响应失败: ${error instanceof Error ? error.message : String(error)}`,
      };
    }

    if (!response.ok) {
      // 如果响应不是 JSON 格式，data 可能是字符串
      if (typeof data === 'string') {
        return {
          ok: false,
          error: `HTTP ${response.status} ${response.statusText}: ${data.substring(0, 500)}`,
        };
      }
      return {
        ok: false,
        error: data?.error || data?.detail || `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    // 验证响应格式
    if (!data || typeof data !== 'object') {
      return {
        ok: false,
        error: `响应格式错误: 返回的数据不是对象格式`,
      };
    }

    return data as ApiResponse<T>;
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * multipart 上传并报告进度（0–100，仅在上传体有 Content-Length 时可靠）
 * 401 时与 apiRequest 一致：先 refresh 换新 token 后整体重试一次，避免长传中途 token 过期被误清登录态。
 */
export function apiUploadWithProgress<T>(
  endpoint: string,
  formData: FormData,
  onProgress?: (percent: number) => void,
  signal?: AbortSignal,
  retriedAfterRefresh = false
): Promise<ApiResponse<T>> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve({ ok: false, error: '已取消' });
      return;
    }
    const token = getAuthToken();
    const xhr = new XMLHttpRequest();
    if (signal) {
      signal.addEventListener(
        'abort',
        () => {
          try {
            xhr.abort();
          } catch {
            /* ignore */
          }
          resolve({ ok: false, error: '已取消' });
        },
        { once: true }
      );
    }
    xhr.open('POST', typeof window === 'undefined' ? `${API_BASE_URL}${endpoint}` : endpoint);
    // 改造说明：不再依赖 cookie
    xhr.withCredentials = false;
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    }
    const sid = (typeof window !== 'undefined' ? (getSessionId() || initSessionId()) : null) || null;
    if (sid) {
      xhr.setRequestHeader('X-Session-Id', sid);
    }
    xhr.upload.onprogress = (ev) => {
      if (!onProgress) return;
      if (ev.lengthComputable && ev.total > 0) {
        onProgress(Math.min(100, Math.round((ev.loaded / ev.total) * 100)));
      } else {
        onProgress(0);
      }
    };
    xhr.onerror = () => {
      resolve({ ok: false, error: '网络错误，上传失败' });
    };
    xhr.onload = () => {
      void (async () => {
        let data: unknown;
        try {
          const text = xhr.responseText || '';
          data = text ? JSON.parse(text) : null;
        } catch {
          resolve({
            ok: false,
            error: `响应格式错误: ${xhr.status}。${(xhr.responseText || '').slice(0, 200)}`,
          });
          return;
        }
        if (xhr.status === 401) {
          console.info('[AUTH-TRACE][DATA_CLIENT] xhr 401 detected', {
            endpoint,
            status: xhr.status,
            strategy: retriedAfterRefresh ? 'after-refresh-retry' : 'refresh-then-retry-once',
            ts: new Date().toISOString(),
            pathname: typeof window !== 'undefined' ? window.location.pathname : '',
            importHint: authDebugImportActiveHint(),
          });
          if (typeof window !== 'undefined' && !retriedAfterRefresh && !endpoint.includes('/auth/logout')) {
            const fresh = await refreshAccessToken();
            if (fresh) {
              setAuthToken(fresh);
              if (signal?.aborted) {
                resolve({ ok: false, error: '已取消' });
                return;
              }
              const retry = await apiUploadWithProgress<T>(endpoint, formData, onProgress, signal, true);
              resolve(retry);
              return;
            }
          }
          if (typeof window !== 'undefined') {
            clearAuthState({ preserveRememberedUsername: true });
            useAuthStore.setState({ isHydrated: true });
          }
          resolve({ ok: false, error: '未授权或登录已过期，请重新登录' });
          return;
        }
        if (xhr.status < 200 || xhr.status >= 300) {
          const d = data as { error?: string; detail?: string };
          resolve({
            ok: false,
            error: d?.error || (typeof d?.detail === 'string' ? d.detail : `HTTP ${xhr.status}`),
          });
          return;
        }
        if (!data || typeof data !== 'object') {
          resolve({ ok: false, error: '响应格式错误' });
          return;
        }
        resolve(data as ApiResponse<T>);
      })();
    };
    xhr.send(formData);
  });
}

