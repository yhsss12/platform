'use client';

import { useAuthStore } from '@/store/authStore';
import type { LoginResponse, AccessTokenOnlyResponse, MeResponse } from './types';
import {
  clearAuthState,
  getAccessToken,
  getRefreshToken,
  getSessionId,
  initSessionId,
  setTokens,
} from '@/lib/auth/session';

// 直接使用后端 URL（CORS 已配置允许所有来源）
// 优先使用环境变量，如果没有则使用默认后端地址
// 浏览器环境统一走 Next.js 的 /api 代理，避免本地/远程端口不一致
const getApiBaseUrl = () => {
  // 浏览器：使用相对路径，由 Next.js rewrites 代理到后端
  if (typeof window !== 'undefined') {
    return '';
  }
  // 服务端环境：可以使用显式后端地址
  return process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
};

const API_BASE_URL = getApiBaseUrl();
const API_BASE = `${API_BASE_URL}/api`;

/** 与 TaskCenterContext 中 BROWSER_IMPORT_STORAGE_KEY 对齐，仅用于 AUTH-TRACE 诊断 */
const BROWSER_IMPORT_STORAGE_KEY = 'eai_task_center_browser_imports_v1';

export function authDebugImportActiveHint(): string {
  if (typeof window === 'undefined') return 'n/a';
  try {
    const raw = window.localStorage.getItem(BROWSER_IMPORT_STORAGE_KEY);
    if (!raw) return 'none';
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return 'parse_err';
    let n = 0;
    for (const t of arr) {
      const st = (t as { status?: string })?.status;
      if (st === 'running' || st === 'queued' || st === 'paused') n += 1;
    }
    return n ? `import_active_like=${n}` : 'import_stored_idle';
  } catch {
    return 'read_err';
  }
}

let refreshMutex: Promise<AccessTokenOnlyResponse | null> | null = null;
let proactiveRefreshTimer: number | null = null;

function decodeJwtExpMs(token: string): number | null {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return null;
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4));
    const json = JSON.parse(atob(b64 + pad));
    return typeof json.exp === 'number' ? json.exp * 1000 : null;
  } catch {
    return null;
  }
}

/** 在 access token 到期前静默续期（默认提前 4 分钟），避免长任务期间仅靠 401 被动刷新 */
export function scheduleAccessTokenProactiveRefresh(accessToken: string | null | undefined): void {
  if (typeof window === 'undefined') return;
  if (proactiveRefreshTimer) {
    clearTimeout(proactiveRefreshTimer);
    proactiveRefreshTimer = null;
  }
  if (!accessToken) return;
  const expMs = decodeJwtExpMs(accessToken);
  if (!expMs) return;
  const earlyMs = 4 * 60 * 1000;
  const delay = Math.max(15_000, expMs - Date.now() - earlyMs);
  proactiveRefreshTimer = window.setTimeout(() => {
    proactiveRefreshTimer = null;
    void (async () => {
      try {
        console.info('[AUTH-TRACE][AUTH_CLIENT] proactive refresh tick', {
          ts: new Date().toISOString(),
          pathname: window.location.pathname,
          importHint: authDebugImportActiveHint(),
        });
        await refreshToken();
      } catch (e) {
        console.info('[AUTH-TRACE][AUTH_CLIENT] proactive refresh failed', {
          err: e instanceof Error ? e.message : String(e),
        });
      } finally {
        const next = useAuthStore.getState().accessToken || getAccessToken();
        scheduleAccessTokenProactiveRefresh(next || undefined);
      }
    })();
  }, delay);
}

async function performRefreshTokenOnce(): Promise<AccessTokenOnlyResponse | null> {
  try {
    const sessionId = getSessionId() || initSessionId();
    const refreshTokenValue = getRefreshToken();
    if (!refreshTokenValue) {
      console.info('[AUTH-TRACE][AUTH_CLIENT] refresh skipped (no refresh_token in sessionStorage)', {
        ts: new Date().toISOString(),
        pathname: typeof window !== 'undefined' ? window.location.pathname : '',
        importHint: authDebugImportActiveHint(),
      });
      return null;
    }
    console.info('[AUTH-TRACE][AUTH_CLIENT] refresh call', {
      ts: new Date().toISOString(),
      from: 'authClient.performRefreshTokenOnce',
      pathname: typeof window !== 'undefined' ? window.location.pathname : '',
      importHint: authDebugImportActiveHint(),
      stack: new Error().stack?.split('\n').slice(1, 5).join(' | '),
    });
    const refreshUrl = API_BASE.startsWith('http') ? `${API_BASE}/auth/refresh` : `${API_BASE}/auth/refresh`;
    const response = await fetch(refreshUrl, {
      method: 'POST',
      credentials: 'omit',
      headers: {
        'Content-Type': 'application/json',
        ...(sessionId ? { 'X-Session-Id': sessionId } : {}),
      },
      body: JSON.stringify({ refresh_token: refreshTokenValue }),
    });

    if (!response.ok) {
      console.info('[AUTH-TRACE][AUTH_CLIENT] refresh failed', {
        ts: new Date().toISOString(),
        status: response.status,
        pathname: typeof window !== 'undefined' ? window.location.pathname : '',
        importHint: authDebugImportActiveHint(),
      });
      return null;
    }

    const data: (AccessTokenOnlyResponse & { refresh_token?: string }) = await response.json();
    if (data.access_token) {
      console.info('[AUTH-TRACE][AUTH_CLIENT] refresh success', {
        ts: new Date().toISOString(),
        token: tokenMask(data.access_token),
        pathname: typeof window !== 'undefined' ? window.location.pathname : '',
        importHint: authDebugImportActiveHint(),
      });
      useAuthStore.getState().setAccessToken(data.access_token);
      setTokens({ access_token: data.access_token, refresh_token: data.refresh_token ?? refreshTokenValue });
      scheduleAccessTokenProactiveRefresh(data.access_token);
      return data;
    }
    return null;
  } catch (error) {
    console.error('Refresh token failed:', error);
    return null;
  }
}

function getRefreshPromise(): Promise<AccessTokenOnlyResponse | null> {
  if (!refreshMutex) {
    refreshMutex = performRefreshTokenOnce().finally(() => {
      refreshMutex = null;
    });
  }
  return refreshMutex;
}

function tokenMask(token: string | null | undefined): string | null {
  if (!token) return null;
  if (token.length < 14) return token;
  return `${token.slice(0, 8)}...${token.slice(-6)}`;
}

// 在开发环境下，在模块加载时输出配置信息
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  console.log('[API Config] API_BASE_URL:', API_BASE_URL);
  console.log('[API Config] API_BASE:', API_BASE);
}

// --- Session-isolated request wrapper ---
async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const { accessToken } = useAuthStore.getState();
  const token = accessToken || getAccessToken();
  const sessionId = initSessionId();
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  // Step 4：自动注入 X-Session-Id
  if (sessionId) {
    headers['X-Session-Id'] = sessionId;
  }

  // 构建完整的 URL
  let url: string;
  if (path.startsWith('http')) {
    url = path;
  } else {
    // 确保 path 以 / 开头，API_BASE 以 / 结尾或没有 /
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    const normalizedBase = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
    url = `${normalizedBase}${normalizedPath}`;
  }
  
  console.info('[AUTH-TRACE][AUTH_CLIENT] request', {
    path,
    method: options.method || 'GET',
    token: tokenMask(token),
    sessionId: sessionId ? `${sessionId.slice(0, 8)}...${sessionId.slice(-6)}` : null,
    credentials: 'omit',
  });
  return fetch(url, {
    ...options,
    headers,
    // Step 6：不再依赖 cookie
    credentials: 'omit',
  });
}

/** 401 续期：与 data-platform 共用串行 refresh，避免并发 refresh 轮换 refresh_token 时互相踩掉 */
async function refreshAccessToken(): Promise<string | null> {
  const data = await getRefreshPromise();
  return data?.access_token ?? null;
}

/**
 * 认证请求（止血策略：禁用 401 自动 refresh，401 直接清理并回登录）。
 */
export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await apiFetch(path, options);

  // Step 5：401 时使用 sessionStorage.refresh_token 主动 refresh（不依赖 cookie）
  if (response.status === 401 && path !== '/auth/refresh') {
    console.info('[AUTH-TRACE][AUTH_CLIENT] 401 detected', {
      path,
      status: response.status,
      strategy: 'refresh-with-sessionStorage',
      ts: new Date().toISOString(),
      pathname: typeof window !== 'undefined' ? window.location.pathname : '',
      importHint: authDebugImportActiveHint(),
    });
    const newToken = await refreshAccessToken();
    if (newToken) {
      console.info('[AUTH-TRACE][AUTH_CLIENT] 401 -> refresh ok, retry once', { path });
      return apiRequest<T>(path, options);
    }
    console.info('[AUTH-TRACE][AUTH_CLIENT] 401 -> refresh failed -> logout redirect', {
      path,
      ts: new Date().toISOString(),
      pathname: typeof window !== 'undefined' ? window.location.pathname : '',
      importHint: authDebugImportActiveHint(),
    });
    clearAuthState({ preserveRememberedUsername: true });
    useAuthStore.setState({ isHydrated: true });
    if (typeof window !== 'undefined') window.location.href = '/login';
    throw new Error('Authentication failed');
  }

  if (response.status === 403) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = typeof (errorData as { detail?: unknown })?.detail === 'string'
      ? (errorData as { detail: string }).detail
      : '';
    if (detail === 'User disabled' || detail === 'Team disabled') {
      clearAuthState({ preserveRememberedUsername: true });
      useAuthStore.setState({ isHydrated: true });
      if (typeof window !== 'undefined') window.location.href = '/login';
      throw new Error(detail === 'Team disabled' ? '所属团队已停用' : '账号已禁用');
    }
    throw new Error(detail || (errorData as { error?: string }).error || 'Forbidden');
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
  }

  // 204 No Content 响应没有 body，直接返回
  if (response.status === 204) {
    return null as T;
  }

  // 检查 Content-Length 或是否有内容
  const contentType = response.headers.get('content-type');
  const contentLength = response.headers.get('content-length');
  
  // 如果没有内容类型或内容长度为 0，返回 null
  if (!contentType || contentLength === '0') {
    return null as T;
  }

  // 尝试解析 JSON，如果失败则返回空对象
  try {
    const text = await response.text();
    if (!text || text.trim() === '') {
      return null as T;
    }
    return JSON.parse(text) as T;
  } catch (error) {
    // 如果解析失败，返回 null（对于 204 或其他无内容响应）
    return null as T;
  }
}

/**
 * GET 请求
 */
export async function apiGet<T>(path: string, options?: RequestInit): Promise<T> {
  return apiRequest<T>(path, { ...options, method: 'GET' });
}

/**
 * POST 请求
 */
export async function apiPost<T>(path: string, body?: any, options?: RequestInit): Promise<T> {
  return apiRequest<T>(path, {
    ...options,
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * PATCH 请求
 */
export async function apiPatch<T>(path: string, body?: any, options?: RequestInit): Promise<T> {
  return apiRequest<T>(path, {
    ...options,
    method: 'PATCH',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * PUT 请求
 */
export async function apiPut<T>(path: string, body?: any, options?: RequestInit): Promise<T> {
  return apiRequest<T>(path, {
    ...options,
    method: 'PUT',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * DELETE 请求
 */
export async function apiDelete<T>(path: string, options?: RequestInit): Promise<T> {
  return apiRequest<T>(path, { ...options, method: 'DELETE' });
}

/**
 * 登录
 */
export async function login(username: string, password: string): Promise<LoginResponse> {
  try {
    // 构建登录URL
    const loginUrl = API_BASE.endsWith('/') 
      ? `${API_BASE}auth/login` 
      : `${API_BASE}/auth/login`;
    
    // 添加调试日志（仅在开发环境）
    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      console.log('[Login] 请求URL:', loginUrl);
      console.log('[Login] API_BASE:', API_BASE);
      console.log('[Login] API_BASE_URL:', API_BASE_URL);
    }
    
    // 请求体字段名仍为 username，与后端 LoginRequest 对齐；语义为登录账号 account_id（非展示名）
    const response = await fetch(loginUrl, {
      method: 'POST',
      credentials: 'omit',
      headers: {
        'Content-Type': 'application/json',
        ...(initSessionId() ? { 'X-Session-Id': initSessionId() } : {}),
      },
      body: JSON.stringify({ username, password }),
    });

    let raw: any;
    try {
      raw = await response.json();
    } catch {
      raw = {};
    }

    if (raw && typeof raw === 'object' && raw.ok === false) {
      const msg = typeof raw.error === 'string' ? raw.error : '登录失败';
      if (msg === '账号已禁用' || msg.includes('已禁用')) {
        throw new Error('账号已禁用');
      }
      if (msg.includes('账号或密码') || msg.includes('密码错误')) {
        throw new Error('账号或密码错误，请重试');
      }
      throw new Error(msg);
    }

    if (!response.ok) {
      const errorData = raw;
      if (response.status === 401) {
        throw new Error('账号或密码错误，请重试');
      }
      if (response.status === 403) {
        throw new Error('账号已禁用');
      }
      if (response.status === 404) {
        throw new Error('登录接口不存在，请检查后端服务是否正常运行');
      }
      if (response.status >= 500) {
        throw new Error('服务器错误，请稍后重试');
      }
      const msg = errorData.detail || errorData.error || `登录失败 (${response.status})`;
      const credentialErrors = [
        'Incorrect username or password',
        'Invalid username or password',
        'Invalid credentials',
        'incorrect username',
        'wrong password',
        '账号或密码错误，请重试',
      ];
      const isCredentialError = typeof msg === 'string' && credentialErrors.some((s) => msg.toLowerCase().includes(s.toLowerCase()));
      throw new Error(isCredentialError ? '账号或密码错误，请重试' : String(msg));
    }

    // 后端返回 { ok, data? } 或直接 { access_token, role, username }
    const tokenData: LoginResponse = raw?.data ?? raw;
    if (!tokenData?.access_token) {
      const errMsg = raw?.error || '登录返回数据异常';
      throw new Error(errMsg);
    }
    useAuthStore.getState().setAccessToken(tokenData.access_token);
    setTokens({ access_token: tokenData.access_token, refresh_token: tokenData.refresh_token ?? null });
    console.info('[AUTH-TRACE][AUTH_CLIENT] login token stored', {
      username,
      token: tokenMask(tokenData.access_token),
    });
    scheduleAccessTokenProactiveRefresh(tokenData.access_token);
    return tokenData;
  } catch (error) {
    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      const errName = error instanceof Error ? error.constructor.name : typeof error;
      const errMsg = error instanceof Error ? error.message : String(error);
      console.warn('[Login] 请求失败:', errName, errMsg, error);
    }
    
    // 捕获网络错误（如 Failed to fetch）
    if (error instanceof TypeError && error.message.includes('fetch')) {
      const actualUrl = API_BASE_URL || 'http://localhost:8000';
      const errorMsg = `无法连接到服务器 (${actualUrl})，请检查：
1. 后端服务是否运行在 ${actualUrl}
2. 浏览器控制台是否有CORS错误
3. 防火墙是否阻止了连接
4. 请尝试访问 ${actualUrl}/docs 确认后端服务是否正常运行`;
      throw new Error(errorMsg);
    }
    // 重新抛出其他错误
    throw error;
  }
}

/**
 * 获取当前用户信息
 * 后端返回 { ok, data } 时自动解包为 data
 */
export async function getMe(): Promise<MeResponse> {
  console.info('[AUTH-TRACE][AUTH_CLIENT] getMe call');
  const raw = await apiGet<{ ok?: boolean; data?: MeResponse } & MeResponse>('/auth/me');
  const user = raw && typeof raw === 'object' && 'data' in raw && (raw as { data?: MeResponse }).data != null
    ? (raw as { data: MeResponse }).data
    : (raw as MeResponse);
  if (!user?.id || (!user?.account_id && !user?.username)) {
    throw new Error('获取用户信息失败');
  }
  console.info('[AUTH-TRACE][AUTH_CLIENT] getMe success', {
    id: user.id,
    account_id: user.account_id,
    username: user.username,
  });
  return user;
}

function unwrapAuthData<T>(raw: unknown): T {
  if (raw && typeof raw === 'object' && 'ok' in raw && (raw as { ok?: boolean }).ok === false) {
    throw new Error(String((raw as { error?: string }).error || '请求失败'));
  }
  if (raw && typeof raw === 'object' && 'data' in raw && (raw as { data?: T }).data !== undefined) {
    return (raw as { data: T }).data;
  }
  return raw as T;
}

/** 修改当前用户展示名（username）；不可改 account_id */
export async function patchProfile(username: string): Promise<MeResponse> {
  const raw = await apiPatch<{ ok?: boolean; data?: MeResponse; error?: string }>('/auth/profile', {
    username,
  });
  return unwrapAuthData<MeResponse>(raw);
}

/** 用户自助修改密码（与管理员重置他人密码接口分离） */
export async function changeOwnPassword(current_password: string, new_password: string): Promise<void> {
  const raw = await apiPost<{ ok?: boolean; error?: string }>('/auth/change-password', {
    current_password,
    new_password,
  });
  if (raw && typeof raw === 'object' && 'ok' in raw && raw.ok === false) {
    throw new Error(String(raw.error || '修改失败'));
  }
}

/**
 * 刷新 access token（供 data-platform client 等调用；与 401 续期共用串行 refresh）
 */
export async function refreshToken(): Promise<AccessTokenOnlyResponse> {
  const data = await getRefreshPromise();
  if (!data?.access_token) {
    throw new Error('Refresh failed');
  }
  return data;
}

/**
 * 通知服务端撤销当前会话（须带 Bearer 与 X-Session-Id）；失败不抛错，由调用方继续清理本地态。
 */
export async function logoutOnServer(): Promise<void> {
  const token = useAuthStore.getState().accessToken || getAccessToken();
  const sessionId = getSessionId() || initSessionId();
  const url = `${API_BASE}/auth/logout`;
  try {
    await fetch(url, {
      method: 'POST',
      credentials: 'omit',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(sessionId ? { 'X-Session-Id': sessionId } : {}),
      },
    });
  } catch (e) {
    console.warn('[AUTH] logoutOnServer failed', e);
  }
}
