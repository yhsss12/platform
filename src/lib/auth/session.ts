'use client';

import { useAuthStore } from '@/store/authStore';

// --- Session-isolated auth (per-tab) ---
// 改造说明：
// - sessionId/token 全部放 sessionStorage（标签页隔离）
// - 不再使用 localStorage 存 token（避免跨标签页串号）
// - remember username 仍可用 localStorage（仅表单回填，不参与身份恢复）
export const SESSION_ID_KEY = 'auth.sessionId';
export const ACCESS_TOKEN_KEY = 'auth.access_token';
export const REFRESH_TOKEN_KEY = 'auth.refresh_token';

// legacy key (do not use for auth anymore; kept for cleanup)
export const LEGACY_AUTH_TOKEN_KEY = 'auth_token';
export const SAVED_USERNAME_KEY = 'auth.savedUsername';
/** 已废弃：不再写入密码；clearAuthState 仍会清除历史残留 key */
export const SAVED_PASSWORD_KEY = 'auth.savedPassword';

function isBrowser(): boolean {
  return typeof window !== 'undefined';
}

function uuidv4(): string {
  // 优先使用浏览器原生 randomUUID
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // fallback：使用 getRandomValues 生成 RFC4122 v4（保证长度=36，兼容后端 varchar(36)）
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    // per RFC4122 section 4.4
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  // 最后兜底：时间戳 + 随机数压缩成固定格式（仍保持 36 长度样式）
  const fallback = `00000000-0000-4000-8000-${String(Date.now()).padStart(12, '0').slice(-12)}`;
  return fallback;
}

/** Step 1：确保每个标签页都有独立 sessionId（存 sessionStorage）。 */
export function initSessionId(): string {
  if (!isBrowser()) return '';
  const existing = window.sessionStorage.getItem(SESSION_ID_KEY);
  if (existing && existing.trim()) return existing;
  const sid = uuidv4();
  window.sessionStorage.setItem(SESSION_ID_KEY, sid);
  return sid;
}

export function getSessionId(): string | null {
  if (!isBrowser()) return null;
  return window.sessionStorage.getItem(SESSION_ID_KEY);
}

/** Step 2：token 存储（sessionStorage）。 */
export function getAccessToken(): string | null {
  if (!isBrowser()) return null;
  return window.sessionStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (!isBrowser()) return null;
  return window.sessionStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(tokens: { access_token: string; refresh_token?: string | null }): void {
  if (!isBrowser()) return;
  const at = (tokens.access_token || '').trim();
  if (at) window.sessionStorage.setItem(ACCESS_TOKEN_KEY, at);
  else window.sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  const rt = (tokens.refresh_token || '').trim();
  if (rt) window.sessionStorage.setItem(REFRESH_TOKEN_KEY, rt);
  else window.sessionStorage.removeItem(REFRESH_TOKEN_KEY);
}

export function clearAuthState(options?: { preserveRememberedUsername?: boolean }): void {
  console.info('[AUTH-TRACE][SESSION] clearAuthState', {
    preserveRememberedUsername: options?.preserveRememberedUsername === true,
    ts: new Date().toISOString(),
    pathname: typeof window !== 'undefined' ? window.location.pathname : '',
  });
  useAuthStore.setState({ accessToken: null, user: null, isHydrated: false });
  if (!isBrowser()) return;

  const preserveRememberedUsername = options?.preserveRememberedUsername === true;
  // 清理 sessionStorage auth（标签页级）
  window.sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  window.sessionStorage.removeItem(REFRESH_TOKEN_KEY);
  // sessionId 默认保留（使同一标签页刷新仍属于同一 session），如需重置可手动 remove

  // 清理 legacy token（同源全局），避免旧逻辑残留影响
  window.localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY);
  window.localStorage.removeItem(SAVED_PASSWORD_KEY);
  if (!preserveRememberedUsername) {
    window.localStorage.removeItem(SAVED_USERNAME_KEY);
  }

  // Remove auth/session residues from sessionStorage.
  for (const k of Object.keys(window.sessionStorage)) {
    if (/auth|token|refresh|session/i.test(k)) {
      // 保留 sessionId；其余清掉
      if (k === SESSION_ID_KEY) continue;
      window.sessionStorage.removeItem(k);
    }
  }
}

