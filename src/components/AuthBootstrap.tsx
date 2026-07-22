'use client';

import { useEffect } from 'react';
import { useAuthStore } from '@/store/authStore';
import { getMe, scheduleAccessTokenProactiveRefresh } from '@/lib/api/authClient';
import { clearAuthState, getAccessToken, initSessionId } from '@/lib/auth/session';

/**
 * AuthBootstrap：应用启动时恢复登录态
 * - 先使用本地 access_token 调 /me 恢复（避免旧 refresh cookie 抢占身份）
 * - 无本地 token 时不自动 refresh，防止被历史 cookie 会话覆盖当前账号
 * - 无论成功失败，最后设置 isHydrated = true
 */
export function AuthBootstrap() {
  const { setAccessToken, setUser, isHydrated } = useAuthStore();

  useEffect(() => {
    if (isHydrated) return; // 已初始化过，不再执行

    const bootstrap = async () => {
      try {
        const sid = initSessionId();
        const storedToken = getAccessToken();
        console.info('[AUTH-TRACE][BOOTSTRAP] start', {
          hasStoredToken: Boolean(storedToken),
          token: storedToken ? `${storedToken.slice(0, 8)}...${storedToken.slice(-6)}` : null,
          sessionId: sid ? `${sid.slice(0, 8)}...${sid.slice(-6)}` : null,
        });
        if (storedToken) {
          setAccessToken(storedToken);
          console.info('[AUTH-TRACE][BOOTSTRAP] before getMe', {
            token: `${storedToken.slice(0, 8)}...${storedToken.slice(-6)}`,
          });
          const user = await getMe();
          console.info('[AUTH-TRACE][BOOTSTRAP] getMe success', {
            id: user?.id,
            username: user?.username,
          });
          setUser(user);
          scheduleAccessTokenProactiveRefresh(storedToken);
        } else {
          console.info('[AUTH-TRACE][BOOTSTRAP] clearAuthState reason=no_stored_token');
          clearAuthState({ preserveRememberedUsername: true });
          useAuthStore.setState({ isHydrated: true });
          return;
        }
      } catch (err) {
        console.info('[AUTH-TRACE][BOOTSTRAP] getMe failed -> clearAuthState', {
          error: err instanceof Error ? err.message : String(err),
        });
        clearAuthState({ preserveRememberedUsername: true });
        console.debug('Auth bootstrap failed, user needs to login', err);
      } finally {
        // 3. 标记已初始化（无论成功失败）
        useAuthStore.setState({ isHydrated: true });
      }
    };

    bootstrap();
  }, [isHydrated, setAccessToken, setUser]);

  return null; // 不渲染任何内容
}

















