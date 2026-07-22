'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { login, getMe } from '@/lib/api/authClient';
import { useAuthStore } from '@/store/authStore';
import type { Role } from '@/lib/api/types';
import { normalizeRole } from '@/lib/api/roleLabels';
import { SAVED_PASSWORD_KEY, SAVED_USERNAME_KEY, clearAuthState, initSessionId } from '@/lib/auth/session';
import { Eye, EyeOff, AlertCircle } from 'lucide-react';
import { FloatingField } from '@/components/auth/FloatingField';
import { useI18n } from '@/components/common/I18nProvider';
import LanguageSwitcher from '@/components/common/LanguageSwitcher';

/** Placeholder to avoid SSR/client hydration mismatch: no locale-dependent text until client has applied epi_locale */
function LoginPlaceholder() {
  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f8fafc',
        backgroundImage: 'linear-gradient(180deg, rgba(2,6,23,0.04) 0%, rgba(248,250,252,0) 55%)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <main
        style={{
          flex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          paddingTop: 80,
          paddingBottom: 48,
          paddingLeft: 16,
          paddingRight: 16,
          boxSizing: 'border-box',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 520,
            borderRadius: 16,
            border: '1px solid rgba(226,232,240,0.8)',
            backgroundColor: '#ffffff',
            boxShadow: '0 18px 60px rgba(2,6,23,0.08)',
            padding: 40,
            minHeight: 380,
          }}
        />
      </main>
    </div>
  );
}

function loadSavedUsername() {
  if (typeof window === 'undefined') return { username: '', saveUsername: false };
  const username = localStorage.getItem(SAVED_USERNAME_KEY) ?? '';
  return {
    username,
    saveUsername: username.length > 0,
  };
}

export default function LoginPage() {
  const router = useRouter();
  const { t } = useI18n();
  const [mounted, setMounted] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [saveUsername, setSaveUsername] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { setUser } = useAuthStore();

  useEffect(() => {
    setMounted(true);
  }, []);
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(SAVED_PASSWORD_KEY);
    }
  }, []);
  useEffect(() => {
    const saved = loadSavedUsername();
    setUsername(saved.username);
    setSaveUsername(saved.saveUsername);
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!username.trim() || !password) {
      setError(t('login.accountPasswordRequired'));
      return;
    }

    setError('');
    setLoading(true);

    try {
      // 切换账号前先统一清理旧认证态，避免旧会话串号
      console.info('[AUTH-TRACE][LOGIN] before clearAuthState', {
        username: username.trim(),
      });
      clearAuthState({ preserveRememberedUsername: true });
      // Step 1：为本标签页初始化 sessionId（标签页隔离）
      console.info('[AUTH-TRACE][LOGIN] sessionId', initSessionId());

      // 1. 登录（获取 access_token，refresh_token 在 HttpOnly cookie）
      const loginData = await login(username, password);
      console.info('[AUTH-TRACE][LOGIN] login success', {
        role: loginData?.role,
        token: loginData?.access_token
          ? `${loginData.access_token.slice(0, 8)}...${loginData.access_token.slice(-6)}`
          : 'null',
      });

      // 2. 获取用户信息
      const user = await getMe();
      console.info('[AUTH-TRACE][LOGIN] getMe success', {
        id: user?.id,
        username: user?.username,
      });
      setUser(user);

      // 3. 根据角色跳转（新角色体系）
      const role: Role = normalizeRole(loginData.role);
      const targetPath = role === 'SUPER_ADMIN' ? '/admin/users' : '/admin/projects';
      router.push(targetPath);
    } catch (err) {
      const msg = err instanceof Error ? err.message : t('login.invalidCredentials');
      if (msg === '账号已禁用') {
        setError(t('login.accountDisabled'));
      } else {
        setError(msg);
      }
      setLoading(false);
    }
  };

  const persistSavedUsername = (saveUser: boolean, user: string) => {
    if (typeof window === 'undefined') return;
    if (saveUser) localStorage.setItem(SAVED_USERNAME_KEY, user);
    else localStorage.removeItem(SAVED_USERNAME_KEY);
  };

  if (!mounted) return <LoginPlaceholder />;
  return (
    <AuthShell
      username={username}
      password={password}
      loading={loading}
      error={error}
      showPassword={showPassword}
      saveUsername={saveUsername}
      onUsernameChange={setUsername}
      onPasswordChange={setPassword}
      onToggleShowPassword={() => setShowPassword((v) => !v)}
      onSaveUsernameChange={setSaveUsername}
      onPersistSaved={persistSavedUsername}
      onSubmit={handleSubmit}
    />
  );
}

type AuthShellProps = {
  username: string;
  password: string;
  loading: boolean;
  error: string;
  showPassword: boolean;
  saveUsername: boolean;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onToggleShowPassword: () => void;
  onSaveUsernameChange: (v: boolean) => void;
  onPersistSaved: (saveUser: boolean, user: string) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
};

function AuthShell({
  username,
  password,
  loading,
  error,
  showPassword,
  saveUsername,
  onUsernameChange,
  onPasswordChange,
  onToggleShowPassword,
  onSaveUsernameChange,
  onPersistSaved,
  onSubmit,
}: AuthShellProps) {
  const { t } = useI18n();
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    onPersistSaved(saveUsername, username);
    onSubmit(e);
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#f8fafc', // bg-slate-50
        backgroundImage:
          'linear-gradient(180deg, rgba(2,6,23,0.04) 0%, rgba(248,250,252,0) 55%)',
        backgroundRepeat: 'no-repeat',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <style>
        {`
          @keyframes auth-card-enter {
            from {
              opacity: 0;
              transform: translateY(6px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }

          @keyframes auth-spinner {
            from {
              transform: rotate(0deg);
            }
            to {
              transform: rotate(360deg);
            }
          }

          @media (max-width: 640px) {
            .auth-card {
              padding: 28px;
              border-radius: 16px;
            }
          }
        `}
      </style>

      <main
        style={{
          flex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          paddingTop: 80, // pt-20
          paddingBottom: 48, // pb-12
          paddingLeft: 16,
          paddingRight: 16,
          boxSizing: 'border-box',
        }}
      >
        <div
          className="auth-card"
          style={{
            width: '100%',
            maxWidth: 520,
            borderRadius: 16, // rounded-2xl
            border: '1px solid rgba(226,232,240,0.8)', // border-slate-200/80
            backgroundColor: '#ffffff',
            boxShadow: '0 18px 60px rgba(2,6,23,0.08)',
            padding: 40, // p-10
            animation: 'auth-card-enter 200ms ease-out',
          }}
        >
          {/* Brand row: flex items-start justify-between */}
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'space-between',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 14,
                minWidth: 0,
                flex: 1,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  width: 48,
                  height: 48,
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 12,
                  border: '1px solid #e2e8f0',
                  backgroundColor: '#ffffff',
                  flexShrink: 0,
                }}
              >
                <img
                  src="/logo/epybots_app_icon.png"
                  alt="ePi Logo"
                  style={{
                    width: 32,
                    height: 32,
                    objectFit: 'contain',
                  }}
                />
              </div>
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  minWidth: 0,
                  paddingTop: 2,
                }}
              >
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 600,
                    letterSpacing: '-0.01em',
                    color: '#0f172a',
                    lineHeight: 1.3,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {t('login.brandName')}
                </div>
                <div
                  style={{
                    marginTop: 6,
                    fontSize: 12,
                    letterSpacing: '0.06em',
                    color: '#64748b',
                    lineHeight: 1.35,
                    maxWidth: 280,
                  }}
                >
                  {t('login.brandSubtitle')}
                </div>
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                marginTop: 2,
                flexShrink: 0,
              }}
            >
              <LanguageSwitcher size="md" />
              <div
                style={{
                  fontSize: 11,
                  letterSpacing: '0.22em',
                  color: '#94a3b8',
                  padding: '6px 8px',
                  borderRadius: 8,
                  border: '1px solid rgba(226,232,240,0.85)',
                  backgroundColor: '#ffffff',
                }}
              >
                {t('login.beta')}
              </div>
            </div>
          </div>

          {/* 表单 */}
          <form
            onSubmit={handleSubmit}
            style={{
              marginTop: 24, // mt-6 标题与第一个输入框
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <FloatingField
              id="login-username"
              label={`${t('login.account')} *`}
              value={username}
              onChange={onUsernameChange}
              type="text"
              autoComplete="username"
            />

            <div style={{ marginTop: 20 }} />
            <FloatingField
              id="login-password"
              label={`${t('login.password')} *`}
              value={password}
              onChange={onPasswordChange}
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              rightSlot={
                <button
                  type="button"
                  onClick={onToggleShowPassword}
                  style={{
                    border: 'none',
                    outline: 'none',
                    background: 'transparent',
                    padding: 4,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    color: '#94a3b8',
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.color = '#475569';
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.color = '#94a3b8';
                  }}
                >
                  {showPassword ? (
                    <EyeOff size={16} strokeWidth={1.5} />
                  ) : (
                    <Eye size={16} strokeWidth={1.5} />
                  )}
                </button>
              }
            />

            {error && (
              <div
                style={{
                  marginTop: 16,
                  borderRadius: 8,
                  border: '1px solid #fecaca',
                  backgroundColor: '#fef2f2',
                  padding: '8px 12px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8,
                }}
              >
                <div style={{ marginTop: 1, color: '#fb7185' }}>
                  <AlertCircle size={14} strokeWidth={1.8} />
                </div>
                <p style={{ margin: 0, fontSize: 12, color: '#b91c1c' }}>{error}</p>
              </div>
            )}

            <div
              style={{
                marginTop: 24,
                display: 'flex',
                alignItems: 'center',
                gap: 24,
              }}
            >
              <label
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 13,
                  color: '#334155',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={saveUsername}
                  onChange={(e) => onSaveUsernameChange(e.target.checked)}
                  style={{
                    width: 16,
                    height: 16,
                    borderRadius: 4,
                    border: '1px solid #cbd5e1',
                    margin: 0,
                  }}
                />
                <span>{t('login.rememberAccount')}</span>
              </label>
            </div>

            {/* 主按钮 */}
            <div style={{ marginTop: 24 }}>
              <button
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  height: 44, // h-11
                  borderRadius: 8, // rounded-lg
                  border: 'none',
                  backgroundColor: '#2563eb',
                  color: '#f9fafb',
                  fontSize: 14,
                  fontWeight: 500,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  cursor: loading ? 'default' : 'pointer',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                  transform: 'translateY(0)',
                  transition: 'background-color 0.15s ease, transform 0.05s ease',
                  opacity: loading ? 0.95 : 1,
                }}
                onMouseEnter={(e) => {
                  if (loading) return;
                  (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#1d4ed8';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#2563eb';
                  (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
                }}
                onMouseDown={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(1px)';
                }}
                onMouseUp={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
                }}
              >
                {loading && (
                  <span
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 999,
                      border: '2px solid rgba(239,246,255,0.5)',
                      borderTopColor: '#eff6ff',
                      animation: 'auth-spinner 0.6s linear infinite',
                    }}
                  />
                )}
                <span>{loading ? t('login.loggingIn') : t('login.login')}</span>
              </button>
            </div>

            <p
              style={{
                marginTop: 24,
                marginBottom: 0,
                fontSize: 11,
                color: '#64748b',
              }}
            >
              {t('login.agreement')}
            </p>
          </form>
        </div>
      </main>

      <footer
        style={{
          padding: '16px 24px',
          fontSize: 11,
          color: '#9ca3af',
          textAlign: 'center',
        }}
      >
        © 2026 ePyBot
      </footer>
    </div>
  );
}

