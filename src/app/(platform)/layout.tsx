'use client';

import { useEffect, useState, useRef, memo, type ReactNode } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import { TaskCenterProvider } from '@/components/task-center';
import TaskCenterMount from '@/components/task-center/TaskCenterMount';
import {
  iconRailItems,
  workspaceSideMenuItems,
  adminUsersNavItem,
  adminAuditNavItem,
  adminProjectsNavItem,
  type NavItem,
} from '@/components/layout/sidebar/navItems';
import {
  isSectionActive,
  isWorkspaceHomeActive,
} from '@/components/layout/sidebar/navActive';
import { ChevronDown, ChevronRight, Check } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import type { Role } from '@/lib/api/types';
import { normalizeRole } from '@/lib/api/roleLabels';
import {
  canSeeLogMenu,
  canSeeUserMenu,
} from '@/lib/permissions/menuVisibility';
import { useI18n } from '@/components/common/I18nProvider';

const PlatformPageMain = memo(function PlatformPageMain({ children }: { children: ReactNode }) {
  return (
    <main style={{ flex: 1, overflow: 'auto', backgroundColor: '#f6f7f9', color: '#111827' }}>
      {children}
    </main>
  );
});

export default function PlatformLayout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const user = useAuthStore((s) => s.user);
  const isHydrated = useAuthStore((s) => s.isHydrated);
  const logout = useAuthStore((s) => s.logout);
  const { t, locale, setLocale } = useI18n();
  const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
  const [isLogoutConfirmOpen, setIsLogoutConfirmOpen] = useState(false);
  const [isLanguageMenuOpen, setIsLanguageMenuOpen] = useState(false);
  const accountMenuRef = useRef<HTMLDivElement | null>(null);
  const [devToast, setDevToast] = useState<string | null>(null);

  // 客户端精确守卫 + 账号菜单点击外部关闭
  useEffect(() => {
    if (isHydrated) {
      // 未登录 → 跳转登录
      if (!user) {
        router.push('/login');
      }
    }

    const handleClickOutside = (event: MouseEvent) => {
      const menuEl = accountMenuRef.current;
      if (!menuEl) return;
      if (!menuEl.contains(event.target as Node)) {
        setIsAccountMenuOpen(false);
        setIsLanguageMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isHydrated, user, pathname, router]);

  const isLoading = !isHydrated || !user;

  const handleLogoutClick = () => {
    setIsAccountMenuOpen(false);
    setIsLanguageMenuOpen(false);
    setIsLogoutConfirmOpen(true);
  };

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#f6f7f9',
      }}>
        <div style={{ fontSize: '14px', color: '#6b7280' }}>{t('common.loading')}</div>
      </div>
    );
  }

  // 判断当前属于哪个域
  const isWorkspaceDomain = pathname.startsWith('/workspace');
  const isAdminDomain =
    pathname.startsWith('/admin') || pathname.startsWith('/devices') || pathname.startsWith('/settings');

  // 按角色计算菜单项（此处 user 一定非空）
  const role: Role = normalizeRole((user as NonNullable<typeof user>).role);
  const rawRole = (user as NonNullable<typeof user>).role;

  const roleLabelMap: Record<Role, string> = {
    SUPER_ADMIN: t('userMenu.roleSuperAdmin'),
    ADMIN: t('userMenu.roleTeamAdmin'),
    OWNER: t('userMenu.roleProjectOwner'),
    USER: t('userMenu.roleUser'),
  };

  const handleConfirmLogout = async () => {
    try {
      await logout();
    } finally {
      setIsLogoutConfirmOpen(false);
      setIsAccountMenuOpen(false);
      setIsLanguageMenuOpen(false);
      router.replace('/login');
    }
  };

  const localeLabelMap: Record<string, string> = {
    'zh-CN': t('userMenu.localeZhCN'),
    en: t('userMenu.localeEn'),
    sv: t('userMenu.localeSv'),
  };

  // 工作台域菜单（工业版 RoboTwin）
  const commonMenuItems = workspaceSideMenuItems;

  const adminMenuItems: NavItem[] = [
    ...(canSeeUserMenu(rawRole) ? [adminUsersNavItem] : []),
    ...(canSeeLogMenu(rawRole) ? [adminAuditNavItem] : []),
  ];

  // 当前 SideMenu 菜单项（根据域和角色；概览页不显示二级菜单）
  const currentSideMenuItems = isAdminDomain ? adminMenuItems : isWorkspaceDomain ? commonMenuItems : [];
  
  // 分离主功能区和系统区
  const mainItems = currentSideMenuItems.filter(item => item.section === 'main');
  const systemItems = currentSideMenuItems.filter(item => item.section === 'system');

  // IconRail 菜单：所有角色均显示 /admin（成员只能进入受限页面）
  const visibleIconRailItems = iconRailItems;

  // 判断 IconRail 菜单项是否激活
  const isIconRailActive = (item: typeof iconRailItems[0]) => {
    if (item.labelKey === 'sidebar.overview') {
      return isWorkspaceHomeActive(pathname);
    }
    if (item.labelKey === 'sidebar.workspace') {
      return isWorkspaceDomain;
    }
    if (item.path === '/admin') {
      return isAdminDomain;
    }
    return isSectionActive(pathname, item.path);
  };

  const iconRailHref = (item: typeof iconRailItems[0]) => {
    if (item.labelKey === 'sidebar.overview') return '/workspace';
    if (item.labelKey === 'sidebar.workspace') return '/workspace/data';
    if (item.path === '/admin') {
      if (canSeeUserMenu(rawRole)) return '/admin/users';
      if (canSeeLogMenu(rawRole)) return '/admin/audit';
      return '/admin/projects';
    }
    return item.path;
  };

  // 判断 SideMenu 菜单项是否激活（禁用项不参与选中）
  const isSideMenuActive = (item: NavItem) => {
    if (item.disabled) return false;
    if (item.children) {
      return (
        isSectionActive(pathname, item.path) ||
        item.children.some((child) => isSectionActive(pathname, child))
      );
    }
    return isSectionActive(pathname, item.path);
  };

  return (
    <TaskCenterProvider>
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column',
      height: '100vh', 
      fontFamily: 'system-ui, -apple-system, sans-serif', 
      backgroundColor: '#f6f7f9'
    }}>
      {/* 顶部 Topbar（第一横栏） */}
      <header style={{
        height: '60px',
        backgroundColor: '#ffffff',
        borderBottom: '1px solid #e5e7eb',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        zIndex: 10,
      }}>
        {/* 左侧：Logo + 品牌名 */}
        <Link
          href="/workspace/data"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            textDecoration: 'none',
            color: 'inherit',
          }}
        >
          <img
            src="/logo/epybots_app_icon.png"
            alt="ePi Logo"
            style={{
              width: 28,
              height: 28,
              objectFit: 'contain',
              flexShrink: 0,
            }}
          />
          <span style={{
            fontSize: '14px',
            fontWeight: '600',
            color: '#111827',
            lineHeight: 1.25,
          }}>
            {t('login.brandName')}
          </span>
        </Link>

        {/* 右侧：用户信息 / 登录入口 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {user ? (
            <div style={{ position: 'relative' }} ref={accountMenuRef}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '6px 12px',
                  color: '#374151',
                  fontSize: '14px',
                  cursor: 'pointer',
                  borderRadius: '6px',
                  transition: 'all 0.2s',
                  border: isAccountMenuOpen ? '1px solid #e5e7eb' : '1px solid transparent',
                  backgroundColor: isAccountMenuOpen ? '#f9fafb' : 'transparent',
                }}
                onMouseEnter={(e) => {
                  if (!isAccountMenuOpen) {
                    e.currentTarget.style.backgroundColor = '#f9fafb';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isAccountMenuOpen) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
                onClick={() => {
                  setIsAccountMenuOpen((open) => !open);
                  if (isLanguageMenuOpen) setIsLanguageMenuOpen(false);
                }}
              >
                {/* 用户头像/图标 */}
                <div style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: '#2563eb',
                  color: '#ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '12px',
                  fontWeight: '600',
                  flexShrink: 0,
                  border: '1px solid #e5e7eb',
                }}>
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <span>{user.username}</span>
                <ChevronDown
                  size={16}
                  style={{
                    color: '#9ca3af',
                    flexShrink: 0,
                    transform: isAccountMenuOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.15s ease',
                  }}
                />
              </div>

              {/* 账号下拉菜单 */}
              {isAccountMenuOpen && (
                <div
                  style={{
                    position: 'absolute',
                    right: 0,
                    marginTop: '8px',
                    width: '220px',
                    backgroundColor: '#ffffff',
                    borderRadius: '8px',
                    boxShadow: '0 10px 40px rgba(15,23,42,0.15)',
                    border: '1px solid #e5e7eb',
                    padding: '8px 0',
                    zIndex: 50,
                  }}
                >
                  {/* 顶部用户信息 */}
                  <div
                    style={{
                      padding: '8px 16px 10px',
                      borderBottom: '1px solid #f3f4f6',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '14px',
                        fontWeight: 600,
                        color: '#111827',
                        marginBottom: '4px',
                      }}
                    >
                      {user.username}
                    </div>
                    <div
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        padding: '2px 8px',
                        borderRadius: '999px',
                        backgroundColor: '#f3f4f6',
                        fontSize: '12px',
                        color: '#4b5563',
                      }}
                    >
                      {roleLabelMap[role] ?? role}
                    </div>
                  </div>

                  <Link
                    href="/account/settings"
                    onClick={() => {
                      setIsAccountMenuOpen(false);
                      setIsLanguageMenuOpen(false);
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 16px',
                      fontSize: '14px',
                      color: '#374151',
                      backgroundColor: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      textDecoration: 'none',
                    }}
                  >
                    {t('userMenu.accountSettings')}
                  </Link>

                  {/* 语言入口（一级菜单单行） */}
                  <button
                    type="button"
                    onClick={() => setIsLanguageMenuOpen((open) => !open)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 16px',
                      fontSize: '14px',
                      color: '#374151',
                      backgroundColor: isLanguageMenuOpen ? '#f9fafb' : 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background-color 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f9fafb';
                    }}
                    onMouseLeave={(e) => {
                      if (!isLanguageMenuOpen) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                  >
                    <span>{t('common.language')}</span>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 8,
                        color: '#6b7280',
                        fontSize: '13px',
                      }}
                    >
                      <span>{localeLabelMap[locale] ?? locale}</span>
                      <ChevronRight size={16} />
                    </span>
                  </button>

                  {/* 语言二级菜单 */}
                  {isLanguageMenuOpen && (
                    <div
                      style={{
                        position: 'absolute',
                        top: 84,
                        right: 232,
                        marginLeft: '8px',
                        width: 180,
                        backgroundColor: '#ffffff',
                        borderRadius: 8,
                        boxShadow: '0 12px 40px rgba(15,23,42,0.18)',
                        border: '1px solid #e5e7eb',
                        padding: '6px 0',
                        zIndex: 60,
                      }}
                    >
                      {(
                        [
                          { value: 'zh-CN', label: t('userMenu.localeZhCN') },
                          { value: 'en', label: t('userMenu.localeEn') },
                          { value: 'sv', label: t('userMenu.localeSv') },
                        ] as const
                      ).map((opt) => {
                        const active = locale === opt.value;
                        return (
                          <button
                            key={opt.value}
                            type="button"
                            onClick={() => {
                              setLocale(opt.value);
                              setIsLanguageMenuOpen(false);
                              setIsAccountMenuOpen(false);
                            }}
                            style={{
                              width: '100%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              padding: '8px 14px',
                              fontSize: '14px',
                              color: '#374151',
                              backgroundColor: active ? 'rgba(37,99,235,0.06)' : 'transparent',
                              border: 'none',
                              cursor: 'pointer',
                              transition: 'background-color 0.15s ease',
                            }}
                            onMouseEnter={(e) => {
                              if (active) return;
                              e.currentTarget.style.backgroundColor = '#f9fafb';
                            }}
                            onMouseLeave={(e) => {
                              if (active) return;
                              e.currentTarget.style.backgroundColor = 'transparent';
                            }}
                          >
                            <span>{opt.label}</span>
                            <span style={{ width: 18, display: 'inline-flex', justifyContent: 'center', color: '#2563eb' }}>
                              {active ? <Check size={16} strokeWidth={2.2} /> : null}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* 菜单项：退出登录 */}
                  <button
                    type="button"
                    onClick={handleLogoutClick}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 16px',
                      fontSize: '14px',
                      color: '#b91c1c',
                      backgroundColor: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#fef2f2';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    {t('userMenu.logout')}
                  </button>
                </div>
              )}
            </div>
          ) : (
            <button
              type="button"
              onClick={() => router.push('/login')}
              style={{
                padding: '6px 12px',
                fontSize: '14px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                color: '#374151',
                cursor: 'pointer',
              }}
            >
              登录
            </button>
          )}
        </div>
      </header>

      {/* 下方内容区（包含左侧栏和主内容） */}
      <div style={{ 
        display: 'flex', 
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
      }}>
        {/* IconRail（最左侧 56px） */}
        <aside style={{
          width: '56px',
          backgroundColor: '#ffffff',
          borderRight: '1px solid #e5e7eb',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '12px 0',
        }}>
          <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
            {visibleIconRailItems.map((item) => {
              const isActive = isIconRailActive(item);
              const Icon = item.icon;
              return (
                <Link
                  key={item.labelKey ?? item.path}
                  href={iconRailHref(item)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '40px',
                    height: '40px',
                    margin: '0 auto',
                    color: isActive ? '#2563eb' : '#6b7280',
                    textDecoration: 'none',
                    backgroundColor: isActive ? '#eff6ff' : 'transparent',
                    borderRadius: '8px',
                    transition: 'all 0.2s',
                    position: 'relative',
                  }}
                  title={item.labelKey ? t(item.labelKey) : item.label}
                  aria-label={item.labelKey ? t(item.labelKey) : item.label}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = '#f3f4f6';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <Icon size={20} />
                  {isActive && (
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      top: '50%',
                      transform: 'translateY(-50%)',
                      width: '3px',
                      height: '24px',
                      backgroundColor: '#2563eb',
                      borderRadius: '0 2px 2px 0',
                    }} />
                  )}
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* SideMenu（第二列 220px） */}
        {(isWorkspaceDomain || isAdminDomain) && (
          <aside style={{
            width: '220px',
            backgroundColor: '#ffffff',
            borderRight: '1px solid #e5e7eb',
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            overflow: 'hidden',
          }}>
            {/* 主功能区（可滚动） */}
            <nav style={{
              flex: 1,
              overflowY: 'auto',
              padding: '8px 0',
            }}>
              {mainItems.map((item) => {
                const isActive = isSideMenuActive(item);
                const Icon = item.icon;
                const isDisabled = item.disabled === true;
                const label = item.labelKey ? t(item.labelKey) : item.label;
                if (isDisabled) {
                  return (
                    <div
                      key={item.path}
                      role="button"
                      title={item.disabledReason ?? '暂未开放'}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        height: '38px',
                        padding: '0 16px',
                        color: '#9ca3af',
                        fontSize: '14px',
                        fontWeight: 400,
                        opacity: 0.6,
                        cursor: 'not-allowed',
                        borderLeft: '2px solid transparent',
                        pointerEvents: 'auto',
                      }}
                      onClick={(e) => {
                        e.preventDefault();
                        if (item.path === '/clean') {
                          setDevToast('功能开发中');
                          setTimeout(() => setDevToast(null), 1800);
                        }
                      }}
                      onMouseDown={(e) => e.preventDefault()}
                    >
                      <Icon size={18} style={{ flexShrink: 0 }} />
                      <span>{label}</span>
                    </div>
                  );
                }
                return (
                  <Link
                    key={item.path}
                    href={item.path}
                    title={label}
                    aria-label={label}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      height: '38px',
                      padding: '0 16px',
                      color: isActive ? '#2563eb' : '#374151',
                      textDecoration: 'none',
                      backgroundColor: isActive ? '#eff6ff' : 'transparent',
                      fontSize: '14px',
                      fontWeight: isActive ? '500' : '400',
                      transition: 'all 0.2s',
                      borderLeft: isActive ? '2px solid #2563eb' : '2px solid transparent',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = '#f9fafb';
                        e.currentTarget.style.color = '#111827';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.color = '#374151';
                      }
                    }}
                  >
                    <Icon size={18} style={{ flexShrink: 0 }} />
                    <span>{label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* 系统区（贴底） */}
            {systemItems.length > 0 && (
              <>
                <div style={{
                  height: '1px',
                  backgroundColor: '#e5e7eb',
                  margin: '8px 0',
                }} />
                <nav style={{
                  padding: '8px 0',
                }}>
                  {systemItems.map((item) => {
                    const isActive = isSideMenuActive(item);
                    const Icon = item.icon;
                    const label = item.labelKey ? t(item.labelKey) : item.label;
                    return (
                      <Link
                        key={item.path}
                        href={item.path}
                        title={label}
                        aria-label={label}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px',
                          height: '38px',
                          padding: '0 16px',
                          color: isActive ? '#2563eb' : '#374151',
                          textDecoration: 'none',
                          backgroundColor: isActive ? '#eff6ff' : 'transparent',
                          fontSize: '14px',
                          fontWeight: isActive ? '500' : '400',
                          transition: 'all 0.2s',
                          borderLeft: isActive ? '2px solid #2563eb' : '2px solid transparent',
                        }}
                        onMouseEnter={(e) => {
                          if (!isActive) {
                            e.currentTarget.style.backgroundColor = '#f9fafb';
                            e.currentTarget.style.color = '#111827';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isActive) {
                            e.currentTarget.style.backgroundColor = 'transparent';
                            e.currentTarget.style.color = '#374151';
                          }
                        }}
                      >
                        <Icon size={18} style={{ flexShrink: 0 }} />
                        <span>{label}</span>
                      </Link>
                    );
                  })}
                </nav>
              </>
            )}
          </aside>
        )}

        {/* 主内容区 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' }}>
          {/* 页面内容 */}
          <PlatformPageMain>{children}</PlatformPageMain>
        </div>
      </div>

      {/* 退出登录确认弹窗 */}
      {isLogoutConfirmOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15,23,42,0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              padding: '24px 24px 20px',
              width: '100%',
              maxWidth: '400px',
              boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
              border: '1px solid #e5e7eb',
            }}
          >
            <h2
              style={{
                fontSize: '18px',
                fontWeight: 600,
                marginBottom: '8px',
                color: '#111827',
              }}
            >
              {t('userMenu.logoutConfirmTitle')}
            </h2>
            <p
              style={{
                fontSize: '14px',
                color: '#6b7280',
                marginBottom: '20px',
              }}
            >
              {t('userMenu.logoutConfirmDescription')}
            </p>
            <div
              style={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: '12px',
              }}
            >
              <button
                type="button"
                onClick={() => setIsLogoutConfirmOpen(false)}
                style={{
                  padding: '8px 14px',
                  fontSize: '14px',
                  borderRadius: '8px',
                  border: '1px solid #d1d5db',
                  backgroundColor: '#ffffff',
                  color: '#374151',
                  cursor: 'pointer',
                }}
              >
                取消
              </button>
              <button
                type="button"
                onClick={handleConfirmLogout}
                style={{
                  padding: '8px 14px',
                  fontSize: '14px',
                  borderRadius: '8px',
                  border: 'none',
                  backgroundColor: '#dc2626',
                  color: '#ffffff',
                  cursor: 'pointer',
                }}
              >
                {t('userMenu.logout')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 后台任务中心：仅在数据资产页/转换页挂载 */}
      <TaskCenterMount />
      {/* 平台级功能开发中提示 Toast */}
      {devToast && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '8px 16px',
            borderRadius: 8,
            backgroundColor: '#111827',
            color: '#ffffff',
            fontSize: 13,
            boxShadow: '0 16px 40px rgba(15,23,42,0.35)',
            zIndex: 2000,
          }}
        >
          {devToast}
        </div>
      )}
    </div>
    </TaskCenterProvider>
  );
}
