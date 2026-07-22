'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/api/authClient';
import {
  assignableUserRolesForEditor,
  canEditOtherUserRole,
  canOwnerScopedUserMutations,
  canTeamAdminScopedUserMutations,
  creatableUserRolesForActor,
  getRoleLabelKey,
  isSuperAdmin,
  isTeamAdminAccount,
  normalizeRole,
  type CreatableUserRole,
} from '@/lib/api/roleLabels';
import { useI18n } from '@/components/common/I18nProvider';
import { useAuthStore } from '@/store/authStore';
import { canSeeUserMenu } from '@/lib/permissions/menuVisibility';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import { Eye, EyeOff } from 'lucide-react';

interface User {
  id: string;
  account_id: string;
  username: string;
  role: string;
  is_active: boolean;
  created_at: string;
  /** 列表接口：综合可用（含团队停用）；缺省时回退 is_active */
  effective_is_active?: boolean;
  /** 列表接口聚合：团队展示名；超管为「平台」 */
  team_name?: string;
  team_id?: string | null;
}

interface UserListResponse {
  items: User[];
  total: number;
  page: number;
  page_size: number;
}

interface TeamOptionsResponse {
  items: TeamOption[];
  total: number;
}

interface TeamOption {
  id: string;
  name: string;
  code: string;
  status?: string;
}

function roleSelectLabel(role: CreatableUserRole, t: (k: string) => string): string {
  if (role === 'ADMIN') return t('adminUsersPage.roleAdmin');
  if (role === 'OWNER') return t('adminUsersPage.roleOwner');
  return t('adminUsersPage.roleMember');
}

export default function UsersPage() {
  const router = useRouter();
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);
  const isHydrated = useAuthStore((s) => s.isHydrated);
  const canCreateUsers = Boolean(
    authUser && (isSuperAdmin(authUser.role) || isTeamAdminAccount(authUser.role))
  );
  const canFullUserAdmin = Boolean(authUser && isSuperAdmin(authUser.role));
  const creatableRoles = useMemo(
    () => (authUser ? creatableUserRolesForActor(authUser.role) : []),
    [authUser]
  );
  const [users, setUsers] = useState<User[]>([]);
  const [total, setTotal] = useState(0);
  const [searchInput, setSearchInput] = useState('');
  const [appliedQuery, setAppliedQuery] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showResetModal, setShowResetModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showRoleModal, setShowRoleModal] = useState(false);
  const [roleEditNew, setRoleEditNew] = useState<CreatableUserRole>('USER');
  const [disableConfirmUser, setDisableConfirmUser] = useState<User | null>(null);
  const [enableConfirmUser, setEnableConfirmUser] = useState<User | null>(null);
  const [toggleActiveLoading, setToggleActiveLoading] = useState(false);
  const [successToast, setSuccessToast] = useState<string | null>(null);
  const [teamsForCreate, setTeamsForCreate] = useState<TeamOption[]>([]);
  const [createModalTeamsLoading, setCreateModalTeamsLoading] = useState(false);
  const [createModalTeamsError, setCreateModalTeamsError] = useState(false);
  const [teamAdminPrimaryTeam, setTeamAdminPrimaryTeam] = useState<TeamOption | null>(null);
  const [showCreatePassword, setShowCreatePassword] = useState(false);

  // 新建用户表单
  const [createForm, setCreateForm] = useState({
    username: '',
    password: '',
    role: 'USER' as CreatableUserRole,
    team_id: '',
  });

  // 重置密码表单
  const [resetForm, setResetForm] = useState({
    new_password: '',
    new_password_confirm: '',
  });

  const showSuccess = (msg: string) => {
    setSuccessToast(msg);
    window.setTimeout(() => setSuccessToast(null), 2200);
  };

  const loadUsers = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      const params = new URLSearchParams();
      if (appliedQuery.trim()) params.set('q', appliedQuery.trim());
      params.set('page', String(page));
      params.set('page_size', String(pageSize));
      const data = await apiGet<UserListResponse>(`/users?${params.toString()}`);
      const items = Array.isArray(data?.items) ? data.items : [];
      const tTotal = typeof data?.total === 'number' ? data.total : 0;
      setUsers(items);
      setTotal(tTotal);
      setPage((prev) => {
        const maxP = Math.max(1, Math.ceil(tTotal / pageSize) || 1);
        return Math.min(prev, maxP);
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('403') || msg.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(msg || t('adminUsersPage.loadFailed'));
    } finally {
      setLoading(false);
    }
  }, [appliedQuery, page, pageSize, router, t]);

  useEffect(() => {
    if (!isHydrated) return;
    if (!authUser) {
      router.replace('/login');
      return;
    }
    if (!canSeeUserMenu(authUser.role)) {
      router.replace('/forbidden');
      return;
    }
    void loadUsers();
  }, [isHydrated, authUser, router, loadUsers]);

  useEffect(() => {
    if (!showCreateModal) return;
    let cancelled = false;
    setCreateModalTeamsLoading(true);
    setCreateModalTeamsError(false);
    if (!canFullUserAdmin) {
      setTeamAdminPrimaryTeam(null);
    }
    void (async () => {
      try {
        const data = await apiGet<TeamOptionsResponse>('/users/team-options');
        const list = Array.isArray(data?.items) ? data.items : [];
        if (cancelled) return;
        if (canFullUserAdmin) {
          setTeamsForCreate(list);
        } else {
          const sorted = [...list].sort((a, b) => a.id.localeCompare(b.id));
          setTeamAdminPrimaryTeam(sorted[0] ?? null);
        }
      } catch {
        if (!cancelled) {
          setCreateModalTeamsError(true);
          if (canFullUserAdmin) {
            setTeamsForCreate([]);
          } else {
            setTeamAdminPrimaryTeam(null);
          }
        }
      } finally {
        if (!cancelled) setCreateModalTeamsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [showCreateModal, canFullUserAdmin]);

  const applySearch = () => {
    setPage(1);
    setAppliedQuery(searchInput.trim());
  };

  const handleCreate = async () => {
    if (!createForm.username.trim()) {
      setError(t('adminUsersPage.usernameRequired'));
      return;
    }
    if (createForm.password.trim().length < 6) {
      setError(t('adminUsersPage.createPasswordTooShort'));
      return;
    }
    if (canFullUserAdmin && createForm.role === 'ADMIN' && !createForm.team_id.trim()) {
      setError(t('adminUsersPage.teamRequired'));
      return;
    }
    try {
      setError('');
      const body: Record<string, unknown> = {
        username: createForm.username.trim(),
        password: createForm.password.trim(),
        role: createForm.role,
      };
      if (canFullUserAdmin && createForm.team_id.trim()) {
        body.team_id = createForm.team_id.trim();
      }
      const created = await apiPost<User>('/users', body);
      setShowCreateModal(false);
      setShowCreatePassword(false);
      setTeamAdminPrimaryTeam(null);
      setCreateModalTeamsError(false);
      setCreateForm({
        username: '',
        password: '',
        role: creatableRoles[0] ?? 'USER',
        team_id: '',
      });
      await loadUsers();
      showSuccess(t('adminUsersPage.createSuccessWithAccount', { account: created.account_id }));
    } catch (err: any) {
      if (err.message?.includes('403') || err.message?.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(err.message || t('adminUsersPage.createFailed'));
    }
  };

  const handleDisableConfirm = async () => {
    const user = disableConfirmUser;
    if (!user) return;
    try {
      setError('');
      setToggleActiveLoading(true);
      await apiPatch<User>(`/users/${user.id}/disable`, {});
      setDisableConfirmUser(null);
      await loadUsers();
      showSuccess(t('adminUsersPage.statusUpdateSuccess'));
    } catch (err: any) {
      if (err.message?.includes('403') || err.message?.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(err.message || t('adminUsersPage.disableFailed'));
    } finally {
      setToggleActiveLoading(false);
    }
  };

  const handleEnableConfirm = async () => {
    const user = enableConfirmUser;
    if (!user) return;
    try {
      setError('');
      setToggleActiveLoading(true);
      await apiPatch<User>(`/users/${user.id}/enable`, {});
      setEnableConfirmUser(null);
      await loadUsers();
      showSuccess(t('adminUsersPage.statusUpdateSuccess'));
    } catch (err: any) {
      if (err.message?.includes('403') || err.message?.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(err.message || t('adminUsersPage.enableFailed'));
    } finally {
      setToggleActiveLoading(false);
    }
  };

  const handleResetPassword = async () => {
    if (!selectedUser) return;
    const p = resetForm.new_password.trim();
    const c = resetForm.new_password_confirm.trim();
    if (!p) {
      setError(t('adminUsersPage.resetPasswordRequired'));
      return;
    }
    if (p.length < 6) {
      setError(t('adminUsersPage.resetPasswordTooShort'));
      return;
    }
    if (p !== c) {
      setError(t('adminUsersPage.resetPasswordMismatch'));
      return;
    }
    try {
      setError('');
      await apiPatch<User>(`/users/${selectedUser.id}/reset-password`, { new_password: p });
      setShowResetModal(false);
      setResetForm({ new_password: '', new_password_confirm: '' });
      setSelectedUser(null);
      await loadUsers();
      showSuccess(t('adminUsersPage.resetPasswordSuccess'));
    } catch (err: any) {
      if (err.message?.includes('403') || err.message?.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(err.message || t('adminUsersPage.resetFailed'));
    }
  };

  const handleSaveRole = async () => {
    if (!selectedUser) return;
    try {
      setError('');
      await apiPatch<User>(`/users/${selectedUser.id}/role`, { role: roleEditNew });
      setShowRoleModal(false);
      setSelectedUser(null);
      await loadUsers();
      showSuccess(t('adminUsersPage.changeRoleSuccess'));
    } catch (err: any) {
      if (err.message?.includes('403') || err.message?.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      setError(err.message || t('adminUsersPage.changeRoleFailed'));
    }
  };

  const openRoleModal = (user: User) => {
    if (!authUser) return;
    const opts = assignableUserRolesForEditor(authUser.role, user.role);
    if (opts.length === 0) return;
    setSelectedUser(user);
    const curN = normalizeRole(user.role);
    const pick: CreatableUserRole =
      curN === 'ADMIN' || curN === 'OWNER' || curN === 'USER'
        ? opts.includes(curN as CreatableUserRole)
          ? (curN as CreatableUserRole)
          : opts[0]
        : opts[0];
    setRoleEditNew(pick);
    setShowRoleModal(true);
  };

  const handleDelete = async () => {
    if (!selectedUser) return;
    try {
      setError('');
      await apiDelete<null>(`/users/${selectedUser.id}`);
      setShowDeleteModal(false);
      setSelectedUser(null);
      await loadUsers();
      showSuccess(t('adminUsersPage.deleteSuccess'));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '';
      if (msg.includes('403') || msg.includes('Forbidden')) {
        router.push('/forbidden');
        return;
      }
      if (
        msg.includes('USER_DELETE_BLOCKED_PROJECT_REFS') ||
        msg.includes('USER_DELETE_BLOCKED_ACTIVE_REFS')
      ) {
        setError(t('adminUsersPage.deleteBlockedRefs'));
        return;
      }
      setError(msg || t('adminUsersPage.deleteFailed'));
    }
  };

  if (!isHydrated || !authUser) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ color: '#6b7280' }}>{t('common.loading')}</div>
      </div>
    );
  }
  if (!canSeeUserMenu(authUser.role)) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ color: '#6b7280' }}>{t('common.loading')}</div>
      </div>
    );
  }

  const maxPage = Math.max(1, Math.ceil(total / pageSize) || 1);
  const th = {
    padding: '12px 16px',
    textAlign: 'left' as const,
    fontSize: 14,
    fontWeight: 500,
    color: '#374151',
    borderBottom: '1px solid #e5e7eb',
  };
  const td = { padding: '12px 16px', fontSize: 14, color: '#111827', borderBottom: '1px solid #e5e7eb' };

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('adminUsersPage.title')}
        subtitle={t('adminUsersPage.subtitle')}
        actions={
          canCreateUsers ? (
            <button
              type="button"
              onClick={() => {
                const opts = authUser ? creatableUserRolesForActor(authUser.role) : [];
                setShowCreatePassword(false);
                setCreateForm({ username: '', password: '', role: opts[0] ?? 'USER', team_id: '' });
                setShowCreateModal(true);
              }}
              style={{
                padding: '8px 16px',
                fontSize: '14px',
                fontWeight: '500',
                color: '#ffffff',
                backgroundColor: '#2563eb',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              {t('adminUsersPage.createButton')}
            </button>
          ) : undefined
        }
      />

      {error && (
        <div style={{
          padding: '12px',
          marginBottom: '20px',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '6px',
          color: '#dc2626',
          fontSize: '14px',
        }}>
          {error}
        </div>
      )}

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
          <input
            type="search"
            placeholder={t('adminUsersPage.searchPlaceholder')}
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') applySearch();
            }}
            style={{
              flex: '1 1 220px',
              minWidth: 200,
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
            }}
          />
          <button
            type="button"
            onClick={applySearch}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              fontWeight: 500,
              color: '#ffffff',
              backgroundColor: '#2563eb',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
            }}
          >
            {t('adminUsersPage.searchButton')}
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, color: '#6b7280' }}>
            <span>{t('adminUsersPage.pageSizeLabel')}</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPage(1);
                setPageSize(Number(e.target.value));
              }}
              style={{
                padding: '6px 10px',
                border: '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
              }}
            >
              {[10, 20, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
      </ModulePageFilterCard>

      <ModulePageTableCard>
        <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb' }}>
              <th style={{ ...th, minWidth: 132 }}>{t('adminUsersPage.tableAccountId')}</th>
              <th style={th}>{t('adminUsersPage.tableUsername')}</th>
              <th style={{ ...th, minWidth: 120, maxWidth: 220 }}>{t('adminUsersPage.tableTeam')}</th>
              <th style={th}>{t('adminUsersPage.tableRole')}</th>
              <th style={th}>{t('adminUsersPage.tableStatus')}</th>
              <th style={th}>{t('adminUsersPage.tableCreatedAt')}</th>
              <th style={{ ...th, minWidth: 280 }}>{t('adminUsersPage.tableActions')}</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={7} style={{ ...td, textAlign: 'center', color: '#6b7280', padding: 32 }}>
                  {t('adminUsersPage.loadingTable')}
                </td>
              </tr>
            ) : (
              users.map((user) => {
              const rowEffectiveActive =
                typeof user.effective_is_active === 'boolean' ? user.effective_is_active : user.is_active;
              const rowIsSuperAdmin = normalizeRole(user.role) === 'SUPER_ADMIN';
              const isSelf = Boolean(authUser && user.id === authUser.id);
              const ownerScopedMutations = canOwnerScopedUserMutations(
                authUser?.role,
                authUser?.id ?? '',
                user
              );
              const teamAdminScopedMutations = canTeamAdminScopedUserMutations(
                authUser?.role,
                authUser?.id ?? '',
                user
              );
              const canAccountMutate =
                !isSelf &&
                ((canFullUserAdmin && !rowIsSuperAdmin) ||
                  ownerScopedMutations ||
                  teamAdminScopedMutations);
              const canShowDelete =
                !isSelf &&
                ((canFullUserAdmin && !rowIsSuperAdmin) || teamAdminScopedMutations);
              const hasRowActions =
                canFullUserAdmin ||
                canEditOtherUserRole(authUser?.role, authUser?.id ?? '', user) ||
                ownerScopedMutations ||
                teamAdminScopedMutations;
              const btnBase = {
                padding: '4px 8px',
                fontSize: '12px' as const,
                backgroundColor: 'transparent' as const,
                borderRadius: '4px',
                cursor: 'pointer' as const,
              };
              return (
              <tr key={user.id}>
                <td style={{ ...td, minWidth: 132, whiteSpace: 'nowrap' }}>{user.account_id}</td>
                <td style={td}>{user.username}</td>
                <td
                  style={{
                    ...td,
                    maxWidth: 220,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={user.team_name ?? ''}
                >
                  {user.team_name ?? '—'}
                </td>
                <td style={td}>{t(getRoleLabelKey(user.role))}</td>
                <td style={td}>
                  <span
                    title={
                      user.is_active && !rowEffectiveActive
                        ? t('adminUsersPage.statusDisabledByTeamTooltip')
                        : undefined
                    }
                    style={{
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: '500',
                    backgroundColor: rowEffectiveActive ? '#d1fae5' : '#fee2e2',
                    color: rowEffectiveActive ? '#065f46' : '#991b1b',
                  }}
                  >
                    {rowEffectiveActive ? t('adminUsersPage.statusEnabled') : t('adminUsersPage.statusDisabled')}
                  </span>
                </td>
                <td style={{ ...td, color: '#6b7280' }}>
                  {new Date(user.created_at).toLocaleString(undefined)}
                </td>
                <td style={td}>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {!hasRowActions ? (
                      <span style={{ fontSize: '13px', color: '#9ca3af' }}>—</span>
                    ) : null}
                    {canEditOtherUserRole(authUser?.role, authUser?.id ?? '', user) ? (
                      <button
                        type="button"
                        onClick={() => openRoleModal(user)}
                        style={{
                          ...btnBase,
                          color: '#7c3aed',
                          border: '1px solid #7c3aed',
                        }}
                      >
                        {t('adminUsersPage.changeRole')}
                      </button>
                    ) : null}
                    {canAccountMutate ? (
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedUser(user);
                          setResetForm({ new_password: '', new_password_confirm: '' });
                          setShowResetModal(true);
                        }}
                        style={{
                          ...btnBase,
                          color: '#2563eb',
                          border: '1px solid #2563eb',
                        }}
                      >
                        {t('adminUsersPage.resetPassword')}
                      </button>
                    ) : null}
                    {canAccountMutate && user.is_active ? (
                      <button
                        type="button"
                        onClick={() => setDisableConfirmUser(user)}
                        style={{
                          ...btnBase,
                          color: '#b45309',
                          border: '1px solid #d97706',
                        }}
                      >
                        {t('adminUsersPage.disable')}
                      </button>
                    ) : null}
                    {canAccountMutate && !user.is_active ? (
                      <button
                        type="button"
                        onClick={() => setEnableConfirmUser(user)}
                        style={{
                          ...btnBase,
                          color: '#059669',
                          border: '1px solid #059669',
                        }}
                      >
                        {t('adminUsersPage.enable')}
                      </button>
                    ) : null}
                    {canShowDelete ? (
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedUser(user);
                          setShowDeleteModal(true);
                        }}
                        style={{
                          ...btnBase,
                          color: '#dc2626',
                          border: '1px solid #dc2626',
                        }}
                      >
                        {t('adminUsersPage.deleteUser')}
                      </button>
                    ) : null}
                  </div>
                </td>
              </tr>
            );
              })
            )}
            {!loading && users.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ ...td, textAlign: 'center', color: '#6b7280', padding: 32 }}>
                  —
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
        {!loading ? (
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
              padding: '12px 16px',
              borderTop: '1px solid #e5e7eb',
              backgroundColor: '#fafafa',
            }}
          >
            <span style={{ fontSize: 14, color: '#6b7280' }}>
              {t('adminUsersPage.paginationTotal', { total })}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                style={{
                  padding: '6px 12px',
                  fontSize: 13,
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  backgroundColor: page <= 1 ? '#f3f4f6' : '#ffffff',
                  cursor: page <= 1 ? 'not-allowed' : 'pointer',
                  color: '#374151',
                }}
              >
                {t('adminUsersPage.pagePrev')}
              </button>
              <span style={{ fontSize: 13, color: '#6b7280' }}>
                {page} / {maxPage}
              </span>
              <button
                type="button"
                disabled={page >= maxPage}
                onClick={() => setPage((p) => p + 1)}
                style={{
                  padding: '6px 12px',
                  fontSize: 13,
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  backgroundColor: page >= maxPage ? '#f3f4f6' : '#ffffff',
                  cursor: page >= maxPage ? 'not-allowed' : 'pointer',
                  color: '#374151',
                }}
              >
                {t('adminUsersPage.pageNext')}
              </button>
            </div>
          </div>
        ) : null}
        </div>
      </ModulePageTableCard>

      {/* 新建用户弹窗 */}
      {showCreateModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            padding: '24px',
            width: '100%',
            maxWidth: '440px',
          }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px' }}>{t('adminUsersPage.createTitle')}</h2>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>
                {t('adminUsersPage.createTeamLabel')}
              </label>
              {canFullUserAdmin ? (
                <select
                  value={createForm.team_id}
                  onChange={(e) => setCreateForm({ ...createForm, team_id: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                  }}
                >
                  <option value="">{t('adminUsersPage.createTeamPlaceholder')}</option>
                  {teamsForCreate.map((tm) => (
                    <option key={tm.id} value={tm.id}>
                      {tm.name} ({tm.code})
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  readOnly
                  value={
                    createModalTeamsLoading
                      ? t('adminUsersPage.loadingTable')
                      : createModalTeamsError
                        ? t('adminUsersPage.createTeamLoadFailed')
                        : teamAdminPrimaryTeam
                          ? `${teamAdminPrimaryTeam.name} (${teamAdminPrimaryTeam.code})`
                          : t('adminUsersPage.createTeamUnknown')
                  }
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                    backgroundColor: '#f3f4f6',
                    color: '#374151',
                    cursor: 'default',
                  }}
                />
              )}
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>
                {t('adminUsersPage.createAccountLabel')}
              </label>
              <input
                type="text"
                readOnly
                tabIndex={-1}
                value={t('adminUsersPage.createAccountAutoGenerated')}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  cursor: 'default',
                }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>{t('adminUsersPage.createUsernameLabel')}</label>
              <input
                type="text"
                value={createForm.username}
                onChange={(e) => setCreateForm({ ...createForm, username: e.target.value })}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>{t('adminUsersPage.createPasswordLabel')}</label>
              <div style={{ position: 'relative', width: '100%' }}>
                <input
                  type={showCreatePassword ? 'text' : 'password'}
                  value={createForm.password}
                  onChange={(e) => setCreateForm({ ...createForm, password: e.target.value })}
                  autoComplete="new-password"
                  style={{
                    width: '100%',
                    padding: '8px 40px 8px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: '14px',
                    boxSizing: 'border-box',
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowCreatePassword((v) => !v)}
                  aria-label={showCreatePassword ? t('adminUsersPage.createPasswordHide') : t('adminUsersPage.createPasswordShow')}
                  title={showCreatePassword ? t('adminUsersPage.createPasswordHide') : t('adminUsersPage.createPasswordShow')}
                  style={{
                    position: 'absolute',
                    right: 4,
                    top: '50%',
                    transform: 'translateY(-50%)',
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
                  {showCreatePassword ? (
                    <EyeOff size={16} strokeWidth={1.5} />
                  ) : (
                    <Eye size={16} strokeWidth={1.5} />
                  )}
                </button>
              </div>
            </div>
            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>{t('adminUsersPage.createRoleLabel')}</label>
              <select
                value={createForm.role}
                onChange={(e) =>
                  setCreateForm({ ...createForm, role: e.target.value as CreatableUserRole })
                }
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              >
                {creatableRoles.map((r) => (
                  <option key={r} value={r}>
                    {roleSelectLabel(r, t)}
                  </option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setShowCreatePassword(false);
                  setTeamAdminPrimaryTeam(null);
                  setCreateModalTeamsError(false);
                  setCreateForm({
                    username: '',
                    password: '',
                    role: creatableRoles[0] ?? 'USER',
                    team_id: '',
                  });
                }}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#374151',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.cancel')}
              </button>
              <button
                onClick={handleCreate}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#ffffff',
                  backgroundColor: '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('common.create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 修改角色 */}
      {showRoleModal && selectedUser && authUser ? (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '8px',
              padding: '24px',
              width: '100%',
              maxWidth: '400px',
            }}
          >
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>
              {t('adminUsersPage.changeRoleTitle')} — {selectedUser.username}
            </h2>
            <div style={{ marginBottom: '12px', fontSize: '14px', color: '#374151' }}>
              <span style={{ fontWeight: 500 }}>{t('adminUsersPage.changeRoleCurrentLabel')}：</span>
              {t(getRoleLabelKey(selectedUser.role))}
            </div>
            <div style={{ marginBottom: '24px' }}>
              <label
                style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}
              >
                {t('adminUsersPage.changeRoleNewLabel')}
              </label>
              <select
                value={roleEditNew}
                onChange={(e) => setRoleEditNew(e.target.value as CreatableUserRole)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              >
                {assignableUserRolesForEditor(authUser.role, selectedUser.role).map((r) => (
                  <option key={r} value={r}>
                    {roleSelectLabel(r, t)}
                  </option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={() => {
                  setShowRoleModal(false);
                  setSelectedUser(null);
                }}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#374151',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.cancel')}
              </button>
              <button
                type="button"
                onClick={handleSaveRole}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#ffffff',
                  backgroundColor: '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.changeRoleSave')}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {/* 重置密码弹窗 */}
      {showResetModal && selectedUser && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            padding: '24px',
            width: '100%',
            maxWidth: '400px',
          }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px' }}>
              {t('adminUsersPage.resetTitlePrefix')}{selectedUser.username}
            </h2>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>{t('adminUsersPage.resetNewPasswordLabel')}</label>
              <input
                type="password"
                autoComplete="new-password"
                value={resetForm.new_password}
                onChange={(e) => setResetForm({ ...resetForm, new_password: e.target.value })}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px' }}>{t('adminUsersPage.resetConfirmPasswordLabel')}</label>
              <input
                type="password"
                autoComplete="new-password"
                value={resetForm.new_password_confirm}
                onChange={(e) => setResetForm({ ...resetForm, new_password_confirm: e.target.value })}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setShowResetModal(false);
                  setResetForm({ new_password: '', new_password_confirm: '' });
                  setSelectedUser(null);
                }}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#374151',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.cancel')}
              </button>
              <button
                type="button"
                onClick={handleResetPassword}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#ffffff',
                  backgroundColor: '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.confirmResetPassword')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 删除用户确认弹窗 */}
      <ConfirmDialog
        open={Boolean(disableConfirmUser)}
        title={t('adminUsersPage.confirmDisableTitle')}
        description={
          disableConfirmUser
            ? `${t('adminUsersPage.confirmDisableDescription')} ${disableConfirmUser.username}`
            : undefined
        }
        confirmText={t('adminUsersPage.confirm')}
        cancelText={t('adminUsersPage.cancel')}
        loading={toggleActiveLoading}
        onCancel={() => {
          if (toggleActiveLoading) return;
          setDisableConfirmUser(null);
        }}
        onConfirm={handleDisableConfirm}
      />

      <ConfirmDialog
        open={Boolean(enableConfirmUser)}
        title={t('adminUsersPage.confirmEnableTitle')}
        description={
          enableConfirmUser
            ? `${t('adminUsersPage.confirmEnableDescription')} ${enableConfirmUser.username}`
            : undefined
        }
        confirmText={t('adminUsersPage.confirm')}
        cancelText={t('adminUsersPage.cancel')}
        loading={toggleActiveLoading}
        onCancel={() => {
          if (toggleActiveLoading) return;
          setEnableConfirmUser(null);
        }}
        onConfirm={handleEnableConfirm}
      />

      {showDeleteModal && selectedUser && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            padding: '24px',
            width: '100%',
            maxWidth: '400px',
          }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '12px', color: '#dc2626' }}>
              {t('adminUsersPage.confirmDeleteTitle')}
            </h2>
            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '24px' }}>
              {t('adminUsersPage.confirmDeleteDescription')}{' '}
              <strong>{selectedUser.username}</strong>
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setSelectedUser(null);
                }}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#374151',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.cancel')}
              </button>
              <button
                type="button"
                onClick={handleDelete}
                style={{
                  padding: '8px 16px',
                  fontSize: '14px',
                  color: '#ffffff',
                  backgroundColor: '#dc2626',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                {t('adminUsersPage.confirmDeleteAction')}
              </button>
            </div>
          </div>
        </div>
      )}

      {successToast && (
        <div
          style={{
            position: 'fixed',
            bottom: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 20px',
            backgroundColor: '#111827',
            color: '#fff',
            borderRadius: 8,
            fontSize: 14,
            zIndex: 2000,
          }}
        >
          {successToast}
        </div>
      )}
    </ModulePageContainer>
  );
}
