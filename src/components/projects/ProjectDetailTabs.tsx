'use client';

import React, { useMemo, useState, useCallback, useEffect } from 'react';
import Link from 'next/link';
import type { Project, ProjectMember, ProjectMemberRole } from '@/lib/projects/types';
import { getRoleLabel } from '@/lib/projects/roleLabels';
import { formatDateTimeByLocale, formatRelativeTimeByLocale } from '@/lib/formatRelativeTime';
import { getSourceLabel } from '@/features/data-platform/api/dataAssetsApi';
import type { ProjectActivity } from '@/lib/projects/projectActivity';
import { useI18n } from '@/components/common/I18nProvider';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import { getRoleLabelKey, normalizeRole } from '@/lib/api/roleLabels';
import { addProjectMember, createProjectUser, removeProjectMember } from '@/lib/projects/projectApi';
import { fetchTeamUsers } from '@/lib/teams/teamsApi';
import type { TeamUserRow } from '@/lib/teams/types';
import type { TaskFrequencyConfig } from '@/features/data-platform/models/frequencyConfigTypes';

type TFunc = (path: string, vars?: Record<string, string | number>) => string;

/** 项目成员表无独立角色列：展示用角色由主库账号角色映射（与后端 members 列表一致） */
function platformAccountToProjectMemberRole(platformRole: string | undefined): ProjectMemberRole {
  const r = normalizeRole(platformRole || 'USER');
  if (r === 'SUPER_ADMIN' || r === 'ADMIN' || r === 'OWNER') return 'Admin';
  return 'Member';
}

function formatProjectActivity(activity: ProjectActivity, t: TFunc): string {
  switch (activity.type) {
    case 'DATA_IMPORTED': {
      const nMatch = activity.message.match(/(\d+)\s*条/);
      const n = nMatch ? parseInt(nMatch[1], 10) : 1;
      const nameMatch = activity.message.match(/导入数据\s*(.+?)(?:\s*条)?$/);
      if (nameMatch && !nMatch) return t('adminProjectDetailPage.activityImportedFile', { name: nameMatch[1].trim() });
      return t('adminProjectDetailPage.activityImportedCount', { n });
    }
    case 'DATA_DELETED':
      return t('adminProjectDetailPage.activityDeletedAsset');
    default:
      return activity.message;
  }
}

/** 采集任务行（与 DaqTask 兼容），用于项目详情-任务 Tab */
export interface CollectionTaskRow {
  id: string;
  taskNumber?: string;
  taskName: string;
  taskDescription?: string;
  collector?: string;
  collectorName?: string;
  owner?: string;
  deviceName?: string;
  deviceId?: string;
  episodeCount?: number;
  durationSec?: number;
  storagePath?: string;
  storageTypes?: string[];
  remark?: string;
  cameraDataFormat?: string;
  frequencyConfig?: TaskFrequencyConfig;
  createdAt: string;
  updatedAt?: string;
  hasJobs?: boolean;
  completedCount?: number;
}

/** 标注任务行（与 LabelTask 兼容），用于项目详情-任务 Tab */
export interface LabelTaskRow {
  id: string;
  backendTaskId?: string;
  taskNo?: number;
  name: string;
  labeler?: string;
  reviewer?: string;
  deviceType?: string;
  dataCount?: number;
  createdAt: string;
  updatedAt?: string;
  completed?: boolean;
  verified?: boolean;
}

type DetailTabKey = 'overview' | 'tasks' | 'data' | 'members';

function badgeStyle(kind: 'neutral' | 'blue' | 'amber' | 'green' | 'red') {
  if (kind === 'blue') return { bg: '#eff6ff', fg: '#1d4ed8', bd: '#bfdbfe' };
  if (kind === 'amber') return { bg: '#fffbeb', fg: '#b45309', bd: '#fde68a' };
  if (kind === 'green') return { bg: '#ecfdf5', fg: '#047857', bd: '#a7f3d0' };
  if (kind === 'red') return { bg: '#fef2f2', fg: '#b91c1c', bd: '#fecaca' };
  return { bg: '#f3f4f6', fg: '#4b5563', bd: '#e5e7eb' };
}

function StatusBadge({ text, kind }: { text: string; kind: 'neutral' | 'blue' | 'amber' | 'green' | 'red' }) {
  const s = badgeStyle(kind);
  return (
    <span
      style={{
        padding: '3px 10px',
        borderRadius: '999px',
        backgroundColor: s.bg,
        color: s.fg,
        border: `1px solid ${s.bd}`,
        fontSize: '12px',
        fontWeight: 600,
        whiteSpace: 'nowrap',
      }}
    >
      {text}
    </span>
  );
}

function CardShell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '14px 16px',
          borderBottom: '1px solid #e5e7eb',
          backgroundColor: '#f9fafb',
          fontSize: '13px',
          fontWeight: 700,
          color: '#111827',
        }}
      >
        {title}
      </div>
      <div style={{ padding: '16px' }}>{children}</div>
    </div>
  );
}

function formatSize(bytes?: number): string {
  if (bytes == null || bytes === 0) return '—';
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(2)} MB`;
}

function formatFileSize(bytes?: number): string {
  if (bytes == null || bytes === 0) return '—';
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(2)} MB`;
}

type ProjectDetailTabsProps = {
  project: Project;
  onUpdateProject: (patch: Partial<Project>) => void;
  collectionTasks?: CollectionTaskRow[];
  labelTasks?: LabelTaskRow[];
  datasetAssets?: Array<{ id: number; name?: string; filename?: string; format?: string; source?: string; created_at?: string; file_size_bytes?: number }>;
  /** 数据集数（与列表卡片同源，来自 getProjectDatasetCount）；未传则回退到 datasetAssets.length */
  datasetCount?: number;
  /** 为 true 时表示当前数据是用「项目名」fallback 查到的，展示迁移提示条 */
  usedProjectNameFallback?: boolean;
  onDeleteDataset?: (id: number) => void;
  /** 最近活动（项目事件流），按 createdAt 倒序已取好，展示前 5/10 条 */
  projectActivities?: ProjectActivity[];
  /** 是否允许管理成员（邀请/移除等），USER 角色应为 false */
  canManageMembers?: boolean;
  /** 是否允许删除数据资产，USER 角色应为 false */
  canDeleteDataset?: boolean;
};

export default function ProjectDetailTabs(props: ProjectDetailTabsProps) {
  const {
    project,
    onUpdateProject,
    collectionTasks,
    labelTasks,
    datasetAssets,
    datasetCount: datasetCountProp,
    usedProjectNameFallback,
    onDeleteDataset,
    projectActivities,
    canManageMembers = false,
    canDeleteDataset = false,
  } = props;
  const { t, locale } = useI18n();
  const [tab, setTab] = useState<DetailTabKey>('overview');
  const [taskModalOpen, setTaskModalOpen] = useState(false);
  const [taskModalTitle, setTaskModalTitle] = useState('');
  const [taskModalTask, setTaskModalTask] = useState<CollectionTaskRow | null>(null);
  const [convertOpen, setConvertOpen] = useState(false);
  const [convertTo, setConvertTo] = useState<'HDF5' | 'LeRobot'>('HDF5');
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteSelectedUserId, setInviteSelectedUserId] = useState('');
  const [teamUsersForInvite, setTeamUsersForInvite] = useState<TeamUserRow[]>([]);
  const [inviteTeamUsersLoading, setInviteTeamUsersLoading] = useState(false);
  const [inviteError, setInviteError] = useState('');
  const [inviteLoading, setInviteLoading] = useState(false);
  const [createUserOpen, setCreateUserOpen] = useState(false);
  const [createUserDisplayName, setCreateUserDisplayName] = useState('');
  const [createUserPassword, setCreateUserPassword] = useState('');
  const [createUserError, setCreateUserError] = useState('');
  const [createUserLoading, setCreateUserLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState<{ text: string; isError?: boolean } | null>(null);
  const [memberPendingRemove, setMemberPendingRemove] = useState<ProjectMember | null>(null);
  const [memberRemoveDialogOpen, setMemberRemoveDialogOpen] = useState(false);
  const [memberRemoveLoading, setMemberRemoveLoading] = useState(false);

  const showToast = useCallback((text: string, isError?: boolean) => {
    setToastMsg({ text, isError });
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  useEffect(() => {
    if (!inviteOpen) {
      setTeamUsersForInvite([]);
      setInviteTeamUsersLoading(false);
      setInviteSelectedUserId('');
      return;
    }
    const tid = (project.teamId ?? '').trim();
    if (!tid) {
      setTeamUsersForInvite([]);
      setInviteTeamUsersLoading(false);
      return;
    }
    let cancelled = false;
    setInviteTeamUsersLoading(true);
    fetchTeamUsers(tid)
      .then((rows) => {
        if (!cancelled) setTeamUsersForInvite(rows);
      })
      .catch(() => {
        if (!cancelled) setTeamUsersForInvite([]);
      })
      .finally(() => {
        if (!cancelled) setInviteTeamUsersLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [inviteOpen, project.teamId]);

  const eligibleTeamUsersForInvite = useMemo(() => {
    const memberIds = new Set((project.members ?? []).map((m) => m.id));
    return teamUsersForInvite.filter((r) => (r.userId || '').trim() && !memberIds.has(r.userId));
  }, [teamUsersForInvite, project.members]);

  const handleRemoveMember = useCallback(
    (member: ProjectMember) => {
      if (!canManageMembers) return;
      setMemberPendingRemove(member);
      setMemberRemoveDialogOpen(true);
    },
    [canManageMembers]
  );

  const confirmRemoveMember = useCallback(() => {
    if (!memberPendingRemove) return;
    setMemberRemoveLoading(true);
    removeProjectMember(project.id, memberPendingRemove.id)
      .then((res) => {
        if (!res.ok) {
          showToast(res.error || '移除失败，请稍后重试', true);
          return;
        }
        const next = project.members.filter((x) => x.id !== memberPendingRemove.id);
        onUpdateProject({ members: next });
        setMemberRemoveDialogOpen(false);
        setMemberPendingRemove(null);
        showToast('成员已移除');
      })
      .finally(() => setMemberRemoveLoading(false));
  }, [memberPendingRemove, onUpdateProject, project.id, project.members, showToast]);

  const owner = useMemo(() => project.members.find((m) => m.role === 'Owner'), [project.members]);
  const taskCount = (collectionTasks?.length ?? 0) + (labelTasks?.length ?? 0);
  const dataCount = datasetCountProp ?? datasetAssets?.length ?? project.datasets?.length ?? 0;

  // 从数据资产或 project.datasets 统计各格式数量（兼容后端大小写）
  const formatDist = useMemo(() => {
    const raw = datasetAssets ?? project.datasets ?? [];
    const norm = (fmt: string | null | undefined) => (fmt || '').toLowerCase();
    return {
      MCAP: raw.filter((d) => norm((d as { format?: string }).format) === 'mcap').length,
      HDF5: raw.filter((d) => norm((d as { format?: string }).format) === 'hdf5').length,
      LeRobot: raw.filter((d) => norm((d as { format?: string }).format) === 'lerobot').length,
    };
  }, [datasetAssets, project.datasets]);

  const activityIcon = (type: ProjectActivity['type']) => {
    if (type.startsWith('TASK_')) return '📋';
    if (type.startsWith('DATA_')) return '📁';
    if (type.startsWith('MEMBER_') || type === 'PROJECT_UPDATED') return '👤';
    return '•';
  };

  const taskStatusKind = (status: string) => {
    if (status === '进行中') return 'blue';
    if (status === '成功') return 'green';
    if (status === '失败') return 'red';
    return 'neutral';
  };

  const tabButton = (k: DetailTabKey, label: string) => {
    const active = tab === k;
    return (
      <button
        type="button"
        onClick={() => setTab(k)}
        style={{
          padding: '10px 14px',
          border: 'none',
          backgroundColor: 'transparent',
          color: active ? '#2563eb' : '#6b7280',
          fontSize: '14px',
          fontWeight: active ? 600 : 400,
          cursor: 'pointer',
          borderBottom: active ? '2px solid #2563eb' : '2px solid transparent',
          marginBottom: '-1px',
        }}
      >
        {label}
      </button>
    );
  };

  const handleConfirmInvite = async () => {
    const teamId = (project.teamId ?? '').trim();
    if (!teamId) {
      setInviteError(t('adminProjectDetailPage.inviteNoTeamBound'));
      return;
    }
    const uid = inviteSelectedUserId.trim();
    if (!uid) {
      setInviteError(t('adminProjectDetailPage.invitePickUserError'));
      return;
    }
    const row = teamUsersForInvite.find((r) => r.userId === uid);
    if (!row) {
      setInviteError(t('adminProjectDetailPage.inviteErrorNotTeamMember'));
      return;
    }
    if (project.members.some((m) => m.id === uid)) {
      setInviteError(t('adminProjectDetailPage.inviteAlreadyMember'));
      return;
    }
    setInviteError('');
    setInviteLoading(true);
    try {
      const upsertRes = await addProjectMember(project.id, uid);
      if (!upsertRes.ok) {
        setInviteError(upsertRes.error || t('adminProjectDetailPage.inviteFailed'));
        return;
      }
      const now = new Date().toISOString();
      const pr = platformAccountToProjectMemberRole(row.platformRole);
      const newMember: ProjectMember = {
        id: uid,
        name: row.username,
        role: pr,
        addedAt: now,
        lastActiveAt: now,
      };
      onUpdateProject({ members: [...project.members, newMember] });
      setInviteSelectedUserId('');
      setInviteOpen(false);
    } catch {
      setInviteError(t('adminProjectDetailPage.inviteFailed'));
    } finally {
      setInviteLoading(false);
    }
  };

  const handleConfirmCreateUser = async () => {
    const teamId = (project.teamId ?? '').trim();
    if (!teamId) {
      setCreateUserError('项目未关联团队，无法创建用户');
      return;
    }
    const display = createUserDisplayName.trim();
    if (!display) {
      setCreateUserError('请输入展示名（用户名）');
      return;
    }
    if (!createUserPassword.trim()) {
      setCreateUserError('请输入密码');
      return;
    }
    setCreateUserError('');
    setCreateUserLoading(true);
    try {
      const res = await createProjectUser(project.id, { username: display, password: createUserPassword });
      if (!res.ok || !res.data) {
        setCreateUserError(res.error || '创建失败，请稍后重试');
        return;
      }
      const now = new Date().toISOString();
      const newMember: ProjectMember = {
        id: res.data.user_id,
        name: res.data.username,
        role: 'Member',
        addedAt: now,
        lastActiveAt: now,
      };
      onUpdateProject({ members: [...project.members, newMember] });
      setCreateUserDisplayName('');
      setCreateUserPassword('');
      setCreateUserOpen(false);
      showToast(`用户已创建，登录账号：${res.data.account_id}`);
    } catch {
      setCreateUserError('创建失败，请稍后重试');
    } finally {
      setCreateUserLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: '4px', borderBottom: '1px solid #e5e7eb' }}>
        {tabButton('overview', t('adminProjectDetailPage.tabOverview'))}
        {tabButton('tasks', t('adminProjectDetailPage.tabTasks'))}
        {tabButton('data', t('adminProjectDetailPage.tabData'))}
        {tabButton('members', t('adminProjectDetailPage.tabMembers'))}
      </div>

      <div style={{ paddingTop: '16px' }}>
        {tab === 'overview' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
            <CardShell title={t('adminProjectDetailPage.projectInfo')}>
              <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr', rowGap: '10px', columnGap: '12px' }}>
                <div style={{ fontSize: '13px', color: '#6b7280' }}>{t('adminProjectDetailPage.creator')}</div>
                <div style={{ fontSize: '13px', color: '#111827', fontWeight: 600 }}>{owner?.name ?? '—'}</div>
                <div style={{ fontSize: '13px', color: '#6b7280' }}>{t('adminProjectDetailPage.memberCount')}</div>
                <div style={{ fontSize: '13px', color: '#111827', fontWeight: 600 }}>{project.members.length}</div>
                <div style={{ fontSize: '13px', color: '#6b7280' }}>{t('adminProjectDetailPage.createdAt')}</div>
                <div style={{ fontSize: '13px', color: '#111827' }}>{formatDateTimeByLocale(project.createdAt, locale)}</div>
                <div style={{ fontSize: '13px', color: '#6b7280' }}>{t('adminProjectDetailPage.lastUpdated')}</div>
                <div style={{ fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(project.updatedAt, locale)}>
                  {formatRelativeTimeByLocale(project.updatedAt, locale, t)}
                </div>
              </div>
            </CardShell>

            <CardShell title={t('adminProjectDetailPage.stats')}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div style={{ padding: '12px', border: '1px solid #e5e7eb', borderRadius: '10px', backgroundColor: '#f9fafb' }}>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px' }}>{t('adminProjectDetailPage.taskCount')}</div>
                  <div style={{ fontSize: '20px', fontWeight: 800, color: '#111827' }}>{taskCount}</div>
                </div>
                <div style={{ padding: '12px', border: '1px solid #e5e7eb', borderRadius: '10px', backgroundColor: '#f9fafb' }}>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px' }}>{t('adminProjectDetailPage.datasetCount')}</div>
                  <div style={{ fontSize: '20px', fontWeight: 800, color: '#111827' }}>{dataCount}</div>
                </div>
              </div>
            </CardShell>

            <CardShell title={t('adminProjectDetailPage.recentActivity')}>
              {(projectActivities ?? []).length === 0 ? (
                <div style={{ fontSize: '13px', color: '#6b7280' }}>{t('adminProjectDetailPage.noActivity')}</div>
              ) : (
                <div style={{ display: 'grid', gap: '10px' }}>
                  {(projectActivities ?? []).map((a) => (
                    <div
                      key={a.id}
                      style={{
                        border: '1px solid #e5e7eb',
                        borderRadius: '10px',
                        padding: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '12px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
                        <span style={{ fontSize: '16px', flexShrink: 0 }}>{activityIcon(a.type)}</span>
                        <span style={{ fontSize: '13px', color: '#111827' }}>{formatProjectActivity(a, t)}</span>
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280', flexShrink: 0 }} title={formatDateTimeByLocale(a.createdAt, locale)}>
                        {formatRelativeTimeByLocale(a.createdAt, locale, t)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardShell>
          </div>
        )}

        {tab === 'tasks' && (
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e5e7eb',
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
              overflow: 'hidden',
            }}
          >
            {collectionTasks !== undefined && (
              <>
                <div style={{ padding: '12px 16px', backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb', fontSize: '13px', fontWeight: 700, color: '#374151' }}>
                  采集任务
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                      {['任务编号', '任务名称', '负责人', '设备', '数量', '时长', '创建时间', '状态', '操作'].map((h) => (
                        <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {collectionTasks.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ padding: '24px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>暂无采集任务</td>
                      </tr>
                    ) : (
                      collectionTasks.map((row) => (
                        <tr key={row.id} style={{ borderBottom: '1px solid #e5e7eb' }} onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')} onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.taskNumber ?? row.id}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{row.taskName}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.owner ?? '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.deviceName ?? '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.episodeCount ?? 0} 条</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.durationSec != null ? `${row.durationSec} 秒` : '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(row.createdAt, locale)}>{formatRelativeTimeByLocale(row.createdAt, locale, t)}</td>
                          <td style={{ padding: '12px', fontSize: '13px' }}>{row.hasJobs ? <StatusBadge text="有作业" kind="blue" /> : <StatusBadge text="未执行" kind="neutral" />}</td>
                          <td style={{ padding: '12px', fontSize: '13px' }}>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                              <Link href={`/collect/jobs?taskId=${row.id}`} style={{ padding: '6px 12px', border: '1px solid #d1d5db', borderRadius: '8px', backgroundColor: '#fff', color: '#374151', fontSize: '13px', cursor: 'pointer', textDecoration: 'none', display: 'inline-block' }}>查看</Link>
                              <button
                                type="button"
                                onClick={() => { setTaskModalTitle(row.taskName); setTaskModalTask(row); setTaskModalOpen(true); }}
                                style={{ padding: '6px 12px', border: '1px solid #d1d5db', borderRadius: '8px', backgroundColor: '#ffffff', color: '#374151', fontSize: '13px', cursor: 'pointer' }}
                              >
                                详情
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </>
            )}
            {labelTasks !== undefined && (
              <>
                <div style={{ padding: '12px 16px', backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb', fontSize: '13px', fontWeight: 700, color: '#374151', marginTop: collectionTasks !== undefined ? '16px' : 0 }}>
                  标注任务
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                      {['任务编号', '任务名称', '标注员', '审核员', '设备类型', '数量', '创建时间', '状态', '操作'].map((h) => (
                        <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {labelTasks.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ padding: '24px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>暂无标注任务</td>
                      </tr>
                    ) : (
                      labelTasks.map((row) => (
                        <tr key={row.id} style={{ borderBottom: '1px solid #e5e7eb' }} onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')} onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.taskNo != null ? String(row.taskNo).padStart(4, '0') : row.id}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{row.name}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.labeler ?? '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.reviewer ?? '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.deviceType ?? '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{row.dataCount != null ? `${row.dataCount} 条` : '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(row.createdAt, locale)}>{formatRelativeTimeByLocale(row.createdAt, locale, t)}</td>
                          <td style={{ padding: '12px', fontSize: '13px' }}>
                            {row.completed ? <StatusBadge text="已完成" kind="green" /> : row.verified ? <StatusBadge text="已校验" kind="blue" /> : <StatusBadge text="进行中" kind="neutral" />}
                          </td>
                          <td style={{ padding: '12px', fontSize: '13px' }}>
                            <Link href={`/label/execute?taskId=${row.backendTaskId ?? row.id}`} style={{ padding: '6px 12px', border: '1px solid #d1d5db', borderRadius: '8px', backgroundColor: '#fff', color: '#374151', fontSize: '13px', cursor: 'pointer', textDecoration: 'none', display: 'inline-block' }}>查看</Link>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </>
            )}
            {collectionTasks === undefined && labelTasks === undefined ? (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                    {['任务名', '类型', '状态', '创建时间', '最近更新', '操作'].map((h) => (
                      <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(project.tasks ?? []).length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ padding: '40px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
                        暂无任务。可去任务管理创建任务（后续接入）。
                      </td>
                    </tr>
                  ) : (
                    (project.tasks ?? []).map((task) => (
                      <tr
                        key={task.id}
                        style={{ borderBottom: '1px solid #e5e7eb' }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{task.name}</td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{task.type}</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>
                          <StatusBadge text={task.status} kind={taskStatusKind(task.status)} />
                        </td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(task.createdAt, locale)}>
                          {formatRelativeTimeByLocale(task.createdAt, locale, t)}
                        </td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(task.updatedAt, locale)}>
                          {formatRelativeTimeByLocale(task.updatedAt, locale, t)}
                        </td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>
                          <button
                            type="button"
                            onClick={() => { setTaskModalTitle(task.name); setTaskModalOpen(true); }}
                            style={{
                              padding: '6px 12px',
                              border: '1px solid #d1d5db',
                              borderRadius: '8px',
                              backgroundColor: '#ffffff',
                              color: '#374151',
                              fontSize: '13px',
                              cursor: 'pointer',
                            }}
                          >
                            查看
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            ) : null}
          </div>
        )}

        {tab === 'data' && (
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e5e7eb',
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
              overflow: 'hidden',
            }}
          >
            {usedProjectNameFallback && (
              <div
                style={{
                  padding: '10px 14px',
                  backgroundColor: '#fffbeb',
                  borderBottom: '1px solid #fde68a',
                  fontSize: '13px',
                  color: '#b45309',
                }}
              >
                检测到历史数据使用项目名绑定，已临时兼容显示。
              </div>
            )}
            {datasetAssets !== undefined ? (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                    {['编号', '文件名', '格式', '来源', '上传时间', '文件大小', '操作'].map((h) => (
                      <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetAssets.length === 0 ? (
                    <tr>
                      <td colSpan={7} style={{ padding: '40px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
                        暂无数据
                      </td>
                    </tr>
                  ) : (
                    datasetAssets.map((d, idx) => {
                      const fmt = (d.format || '').toLowerCase();
                      const formatLabel = fmt === 'mcap' ? 'MCAP' : fmt === 'lerobot' ? 'LeRobot' : fmt === 'hdf5' ? 'HDF5' : (d.format || '—');
                      return (
                        <tr
                          key={d.id}
                          style={{ borderBottom: '1px solid #e5e7eb' }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                        >
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{String(idx + 1).padStart(4, '0')}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{d.filename ?? d.name}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{formatLabel}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{getSourceLabel(d.source, t)}</td>
                          <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={d.created_at ?? ''}>{d.created_at ? formatRelativeTimeByLocale(d.created_at, locale, t) : '—'}</td>
                          <td style={{ padding: '12px', fontSize: '13px' }}>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <Link
                                href={`/labeling?datasetId=${d.id}`}
                                style={{
                                  padding: '6px 10px',
                                  border: '1px solid #d1d5db',
                                  borderRadius: '8px',
                                  backgroundColor: '#fff',
                                  fontSize: '13px',
                                  cursor: 'pointer',
                                  textDecoration: 'none',
                                  color: '#374151',
                                }}
                              >
                                查看
                              </Link>
                              <button
                                type="button"
                                disabled={!canDeleteDataset || !onDeleteDataset}
                                title={
                                  !canDeleteDataset
                                    ? '当前角色无删除数据权限'
                                    : undefined
                                }
                                onClick={() => {
                                  if (!canDeleteDataset || !onDeleteDataset) return;
                                  onDeleteDataset(d.id);
                                }}
                                style={{
                                  padding: '6px 10px',
                                  border: '1px solid #fecaca',
                                  borderRadius: '8px',
                                  backgroundColor: '#fff',
                                  fontSize: '13px',
                                  cursor: !canDeleteDataset || !onDeleteDataset ? 'not-allowed' : 'pointer',
                                  color: !canDeleteDataset || !onDeleteDataset ? '#9ca3af' : '#dc2626',
                                  opacity: !canDeleteDataset || !onDeleteDataset ? 0.6 : 1,
                                }}
                              >
                                删除
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                    {['数据集名', '格式', '大小', '创建时间', '来源任务', '操作'].map((h) => (
                      <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(project.datasets ?? []).length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ padding: '40px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
                        暂无数据集
                      </td>
                    </tr>
                  ) : (
                    (project.datasets ?? []).map((d) => (
                      <tr
                        key={d.id}
                        style={{ borderBottom: '1px solid #e5e7eb' }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{d.name}</td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{d.format}</td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{formatSize(d.sizeBytes)}</td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(d.createdAt, locale)}>
                          {formatRelativeTimeByLocale(d.createdAt, locale, t)}
                        </td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{d.sourceTaskId ?? '—'}</td>
                        <td style={{ padding: '12px', fontSize: '13px' }}>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <button
                              type="button"
                              onClick={() => {
                                showToast('TODO：下载功能开发中');
                              }}
                              style={{ padding: '6px 10px', border: '1px solid #d1d5db', borderRadius: '8px', backgroundColor: '#fff', fontSize: '13px', cursor: 'pointer' }}
                            >
                              下载
                            </button>
                            <button type="button" onClick={() => setConvertOpen(true)} style={{ padding: '6px 10px', border: '1px solid #d1d5db', borderRadius: '8px', backgroundColor: '#fff', fontSize: '13px', cursor: 'pointer' }}>
                              转换
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            )}
          </div>
        )}

        {tab === 'members' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginBottom: '12px' }}>
              <button
                type="button"
                onClick={() => {
                  if (!canManageMembers) return;
                  setCreateUserError('');
                  setCreateUserOpen(true);
                }}
                disabled={!canManageMembers}
                title={!canManageMembers ? '当前角色无创建用户权限' : undefined}
                style={{
                  height: '38px',
                  padding: '0 14px',
                  borderRadius: '10px',
                  border: '1px solid #d1d5db',
                  backgroundColor: canManageMembers ? '#ffffff' : '#f3f4f6',
                  color: canManageMembers ? '#374151' : '#9ca3af',
                  fontSize: '14px',
                  cursor: canManageMembers ? 'pointer' : 'not-allowed',
                  fontWeight: 600,
                }}
              >
                新建用户
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!canManageMembers) return;
                  setInviteOpen(true);
                }}
                disabled={!canManageMembers}
                title={!canManageMembers ? '当前角色无邀请成员权限' : undefined}
                style={{
                  height: '38px',
                  padding: '0 14px',
                  borderRadius: '10px',
                  border: 'none',
                  backgroundColor: canManageMembers ? '#2563eb' : '#e5e7eb',
                  color: canManageMembers ? '#ffffff' : '#9ca3af',
                  fontSize: '14px',
                  cursor: canManageMembers ? 'pointer' : 'not-allowed',
                  fontWeight: 600,
                }}
              >
                邀请成员
              </button>
            </div>
            <div
              style={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                border: '1px solid #e5e7eb',
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                overflow: 'hidden',
              }}
            >
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                    {['用户名', '角色', '加入时间', '最后活跃', '操作'].map((h) => (
                      <th key={h} style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: 700, color: '#374151' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(project.members ?? []).map((m) => (
                    <tr
                      key={m.id}
                      style={{ borderBottom: '1px solid #e5e7eb' }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <td style={{ padding: '12px', fontSize: '13px', color: '#111827', fontWeight: 600 }}>{m.name}</td>
                      <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>{getRoleLabel(m.role)}</td>
                      <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={formatDateTimeByLocale(m.addedAt, locale)}>
                        {formatRelativeTimeByLocale(m.addedAt, locale, t)}
                      </td>
                      <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }} title={m.lastActiveAt ? formatDateTimeByLocale(m.lastActiveAt, locale) : ''}>
                        {m.lastActiveAt ? formatRelativeTimeByLocale(m.lastActiveAt, locale, t) : '—'}
                      </td>
                      <td style={{ padding: '12px', fontSize: '13px', textAlign: 'center' }}>
                        {m.role === 'Owner' || m.id === project.ownerId ? (
                          '—'
                        ) : (
                          <button
                            type="button"
                            disabled={!canManageMembers}
                            title={!canManageMembers ? '当前角色无移除成员权限' : undefined}
                            onClick={() => handleRemoveMember(m)}
                            style={{
                              padding: '6px 12px',
                              border: '1px solid #fecaca',
                              borderRadius: '8px',
                              color: !canManageMembers ? '#9ca3af' : '#dc2626',
                              backgroundColor: '#ffffff',
                              fontSize: '13px',
                              cursor: !canManageMembers ? 'not-allowed' : 'pointer',
                              opacity: !canManageMembers ? 0.6 : 1,
                            }}
                          >
                            移除
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {taskModalOpen && (
        <div
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15,23,42,0.45)', zIndex: 1200, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
          onClick={() => setTaskModalOpen(false)}
        >
          <div
            style={{ width: '560px', maxWidth: '96vw', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 24px 80px rgba(15,23,42,0.18)', overflow: 'hidden' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>任务详情</div>
              <button type="button" onClick={() => setTaskModalOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '18px' }}>✕</button>
            </div>
            <div style={{ padding: '18px' }}>
              <div style={{ fontSize: '14px', fontWeight: 800, color: '#111827', marginBottom: '12px' }}>{taskModalTitle}</div>
              {taskModalTask ? (
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', rowGap: '10px', columnGap: '12px' }}>
                  <div style={{ fontSize: '13px', color: '#6b7280' }}>任务编号</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{taskModalTask.taskNumber ?? taskModalTask.id}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>任务名称</div>
                  <div style={{ fontSize: '13px', color: '#111827', fontWeight: 600 }}>{taskModalTask.taskName}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>任务描述</div>
                  <div style={{ fontSize: '13px', color: '#111827', whiteSpace: 'pre-wrap' }}>{taskModalTask.taskDescription?.trim() || '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>采集人员</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{(taskModalTask.collectorName || taskModalTask.collector || '').trim() || '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>任务负责人</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{taskModalTask.owner?.trim() || '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>设备</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>
                    {taskModalTask.deviceName?.trim() || '—'}
                    {taskModalTask.deviceId ? <span style={{ color: '#6b7280' }}>（{taskModalTask.deviceId}）</span> : null}
                  </div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>数量</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{taskModalTask.episodeCount != null ? `${taskModalTask.episodeCount} 条` : '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>时长</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{taskModalTask.durationSec != null ? `${taskModalTask.durationSec} 秒` : '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>存储路径</div>
                  <div style={{ fontSize: '13px', color: '#111827', wordBreak: 'break-all' }}>{taskModalTask.storagePath?.trim() || '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>存储类型</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{(taskModalTask.storageTypes ?? []).length ? (taskModalTask.storageTypes ?? []).join('、') : '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>备注</div>
                  <div style={{ fontSize: '13px', color: '#111827', whiteSpace: 'pre-wrap' }}>{taskModalTask.remark?.trim() || '—'}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>创建时间</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{formatDateTimeByLocale(taskModalTask.createdAt, locale)}</div>

                  <div style={{ fontSize: '13px', color: '#6b7280' }}>更新时间</div>
                  <div style={{ fontSize: '13px', color: '#111827' }}>{taskModalTask.updatedAt ? formatDateTimeByLocale(taskModalTask.updatedAt, locale) : '—'}</div>
                </div>
              ) : (
                <div style={{ fontSize: '13px', color: '#6b7280' }}>未选择任务</div>
              )}
            </div>
          </div>
        </div>
      )}

      {convertOpen && (
        <div
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15,23,42,0.45)', zIndex: 1200, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
          onClick={() => setConvertOpen(false)}
        >
          <div
            style={{ width: '520px', maxWidth: '96vw', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 24px 80px rgba(15,23,42,0.18)', overflow: 'hidden' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>转换数据集</div>
              <button type="button" onClick={() => setConvertOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '18px' }}>✕</button>
            </div>
            <div style={{ padding: '18px' }}>
              <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '10px' }}>选择目标格式（仅 UI）</div>
              <select value={convertTo} onChange={(e) => setConvertTo(e.target.value as 'HDF5' | 'LeRobot')} style={{ width: '100%', height: '40px', padding: '0 12px', border: '1px solid #d1d5db', borderRadius: '10px', fontSize: '14px', marginBottom: '16px', boxSizing: 'border-box' }}>
                <option value="HDF5">HDF5</option>
                <option value="LeRobot">LeRobot</option>
              </select>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button type="button" onClick={() => setConvertOpen(false)} style={{ height: '38px', padding: '0 14px', borderRadius: '10px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: 'pointer', fontSize: '14px' }}>取消</button>
                <button
                  type="button"
                  onClick={() => {
                    setConvertOpen(false);
                    showToast(`TODO：转换为 ${convertTo}`);
                  }}
                  style={{ height: '38px', padding: '0 14px', borderRadius: '10px', border: 'none', backgroundColor: '#2563eb', color: '#fff', cursor: 'pointer', fontSize: '14px', fontWeight: 600 }}
                >
                  确认
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={memberRemoveDialogOpen}
        title="移除成员"
        description={memberPendingRemove ? `确认移除成员「${memberPendingRemove.name}」？` : ''}
        loading={memberRemoveLoading}
        confirmText="确认"
        cancelText="取消"
        onCancel={() => {
          if (memberRemoveLoading) return;
          setMemberRemoveDialogOpen(false);
          setMemberPendingRemove(null);
        }}
        onConfirm={confirmRemoveMember}
      />

      {toastMsg && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 14,
            fontWeight: 500,
            zIndex: 1700,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: toastMsg.isError ? '#fef2f2' : 'rgba(17,24,39,0.92)',
            color: toastMsg.isError ? '#b91c1c' : '#fff',
          }}
        >
          {toastMsg.text}
        </div>
      )}

      {createUserOpen && canManageMembers && (
        <div
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15,23,42,0.45)', zIndex: 1200, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
          onClick={() => !createUserLoading && setCreateUserOpen(false)}
        >
          <div
            style={{ width: '520px', maxWidth: '96vw', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 24px 80px rgba(15,23,42,0.18)', overflow: 'hidden' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>新建用户</div>
              <button
                type="button"
                disabled={createUserLoading}
                onClick={() => setCreateUserOpen(false)}
                style={{ background: 'none', border: 'none', cursor: createUserLoading ? 'default' : 'pointer', color: '#6b7280', fontSize: '18px' }}
              >
                ✕
              </button>
            </div>
            <div style={{ padding: '14px 18px 18px' }}>
              <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '6px' }}>展示名（非登录账号）</div>
              <input
                type="text"
                value={createUserDisplayName}
                onChange={(e) => {
                  setCreateUserError('');
                  setCreateUserDisplayName(e.target.value);
                }}
                disabled={createUserLoading}
                placeholder="例如：张三"
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '10px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  marginBottom: '12px',
                }}
              />
              <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '6px' }}>初始密码</div>
              <input
                type="password"
                value={createUserPassword}
                onChange={(e) => {
                  setCreateUserError('');
                  setCreateUserPassword(e.target.value);
                }}
                disabled={createUserLoading}
                placeholder="登录密码"
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '10px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  marginBottom: '12px',
                }}
              />
              <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '10px' }}>
                登录账号由系统按团队规则自动生成，创建成功后在提示中展示。
              </div>
              {createUserError && (
                <div style={{ fontSize: '12px', color: '#b91c1c', marginBottom: '12px' }}>{createUserError}</div>
              )}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  disabled={createUserLoading}
                  onClick={() => setCreateUserOpen(false)}
                  style={{ height: '38px', padding: '0 14px', borderRadius: '10px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: createUserLoading ? 'default' : 'pointer', fontSize: '14px' }}
                >
                  取消
                </button>
                <button
                  type="button"
                  onClick={() => void handleConfirmCreateUser()}
                  disabled={createUserLoading || !(project.teamId ?? '').trim()}
                  style={{
                    height: '38px',
                    padding: '0 14px',
                    borderRadius: '10px',
                    border: 'none',
                    backgroundColor: createUserLoading || !(project.teamId ?? '').trim() ? '#93c5fd' : '#2563eb',
                    color: '#fff',
                    cursor: createUserLoading || !(project.teamId ?? '').trim() ? 'default' : 'pointer',
                    fontSize: '14px',
                    fontWeight: 600,
                  }}
                >
                  {createUserLoading ? '创建中...' : '创建'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {inviteOpen && canManageMembers && (
        <div
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15,23,42,0.45)', zIndex: 1200, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
          onClick={() => setInviteOpen(false)}
        >
          <div
            style={{ width: '520px', maxWidth: '96vw', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 24px 80px rgba(15,23,42,0.18)', overflow: 'hidden' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>邀请成员</div>
              <button type="button" onClick={() => setInviteOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '18px' }}>✕</button>
            </div>
            <div style={{ padding: '14px 18px 18px' }}>
              <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '8px' }}>
                {t('adminProjectDetailPage.inviteSelectUserLabel')}
              </div>
              {inviteTeamUsersLoading ? (
                <div style={{ fontSize: '13px', color: '#6b7280', marginBottom: '12px' }}>
                  {t('adminProjectDetailPage.inviteLoadingTeamUsers')}
                </div>
              ) : null}
              <select
                value={inviteSelectedUserId}
                onChange={(e) => {
                  setInviteError('');
                  setInviteSelectedUserId(e.target.value);
                }}
                disabled={!(project.teamId ?? '').trim() || inviteTeamUsersLoading}
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '10px',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box',
                  marginBottom: '12px',
                  backgroundColor: !(project.teamId ?? '').trim() || inviteTeamUsersLoading ? '#f3f4f6' : '#fff',
                }}
              >
                <option value="">{t('adminProjectDetailPage.inviteSelectPlaceholder')}</option>
                {eligibleTeamUsersForInvite.map((row) => (
                  <option key={row.userId} value={row.userId}>
                    {row.username}
                    {row.platformRole
                      ? ` · ${t(getRoleLabelKey(row.platformRole))}`
                      : ''}
                  </option>
                ))}
              </select>
              {inviteError && (
                <div style={{ fontSize: '12px', color: '#b91c1c', marginBottom: '12px' }}>
                  {inviteError}
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button type="button" onClick={() => setInviteOpen(false)} style={{ height: '38px', padding: '0 14px', borderRadius: '10px', border: '1px solid #d1d5db', backgroundColor: '#fff', color: '#374151', cursor: 'pointer', fontSize: '14px' }}>取消</button>
                <button
                  type="button"
                  onClick={handleConfirmInvite}
                  disabled={
                    inviteLoading ||
                    !(project.teamId ?? '').trim() ||
                    inviteTeamUsersLoading ||
                    !inviteSelectedUserId
                  }
                  style={{
                    height: '38px',
                    padding: '0 14px',
                    borderRadius: '10px',
                    border: 'none',
                    backgroundColor: inviteLoading ? '#93c5fd' : '#2563eb',
                    color: '#fff',
                    cursor: inviteLoading ? 'default' : 'pointer',
                    fontSize: '14px',
                    fontWeight: 600,
                  }}
                >
                  {inviteLoading ? '邀请中...' : '邀请'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
