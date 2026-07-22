'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import ProjectTabs, { type ProjectListTabKey } from '@/components/projects/ProjectTabs';
import ProjectGrid from '@/components/projects/ProjectGrid';
import CreateProjectModal from '@/components/projects/CreateProjectModal';
import type { Project } from '@/lib/projects/types';
import type { CreateProjectInput } from '@/lib/projects/createProject';
import * as projectService from '@/lib/projects/projectService';
import { recordProjectActivityAndTouch } from '@/lib/projects/projectService';
import { useAuthStore } from '@/store/authStore';
import { fetchTaskList } from '@/lib/daq/fetchTaskList';
import {
  canCreateProject,
  canDeleteProjectForUser,
  canEditProjectForUser,
  normalizeRole,
  type TeamAdminScope,
} from '@/lib/api/roleLabels';
import { fetchProjectPermissionsContext } from '@/lib/projects/projectApi';
import { getLabelTasks, labelTaskRowToTask } from '@/features/asset-viewer/api/labelApi';
import { getProjectDatasetCount } from '@/lib/dataAssets/datasetQueries';
import { useI18n } from '@/components/common/I18nProvider';

function useToast() {
  const [message, setMessage] = useState<string>('');
  React.useEffect(() => {
    if (!message) return;
    const t = setTimeout(() => setMessage(''), 2200);
    return () => clearTimeout(t);
  }, [message]);
  return { message, show: setMessage };
}

function Toast({ message }: { message: string }) {
  const { t } = useI18n();
  if (!message) return null;
  return (
    <div
      style={{
        position: 'fixed',
        left: '50%',
        bottom: '22px',
        transform: 'translateX(-50%)',
        backgroundColor: 'rgba(17,24,39,0.92)',
        color: '#ffffff',
        padding: '10px 14px',
        borderRadius: '10px',
        fontSize: '13px',
        boxShadow: '0 18px 60px rgba(15,23,42,0.25)',
        zIndex: 1500,
      }}
    >
      {message}
    </div>
  );
}

export default function AdminProjectsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const teamFilterId = (searchParams.get('teamId') || '').trim() || null;
  const teamFilterName = (searchParams.get('teamName') || '').trim() || null;
  const toast = useToast();
  const user = useAuthStore((s) => s.user);
  const userRole = normalizeRole(user?.role);
  const canCreate = canCreateProject(userRole);
  const isMember = userRole === 'USER';
  /** URL 带 teamId 且非普通成员：团队管理「查看项目」等入口，列表数据已按 team_id 过滤，不应再套「我的项目」客户端筛选 */
  const isPrivilegedTeamView = Boolean(teamFilterId) && !isMember;
  const [teamAdminScope, setTeamAdminScope] = useState<TeamAdminScope | undefined>(undefined);
  const { t } = useI18n();

  const [tab, setTab] = useState<ProjectListTabKey>(isMember ? 'shared' : 'my');
  const listTab: ProjectListTabKey = useMemo(() => {
    if (isPrivilegedTeamView) {
      if (tab === 'archived') return 'archived';
      if (tab === 'team_active') return 'team_active';
      return 'team_active';
    }
    return tab;
  }, [isPrivilegedTeamView, tab]);
  const [search, setSearch] = useState('');
  const [projects, setProjects] = useState<Project[]>([]);
  const [createOpen, setCreateOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [deleteInput, setDeleteInput] = useState('');
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [collectionTasks, setCollectionTasks] = useState<{ id: string; projectId?: string }[]>([]);
  const [labelTasks, setLabelTasks] = useState<{ id: string; projectId?: string }[]>([]);
  /** 按 project_id 统计的任务数、数据数（来自 GET /api/projects?with_stats=true） */
  const [projectStatsMap, setProjectStatsMap] = useState<
    Record<string, { label_task_count: number; dataset_count: number }>
  >({});
  /** 兼容：无 stats 时仍用 getProjectDatasetCount 回填数据数 */
  const [datasetCountMap, setDatasetCountMap] = useState<Record<string, number>>({});

  const currentUserId = user?.id ?? '';

  /** 共享项目：非负责人，且在 project_members 有记录（优先接口 viewer 字段，兼容详情页拉过 members 的旧逻辑） */
  const projectQualifiesAsShared = useCallback(
    (p: Project) => {
      const uid = currentUserId.trim();
      const oid = (p.ownerId ?? '').trim();
      if (!uid || oid === uid) return false;
      if (p.viewerInProjectMembers === true) return true;
      if (p.viewerInProjectMembers === false) return false;
      return (p.members ?? []).some((m) => m.id === uid);
    },
    [currentUserId]
  );

  useEffect(() => {
    let alive = true;
    fetchProjectPermissionsContext().then((res) => {
      if (!alive) return;
      if (res.ok && res.data) {
        const raw = res.data.team_admin_team_ids;
        if (raw === null) setTeamAdminScope(null);
        else if (Array.isArray(raw)) setTeamAdminScope(raw);
        else setTeamAdminScope([]);
      } else {
        setTeamAdminScope([]);
      }
    });
    return () => {
      alive = false;
    };
  }, []);

  const getProjectCaps = useCallback(
    (p: Project) => ({
      canEdit: canEditProjectForUser(userRole, p, currentUserId, teamAdminScope),
      canDelete: canDeleteProjectForUser(userRole, p, currentUserId, teamAdminScope),
    }),
    [userRole, currentUserId, teamAdminScope]
  );

  /**
   * 团队管理员创建项目时自动绑定团队（不依赖手动下拉）：
   * - URL ?teamId= 且在辖区列表中 → 用该团队
   * - 仅辖 1 个团队 → 用该 id
   * - 辖多个团队且未带 teamId 参数 → 暂用 team_admin_team_ids 首项（与列表默认可见范围一致；后续可接全局「当前团队」）
   * 超管：仅当带 ?teamId= 时写入 teamId（可选）
   */
  const resolvedTeamIdForCreate = useMemo(() => {
    if (userRole === 'SUPER_ADMIN') {
      return teamFilterId || undefined;
    }
    if (userRole === 'ADMIN') {
      const ids = Array.isArray(teamAdminScope) ? teamAdminScope : [];
      if (teamFilterId && ids.includes(teamFilterId)) return teamFilterId;
      if (ids.length === 1) return ids[0];
      if (ids.length > 1) {
        if (teamFilterId && ids.includes(teamFilterId)) return teamFilterId;
        return ids[0];
      }
      return teamFilterId || undefined;
    }
    return teamFilterId || undefined;
  }, [userRole, teamAdminScope, teamFilterId]);

  const refresh = useCallback(async () => {
    const result = await projectService.listAsync(true, teamFilterId);
    const list = Array.isArray(result) ? result : result.projects;
    const stats = Array.isArray(result) ? undefined : result.stats;
    setProjects(list);
    setProjectStatsMap(stats ?? {});
  }, [teamFilterId]);

  const loadStats = useCallback(async () => {
    try {
      const [tasks, labelRes] = await Promise.all([
        fetchTaskList(),
        getLabelTasks({ limit: 500 }),
      ]);
      setCollectionTasks(tasks);
      if (labelRes.ok && labelRes.data) {
        const mapped = labelRes.data.map((row, i) => labelTaskRowToTask(row, i + 1));
        setLabelTasks(mapped);
      } else {
        setLabelTasks([]);
      }
    } catch {
      setCollectionTasks([]);
      setLabelTasks([]);
    }
  }, []);

  /** 仅当无 stats 时对当前 tab 项目用 getProjectDatasetCount 回填数据数（兼容） */
  const refreshCounts = useCallback((projectList: Project[]) => {
    if (projectList.length === 0) return;
    Promise.all(
      projectList.map((p) => getProjectDatasetCount(p.id, p.name).then((n) => [p.id, n] as const))
    ).then((counts) => {
      setDatasetCountMap((prev) => ({ ...prev, ...Object.fromEntries(counts) }));
    });
  }, []);

  useEffect(() => {
    refresh();
    void loadStats();
  }, [refresh, loadStats]);

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        refresh();
        void loadStats();
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [refresh, loadStats]);

  useEffect(() => {
    if (!isPrivilegedTeamView && tab === 'team_active') {
      setTab(isMember ? 'shared' : 'my');
    }
  }, [isPrivilegedTeamView, tab, isMember]);

  // 当前 tab 的项目列表变化时，只对当前 tab 的项目拉取数据数
  const currentList = useMemo(() => {
    const all = projects;
    if (isPrivilegedTeamView) {
      if (listTab === 'archived') {
        return all.filter((p) => p.status === '已归档');
      }
      return all.filter((p) => p.status !== '已归档');
    }
    if (listTab === 'my') {
      return all.filter((p) => p.ownerId === currentUserId && p.status !== '已归档');
    }
    if (listTab === 'shared') {
      return all.filter((p) => p.status !== '已归档' && projectQualifiesAsShared(p));
    }
    return all.filter((p) => p.status === '已归档');
  }, [projects, listTab, isPrivilegedTeamView, currentUserId, projectQualifiesAsShared]);

  useEffect(() => {
    refreshCounts(currentList);
  }, [currentList, refreshCounts]);

  // 数据资产导入/删除后刷新当前 tab 的数据数
  useEffect(() => {
    const onDatasetsChanged = () => refreshCounts(currentList);
    window.addEventListener('datasets-changed', onDatasetsChanged);
    return () => window.removeEventListener('datasets-changed', onDatasetsChanged);
  }, [currentList, refreshCounts]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return currentList;
    return currentList.filter((p) => {
      const tagStr = (p.tags ?? []).join(' ').toLowerCase();
      return (
        p.name.toLowerCase().includes(q) ||
        p.id.toLowerCase().includes(q) ||
        (tagStr.length > 0 && tagStr.includes(q))
      );
    });
  }, [currentList, search]);

  const statsMap = useMemo(() => {
    const map: Record<string, { taskCount: number; dataCount: number | undefined }> = {};
    filtered.forEach((p) => {
      const collectionCount = collectionTasks.filter(
        (t) => (t as { projectId?: string }).projectId === p.id
      ).length;
      const labelCount = projectStatsMap[p.id]?.label_task_count ?? labelTasks.filter((t) => t.projectId === p.id).length;
      const dataCount = projectStatsMap[p.id]?.dataset_count ?? datasetCountMap[p.id];
      map[p.id] = { taskCount: collectionCount + labelCount, dataCount };
    });
    return map;
  }, [filtered, collectionTasks, labelTasks, projectStatsMap, datasetCountMap]);

  const handleCreateSubmit = useCallback(
    async (input: CreateProjectInput) => {
      if (!canCreate) {
        toast.show(t('adminProjectsPage.noCreatePermissionHint'));
        return;
      }
      if (!user) {
      toast.show(t('login.invalidCredentials'));
        return;
      }
      if (userRole === 'ADMIN' && !resolvedTeamIdForCreate) {
        toast.show(t('adminProjectsPage.createNeedTeamContext'));
        return;
      }
      const full: CreateProjectInput = {
        ...input,
        ownerId: user.id,
        ownerName: user.username,
        ...(userRole === 'ADMIN'
          ? { teamId: resolvedTeamIdForCreate }
          : teamFilterId
            ? { teamId: teamFilterId }
            : {}),
      };
      try {
        const created = await projectService.create(full);
        recordProjectActivityAndTouch(
          created.id,
          'PROJECT_UPDATED',
          `${full.ownerName ?? user?.username} 创建了项目`,
          full.ownerName ?? user?.username ?? '当前用户'
        );
        await refresh();
        toast.show(t('adminProjectsPage.createSuccess'));
        setCreateOpen(false);
      } catch (e) {
        toast.show(`${t('feedback.requestFailed')}: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [canCreate, user, refresh, toast, t, teamFilterId, userRole, resolvedTeamIdForCreate]
  );

  const handleEditSubmit = useCallback(
    async (input: CreateProjectInput) => {
      if (!editingProject) return;
      try {
        await projectService.update(editingProject.id, {
          name: input.name.trim(),
          description: input.description?.trim() || undefined,
          tags: Array.isArray(input.tags) ? input.tags.slice(0, 4) : [],
        });
        recordProjectActivityAndTouch(
          editingProject.id,
          'PROJECT_UPDATED',
          `${user?.username ?? '当前用户'} 更新了项目信息`,
          user?.username ?? '当前用户'
        );
        await refresh();
        toast.show(t('feedback.saveSuccess'));
      } catch (e) {
        toast.show(`${t('feedback.requestFailed')}: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setEditingProject(null);
        setCreateOpen(false);
      }
    },
    [editingProject, refresh, toast, t, user]
  );

  const archiveProject = useCallback(
    async (p: Project) => {
      if (p.status === '已归档') return;
      try {
        await projectService.archive(p.id);
        recordProjectActivityAndTouch(
          p.id,
          'PROJECT_UPDATED',
          `${user?.username ?? '当前用户'} 归档了项目`,
          user?.username ?? '当前用户'
        );
        await refresh();
        toast.show(t('adminProjectsPage.archiveSuccess'));
      } catch (e) {
        toast.show(t('adminProjectsPage.archiveFailed'));
      }
    },
    [refresh, toast, t, user]
  );

  const requestDelete = useCallback(
    (p: Project) => {
      if (!canDeleteProjectForUser(userRole, p, currentUserId, teamAdminScope)) {
        toast.show('当前角色无删除该项目权限');
        return;
      }
      setDeleteTarget(p);
      setDeleteInput('');
      setDeleteOpen(true);
    },
    [userRole, currentUserId, teamAdminScope, toast]
  );

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    if (deleteInput !== deleteTarget.name) return;
    if (deleteLoading) return;
    setDeleteLoading(true);
    setDeleteOpen(false);
    setDeleteTarget(null);
    setDeleteInput('');
    const res = await projectService.removeAsync(deleteTarget.id);
    await refresh();
    toast.show(res.ok ? '已删除' : `删除失败：${res.error ?? ''}`);
    setDeleteLoading(false);
  }, [deleteTarget, deleteInput, deleteLoading, refresh, toast]);

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '14px',
          flexWrap: 'wrap',
          marginBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap', minWidth: 0 }}>
            <h2 style={{ fontSize: '20px', fontWeight: 800, color: '#111827', margin: 0 }}>
              {teamFilterId
                ? `${t('adminProjectsPage.title')} · ${teamFilterName || teamFilterId}`
                : t('adminProjectsPage.title')}
            </h2>
            {teamFilterId ? (
              <button
                type="button"
                onClick={() => router.replace('/admin/projects')}
                style={{
                  fontSize: '13px',
                  fontWeight: 600,
                  color: '#2563eb',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline',
                }}
              >
                查看全部项目
              </button>
            ) : null}
          </div>
          <div style={{ minWidth: '360px', maxWidth: '100%', flex: '0 1 auto' }}>
              <ProjectTabs
                value={listTab}
                onChange={setTab}
                withBorder={false}
                teamScoped={isPrivilegedTeamView}
                disabledKeys={isMember ? ['my'] : []}
                disabledReasons={
                  isMember
                    ? {
                        my: t('adminProjectsPage.noCreatePermissionHint'),
                      }
                    : undefined
                }
              />
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder={t('adminProjectsPage.searchPlaceholder')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: '320px',
              maxWidth: '100%',
              height: '40px',
              padding: '0 12px',
              border: '1px solid #d1d5db',
              borderRadius: '10px',
              fontSize: '14px',
              outline: 'none',
              backgroundColor: '#ffffff',
            }}
          />
            <button
              type="button"
              onClick={() => {
                if (!canCreate) return;
                setEditingProject(null);
                setCreateOpen(true);
              }}
              disabled={!canCreate}
              title={!canCreate ? t('adminProjectsPage.noCreatePermissionHint') : undefined}
            style={{
              height: '40px',
              padding: '0 14px',
              borderRadius: '10px',
              border: 'none',
              backgroundColor: canCreate ? '#2563eb' : '#e5e7eb',
              color: canCreate ? '#ffffff' : '#9ca3af',
              fontSize: '14px',
              cursor: canCreate ? 'pointer' : 'not-allowed',
              fontWeight: 700,
              flexShrink: 0,
            }}
          >
            + {t('adminProjectsPage.createButton')}
          </button>
        </div>
      </div>

      <ProjectGrid
        projects={filtered}
        statsMap={statsMap}
        emptyHintI18nKey={
          isPrivilegedTeamView
            ? listTab === 'archived'
              ? 'adminProjectsPage.emptyHintTeamArchived'
              : 'adminProjectsPage.emptyHintTeamActive'
            : 'adminProjectsPage.emptyHint'
        }
        showCreateInEmpty={false}
        onCreate={() => {
          if (!canCreate) {
            toast.show(t('adminProjectsPage.noCreatePermissionHint'));
            return;
          }
          setEditingProject(null);
          setCreateOpen(true);
        }}
        onEnter={(p) => router.push(`/admin/projects/${p.id}`)}
        onEdit={(p) => {
          setEditingProject(p);
          setCreateOpen(true);
        }}
        onArchive={archiveProject}
        onDelete={requestDelete}
        getCaps={getProjectCaps}
      />

      <CreateProjectModal
        open={createOpen}
        onClose={() => {
          setCreateOpen(false);
          setEditingProject(null);
        }}
        onSubmit={editingProject ? handleEditSubmit : handleCreateSubmit}
        initialValues={
          editingProject
            ? {
                name: editingProject.name,
                description: editingProject.description,
                tags: editingProject.tags,
              }
            : undefined
        }
        title={editingProject ? t('adminProjectsPage.editProject') : t('adminProjectsPage.createTitle')}
        confirmText={editingProject ? t('common.save') : t('adminProjectsPage.confirmCreate')}
      />

      {deleteOpen && deleteTarget && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15,23,42,0.45)',
            zIndex: 1600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '16px',
          }}
          onClick={() => setDeleteOpen(false)}
        >
          <div
            style={{
              width: '520px',
              maxWidth: '96vw',
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              border: '1px solid #e5e7eb',
              boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                padding: '16px 18px',
                borderBottom: '1px solid #e5e7eb',
                display: 'flex',
                justifyContent: 'space-between',
                gap: '10px',
              }}
            >
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>删除项目</div>
              <button
                type="button"
                onClick={() => setDeleteOpen(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '18px' }}
              >
                ✕
              </button>
            </div>
            <div style={{ padding: '18px' }}>
              <div style={{ fontSize: '14px', color: '#111827', marginBottom: '8px' }}>
                {t('adminProjectsPage.deleteDescription')}{' '}
                <span style={{ fontWeight: 800 }}>{deleteTarget.name}</span>
              </div>
              <input
                value={deleteInput}
                onChange={(e) => setDeleteInput(e.target.value)}
                placeholder={t('adminProjectsPage.deleteInputPlaceholder')}
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '10px',
                  fontSize: '14px',
                  outline: 'none',
                  backgroundColor: '#ffffff',
                  boxSizing: 'border-box',
                  marginTop: '10px',
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '16px' }}>
                <button
                  type="button"
                  onClick={() => setDeleteOpen(false)}
                  style={{
                    height: '38px',
                    padding: '0 14px',
                    borderRadius: '10px',
                    border: '1px solid #d1d5db',
                    backgroundColor: '#ffffff',
                    color: '#374151',
                    cursor: 'pointer',
                    fontSize: '14px',
                  }}
                >
                  {t('adminProjectsPage.cancel')}
                </button>
                <button
                  type="button"
                  onClick={confirmDelete}
                  disabled={deleteLoading || deleteInput !== deleteTarget.name}
                  style={{
                    height: '38px',
                    padding: '0 14px',
                    borderRadius: '10px',
                    border: 'none',
                    backgroundColor: !deleteLoading && deleteInput === deleteTarget.name ? '#dc2626' : '#f3f4f6',
                    color: !deleteLoading && deleteInput === deleteTarget.name ? '#ffffff' : '#9ca3af',
                    cursor: !deleteLoading && deleteInput === deleteTarget.name ? 'pointer' : 'not-allowed',
                    fontSize: '14px',
                    fontWeight: 700,
                  }}
                >
                  {deleteLoading ? t('common.loading') : t('adminProjectsPage.confirmDelete')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <Toast message={toast.message} />
    </div>
  );
}
