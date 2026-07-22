'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, MoreHorizontal } from 'lucide-react';
import ProjectDetailTabs, { type CollectionTaskRow, type LabelTaskRow } from '@/components/projects/ProjectDetailTabs';
import type { Project } from '@/lib/projects/types';
import * as projectService from '@/lib/projects/projectService';
import { recordProjectActivityAndTouch } from '@/lib/projects/projectService';
import { formatRelativeTime } from '@/lib/formatRelativeTime';
import { loadDaqTasks } from '@/features/data-platform/storage/daqTasks';
import { fetchTaskList } from '@/lib/daq/fetchTaskList';
import { listJobs } from '@/features/data-platform/api/jobApi';
import { deleteDataAsset, type DataAssetItem } from '@/features/data-platform/api/dataAssetsApi';
import { getProjectDatasetCount, fetchProjectDatasets } from '@/lib/dataAssets/datasetQueries';
import {
  fetchProjectPermissionsContext,
  fetchProjectStats,
  fetchProjectLabelTasks,
  fetchProjectMembers,
} from '@/lib/projects/projectApi';
import { getProjectActivities } from '@/lib/projects/projectActivity';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import { useI18n } from '@/components/common/I18nProvider';
import { useAuthStore } from '@/store/authStore';
import {
  canDeleteProjectForUser,
  canEditProjectForUser,
  canManageProjectMembersForUser,
  normalizeRole,
  type TeamAdminScope,
} from '@/lib/api/roleLabels';

export default function ProjectDetailPage() {
  const params = useParams();
  const projectId = params.projectId as string;
  const router = useRouter();

  const [project, setProject] = useState<Project | null>(null);
  const [collectionTasks, setCollectionTasks] = useState<CollectionTaskRow[]>([]);
  const [labelTasks, setLabelTasks] = useState<LabelTaskRow[]>([]);
  const [datasetAssets, setDatasetAssets] = useState<DataAssetItem[]>([]);
  /** 数据集数（与卡片同源：getProjectDatasetCount），未加载前为 undefined */
  const [datasetCount, setDatasetCount] = useState<number | undefined>(undefined);
  /** 为 true 表示本次数据是用「项目名」fallback 查到的，DB 里存的是项目名而非项目ID */
  const [usedProjectNameFallback, setUsedProjectNameFallback] = useState(false);
  const [projectActivities, setProjectActivities] = useState<Awaited<ReturnType<typeof getProjectActivities>>>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [toastMsg, setToastMsg] = useState('');
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteInput, setDeleteInput] = useState('');
  const [assetDeleteOpen, setAssetDeleteOpen] = useState(false);
  const [assetDeleteId, setAssetDeleteId] = useState<number | null>(null);
  const [assetDeleteLoading, setAssetDeleteLoading] = useState(false);
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);
  const role = normalizeRole(authUser?.role);
  const [teamAdminScope, setTeamAdminScope] = useState<TeamAdminScope | undefined>(undefined);

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

  const projectCaps = useMemo(() => {
    if (!project) {
      return { canEdit: false, canDelete: false, canManageMembers: false };
    }
    const uid = authUser?.id ?? '';
    return {
      canEdit: canEditProjectForUser(role, project, uid, teamAdminScope),
      canDelete: canDeleteProjectForUser(role, project, uid, teamAdminScope),
      canManageMembers: canManageProjectMembersForUser(role, project, uid, teamAdminScope),
    };
  }, [project, role, authUser?.id, teamAdminScope]);

  const { canEdit: canEditProject, canDelete: canDeleteProject, canManageMembers } = projectCaps;
  const canDeleteDataset = canEditProject;

  useEffect(() => {
    let active = true;
    projectService.getAsync(projectId).then((fromApi) => {
      if (!active || !fromApi) return;
      setProject(fromApi);
    });
    return () => {
      active = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return;
    let active = true;
    fetchProjectMembers(projectId).then((res) => {
      if (!active) return;
      if (!res.ok || !res.data) return;
      const now = new Date().toISOString();
      const members = res.data.items.map((m) => ({
        id: m.user_id,
        name: m.username,
        role: m.role,
        addedAt: now,
        lastActiveAt: now,
      }));
      setProject((prev) => (prev ? { ...prev, members } : prev));
    });
    return () => {
      active = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return;
    let active = true;
    (async () => {
      // 仅使用后端按 project_id 拉取该项目下的采集任务；
      // 避免使用 localStorage 的历史数据造成项目/账号之间的可见性串扰。
      const tasksFromApi = await fetchTaskList(projectId);
      const byProject = tasksFromApi.filter((t) => t.projectId === projectId);

      let jobs: Array<{ taskId?: string; collector?: string }> = [];
      try {
        const jobsRes = await listJobs();
        if (jobsRes.ok && jobsRes.data) {
          jobs = jobsRes.data.map((j) => ({ taskId: (j as any).taskId || (j as any).task_id, collector: (j as any).collector || (j as any).operator_name || (j as any).operatorName }));
        }
      } catch {
        jobs = [];
      }
      const collectorsByTaskId = new Map<string, Set<string>>();
      for (const job of jobs) {
        const tid = job.taskId;
        if (!tid) continue;
        const name = String(job.collector ?? '').trim();
        if (!name) continue;
        if (!collectorsByTaskId.has(tid)) collectorsByTaskId.set(tid, new Set<string>());
        collectorsByTaskId.get(tid)!.add(name);
      }

      const enriched = byProject.map((task) => {
        const existing = String(task.collectorName || task.collector || '').trim();
        if (existing) return task;
        const names = Array.from(collectorsByTaskId.get(task.id) ?? []);
        if (names.length === 0) return task;
        return { ...task, collectorName: names.join('、'), collector: names[0] };
      });

      if (active) setCollectionTasks(enriched);
    })();
    return () => {
      active = false;
    };
  }, [projectId]);

  /** 标注任务按 project_id 从后端拉取（表与表通过 project_id 关联） */
  useEffect(() => {
    if (!projectId) return;
    fetchProjectLabelTasks(projectId).then((res) => {
      if (res.ok && res.data) {
        const rows: LabelTaskRow[] = res.data.items.map((item) => ({
          id: String(item.id),
          backendTaskId: item.task_id,
          name: item.name,
          createdAt: item.created_at,
          updatedAt: item.updated_at,
        }));
        setLabelTasks(rows);
      } else setLabelTasks([]);
    });
  }, [projectId]);

  const loadDatasetAssets = useCallback(async () => {
    if (!projectId) return;
    // 优先用已加载的 project（保证有 name 做 fallback），避免首帧 project 未注入时只按 id 查导致 0 条
    const proj = project;
    const projectName = (proj?.name ?? '').trim();
    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      console.log('[project-data] projectId=', proj?.id ?? projectId, 'projectName=', projectName);
    }
    try {
      const { items, usedFallback, rawResp } = await fetchProjectDatasets(projectId, projectName || undefined);
      setDatasetAssets(items);
      setUsedProjectNameFallback(usedFallback);
      // 数量由 fetchProjectStats(project_id) 统一提供，此处不覆盖

      const count = await getProjectDatasetCount(projectId, projectName || undefined);
      if (count > 0 && items.length === 0) {
        try {
          console.error('[project-data] resp=', JSON.stringify(rawResp, null, 2));
        } catch (_) {
          console.error('[project-data] resp (no stringify)=', rawResp);
        }
        setToastMsg('检测到项目存在数据资产，但列表未加载成功，请检查接口返回结构或 project 绑定字段（详见控制台 [project-data] resp）');
      }
    } catch (e) {
      console.error('[项目详情] 加载项目数据资产失败:', e);
      setDatasetAssets([]);
      setDatasetCount(0);
      setUsedProjectNameFallback(false);
    }
  }, [projectId, project]);

  useEffect(() => {
    loadDatasetAssets();
  }, [loadDatasetAssets]);

  // project 从 store 注入后再拉一次，确保 projectName 可用（DB 存项目名时按 name 回退）
  useEffect(() => {
    if (project?.id) loadDatasetAssets();
  }, [project?.id, project?.name, loadDatasetAssets]);

  // project 加载完成后按 project_id 拉取统计（任务数、数据数）
  useEffect(() => {
    if (!projectId) return;
    fetchProjectStats(projectId).then((res) => {
      if (res.ok && res.data) {
        setDatasetCount(res.data.dataset_count);
      }
    });
  }, [projectId]);

  // 页面重新可见时重新拉取数据
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible' && projectId) loadDatasetAssets();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [loadDatasetAssets, projectId]);

  // 数据资产导入/删除后刷新列表与统计
  useEffect(() => {
    const onDatasetsChanged = () => loadDatasetAssets();
    window.addEventListener('datasets-changed', onDatasetsChanged);
    return () => window.removeEventListener('datasets-changed', onDatasetsChanged);
  }, [loadDatasetAssets]);

  useEffect(() => {
    if (projectId && typeof window !== 'undefined') {
      setProjectActivities(getProjectActivities(projectId, 10));
    }
  }, [projectId]);

  useEffect(() => {
    if (!toastMsg) return;
    const t = setTimeout(() => setToastMsg(''), 2200);
    return () => clearTimeout(t);
  }, [toastMsg]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      const el = menuRef.current;
      if (!el) return;
      if (e.target instanceof Node && !el.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [menuOpen]);

  const handleUpdateProject = useCallback(
    async (patch: Partial<Project>) => {
      if (!project) return;
      try {
        const updated = await projectService.update(project.id, patch);
        // 与 projectService 一致：无 members 的 patch 须保留页面已有成员（内存缓存可能仍为 []）。
        setProject((prev) => {
          if (!prev || prev.id !== updated.id) return updated;
          return {
            ...updated,
            members: patch.members !== undefined ? patch.members : prev.members,
          };
        });
        recordProjectActivityAndTouch(project.id, 'PROJECT_UPDATED', '项目信息已更新', '当前用户');
        setProjectActivities(getProjectActivities(project.id, 10));
      } catch (e) {
        setToastMsg(`更新失败：${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [project]
  );

  const archiveProject = useCallback(async () => {
    if (!project || project.status === '已归档') return;
    try {
      const updated = await projectService.archive(project.id);
      setProject(updated);
      setToastMsg(t('adminProjectDetailPage.archiveSuccess'));
    } catch (e) {
      setToastMsg(`归档失败：${e instanceof Error ? e.message : String(e)}`);
    }
  }, [project, t]);

  const deleteProject = useCallback(async () => {
    if (!project) return;
    if (!canDeleteProject) {
      setToastMsg('当前角色无删除项目权限');
      return;
    }
    const res = await projectService.removeAsync(project.id);
    setToastMsg(res.ok ? t('adminProjectDetailPage.deleteSuccess') : `${t('feedback.error')}: ${res.error ?? ''}`);
    router.push('/admin/projects');
  }, [canDeleteProject, project, router, t]);

  const statusBadge = (status: Project['status']) => {
    const styles =
      status === '进行中'
        ? { bg: '#eff6ff', fg: '#1d4ed8', bd: '#bfdbfe' }
        : status === '已暂停'
          ? { bg: '#fffbeb', fg: '#b45309', bd: '#fde68a' }
          : { bg: '#f3f4f6', fg: '#4b5563', bd: '#e5e7eb' };
    return (
      <span
        style={{
          padding: '4px 10px',
          borderRadius: '999px',
          backgroundColor: styles.bg,
          color: styles.fg,
          border: `1px solid ${styles.bd}`,
          fontSize: '12px',
          fontWeight: 700,
          whiteSpace: 'nowrap',
        }}
      >
        {status}
      </span>
    );
  };

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0, flex: 1 }}>
          <button
            type="button"
            onClick={() => router.push('/admin/projects')}
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              border: '1px solid #e5e7eb',
              backgroundColor: '#ffffff',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#374151',
              flexShrink: 0,
            }}
            title={t('adminProjectDetailPage.backTooltip')}
          >
            <ArrowLeft size={18} />
          </button>

          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              <h2
                style={{
                  fontSize: '20px',
                  fontWeight: 800,
                  color: '#111827',
                  margin: 0,
                  maxWidth: '72vw',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={project?.name ?? projectId}
              >
                {project?.name ?? `项目 ${projectId}`}
              </h2>
              {project && statusBadge(project.status)}
            </div>
            {project && (project.tags?.length > 0 || project.description) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px', flexWrap: 'wrap' }}>
                {(project.tags ?? []).slice(0, 4).map((t) => (
                  <span
                    key={t}
                    style={{
                      padding: '2px 8px',
                      borderRadius: '999px',
                      backgroundColor: '#f3f4f6',
                      color: '#4b5563',
                      fontSize: '12px',
                      border: '1px solid #e5e7eb',
                    }}
                  >
                    {t}
                  </span>
                ))}
                {project.description && (
                  <span style={{ fontSize: '13px', color: '#6b7280', maxWidth: '400px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={project.description}>
                    {project.description}
                  </span>
                )}
              </div>
            )}
            {project && (
              <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }} title={project.updatedAt}>
                {t('adminProjectDetailPage.lastUpdatedPrefix')}{formatRelativeTime(project.updatedAt)}
              </div>
            )}
          </div>
        </div>

        {project && (
          <div ref={menuRef} style={{ position: 'relative', flexShrink: 0 }}>
            <button
              type="button"
              onClick={() => {
                if (!canEditProject && !canDeleteProject) return;
                setMenuOpen((v) => !v);
              }}
              disabled={!canEditProject && !canDeleteProject}
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '10px',
                border: '1px solid #e5e7eb',
                backgroundColor: '#ffffff',
                cursor: canEditProject || canDeleteProject ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: canEditProject || canDeleteProject ? '#374151' : '#9ca3af',
                opacity: canEditProject || canDeleteProject ? 1 : 0.6,
              }}
              title={
                canEditProject || canDeleteProject ? '更多操作' : '当前角色无项目管理权限'
              }
            >
              <MoreHorizontal size={18} />
            </button>

            {menuOpen && (canEditProject || canDeleteProject) && (
              <div
                style={{
                  position: 'absolute',
                  right: 0,
                  top: '46px',
                  width: '190px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '10px',
                  boxShadow: '0 18px 60px rgba(15,23,42,0.12)',
                  padding: '6px',
                  zIndex: 40,
                }}
              >
                {canEditProject && (
                  <button
                    type="button"
                    onClick={() => { setMenuOpen(false); setToastMsg(t('adminProjectDetailPage.editProject')); }}
                    style={{ width: '100%', textAlign: 'left', padding: '8px 10px', border: 'none', backgroundColor: 'transparent', cursor: 'pointer', borderRadius: '8px', fontSize: '13px', color: '#111827' }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {t('adminProjectDetailPage.editProject')}
                  </button>
                )}

                {canEditProject && project.status !== '已归档' && (
                  <button
                    type="button"
                  onClick={() => { setMenuOpen(false); archiveProject(); }}
                    style={{ width: '100%', textAlign: 'left', padding: '8px 10px', border: 'none', backgroundColor: 'transparent', cursor: 'pointer', borderRadius: '8px', fontSize: '13px', color: '#111827' }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {t('adminProjectDetailPage.archiveProject')}
                  </button>
                )}

                {canEditProject && canDeleteProject && (
                  <div style={{ height: '1px', backgroundColor: '#f3f4f6', margin: '6px 2px' }} />
                )}

                {canDeleteProject && (
                  <button
                    type="button"
                    onClick={() => {
                      setMenuOpen(false);
                      setDeleteInput('');
                      setDeleteOpen(true);
                    }}
                    style={{ width: '100%', textAlign: 'left', padding: '8px 10px', border: 'none', backgroundColor: 'transparent', cursor: 'pointer', borderRadius: '8px', fontSize: '13px', color: '#dc2626' }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#fef2f2')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {t('adminProjectDetailPage.deleteProject')}
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {!project ? (
        <div
          style={{
            backgroundColor: '#ffffff',
            borderRadius: '12px',
            border: '1px solid #e5e7eb',
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            padding: '40px 24px',
            textAlign: 'center',
            color: '#6b7280',
          }}
        >
          {t('adminProjectDetailPage.notFound')}（ID：{projectId}）
        </div>
      ) : (
        <div
          style={{
            backgroundColor: '#ffffff',
            borderRadius: '12px',
            border: '1px solid #e5e7eb',
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            padding: '16px',
          }}
        >
          <ProjectDetailTabs
            project={project}
            onUpdateProject={handleUpdateProject}
            collectionTasks={collectionTasks}
            labelTasks={labelTasks}
            datasetAssets={datasetAssets}
            datasetCount={datasetCount}
            usedProjectNameFallback={usedProjectNameFallback}
            projectActivities={projectActivities}
            onDeleteDataset={(id) => {
              setAssetDeleteId(id);
              setAssetDeleteOpen(true);
            }}
            canManageMembers={canManageMembers}
            canDeleteDataset={canDeleteDataset}
          />
        </div>
      )}

      {deleteOpen && project && (
        <div
          style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15,23,42,0.45)', zIndex: 1600, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
          onClick={() => setDeleteOpen(false)}
        >
          <div
            style={{ width: '520px', maxWidth: '96vw', backgroundColor: '#ffffff', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 24px 80px rgba(15,23,42,0.18)', overflow: 'hidden' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
              <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>{t('adminProjectDetailPage.deleteTitle')}</div>
              <button type="button" onClick={() => setDeleteOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: '18px' }}>✕</button>
            </div>
            <div style={{ padding: '18px' }}>
              <div style={{ fontSize: '14px', color: '#111827', marginBottom: '8px' }}>
                {t('adminProjectDetailPage.deleteDescription')}{' '}
                <span style={{ fontWeight: 800 }}>{project.name}</span>
              </div>
              <input
                value={deleteInput}
                onChange={(e) => setDeleteInput(e.target.value)}
                placeholder={t('adminProjectDetailPage.deleteInputPlaceholder')}
                style={{ width: '100%', height: '40px', padding: '0 12px', border: '1px solid #d1d5db', borderRadius: '10px', fontSize: '14px', outline: 'none', backgroundColor: '#ffffff', boxSizing: 'border-box', marginTop: '10px' }}
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '16px' }}>
                <button type="button" onClick={() => setDeleteOpen(false)} style={{ height: '38px', padding: '0 14px', borderRadius: '10px', border: '1px solid #d1d5db', backgroundColor: '#ffffff', color: '#374151', cursor: 'pointer', fontSize: '14px' }}>{t('adminProjectDetailPage.cancel')}</button>
                <button
                  type="button"
                  onClick={deleteProject}
                  disabled={deleteInput !== project.name}
                  style={{
                    height: '38px',
                    padding: '0 14px',
                    borderRadius: '10px',
                    border: 'none',
                    backgroundColor: deleteInput === project.name ? '#dc2626' : '#f3f4f6',
                    color: deleteInput === project.name ? '#ffffff' : '#9ca3af',
                    cursor: deleteInput === project.name ? 'pointer' : 'not-allowed',
                    fontSize: '14px',
                    fontWeight: 700,
                  }}
                >
                  {t('adminProjectDetailPage.confirmDelete')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {toastMsg && (
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
          {toastMsg}
        </div>
      )}

      <ConfirmDialog
        open={assetDeleteOpen}
        title="确认删除"
        description="删除后将无法恢复，确定要删除该数据吗？"
        confirmText="删除"
        cancelText="取消"
        loading={assetDeleteLoading}
        onCancel={() => {
          if (assetDeleteLoading) return;
          setAssetDeleteOpen(false);
          setAssetDeleteId(null);
        }}
        onConfirm={async () => {
          if (!assetDeleteId || assetDeleteLoading) return;
          setAssetDeleteLoading(true);
          try {
            const res = await deleteDataAsset(assetDeleteId);
            if (res.ok) {
              recordProjectActivityAndTouch(
                projectId,
                'DATA_DELETED',
                '删除了数据资产',
                '当前用户',
                String(assetDeleteId),
              );
              setToastMsg(t('adminProjectDetailPage.deleteSuccess'));
              await loadDatasetAssets();
              setProjectActivities(getProjectActivities(projectId, 10));
              if (typeof window !== 'undefined') {
                window.dispatchEvent(new Event('datasets-changed'));
              }
            } else {
              setToastMsg(t('adminProjectDetailPage.deleteFailed'));
            }
          } catch (e) {
            setToastMsg(`删除失败：${e instanceof Error ? e.message : String(e)}`);
          } finally {
            setAssetDeleteLoading(false);
            setAssetDeleteOpen(false);
            setAssetDeleteId(null);
          }
        }}
      />
    </div>
  );
}
