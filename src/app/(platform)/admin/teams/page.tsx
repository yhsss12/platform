'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { isSuperAdmin } from '@/lib/api/roleLabels';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageFilterCard,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import { useI18n } from '@/components/common/I18nProvider';
import type { Team, TeamAdmin, TeamAdminCandidateUser } from '@/lib/teams/types';
import {
  fetchTeams,
  createTeamApi,
  patchTeamApi,
  fetchTeamAdmins,
  addTeamAdminApi,
  removeTeamAdminApi,
  fetchTeamAdminCandidateOptions,
} from '@/lib/teams/teamsApi';
import { TeamDetailModal } from '@/components/admin/teams/TeamDetailModal';
import { TeamCreateModal } from '@/components/admin/teams/TeamCreateModal';
import { TeamEditModal } from '@/components/admin/teams/TeamEditModal';
import { ManageTeamAdminsModal } from '@/components/admin/teams/ManageTeamAdminsModal';
import { TeamDeleteModal } from '@/components/admin/teams/TeamDeleteModal';

type StatusFilter = 'all' | 'active' | 'inactive';

export default function AdminTeamsPage() {
  const { t } = useI18n();
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const isHydrated = useAuthStore((s) => s.isHydrated);
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState('');

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  const [detailTeam, setDetailTeam] = useState<Team | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [editTeam, setEditTeam] = useState<Team | null>(null);
  const [manageTeam, setManageTeam] = useState<Team | null>(null);
  const [manageAdmins, setManageAdmins] = useState<TeamAdmin[]>([]);
  const [manageCandidates, setManageCandidates] = useState<TeamAdminCandidateUser[]>([]);
  const [manageLoading, setManageLoading] = useState(false);
  const [manageAdminsFetchError, setManageAdminsFetchError] = useState('');
  const [manageCandidatesFetchError, setManageCandidatesFetchError] = useState('');
  const [teamToDelete, setTeamToDelete] = useState<Team | null>(null);
  const [deleteResultBanner, setDeleteResultBanner] = useState<{ text: string; isError?: boolean } | null>(null);

  const loadTeams = useCallback(async () => {
    setListError('');
    setLoading(true);
    try {
      const list = await fetchTeams();
      setTeams(list);
    } catch (e) {
      setListError(e instanceof Error ? e.message : String(e));
      setTeams([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isHydrated) return;
    if (!user) {
      router.replace('/login');
      return;
    }
    if (!isSuperAdmin(user.role)) {
      router.replace('/forbidden');
      return;
    }
    void loadTeams();
  }, [isHydrated, user, router, loadTeams]);

  useEffect(() => {
    if (!deleteResultBanner) return;
    const tmr = setTimeout(() => setDeleteResultBanner(null), 6000);
    return () => clearTimeout(tmr);
  }, [deleteResultBanner]);

  useEffect(() => {
    if (!manageTeam) {
      setManageAdmins([]);
      setManageCandidates([]);
      setManageAdminsFetchError('');
      setManageCandidatesFetchError('');
      return;
    }
    let cancelled = false;
    setManageLoading(true);
    setManageAdminsFetchError('');
    setManageCandidatesFetchError('');
    (async () => {
      let admins: TeamAdmin[] = [];
      let adminsOk = false;
      let cands: TeamAdminCandidateUser[] = [];
      try {
        admins = await fetchTeamAdmins(manageTeam.id);
        adminsOk = true;
      } catch (e) {
        if (!cancelled) {
          setManageAdminsFetchError(e instanceof Error ? e.message : String(e));
        }
      }
      try {
        cands = await fetchTeamAdminCandidateOptions(manageTeam.id);
      } catch (e) {
        if (!cancelled) {
          setManageCandidatesFetchError(e instanceof Error ? e.message : String(e));
        }
      }
      if (!cancelled) {
        setManageAdmins(admins);
        setManageCandidates(cands);
        if (adminsOk) {
          setTeams((prev) =>
            prev.map((t) =>
              t.id === manageTeam.id ? { ...t, adminCount: admins.length } : t,
            ),
          );
        }
      }
      if (!cancelled) setManageLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [manageTeam]);

  const filteredTeams = useMemo(() => {
    const q = search.trim().toLowerCase();
    return teams.filter((team) => {
      if (statusFilter === 'active' && team.status !== 'active') return false;
      if (statusFilter === 'inactive' && team.status !== 'inactive') return false;
      if (!q) return true;
      return team.name.toLowerCase().includes(q) || team.code.toLowerCase().includes(q);
    });
  }, [teams, search, statusFilter]);

  const openProjects = (team: Team) => {
    const q = new URLSearchParams();
    q.set('teamId', team.id);
    q.set('teamName', team.name);
    router.push(`/admin/projects?${q.toString()}`);
  };

  const toggleTeamStatus = async (team: Team) => {
    const next = team.status === 'active' ? 'inactive' : 'active';
    try {
      await patchTeamApi(team.id, { status: next });
      await loadTeams();
    } catch {
      /* 静默失败时可加 toast；保持最小实现 */
    }
  };

  const handleApplyAdmins = async (teamId: string, snapshot: TeamAdmin[], next: TeamAdmin[]) => {
    const prevU = new Set(snapshot.map((a) => a.userId));
    const nextU = new Set(next.map((a) => a.userId));
    for (const uid of prevU) {
      if (!nextU.has(uid)) {
        await removeTeamAdminApi(teamId, uid);
      }
    }
    for (const uid of nextU) {
      if (!prevU.has(uid)) {
        await addTeamAdminApi(teamId, uid);
      }
    }
    await loadTeams();
  };

  const detailLabels = {
    title: t('teamsPage.detailTitle'),
    fieldName: t('teamsPage.fieldName'),
    fieldCode: t('teamsPage.fieldCode'),
    fieldDescription: t('teamsPage.fieldDescription'),
    fieldStatus: t('teamsPage.fieldStatus'),
    fieldCreatedAt: t('teamsPage.fieldCreatedAt'),
    fieldAdminCount: t('teamsPage.fieldAdminCount'),
    fieldUserCount: t('teamsPage.fieldUserCount'),
    fieldProjectCount: t('teamsPage.fieldProjectCount'),
    statusActive: t('teamsPage.statusLabelActive'),
    statusInactive: t('teamsPage.statusLabelInactive'),
    close: t('teamsPage.close'),
  };

  const createLabels = {
    title: t('teamsPage.createTitle'),
    fieldName: t('teamsPage.fieldName'),
    fieldCode: t('teamsPage.fieldCode'),
    fieldDescription: t('teamsPage.fieldDescription'),
    fieldStatus: t('teamsPage.fieldStatus'),
    cancel: t('teamsPage.cancel'),
    create: t('teamsPage.create'),
    nameRequired: t('teamsPage.nameRequired'),
    codeRequired: t('teamsPage.codeRequired'),
    statusActive: t('teamsPage.statusLabelActive'),
    statusInactive: t('teamsPage.statusLabelInactive'),
  };

  const editLabels = {
    title: t('teamsPage.editTitle'),
    fieldName: t('teamsPage.fieldName'),
    fieldCode: t('teamsPage.fieldCode'),
    fieldCodeReadonly: t('teamsPage.fieldCodeReadonly'),
    fieldDescription: t('teamsPage.fieldDescription'),
    fieldStatus: t('teamsPage.fieldStatus'),
    cancel: t('teamsPage.cancel'),
    save: t('teamsPage.save'),
    nameRequired: t('teamsPage.nameRequired'),
    statusActive: t('teamsPage.statusLabelActive'),
    statusInactive: t('teamsPage.statusLabelInactive'),
  };

  const manageLabels = {
    title: t('teamsPage.manageAdminsTitle'),
    cancel: t('teamsPage.cancel'),
    done: t('teamsPage.done'),
    adminsSectionList: t('teamsPage.adminsSectionList'),
    adminsSectionAdd: t('teamsPage.adminsSectionAdd'),
    selectUserPlaceholder: t('teamsPage.selectUserPlaceholder'),
    addButton: t('teamsPage.addButton'),
    remove: t('teamsPage.remove'),
    tableUsername: t('teamsPage.tableUsername'),
    tableStatus: t('teamsPage.tableStatus'),
    userStatusActive: t('teamsPage.userStatusActive'),
    userStatusInactive: t('teamsPage.userStatusInactive'),
    noAdmins: t('teamsPage.noAdmins'),
  };

  const th = {
    padding: '12px 16px',
    textAlign: 'left' as const,
    fontSize: 14,
    fontWeight: 500,
    color: '#374151',
    borderBottom: '1px solid #e5e7eb',
  };

  const td = { padding: '12px 16px', fontSize: 14, color: '#111827', borderBottom: '1px solid #e5e7eb' };

  const linkBtn = {
    padding: '4px 8px',
    fontSize: 12,
    color: '#2563eb',
    backgroundColor: 'transparent',
    border: '1px solid #2563eb',
    borderRadius: 4,
    cursor: 'pointer',
    marginRight: 6,
    marginBottom: 4,
  };

  if (!isHydrated || !user) {
    return (
      <ModulePageContainer>
        <div style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>{t('common.loading')}</div>
      </ModulePageContainer>
    );
  }
  if (!isSuperAdmin(user.role)) {
    return (
      <ModulePageContainer>
        <div style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>{t('common.loading')}</div>
      </ModulePageContainer>
    );
  }

  if (loading) {
    return (
      <ModulePageContainer>
        <div style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>{t('common.loading')}</div>
      </ModulePageContainer>
    );
  }

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('teamsPage.title')}
        actions={
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
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
            {t('teamsPage.newTeam')}
          </button>
        }
      />

      {deleteResultBanner ? (
        <div
          style={{
            padding: '12px',
            marginBottom: 16,
            backgroundColor: deleteResultBanner.isError ? '#fef2f2' : '#ecfdf5',
            border: `1px solid ${deleteResultBanner.isError ? '#fecaca' : '#a7f3d0'}`,
            borderRadius: 6,
            color: deleteResultBanner.isError ? '#dc2626' : '#047857',
            fontSize: 14,
            whiteSpace: 'pre-wrap',
          }}
        >
          {deleteResultBanner.text}
        </div>
      ) : null}

      {listError ? (
        <div
          style={{
            padding: '12px',
            marginBottom: 20,
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 6,
            color: '#dc2626',
            fontSize: 14,
          }}
        >
          {listError}
        </div>
      ) : null}

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
          <input
            type="search"
            placeholder={t('teamsPage.searchPlaceholder')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              flex: '1 1 220px',
              minWidth: 200,
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 14, color: '#6b7280' }}>{t('teamsPage.colStatus')}</span>
            {(['all', 'active', 'inactive'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setStatusFilter(k)}
                style={{
                  padding: '6px 12px',
                  fontSize: 13,
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  cursor: 'pointer',
                  backgroundColor: statusFilter === k ? '#eff6ff' : '#ffffff',
                  color: statusFilter === k ? '#1d4ed8' : '#374151',
                }}
              >
                {k === 'all' ? t('teamsPage.statusAll') : k === 'active' ? t('teamsPage.statusActive') : t('teamsPage.statusInactive')}
              </button>
            ))}
          </div>
        </div>
      </ModulePageFilterCard>

      <ModulePageTableCard>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ backgroundColor: '#f9fafb' }}>
                <th style={th}>{t('teamsPage.colName')}</th>
                <th style={th}>{t('teamsPage.colCode')}</th>
                <th style={th}>{t('teamsPage.colStatus')}</th>
                <th style={th}>{t('teamsPage.colAdmins')}</th>
                <th style={th}>{t('teamsPage.colUsers')}</th>
                <th style={th}>{t('teamsPage.colProjects')}</th>
                <th style={th}>{t('teamsPage.colCreated')}</th>
                <th style={{ ...th, minWidth: 340 }}>{t('teamsPage.colActions')}</th>
              </tr>
            </thead>
            <tbody>
              {filteredTeams.map((team) => (
                <tr key={team.id}>
                  <td style={td}>{team.name}</td>
                  <td style={td}>
                    <code style={{ fontSize: 13, color: '#4b5563' }}>{team.code}</code>
                  </td>
                  <td style={td}>
                    <span
                      style={{
                        padding: '4px 8px',
                        borderRadius: 4,
                        fontSize: 12,
                        fontWeight: 500,
                        backgroundColor: team.status === 'active' ? '#d1fae5' : '#fee2e2',
                        color: team.status === 'active' ? '#065f46' : '#991b1b',
                      }}
                    >
                      {team.status === 'active' ? t('teamsPage.statusLabelActive') : t('teamsPage.statusLabelInactive')}
                    </span>
                  </td>
                  <td style={td}>{team.adminCount}</td>
                  <td style={td}>{team.userCount}</td>
                  <td style={td}>{team.projectCount}</td>
                  <td style={{ ...td, color: '#6b7280', whiteSpace: 'nowrap' }}>{new Date(team.createdAt).toLocaleString()}</td>
                  <td style={td}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      <button type="button" style={linkBtn} onClick={() => setDetailTeam(team)}>
                        {t('teamsPage.actionDetail')}
                      </button>
                      <button type="button" style={linkBtn} onClick={() => openProjects(team)}>
                        {t('teamsPage.actionProjects')}
                      </button>
                      <button type="button" style={linkBtn} onClick={() => setEditTeam(team)}>
                        {t('teamsPage.actionEdit')}
                      </button>
                      <button type="button" style={linkBtn} onClick={() => setManageTeam(team)}>
                        {t('teamsPage.actionManageAdmins')}
                      </button>
                      <button
                        type="button"
                        style={{
                          ...linkBtn,
                          color: team.status === 'active' ? '#dc2626' : '#059669',
                          borderColor: team.status === 'active' ? '#dc2626' : '#059669',
                        }}
                        onClick={() => void toggleTeamStatus(team)}
                      >
                        {team.status === 'active' ? t('teamsPage.actionDisable') : t('teamsPage.actionEnable')}
                      </button>
                      <button
                        type="button"
                        style={{
                          ...linkBtn,
                          color: '#dc2626',
                          borderColor: '#dc2626',
                          backgroundColor: '#fef2f2',
                        }}
                        onClick={() => setTeamToDelete(team)}
                      >
                        {t('teamsPage.actionDelete')}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredTeams.length === 0 ? (
            <div style={{ padding: 32, textAlign: 'center', color: '#6b7280', fontSize: 14 }}>—</div>
          ) : null}
        </div>
      </ModulePageTableCard>

      {detailTeam ? (
        <TeamDetailModal
          team={detailTeam}
          adminCount={detailTeam.adminCount}
          projectCount={detailTeam.projectCount}
          labels={detailLabels}
          onClose={() => setDetailTeam(null)}
        />
      ) : null}

      {createOpen ? (
        <TeamCreateModal
          labels={createLabels}
          onClose={() => setCreateOpen(false)}
          onCreate={async (payload) => {
            await createTeamApi(payload);
            await loadTeams();
          }}
        />
      ) : null}

      {editTeam ? (
        <TeamEditModal
          team={editTeam}
          labels={editLabels}
          onClose={() => setEditTeam(null)}
          onSave={async (patch) => {
            await patchTeamApi(editTeam.id, patch);
            await loadTeams();
          }}
        />
      ) : null}

      {manageTeam && manageLoading ? (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.35)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 999,
            fontSize: 14,
            color: '#374151',
          }}
        >
          {t('common.loading')}
        </div>
      ) : null}

      {manageTeam && !manageLoading ? (
        <ManageTeamAdminsModal
          key={manageTeam.id}
          teamId={manageTeam.id}
          teamName={manageTeam.name}
          admins={manageAdmins}
          candidates={manageCandidates}
          adminsFetchError={manageAdminsFetchError}
          candidatesFetchError={manageCandidatesFetchError}
          labels={manageLabels}
          onClose={() => setManageTeam(null)}
          onApply={async (next) => {
            const snapshot = [...manageAdmins];
            await handleApplyAdmins(manageTeam.id, snapshot, next);
          }}
        />
      ) : null}

      {teamToDelete ? (
        <TeamDeleteModal
          team={teamToDelete}
          labels={{
            title: t('teamsPage.deleteTeamTitle'),
            warning: t('teamsPage.deleteTeamWarning'),
            nameLabel: t('teamsPage.deleteTeamNameLabel'),
            namePlaceholder: t('teamsPage.deleteTeamNamePlaceholder'),
            cancel: t('teamsPage.cancel'),
            confirm: t('teamsPage.deleteTeamConfirm'),
            deleting: t('teamsPage.deleteTeamDeleting'),
            nameMismatch: t('teamsPage.deleteTeamNameMismatch'),
          }}
          onClose={() => setTeamToDelete(null)}
          onSuccess={(summary) => {
            void loadTeams();
            let text = t('teamsPage.deleteTeamSuccess', {
              name: summary.team_name,
              projects: summary.projects_deleted,
              orphanUsers: summary.users_deleted,
            });
            if (summary.storage_warnings?.length) {
              text += `\n${t('teamsPage.deleteTeamStorageHint', {
                detail: summary.storage_warnings.join('；'),
              })}`;
            }
            setDeleteResultBanner({ text });
          }}
        />
      ) : null}
    </ModulePageContainer>
  );
}
