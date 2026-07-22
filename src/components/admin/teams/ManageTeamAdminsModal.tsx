'use client';

import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import type { TeamAdmin, TeamAdminCandidateUser } from '@/lib/teams/types';

const overlay: CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1000,
};

const card: CSSProperties = {
  backgroundColor: '#ffffff',
  borderRadius: 8,
  padding: 24,
  width: '100%',
  maxWidth: 720,
  maxHeight: '90vh',
  overflow: 'auto',
  boxSizing: 'border-box',
};

const th: CSSProperties = {
  padding: '10px 12px',
  textAlign: 'left',
  fontSize: 13,
  fontWeight: 500,
  color: '#374151',
  borderBottom: '1px solid #e5e7eb',
};

const td: CSSProperties = {
  padding: '10px 12px',
  fontSize: 13,
  color: '#111827',
  borderBottom: '1px solid #f3f4f6',
};

type Labels = {
  title: string;
  cancel: string;
  done: string;
  adminsSectionList: string;
  adminsSectionAdd: string;
  selectUserPlaceholder: string;
  addButton: string;
  remove: string;
  tableUsername: string;
  tableStatus: string;
  userStatusActive: string;
  userStatusInactive: string;
  noAdmins: string;
};

type Props = {
  teamId: string;
  teamName: string;
  admins: TeamAdmin[];
  candidates: TeamAdminCandidateUser[];
  /** 拉取当前管理员失败时展示（与候选拉取独立，避免一侧失败清空另一侧） */
  adminsFetchError?: string;
  candidatesFetchError?: string;
  labels: Labels;
  onClose: () => void;
  onApply: (next: TeamAdmin[]) => Promise<void>;
};

export function ManageTeamAdminsModal({
  teamId,
  teamName,
  admins,
  candidates,
  adminsFetchError = '',
  candidatesFetchError = '',
  labels,
  onClose,
  onApply,
}: Props) {
  const [local, setLocal] = useState<TeamAdmin[]>(() => admins.map((a) => ({ ...a, teamId: a.teamId || teamId })));
  const [selectedId, setSelectedId] = useState('');
  const [saving, setSaving] = useState(false);
  const [applyError, setApplyError] = useState('');

  useEffect(() => {
    setLocal(admins.map((a) => ({ ...a, teamId: a.teamId || teamId })));
  }, [admins, teamId]);

  /** 已绑定为团队管理员的用户 ID（与候选 user id 对齐，统一 string） */
  const boundUserIds = useMemo(() => {
    const s = new Set<string>();
    local.forEach((a) => {
      const uid = String(a.userId || '').trim();
      if (uid) s.add(uid);
    });
    return s;
  }, [local]);

  const availableCandidates = useMemo(
    () => candidates.filter((c) => !boundUserIds.has(String(c.id || '').trim())),
    [candidates, boundUserIds],
  );

  const remove = (id: string) => {
    setLocal((prev) => prev.filter((a) => a.id !== id));
  };

  const add = () => {
    const c = availableCandidates.find((x) => x.id === selectedId);
    if (!c || boundUserIds.has(String(c.id || '').trim())) return;
    const newAdmin: TeamAdmin = {
      id: `ta-new-${Date.now()}`,
      teamId,
      userId: c.id,
      username: c.username,
      displayName: c.displayName,
      email: c.email,
      status: c.status,
    };
    setLocal((prev) => [...prev, newAdmin]);
    setSelectedId('');
  };

  const handleDone = async () => {
    setApplyError('');
    setSaving(true);
    try {
      const normalized = local.map((a) => ({ ...a, teamId }));
      await onApply(normalized);
      onClose();
    } catch (e) {
      setApplyError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={overlay} role="dialog" aria-modal>
      <div style={card}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: '0 0 4px', color: '#111827' }}>{labels.title}</h2>
        <p style={{ margin: '0 0 20px', fontSize: 13, color: '#6b7280' }}>{teamName}</p>

        {applyError ? (
          <div
            style={{
              padding: '10px 12px',
              marginBottom: 16,
              backgroundColor: '#fef2f2',
              border: '1px solid #fecaca',
              borderRadius: 6,
              color: '#dc2626',
              fontSize: 13,
            }}
          >
            {applyError}
          </div>
        ) : null}

        {adminsFetchError ? (
          <div
            style={{
              padding: '10px 12px',
              marginBottom: 12,
              backgroundColor: '#fffbeb',
              border: '1px solid #fde68a',
              borderRadius: 6,
              color: '#92400e',
              fontSize: 13,
            }}
          >
            {adminsFetchError}
          </div>
        ) : null}

        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px', color: '#374151' }}>{labels.adminsSectionList}</h3>
        {local.length === 0 && !adminsFetchError ? (
          <p style={{ fontSize: 13, color: '#9ca3af', marginBottom: 20 }}>{labels.noAdmins}</p>
        ) : null}
        {local.length > 0 ? (
          <div
            style={{
              marginBottom: 24,
              border: '1px solid #e5e7eb',
              borderRadius: 8,
              overflow: 'hidden',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f9fafb' }}>
                  <th style={th}>{labels.tableUsername}</th>
                  <th style={th}>{labels.tableStatus}</th>
                  <th style={{ ...th, width: 88 }}>{labels.remove}</th>
                </tr>
              </thead>
              <tbody>
                {local.map((a) => (
                  <tr key={`${a.userId}-${a.id}`}>
                    <td style={td}>
                      <div style={{ fontWeight: 500 }}>
                        {a.displayName && a.displayName !== a.username ? a.displayName : a.username}
                      </div>
                      {a.displayName && a.displayName !== a.username ? (
                        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>{a.username}</div>
                      ) : null}
                      {a.email ? (
                        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>{a.email}</div>
                      ) : null}
                    </td>
                    <td style={td}>
                      <span
                        style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 12,
                          fontWeight: 500,
                          backgroundColor: a.status === 'active' ? '#d1fae5' : '#fee2e2',
                          color: a.status === 'active' ? '#065f46' : '#991b1b',
                        }}
                      >
                        {a.status === 'active' ? labels.userStatusActive : labels.userStatusInactive}
                      </span>
                    </td>
                    <td style={td}>
                      <button
                        type="button"
                        onClick={() => remove(a.id)}
                        style={{
                          padding: '4px 8px',
                          fontSize: 12,
                          color: '#dc2626',
                          backgroundColor: 'transparent',
                          border: '1px solid #dc2626',
                          borderRadius: 4,
                          cursor: 'pointer',
                        }}
                      >
                        {labels.remove}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {candidatesFetchError ? (
          <div
            style={{
              padding: '10px 12px',
              marginBottom: 12,
              backgroundColor: '#fffbeb',
              border: '1px solid #fde68a',
              borderRadius: 6,
              color: '#92400e',
              fontSize: 13,
            }}
          >
            {candidatesFetchError}
          </div>
        ) : null}

        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px', color: '#374151' }}>{labels.adminsSectionAdd}</h3>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 24 }}>
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            style={{
              flex: '1 1 220px',
              minWidth: 200,
              padding: '8px 12px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
            }}
          >
            <option value="">{labels.selectUserPlaceholder}</option>
            {availableCandidates.map((c) => (
              <option key={c.id} value={c.id}>
                {c.displayName && c.displayName !== c.username
                  ? `${c.username} (${c.displayName})`
                  : c.username}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={add}
            disabled={!selectedId}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: selectedId ? '#ffffff' : '#9ca3af',
              backgroundColor: selectedId ? '#2563eb' : '#e5e7eb',
              border: 'none',
              borderRadius: 6,
              cursor: selectedId ? 'pointer' : 'not-allowed',
            }}
          >
            {labels.addButton}
          </button>
        </div>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
          <button
            type="button"
            disabled={saving}
            onClick={onClose}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: '#374151',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              cursor: saving ? 'not-allowed' : 'pointer',
            }}
          >
            {labels.cancel}
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={() => void handleDone()}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: '#ffffff',
              backgroundColor: saving ? '#93c5fd' : '#2563eb',
              border: 'none',
              borderRadius: 6,
              cursor: saving ? 'not-allowed' : 'pointer',
            }}
          >
            {labels.done}
          </button>
        </div>
      </div>
    </div>
  );
}
