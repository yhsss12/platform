'use client';

import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import type { TeamAdminCandidateUser, TeamUserRow } from '@/lib/teams/types';

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
  sectionList: string;
  sectionAdd: string;
  selectUserPlaceholder: string;
  addButton: string;
  remove: string;
  tableUsername: string;
  tableDisplayName: string;
  tableEmail: string;
  tableStatus: string;
  userStatusActive: string;
  userStatusInactive: string;
  empty: string;
};

type Props = {
  teamId: string;
  teamName: string;
  users: TeamUserRow[];
  candidates: TeamAdminCandidateUser[];
  labels: Labels;
  onClose: () => void;
  onApply: (next: TeamUserRow[]) => Promise<void>;
};

export function ManageTeamUsersModal({ teamId, teamName, users, candidates, labels, onClose, onApply }: Props) {
  const [local, setLocal] = useState<TeamUserRow[]>(() => users.map((a) => ({ ...a, teamId: a.teamId || teamId })));
  const [selectedId, setSelectedId] = useState('');
  const [saving, setSaving] = useState(false);
  const [applyError, setApplyError] = useState('');

  useEffect(() => {
    setLocal(users.map((a) => ({ ...a, teamId: a.teamId || teamId })));
  }, [users, teamId]);

  const userIds = useMemo(() => new Set(local.map((a) => a.userId)), [local]);

  const availableCandidates = useMemo(
    () => candidates.filter((c) => !userIds.has(c.id)),
    [candidates, userIds],
  );

  const remove = (id: string) => {
    setLocal((prev) => prev.filter((a) => a.id !== id));
  };

  const add = () => {
    const c = candidates.find((x) => x.id === selectedId);
    if (!c || userIds.has(c.id)) return;
    const row: TeamUserRow = {
      id: `tu-new-${Date.now()}`,
      teamId,
      userId: c.id,
      username: c.username,
      displayName: c.displayName,
      email: c.email,
      status: c.status,
    };
    setLocal((prev) => [...prev, row]);
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

        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px', color: '#374151' }}>{labels.sectionList}</h3>
        {local.length === 0 ? (
          <p style={{ fontSize: 13, color: '#9ca3af', marginBottom: 20 }}>{labels.empty}</p>
        ) : (
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
                  <th style={th}>{labels.tableDisplayName}</th>
                  <th style={th}>{labels.tableEmail}</th>
                  <th style={th}>{labels.tableStatus}</th>
                  <th style={{ ...th, width: 88 }}>{labels.remove}</th>
                </tr>
              </thead>
              <tbody>
                {local.map((a) => (
                  <tr key={a.id}>
                    <td style={td}>{a.username}</td>
                    <td style={td}>{a.displayName}</td>
                    <td style={td}>{a.email}</td>
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
        )}

        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px', color: '#374151' }}>{labels.sectionAdd}</h3>
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
                {c.displayName} ({c.username})
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
