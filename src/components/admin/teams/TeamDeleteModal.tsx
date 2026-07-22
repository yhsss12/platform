'use client';

import React, { useState } from 'react';
import type { Team } from '@/lib/teams/types';
import { deleteTeamHardApi, type DeleteTeamSummary } from '@/lib/teams/teamsApi';

type Labels = {
  title: string;
  warning: string;
  nameLabel: string;
  namePlaceholder: string;
  cancel: string;
  confirm: string;
  deleting: string;
  nameMismatch: string;
};

export function TeamDeleteModal({
  team,
  labels,
  onClose,
  onSuccess,
}: {
  team: Team;
  labels: Labels;
  onClose: () => void;
  onSuccess: (summary: DeleteTeamSummary) => void;
}) {
  const [nameInput, setNameInput] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleConfirm = async () => {
    const expected = team.name.trim();
    if (nameInput.trim() !== expected) {
      setError(labels.nameMismatch);
      return;
    }
    setError('');
    setLoading(true);
    try {
      const summary = await deleteTeamHardApi(team.id, nameInput.trim());
      onSuccess(summary);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15,23,42,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1600,
        padding: '16px',
      }}
      onClick={(ev) => {
        if (ev.target === ev.currentTarget && !loading) onClose();
      }}
    >
      <div
        style={{
          width: '480px',
          maxWidth: '96vw',
          backgroundColor: '#ffffff',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ padding: '16px 18px', borderBottom: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 16, fontWeight: 800, color: '#b91c1c' }}>{labels.title}</div>
        </div>
        <div style={{ padding: '14px 18px 18px' }}>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.6, marginBottom: 14, whiteSpace: 'pre-wrap' }}>
            {labels.warning}
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', marginBottom: 6 }}>{labels.nameLabel}</div>
          <input
            type="text"
            value={nameInput}
            onChange={(e) => {
              setNameInput(e.target.value);
              setError('');
            }}
            disabled={loading}
            placeholder={labels.namePlaceholder}
            style={{
              width: '100%',
              height: 40,
              padding: '0 12px',
              border: '1px solid #d1d5db',
              borderRadius: 10,
              fontSize: 14,
              boxSizing: 'border-box',
              marginBottom: 10,
            }}
          />
          {error ? (
            <div style={{ fontSize: 12, color: '#b91c1c', marginBottom: 12 }}>{error}</div>
          ) : null}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <button
              type="button"
              disabled={loading}
              onClick={onClose}
              style={{
                height: 38,
                padding: '0 14px',
                borderRadius: 10,
                border: '1px solid #d1d5db',
                backgroundColor: '#fff',
                color: '#374151',
                cursor: loading ? 'default' : 'pointer',
                fontSize: 14,
              }}
            >
              {labels.cancel}
            </button>
            <button
              type="button"
              disabled={loading || !nameInput.trim()}
              onClick={() => void handleConfirm()}
              style={{
                height: 38,
                padding: '0 14px',
                borderRadius: 10,
                border: 'none',
                backgroundColor: loading || !nameInput.trim() ? '#fca5a5' : '#dc2626',
                color: '#fff',
                cursor: loading || !nameInput.trim() ? 'default' : 'pointer',
                fontSize: 14,
                fontWeight: 600,
              }}
            >
              {loading ? labels.deleting : labels.confirm}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
