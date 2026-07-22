'use client';

import { useState, useEffect } from 'react';
import type { CSSProperties } from 'react';
import type { Team, TeamStatus } from '@/lib/teams/types';

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
  maxWidth: 440,
  boxSizing: 'border-box',
};

const inputStyle: CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  border: '1px solid #d1d5db',
  borderRadius: 6,
  fontSize: 14,
  boxSizing: 'border-box',
};

type Labels = {
  title: string;
  fieldName: string;
  fieldCode: string;
  fieldCodeReadonly: string;
  fieldDescription: string;
  fieldStatus: string;
  cancel: string;
  save: string;
  nameRequired: string;
  statusActive: string;
  statusInactive: string;
};

type Props = {
  team: Team;
  labels: Labels;
  onClose: () => void;
  onSave: (patch: Pick<Team, 'name' | 'description' | 'status'>) => Promise<void>;
};

export function TeamEditModal({ team, labels, onClose, onSave }: Props) {
  const [name, setName] = useState(team.name);
  const [description, setDescription] = useState(team.description);
  const [status, setStatus] = useState<TeamStatus>(team.status);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    setName(team.name);
    setDescription(team.description);
    setStatus(team.status);
  }, [team]);

  const handleSubmit = async () => {
    const n = name.trim();
    if (!n) {
      setError(labels.nameRequired);
      return;
    }
    setError('');
    setSubmitting(true);
    try {
      await onSave({ name: n, description: description.trim(), status });
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={overlay} role="dialog" aria-modal>
      <div style={card}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: '0 0 16px' }}>{labels.title}</h2>
        {error ? (
          <div
            style={{
              padding: '10px 12px',
              marginBottom: 12,
              backgroundColor: '#fef2f2',
              border: '1px solid #fecaca',
              borderRadius: 6,
              color: '#dc2626',
              fontSize: 13,
            }}
          >
            {error}
          </div>
        ) : null}
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, marginBottom: 6 }}>{labels.fieldName}</label>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} style={inputStyle} />
        </div>
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, marginBottom: 6 }}>{labels.fieldCode}</label>
          <input type="text" value={team.code} readOnly style={{ ...inputStyle, backgroundColor: '#f9fafb', color: '#6b7280' }} />
          <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>{labels.fieldCodeReadonly}</div>
        </div>
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, marginBottom: 6 }}>{labels.fieldDescription}</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            style={{ ...inputStyle, resize: 'vertical' as const }}
          />
        </div>
        <div style={{ marginBottom: 20 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, marginBottom: 6 }}>{labels.fieldStatus}</label>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as TeamStatus)}
            style={inputStyle}
          >
            <option value="active">{labels.statusActive}</option>
            <option value="inactive">{labels.statusInactive}</option>
          </select>
        </div>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
          <button
            type="button"
            disabled={submitting}
            onClick={onClose}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: '#374151',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              cursor: submitting ? 'not-allowed' : 'pointer',
            }}
          >
            {labels.cancel}
          </button>
          <button
            type="button"
            disabled={submitting}
            onClick={() => void handleSubmit()}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: '#ffffff',
              backgroundColor: submitting ? '#93c5fd' : '#2563eb',
              border: 'none',
              borderRadius: 6,
              cursor: submitting ? 'not-allowed' : 'pointer',
            }}
          >
            {labels.save}
          </button>
        </div>
      </div>
    </div>
  );
}
