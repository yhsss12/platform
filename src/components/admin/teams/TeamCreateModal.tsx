'use client';

import { useState } from 'react';
import type { CSSProperties } from 'react';
import type { TeamStatus } from '@/lib/teams/types';

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
  fieldDescription: string;
  fieldStatus: string;
  cancel: string;
  create: string;
  nameRequired: string;
  codeRequired: string;
  statusActive: string;
  statusInactive: string;
};

export type TeamCreatePayload = {
  name: string;
  code: string;
  description: string;
  status: TeamStatus;
};

type Props = {
  labels: Labels;
  onClose: () => void;
  onCreate: (payload: TeamCreatePayload) => Promise<void>;
};

export function TeamCreateModal({ labels, onClose, onCreate }: Props) {
  const [name, setName] = useState('');
  const [code, setCode] = useState('');
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<TeamStatus>('active');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    const n = name.trim();
    const c = code.trim();
    if (!n) {
      setError(labels.nameRequired);
      return;
    }
    if (!c) {
      setError(labels.codeRequired);
      return;
    }
    setError('');
    setSubmitting(true);
    try {
      await onCreate({
        name: n,
        code: c,
        description: description.trim(),
        status,
      });
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
          <input type="text" value={code} onChange={(e) => setCode(e.target.value)} style={inputStyle} />
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
            {labels.create}
          </button>
        </div>
      </div>
    </div>
  );
}
