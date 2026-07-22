'use client';

import type { CSSProperties } from 'react';
import type { Team } from '@/lib/teams/types';

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
  maxWidth: 480,
  maxHeight: '90vh',
  overflow: 'auto',
  boxSizing: 'border-box',
};

function row(label: string, value: string) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, color: '#111827' }}>{value}</div>
    </div>
  );
}

type Props = {
  team: Team;
  adminCount: number;
  projectCount: number;
  labels: {
    title: string;
    fieldName: string;
    fieldCode: string;
    fieldDescription: string;
    fieldStatus: string;
    fieldCreatedAt: string;
    fieldAdminCount: string;
    fieldUserCount: string;
    fieldProjectCount: string;
    statusActive: string;
    statusInactive: string;
    close: string;
  };
  onClose: () => void;
};

export function TeamDetailModal({ team, adminCount, projectCount, labels, onClose }: Props) {
  const statusLabel = team.status === 'active' ? labels.statusActive : labels.statusInactive;
  return (
    <div style={overlay} role="dialog" aria-modal>
      <div style={card}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: '0 0 20px', color: '#111827' }}>{labels.title}</h2>
        {row(labels.fieldName, team.name)}
        {row(labels.fieldCode, team.code)}
        {row(labels.fieldDescription, team.description.trim() ? team.description : '—')}
        {row(labels.fieldStatus, statusLabel)}
        {row(labels.fieldCreatedAt, new Date(team.createdAt).toLocaleString())}
        {row(labels.fieldAdminCount, String(adminCount))}
        {row(labels.fieldUserCount, String(team.userCount))}
        {row(labels.fieldProjectCount, String(projectCount))}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 20 }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              color: '#374151',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              cursor: 'pointer',
            }}
          >
            {labels.close}
          </button>
        </div>
      </div>
    </div>
  );
}
