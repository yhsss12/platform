'use client';

import Link from 'next/link';
import type { ReactNode } from 'react';
import { ArrowLeft } from 'lucide-react';

export function SimAssetBackLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        marginBottom: 16,
        fontSize: 13,
        color: '#2563eb',
        textDecoration: 'none',
      }}
    >
      <ArrowLeft size={16} strokeWidth={1.75} />
      {label}
    </Link>
  );
}

export function SimAssetToast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="status"
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 1000,
        maxWidth: 360,
        padding: '12px 16px',
        borderRadius: 10,
        backgroundColor: '#111827',
        color: '#fff',
        fontSize: 13,
        lineHeight: 1.5,
        boxShadow: '0 8px 24px rgba(15, 23, 42, 0.2)',
      }}
    >
      {message}
    </div>
  );
}

export const formFieldLabelStyle = {
  display: 'block',
  fontSize: 13,
  fontWeight: 500,
  color: '#374151',
  marginBottom: 6,
} as const;

export const formControlStyle = {
  width: '100%',
  padding: '8px 12px',
  borderRadius: 8,
  border: '1px solid #d1d5db',
  fontSize: 14,
  backgroundColor: '#fff',
  boxSizing: 'border-box' as const,
};

export function FormField({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={formFieldLabelStyle}>{label}</label>
      {children}
      {hint ? (
        <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280', lineHeight: 1.4 }}>{hint}</p>
      ) : null}
    </div>
  );
}

export function StepSection({
  step,
  title,
  children,
}: {
  step: number;
  title: string;
  children: ReactNode;
}) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          marginBottom: 12,
        }}
      >
        <span
          style={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            backgroundColor: '#eff6ff',
            color: '#2563eb',
            fontSize: 13,
            fontWeight: 600,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {step}
        </span>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: '#111827' }}>{title}</h3>
      </div>
      {children}
    </div>
  );
}
