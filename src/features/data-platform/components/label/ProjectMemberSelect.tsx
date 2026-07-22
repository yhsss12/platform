'use client';

import { useState, useEffect, useRef } from 'react';
import { fetchProjectMembers, type ProjectMemberItem } from '@/lib/projects/projectApi';

type ProjectMemberSelectProps = {
  projectId: string;
  value: string;
  onChange: (username: string) => void;
  placeholder: string;
  disabled?: boolean;
  error?: string;
  label: string;
};

/**
 * 仅从项目成员（GET /api/projects/:id/members）选择用户名，禁止键盘手输。
 */
export default function ProjectMemberSelect({
  projectId,
  value,
  onChange,
  placeholder,
  disabled = false,
  error,
  label,
}: ProjectMemberSelectProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [members, setMembers] = useState<ProjectMemberItem[]>([]);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const pid = (projectId || '').trim();

  useEffect(() => {
    if (!pid) {
      setMembers([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    fetchProjectMembers(pid)
      .then((res) => {
        if (cancelled) return;
        if (res.ok && res.data?.items) setMembers(res.data.items);
        else setMembers([]);
      })
      .catch(() => {
        if (!cancelled) setMembers([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [pid]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const el = wrapRef.current;
      if (el && !el.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const display = (value || '').trim();
  const canPick = !!pid && !disabled;

  return (
    <div ref={wrapRef} style={{ position: 'relative' }}>
      <div
        style={{
          display: 'block',
          fontSize: '14px',
          fontWeight: 500,
          color: '#374151',
          marginBottom: '8px',
        }}
      >
        {label}
      </div>
      <button
        type="button"
        disabled={!canPick}
        onClick={() => canPick && setOpen((o) => !o)}
        style={{
          width: '100%',
          height: '40px',
          padding: '0 12px',
          backgroundColor: disabled || !pid ? '#f9fafb' : '#ffffff',
          border: error ? '1px solid #ef4444' : '1px solid #d1d5db',
          borderRadius: '6px',
          color: display ? '#111827' : '#9ca3af',
          fontSize: '14px',
          textAlign: 'left',
          cursor: !canPick ? 'not-allowed' : 'pointer',
          boxSizing: 'border-box',
        }}
      >
        {loading && pid ? '加载成员…' : display || (pid ? placeholder : '请先选择所属项目')}
      </button>
      {error && (
        <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>{error}</div>
      )}
      {open && canPick && (
        <div
          role="listbox"
          style={{
            position: 'absolute',
            zIndex: 50,
            left: 0,
            right: 0,
            marginTop: 4,
            maxHeight: 220,
            overflowY: 'auto',
            background: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            boxShadow: '0 10px 30px rgba(0,0,0,0.12)',
          }}
        >
          {members.length === 0 && !loading ? (
            <div style={{ padding: '12px', fontSize: 13, color: '#6b7280' }}>暂无成员</div>
          ) : (
            members.map((m) => {
              const un = (m.username || '').trim();
              if (!un) return null;
              return (
                <button
                  key={m.user_id}
                  type="button"
                  role="option"
                  onClick={() => {
                    onChange(un);
                    setOpen(false);
                  }}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '10px 12px',
                    border: 'none',
                    borderBottom: '1px solid #f3f4f6',
                    background: un === display ? '#eff6ff' : '#fff',
                    textAlign: 'left',
                    cursor: 'pointer',
                    fontSize: 14,
                  }}
                >
                  {un}
                </button>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
