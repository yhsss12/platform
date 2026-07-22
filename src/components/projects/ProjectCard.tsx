'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MoreHorizontal } from 'lucide-react';
import type { Project } from '@/lib/projects/types';
import { useI18n } from '@/components/common/I18nProvider';
import { formatDateTimeByLocale, formatRelativeTimeByLocale } from '@/lib/formatRelativeTime';

function statusStyle(status: Project['status']) {
  if (status === '进行中') {
    return { bg: '#eff6ff', fg: '#1d4ed8', border: '#bfdbfe' };
  }
  if (status === '已暂停') {
    return { bg: '#fffbeb', fg: '#b45309', border: '#fde68a' };
  }
  return { bg: '#f3f4f6', fg: '#4b5563', border: '#e5e7eb' };
}

function getProjectStatusLabel(status: Project['status'], t: (path: string) => string): string {
  if (status === '进行中') return t('adminProjectsPage.statusActive');
  if (status === '已暂停') return t('adminProjectsPage.statusPaused');
  if (status === '已归档') return t('adminProjectsPage.statusArchived');
  return status ?? '—';
}

export default function ProjectCard({
  project,
  stats,
  onEnter,
  onEdit,
  onArchive,
  onDelete,
  canEdit = false,
  canDelete = false,
}: {
  project: Project;
  /** 真实统计（任务数/数据数由 projectId 反查得到）；dataCount 未加载时为 undefined，展示 — */
  stats?: { taskCount: number; dataCount: number | undefined };
  onEnter: () => void;
  onEdit: () => void;
  onArchive: () => void;
  onDelete: () => void;
  canEdit?: boolean;
  canDelete?: boolean;
}) {
  const { t, locale } = useI18n();
  const badge = statusStyle(project.status);
  const tags = (project.tags ?? []).slice(0, 2);
  const taskCount = stats?.taskCount ?? project.tasks?.length ?? 0;
  const datasetCount = stats?.dataCount;
  const datasetDisplay = datasetCount !== undefined ? String(datasetCount) : '—';
  const memberCount =
    project.members && project.members.length > 0 ? project.members.length : (project.memberCount ?? 0);
  const updatedText = useMemo(() => formatRelativeTimeByLocale(project.updatedAt, locale, t), [project.updatedAt, locale, t]);
  const updatedTooltip = useMemo(() => formatDateTimeByLocale(project.updatedAt, locale), [project.updatedAt, locale]);

  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const showMoreMenu = canEdit || canDelete;

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

  return (
    <div
      style={{
        backgroundColor: '#ffffff',
        border: '1px solid #e5e7eb',
        borderRadius: '12px',
        padding: '18px',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
        position: 'relative',
        minHeight: '220px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#cbd5e1';
        e.currentTarget.style.boxShadow = '0 10px 35px rgba(15,23,42,0.06)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#e5e7eb';
        e.currentTarget.style.boxShadow = '0 1px 2px 0 rgba(0, 0, 0, 0.05)';
      }}
    >
      {/* 顶部：名称 + 状态徽标 */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontSize: '16px',
              fontWeight: 700,
              color: '#111827',
              lineHeight: 1.2,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: '100%',
            }}
            title={project.name}
          >
            {project.name}
          </div>
          {tags.length > 0 && (
            <div style={{ display: 'flex', gap: '6px', marginTop: '8px', flexWrap: 'wrap' }}>
              {tags.map((t) => (
                <span
                  key={t}
                  style={{
                    padding: '2px 10px',
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
            </div>
          )}
        </div>

        <span
          style={{
            flexShrink: 0,
            padding: '4px 10px',
            borderRadius: '999px',
            backgroundColor: badge.bg,
            color: badge.fg,
            border: `1px solid ${badge.border}`,
            fontSize: '12px',
            fontWeight: 600,
          }}
        >
          {getProjectStatusLabel(project.status, t)}
        </span>
      </div>

      {/* 中部：三块指标（任务/数据来自 API；成员数优先 members，列表页用 API member_count） */}
      <div style={{ display: 'flex', gap: '10px' }}>
        {[
          { label: t('adminProjectsPage.statTasks'), value: taskCount },
          { label: t('adminProjectsPage.statData'), value: datasetDisplay },
          { label: t('adminProjectsPage.statUsers'), value: memberCount },
        ].map((m) => (
          <div
            key={m.label}
            style={{
              flex: 1,
              border: '1px solid #e5e7eb',
              borderRadius: '10px',
              backgroundColor: '#f9fafb',
              padding: '12px 12px 10px',
              minWidth: 0,
            }}
          >
            <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '6px' }}>{m.label}</div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#111827', lineHeight: 1.1 }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* 底部：最后更新 + 按钮 */}
      <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ fontSize: '12px', color: '#6b7280' }} title={updatedTooltip}>
          {t('adminProjectsPage.lastUpdatedPrefix')}{updatedText}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            type="button"
            onClick={onEnter}
            style={{
              height: '34px',
              padding: '0 14px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: '#2563eb',
              color: '#ffffff',
              fontSize: '13px',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            {t('adminProjectsPage.enterProject')}
          </button>

          {showMoreMenu ? (
            <div ref={menuRef} style={{ position: 'relative' }}>
              <button
                type="button"
                onClick={() => setMenuOpen((v) => !v)}
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#ffffff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#374151',
                }}
                title={t('adminProjectsPage.moreActions')}
              >
                <MoreHorizontal size={18} />
              </button>

              {menuOpen && (
                <div
                  style={{
                    position: 'absolute',
                    right: 0,
                    top: '40px',
                    width: '180px',
                    backgroundColor: '#ffffff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '10px',
                    boxShadow: '0 18px 60px rgba(15,23,42,0.12)',
                    padding: '6px',
                    zIndex: 30,
                  }}
                >
                  {canEdit && (
                    <>
                      <button
                        type="button"
                        onClick={() => {
                          setMenuOpen(false);
                          onEdit();
                        }}
                        style={{
                          width: '100%',
                          textAlign: 'left',
                          padding: '8px 10px',
                          border: 'none',
                          backgroundColor: 'transparent',
                          cursor: 'pointer',
                          borderRadius: '8px',
                          fontSize: '13px',
                          color: '#111827',
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                      >
                        {t('adminProjectsPage.editProject')}
                      </button>

                      {project.status !== '已归档' && (
                        <button
                          type="button"
                          onClick={() => {
                            setMenuOpen(false);
                            onArchive();
                          }}
                          style={{
                            width: '100%',
                            textAlign: 'left',
                            padding: '8px 10px',
                            border: 'none',
                            backgroundColor: 'transparent',
                            cursor: 'pointer',
                            borderRadius: '8px',
                            fontSize: '13px',
                            color: '#111827',
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                        >
                          {t('adminProjectsPage.archiveProject')}
                        </button>
                      )}
                    </>
                  )}

                  {canEdit && canDelete && (
                    <div style={{ height: '1px', backgroundColor: '#f3f4f6', margin: '6px 2px' }} />
                  )}

                  {canDelete && (
                    <button
                      type="button"
                      onClick={() => {
                        setMenuOpen(false);
                        onDelete();
                      }}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: '8px 10px',
                        border: 'none',
                        backgroundColor: 'transparent',
                        cursor: 'pointer',
                        borderRadius: '8px',
                        fontSize: '13px',
                        color: '#dc2626',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#fef2f2')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      {t('adminProjectsPage.deleteProject')}
                    </button>
                  )}
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
