'use client';

import React, { useEffect, useMemo, useState } from 'react';
import type { Project } from '@/lib/projects/types';
import ProjectCard from '@/components/projects/ProjectCard';
import { useI18n } from '@/components/common/I18nProvider';

function useWindowWidth() {
  const [w, setW] = useState<number>(0);
  useEffect(() => {
    const update = () => setW(window.innerWidth);
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);
  return w;
}

export default function ProjectGrid({
  projects,
  statsMap,
  showCreateInEmpty = false,
  onCreate,
  onEnter,
  onEdit,
  onArchive,
  onDelete,
  getCaps,
  emptyHintI18nKey = 'adminProjectsPage.emptyHint',
}: {
  projects: Project[];
  /** 按 projectId 的真实任务数/数据数；dataCount 未加载时为 undefined，卡片显示 — */
  statsMap?: Record<string, { taskCount: number; dataCount: number | undefined }>;
  showCreateInEmpty?: boolean;
  onCreate?: () => void;
  onEnter: (project: Project) => void;
  onEdit: (project: Project) => void;
  onArchive: (project: Project) => void;
  onDelete: (project: Project) => void;
  /** 与后端项目写权限一致（团队辖区 / owner_id / 超管） */
  getCaps: (project: Project) => { canEdit: boolean; canDelete: boolean };
  /** 空列表副文案 i18n key（默认「创建项目或等待他人共享」） */
  emptyHintI18nKey?: string;
}) {
  const { t } = useI18n();
  const width = useWindowWidth();
  const cols = useMemo(() => {
    if (width >= 1280) return 3;
    if (width >= 900) return 2;
    return 1;
  }, [width]);

  if (projects.length === 0) {
    return (
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '12px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
          padding: '56px 24px',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: '18px', fontWeight: 700, color: '#111827', marginBottom: '8px' }}>{t('adminProjectsPage.emptyTitle')}</div>
        <div style={{ fontSize: '14px', color: '#6b7280', marginBottom: showCreateInEmpty ? '18px' : 0 }}>
          {t(emptyHintI18nKey)}
        </div>
        {showCreateInEmpty && onCreate && (
          <button
            type="button"
            onClick={onCreate}
            style={{
              height: '38px',
              padding: '0 16px',
              borderRadius: '10px',
              border: 'none',
              backgroundColor: '#2563eb',
              color: '#ffffff',
              fontSize: '14px',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            {t('adminProjectsPage.createButton')}
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gap: '16px',
      }}
    >
      {projects.map((p) => {
        const { canEdit, canDelete } = getCaps(p);
        return (
          <ProjectCard
            key={p.id}
            project={p}
            stats={statsMap?.[p.id]}
            onEnter={() => onEnter(p)}
            onEdit={() => onEdit(p)}
            onArchive={() => onArchive(p)}
            onDelete={() => onDelete(p)}
            canEdit={canEdit}
            canDelete={canDelete}
          />
        );
      })}
    </div>
  );
}
