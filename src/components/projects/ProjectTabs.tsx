'use client';

import React from 'react';
import { useI18n } from '@/components/common/I18nProvider';

export type ProjectListTabKey = 'my' | 'shared' | 'archived' | 'team_active';

const TAB_KEYS: Record<ProjectListTabKey, string> = {
  my: 'adminProjectsPage.tabMine',
  shared: 'adminProjectsPage.tabShared',
  archived: 'adminProjectsPage.tabArchived',
  team_active: 'adminProjectsPage.tabTeamAll',
};

export default function ProjectTabs({
  value,
  onChange,
  withBorder = true,
  disabledKeys = [],
  disabledReasons,
  teamScoped = false,
}: {
  value: ProjectListTabKey;
  onChange: (key: ProjectListTabKey) => void;
  withBorder?: boolean;
  disabledKeys?: ProjectListTabKey[];
  disabledReasons?: Partial<Record<ProjectListTabKey, string>>;
  /** 从团队管理带 teamId 进入：仅「团队内未归档」+「已归档」两档，与按负责人/共享拆分解耦 */
  teamScoped?: boolean;
}) {
  const { t } = useI18n();
  const keys: ProjectListTabKey[] = teamScoped
    ? ['team_active', 'archived']
    : ['my', 'shared', 'archived'];

  return (
    <div
      style={{
        display: 'flex',
        gap: '4px',
        borderBottom: withBorder ? '1px solid #e5e7eb' : 'none',
      }}
    >
      {keys.map((k) => {
        const active = value === k;
        const disabled = disabledKeys.includes(k);
        return (
          <button
            key={k}
            type="button"
            onClick={() => {
              if (disabled) return;
              onChange(k);
            }}
            disabled={disabled}
            title={disabled && disabledReasons?.[k] ? disabledReasons[k] : undefined}
            style={{
              padding: '10px 14px',
              border: 'none',
              backgroundColor: 'transparent',
              color: disabled ? '#9ca3af' : active ? '#2563eb' : '#6b7280',
              fontSize: '14px',
              fontWeight: active ? 600 : 400,
              cursor: disabled ? 'not-allowed' : 'pointer',
              borderBottom: active ? '2px solid #2563eb' : '2px solid transparent',
              marginBottom: '-1px',
              transition: 'color 0.15s ease',
            }}
          >
            {t(TAB_KEYS[k])}
          </button>
        );
      })}
    </div>
  );
}

