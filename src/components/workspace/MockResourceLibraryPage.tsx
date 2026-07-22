'use client';

import { useMemo, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import type { ResourceItem } from '@/lib/mock/workspacePagesMock';
import {
  FilterInput,
  GhostButton,
  MockActionHint,
  PrimaryButton,
  SecondaryButton,
  SectionCard,
  StatusBadge,
  WS,
} from './workspaceUi';

export interface MockResourceLibraryPageProps {
  title: string;
  subtitle: string;
  resources: ResourceItem[];
  createLabel?: string;
  importLabel?: string;
}

export default function MockResourceLibraryPage({
  title,
  subtitle,
  resources,
  createLabel = '新建',
  importLabel = '导入',
}: MockResourceLibraryPageProps) {
  const [search, setSearch] = useState('');
  const [tagFilter, setTagFilter] = useState('');

  const allTags = useMemo(() => {
    const set = new Set<string>();
    resources.forEach((r) => r.tags.forEach((t) => set.add(t)));
    return [...set];
  }, [resources]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return resources.filter((r) => {
      if (tagFilter && !r.tags.includes(tagFilter)) return false;
      if (!q) return true;
      return (
        r.name.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)
      );
    });
  }, [resources, search, tagFilter]);

  const mockToast = (action: string) => window.alert(`${action}功能开发中，敬请期待。`);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={title}
        subtitle={subtitle}
        actions={
          <>
            <SecondaryButton onClick={() => mockToast(importLabel)}>{importLabel}</SecondaryButton>
            <PrimaryButton onClick={() => mockToast(createLabel)}>{createLabel}</PrimaryButton>
          </>
        }
      />

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <FilterInput value={search} onChange={setSearch} placeholder="搜索资源名称、说明…" />
          <select
            value={tagFilter}
            onChange={(e) => setTagFilter(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              fontSize: 14,
              backgroundColor: '#fff',
            }}
          >
            <option value="">全部标签</option>
            {allTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <span style={{ fontSize: 13, color: '#6b7280' }}>
            共 {filtered.length} 项资源 <MockActionHint />
          </span>
        </div>
      </ModulePageFilterCard>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: WS.gap,
        }}
      >
        {filtered.map((r) => (
          <SectionCard key={r.id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 600, color: '#111827' }}>{r.name}</div>
                <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4, fontFamily: 'monospace' }}>{r.id}</div>
              </div>
              <StatusBadge status={r.status} />
            </div>
            <p style={{ fontSize: 13, color: '#4b5563', lineHeight: 1.5, margin: '0 0 12px' }}>{r.description}</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
              {r.tags.map((tag) => (
                <span
                  key={tag}
                  style={{
                    fontSize: 11,
                    padding: '2px 8px',
                    borderRadius: 4,
                    backgroundColor: '#eff6ff',
                    color: '#1d4ed8',
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
            <div style={{ fontSize: 12, color: '#9ca3af', marginBottom: 12 }}>
              {r.category} · {r.version} · 更新 {r.updatedAt}
            </div>
            <div style={{ display: 'flex', gap: 8, borderTop: '1px solid #f3f4f6', paddingTop: 12 }}>
              <GhostButton onClick={() => mockToast('查看详情')}>详情</GhostButton>
              <GhostButton onClick={() => mockToast('编辑')}>编辑</GhostButton>
              <GhostButton onClick={() => mockToast('复制到实验')}>复制</GhostButton>
            </div>
          </SectionCard>
        ))}
      </div>
      {filtered.length === 0 && (
        <SectionCard>
          <p style={{ textAlign: 'center', color: '#6b7280', margin: 24 }}>暂无匹配资源</p>
        </SectionCard>
      )}
    </ModulePageContainer>
  );
}
