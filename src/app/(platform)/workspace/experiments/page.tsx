'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import {
  FilterInput,
  GhostButton,
  MockActionHint,
  PrimaryButton,
  StatusBadge,
} from '@/components/workspace/workspaceUi';
import { experimentBatches } from '@/lib/mock/workspacePagesMock';

export default function ExperimentsPage() {
  const { t } = useI18n();
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return experimentBatches;
    return experimentBatches.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.scene.toLowerCase().includes(q) ||
        e.policy.toLowerCase().includes(q)
    );
  }, [search]);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('workspacePages.simulationRecordsTitle')}
        subtitle={t('workspacePages.simulationRecordsSubtitle')}
        actions={<PrimaryButton onClick={() => window.alert('已创建新批次')}>新建批次</PrimaryButton>}
      />

      <ModulePageFilterCard>
        <FilterInput value={search} onChange={setSearch} placeholder="搜索实验名称、场景、策略…" />
        <span style={{ fontSize: 13, color: '#6b7280', marginLeft: 12 }}>
          共 {filtered.length} 个批次 <MockActionHint />
        </span>
      </ModulePageFilterCard>

      <ModulePageTableCard>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
              {[
                '批次 ID',
                '实验名称',
                '场景',
                '策略',
                '运行轮次',
                '成功率',
                '创建人',
                '创建时间',
                '状态',
                '操作',
              ].map((h) => (
                <th
                  key={h}
                  style={{
                    textAlign: 'left',
                    padding: '12px 16px',
                    fontWeight: 600,
                    color: '#374151',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => (
              <tr key={e.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#6b7280' }}>{e.id}</td>
                <td style={{ padding: '12px 16px', fontWeight: 500, color: '#111827' }}>{e.name}</td>
                <td style={{ padding: '12px 16px' }}>{e.scene}</td>
                <td style={{ padding: '12px 16px' }}>{e.policy}</td>
                <td style={{ padding: '12px 16px' }}>{e.rounds}</td>
                <td
                  style={{
                    padding: '12px 16px',
                    fontWeight: 600,
                    color: e.successRate >= 85 ? '#059669' : '#d97706',
                  }}
                >
                  {e.successRate}%
                </td>
                <td style={{ padding: '12px 16px' }}>{e.creator}</td>
                <td style={{ padding: '12px 16px', color: '#6b7280' }}>{e.createdAt}</td>
                <td style={{ padding: '12px 16px' }}>
                  <StatusBadge status={e.status} />
                </td>
                <td style={{ padding: '12px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <GhostButton onClick={() => window.alert(`查看 ${e.id} 详情`)}>
                      查看详情
                    </GhostButton>
                    <Link
                      href="/workspace/replay"
                      style={{ fontSize: 13, color: '#2563eb', textDecoration: 'none' }}
                    >
                      查看回放
                    </Link>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ModulePageTableCard>
    </ModulePageContainer>
  );
}
