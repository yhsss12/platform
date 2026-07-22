'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import {
  listRegistryResources,
  registryStatusLabel,
  type RegistryResource,
} from '@/lib/api/resourceRegistryClient';
import {
  FilterInput,
  GhostButton,
  PrimaryButton,
  SecondaryButton,
  StatusBadge,
  WS,
} from './workspaceUi';

const thStyle: React.CSSProperties = {
  padding: '12px 16px',
  textAlign: 'left',
  borderBottom: '1px solid #e5e7eb',
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  backgroundColor: '#f9fafb',
  whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  fontSize: 13,
  borderBottom: '1px solid #f3f4f6',
  verticalAlign: 'top',
};

function physicsProxyStatusLabel(status: string, metadata: Record<string, unknown>): string {
  if (metadata.validating === true || status === 'draft') return '验证中';
  return status === 'available' ? '可用' : registryStatusLabel(status);
}

function proxyStatusBadge(status: string, metadata: Record<string, unknown>): 'active' | 'running' {
  if (metadata.validating === true || status === 'draft') return 'running';
  return status === 'available' ? 'active' : 'running';
}

function rowFromResource(resource: RegistryResource) {
  const meta = resource.metadata ?? resource.physicsProxy ?? {};
  return {
    id: resource.assetId,
    name: resource.name,
    proxyType: String(meta.proxyType ?? '—'),
    applicableTasks: String(meta.applicableTasks ?? '—'),
    physicalObjects: String(meta.physicalObjects ?? '—'),
    inputVariables: String(meta.inputVariables ?? '—'),
    outputVariables: String(meta.outputVariables ?? '—'),
    trainingMethod: String(meta.trainingMethod ?? '—'),
    errorMetric: String(meta.errorMetric ?? '—'),
    speedup: String(meta.speedup ?? '—'),
    status: resource.status,
    metadata: meta,
  };
}

export interface PhysicsProxyLibraryTableProps {
  title: string;
  subtitle: string;
  onToast?: (message: string) => void;
}

export function PhysicsProxyLibraryTable({
  title,
  subtitle,
  onToast,
}: PhysicsProxyLibraryTableProps) {
  const [search, setSearch] = useState('');
  const [resources, setResources] = useState<RegistryResource[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const response = await listRegistryResources({ resourceType: 'physics_proxy' });
      setResources(response.resources);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载物理代理模型失败');
      setResources([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const rows = useMemo(() => resources.map(rowFromResource), [resources]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (row) =>
        row.name.toLowerCase().includes(q) ||
        row.proxyType.toLowerCase().includes(q) ||
        row.applicableTasks.toLowerCase().includes(q) ||
        row.physicalObjects.toLowerCase().includes(q)
    );
  }, [rows, search]);

  const notify = (message: string) => {
    if (onToast) {
      onToast(message);
      return;
    }
    window.alert(message);
  };

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={title}
        subtitle={subtitle}
        actions={
          <>
            <SecondaryButton onClick={() => notify('导入物理代理模型')}>导入</SecondaryButton>
            <PrimaryButton onClick={() => notify('新建物理代理模型')}>新建</PrimaryButton>
          </>
        }
      />

      <ModulePageFilterCard>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <FilterInput value={search} onChange={setSearch} placeholder="搜索模型名称、代理类型、适用任务…" />
          <span style={{ fontSize: 13, color: '#6b7280' }}>
            {loading ? '加载中…' : `共 ${filtered.length} 项模型`}
          </span>
          {loadError ? (
            <>
              <span style={{ fontSize: 13, color: '#b45309' }}>{loadError}</span>
              <SecondaryButton onClick={() => void refresh()}>重试</SecondaryButton>
            </>
          ) : null}
        </div>
      </ModulePageFilterCard>

      <div style={{ ...WS.card, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', minWidth: 1400, borderCollapse: 'collapse', backgroundColor: '#fff' }}>
            <thead>
              <tr>
                {[
                  '模型名称',
                  '代理类型',
                  '适用任务',
                  '物理对象',
                  '输入变量',
                  '输出变量',
                  '训练方法',
                  '误差指标',
                  '加速倍率',
                  '状态',
                  '操作',
                ].map((h) => (
                  <th key={h} style={thStyle}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={11} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280' }}>
                    正在加载…
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={11} style={{ ...tdStyle, textAlign: 'center', color: '#6b7280' }}>
                    {loadError ?? '暂无物理代理模型'}
                  </td>
                </tr>
              ) : (
                filtered.map((row) => (
                  <tr
                    key={row.id}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f9fafb';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <td style={{ ...tdStyle, fontFamily: 'ui-monospace, monospace', fontSize: 12, fontWeight: 500 }}>
                      {row.name}
                    </td>
                    <td style={tdStyle}>{row.proxyType}</td>
                    <td style={tdStyle}>{row.applicableTasks}</td>
                    <td style={tdStyle}>{row.physicalObjects}</td>
                    <td style={{ ...tdStyle, maxWidth: 180 }}>{row.inputVariables}</td>
                    <td style={{ ...tdStyle, maxWidth: 160 }}>{row.outputVariables}</td>
                    <td style={tdStyle}>{row.trainingMethod}</td>
                    <td style={tdStyle}>{row.errorMetric}</td>
                    <td style={{ ...tdStyle, fontWeight: 600, color: '#2563eb' }}>{row.speedup}</td>
                    <td style={tdStyle}>
                      <StatusBadge
                        status={proxyStatusBadge(row.status, row.metadata)}
                        label={physicsProxyStatusLabel(row.status, row.metadata)}
                      />
                    </td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        <GhostButton onClick={() => notify(`查看 ${row.name} 详情`)}>详情</GhostButton>
                        <GhostButton onClick={() => notify(`开始验证 ${row.name}`)}>验证</GhostButton>
                        <GhostButton onClick={() => notify(`部署 ${row.name}`)}>部署</GhostButton>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </ModulePageContainer>
  );
}
