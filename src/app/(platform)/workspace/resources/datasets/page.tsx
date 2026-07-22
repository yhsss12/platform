'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import ListFooterBar from '@/components/common/ListFooterBar';
import TaskFilterBar from '@/components/tasks/TaskFilterBar';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { WorkspaceDatasetTable } from '@/components/workspace/WorkspaceDatasetTable';
import { WorkspaceDatasetDetailDrawer } from '@/components/workspace/WorkspaceDatasetDetailDrawer';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import type { Dataset } from '@/types/benchmark';

export default function ResourceDatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [detailDataset, setDetailDataset] = useState<Dataset | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const response = await listWorkspaceDatasets();
      setDatasets(response.datasets);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载数据集失败');
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return datasets;
    return datasets.filter((d) =>
      [d.id, d.name, d.sourceJobId, d.sourceTaskTemplateId ?? '', d.format]
        .join(' ')
        .toLowerCase()
        .includes(q)
    );
  }, [datasets, search]);

  const paged = useMemo(
    () => filtered.slice((page - 1) * pageSize, page * pageSize),
    [filtered, page, pageSize]
  );

  const allPageSelected =
    paged.length > 0 && paged.every((d) => selectedIds.has(d.id));

  const toggleRow = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (allPageSelected) {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        paged.forEach((d) => next.delete(d.id));
        return next;
      });
    } else {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        paged.forEach((d) => next.add(d.id));
        return next;
      });
    }
  };

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="数据集"
        subtitle="平台登记的数据集索引，来源于仿真生成任务。完整数据操作请前往数据中心。"
        actions={
          <Link href="/workspace/data" style={{ textDecoration: 'none' }}>
            <SecondaryButton>前往数据中心</SecondaryButton>
          </Link>
        }
      />

      <ModulePageFilterCard>
        <TaskFilterBar
          searchValue={search}
          onSearchChange={setSearch}
          searchPlaceholder="搜索数据集"
          filters={[]}
          onReset={() => setSearch('')}
        />
        {loadError ? (
          <span style={{ fontSize: 13, color: '#b45309', marginLeft: 12 }}>{loadError}</span>
        ) : null}
      </ModulePageFilterCard>

      <ModulePageTableCard>
        {loading ? (
          <p style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>正在加载数据集…</p>
        ) : (
          <WorkspaceDatasetTable
            datasets={paged}
            selectedIds={selectedIds}
            onToggleRow={toggleRow}
            onToggleSelectAll={toggleSelectAll}
            allPageSelected={allPageSelected}
            onOpenDetail={setDetailDataset}
            emptyMessage="暂无数据集，请前往数据中心生成数据。"
          />
        )}
        {!loading && filtered.length > 0 ? (
          <ListFooterBar
            variant="inline"
            total={filtered.length}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size);
              setPage(1);
            }}
          />
        ) : null}
      </ModulePageTableCard>

      <WorkspaceDatasetDetailDrawer
        dataset={detailDataset}
        onClose={() => setDetailDataset(null)}
        onTrain={(dataset) => router.push(`/workspace/training?dataset=${encodeURIComponent(dataset.id)}`)}
        onBuilt={() => void refresh()}
      />
    </ModulePageContainer>
  );
}
