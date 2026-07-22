'use client';

import { useRouter } from 'next/navigation';
import {
  dataAssetRequiresAgentSync,
  getSourceLabel,
  isDataAssetSynced,
} from '@/features/data-platform/api/dataAssetsApi';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import type { Project } from '@/lib/projects/types';
import type { DataAssetItem } from '@/features/data-platform/api/dataAssetsApi';
import { useI18n } from '@/components/common/I18nProvider';

/** 表格行：与 DataAssetItem 对齐，兼容 code、filename、project_id/project_name */
export type DataAssetRow = DataAssetItem;

interface DatasetTableProps {
  datasets: DataAssetRow[];
  loading: boolean;
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onDelete?: (id: number) => void;
  onLabel?: (dataset: DataAssetRow) => void;
  onConvert?: (dataset: DataAssetRow) => void;
  onExport?: (dataset: DataAssetRow) => void;
  onSync?: (dataset: DataAssetRow) => void;
  /** 资产详情（文件名、格式、设备、操作人、转换来源等） */
  onDetail?: (dataset: DataAssetRow) => void;
  /** 当前正在导出的资产 ID（行内导出按钮 loading） */
  exportingAssetId?: number | null;
  /** 多选：当前选中的 ID 集合 */
  selectedIds?: Set<number>;
  onSelectionChange?: (ids: Set<number>) => void;
  projectList?: Project[];
}

const btnSecondary = {
  padding: '4px 12px',
  backgroundColor: 'transparent',
  border: '1px solid #d1d5db',
  borderRadius: '4px',
  color: '#374151',
  fontSize: '12px',
  cursor: 'pointer' as const,
  transition: 'background-color 0.15s',
};

const btnDisabled = {
  backgroundColor: '#f3f4f6',
  border: '1px solid #e5e7eb',
  color: '#9ca3af',
  cursor: 'not-allowed' as const,
  opacity: 1,
};

export default function DatasetTable({
  datasets,
  loading,
  total,
  page,
  pageSize,
  onPageChange,
  onDelete,
  onLabel,
  onConvert,
  onExport,
  onSync,
  onDetail,
  exportingAssetId = null,
  selectedIds = new Set(),
  onSelectionChange,
  projectList = [],
}: DatasetTableProps) {
  const router = useRouter();
  const { t } = useI18n();

  const renderProjectName = (row: DataAssetRow): string => {
    if (row.project_name) return row.project_name;
    const pv = row.project_id ?? null;
    if (!pv) return '—';
    const byId = projectList.find((p) => p.id === pv);
    if (byId) return byId.name;
    const byName = projectList.find((p) => p.name === pv);
    if (byName) return byName.name;
    return pv;
  };
  // 格式化文件大小（转换为 MB/GB）
  const formatFileSize = (bytes: number): string => {
    if (!bytes) return '—';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      const gb = mb / 1024;
      return `${gb.toFixed(2)} GB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  const getFormatValue = (dataset: DataAssetRow): string | null => {
    let fmt = (dataset.format || '').toLowerCase();
    if (!fmt) {
      const name = dataset.filename || '';
      const lowerName = name.toLowerCase();
      if (/\.(hdf5|h5)$/.test(lowerName)) fmt = 'hdf5';
      else if (/\.mcap$/.test(lowerName)) fmt = 'mcap';
      else if (/\.zip$/.test(lowerName)) fmt = 'lerobot';
    }
    return fmt || null;
  };

  const renderFormatText = (dataset: DataAssetRow) => {
    const fmt = getFormatValue(dataset);
    if (!fmt) {
      return '—';
    }

    if (fmt === 'hdf5') return 'HDF5';
    if (fmt === 'mcap') return 'MCAP';
    if (fmt === 'lerobot') return 'LeRobot';
    return fmt.toUpperCase();
  };

  const hasSelection = onSelectionChange != null;
  const isSynced = (dataset: DataAssetRow) => isDataAssetSynced(dataset);
  const isLerobot = (dataset: DataAssetRow) => getFormatValue(dataset) === 'lerobot';
  const isHdf5 = (dataset: DataAssetRow) => getFormatValue(dataset) === 'hdf5';
  const syncLabel = (dataset: DataAssetRow) => {
    const s = String(dataset.sync_status || '').trim().toLowerCase();
    if (s === 'syncing') return '同步中';
    if (s === 'failed') return '同步失败';
    if (isSynced(dataset)) return '已同步';
    return '未同步';
  };
  const currentPageIds = datasets.map((d) => d.id);
  const allSelectedOnPage = currentPageIds.length > 0 && currentPageIds.every((id) => selectedIds.has(id));

  const toggleRow = (id: number) => {
    if (!onSelectionChange) return;
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onSelectionChange(next);
  };

  const toggleSelectAll = () => {
    if (!onSelectionChange) return;
    if (allSelectedOnPage) {
      const next = new Set(selectedIds);
      currentPageIds.forEach((id) => next.delete(id));
      onSelectionChange(next);
    } else {
      const next = new Set(selectedIds);
      currentPageIds.forEach((id) => next.add(id));
      onSelectionChange(next);
    }
  };

  if (loading) {
    return (
      <div style={{
        padding: '48px',
        textAlign: 'center',
        color: '#6b7280',
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
        fontSize: '14px',
      }}>
        {t('common.loading')}
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: '#ffffff',
      borderRadius: '8px',
      border: '1px solid #e5e7eb',
      boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      overflow: 'hidden',
    }}>
      <table style={{
        width: '100%',
        borderCollapse: 'collapse',
      }}>
        <thead>
          <tr style={{
            backgroundColor: '#f9fafb',
            borderBottom: '1px solid #e5e7eb',
          }}>
            {hasSelection && (
              <th style={{ padding: '12px', width: 40, textAlign: 'center', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
                <input
                  type="checkbox"
                  checked={allSelectedOnPage}
                  onChange={toggleSelectAll}
                  style={{ cursor: 'pointer', width: 16, height: 16 }}
                  title={allSelectedOnPage ? '取消全选' : '全选当前页'}
                />
              </th>
            )}
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.fileName')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.format')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.source')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.operator')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.updatedAt')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.project')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('dataPage.fileSize')}
            </th>
            <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>
              {t('common.actions')}
            </th>
          </tr>
        </thead>
        <tbody>
          {datasets.length === 0 ? (
            <tr>
              <td colSpan={hasSelection ? 9 : 8} style={{
                padding: '48px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}>
                {t('common.noData')}
              </td>
            </tr>
          ) : (
            datasets.map((dataset) => (
              <tr
                key={dataset.id}
                style={{
                  borderBottom: '1px solid #e5e7eb',
                  transition: 'background-color 0.15s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                {hasSelection && (
                  <td style={{ padding: '12px', width: 40, textAlign: 'center', verticalAlign: 'middle' }}>
                    <input
                      type="checkbox"
                      checked={selectedIds.has(dataset.id)}
                      onChange={() => toggleRow(dataset.id)}
                      onClick={(e) => e.stopPropagation()}
                      style={{ cursor: 'pointer', width: 16, height: 16 }}
                    />
                  </td>
                )}
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span>{dataset.filename}</span>
                      {onSync && String(dataset.source || '').trim().toLowerCase() === 'collect' && (
                        (() => {
                          const st = String(dataset.sync_status || '').trim().toLowerCase();
                          const label = syncLabel(dataset);
                          const style =
                            st === 'syncing'
                              ? { fontSize: '11px', padding: '2px 6px', borderRadius: 4, backgroundColor: '#e5e7eb', color: '#374151' }
                              : st === 'failed'
                                ? { fontSize: '11px', padding: '2px 6px', borderRadius: 4, backgroundColor: '#fee2e2', color: '#991b1b' }
                                : label === '已同步'
                                  ? { fontSize: '11px', padding: '2px 6px', borderRadius: 4, backgroundColor: '#d1fae5', color: '#065f46' }
                                  : { fontSize: '11px', padding: '2px 6px', borderRadius: 4, backgroundColor: '#fef3c7', color: '#92400e' };
                          return <span style={style}>{label}</span>;
                        })()
                      )}
                    </div>
                    {String(dataset.source || '').trim().toLowerCase() === 'collect' &&
                      dataset.collect_episode_rel_path != null &&
                      String(dataset.collect_episode_rel_path).trim() !== '' && (
                        <div style={{ fontSize: '11px', color: '#6b7280', lineHeight: 1.35 }}>
                          <span title="相对作业目录的 episode 路径">{dataset.collect_episode_rel_path}</span>
                          {dataset.collect_episode_on_device === false && (
                            <span style={{ marginLeft: 8, color: '#b45309', fontWeight: 500 }}>
                              采集端已无此目录
                            </span>
                          )}
                        </div>
                      )}
                  </div>
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {renderFormatText(dataset)}
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {getSourceLabel(dataset.source, t)}
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {(dataset.operator_name || '').trim() || '—'}
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {formatDateTimeMinuteYmdSlash(dataset.updated_at ?? dataset.created_at)}
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {renderProjectName(dataset)}
                </td>
                <td style={{ padding: '12px', fontSize: '13px', color: '#111827' }}>
                  {formatFileSize(dataset.file_size_bytes)}
                </td>
                <td style={{ padding: '12px', fontSize: '13px' }}>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {(() => {
                      const synced = isSynced(dataset);
                      const lerobot = isLerobot(dataset);
                      const hdf5 = isHdf5(dataset);
                      const viewDisabled = !synced || lerobot;
                      const labelDisabled = !synced || lerobot;
                      const convertDisabled = !synced || lerobot || hdf5;
                      const exportDisabled = exportingAssetId === dataset.id || !synced;
                      const viewTitle = !synced ? '该数据尚未同步，暂不可查看' : (lerobot ? 'LeRobot 数据不支持该操作' : undefined);
                      const labelTitle = !synced ? '该数据尚未同步，暂不可标注' : (lerobot ? 'LeRobot 数据不支持该操作' : undefined);
                      const convertTitle = !synced
                        ? '该数据尚未同步，暂不可转换'
                        : (lerobot ? 'LeRobot 数据不支持该操作' : (hdf5 ? 'HDF5 数据不支持转换' : undefined));
                      const exportTitle = !synced ? '该数据尚未同步，暂不可导出' : undefined;
                      return (
                        <>
                    {onDetail && (
                      <button
                        type="button"
                        onClick={() => onDetail(dataset)}
                        style={{ ...btnSecondary }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f9fafb';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        {t('dataPage.detail')}
                      </button>
                    )}
                    <button
                      onClick={() => router.push(`/data/view?assetId=${dataset.id}`)}
                      disabled={viewDisabled}
                      title={viewTitle}
                      style={{ ...btnSecondary, ...(viewDisabled ? btnDisabled : {}) }}
                      onMouseEnter={(e) => { if (!viewDisabled) e.currentTarget.style.backgroundColor = '#f9fafb'; }}
                      onMouseLeave={(e) => { if (!viewDisabled) e.currentTarget.style.backgroundColor = 'transparent'; }}
                    >
                      {t('common.view')}
                    </button>
                    {onLabel && (
                      <button
                        onClick={() => onLabel(dataset)}
                        disabled={labelDisabled}
                        title={labelTitle}
                        style={{ ...btnSecondary, ...(labelDisabled ? btnDisabled : {}) }}
                        onMouseEnter={(e) => { if (!labelDisabled) e.currentTarget.style.backgroundColor = '#f9fafb'; }}
                        onMouseLeave={(e) => { if (!labelDisabled) e.currentTarget.style.backgroundColor = 'transparent'; }}
                      >
                        {t('dataPage.annotate')}
                      </button>
                    )}
                    {onConvert && (
                      <button
                        onClick={() => onConvert(dataset)}
                        disabled={convertDisabled}
                        title={convertTitle}
                        style={{ ...btnSecondary, ...(convertDisabled ? btnDisabled : {}) }}
                        onMouseEnter={(e) => { if (!convertDisabled) e.currentTarget.style.backgroundColor = '#f9fafb'; }}
                        onMouseLeave={(e) => { if (!convertDisabled) e.currentTarget.style.backgroundColor = 'transparent'; }}
                      >
                        {t('dataPage.transform')}
                      </button>
                    )}
                    {onExport && (
                      <button
                        type="button"
                        disabled={exportDisabled}
                        title={exportTitle}
                        onClick={() => onExport(dataset)}
                        style={{ ...btnSecondary, ...(exportDisabled ? btnDisabled : {}) }}
                        onMouseEnter={(e) => { if (!exportDisabled && exportingAssetId !== dataset.id) e.currentTarget.style.backgroundColor = '#f9fafb'; }}
                        onMouseLeave={(e) => { if (!exportDisabled) e.currentTarget.style.backgroundColor = 'transparent'; }}
                      >
                        {exportingAssetId === dataset.id ? `${t('dataPage.export')}...` : t('dataPage.export')}
                      </button>
                    )}
                    {onSync && !lerobot && (
                      (() => {
                        const needsAgentSync = dataAssetRequiresAgentSync(dataset);
                        const st = String(dataset.sync_status || '').trim().toLowerCase();
                        if (!needsAgentSync) {
                          return (
                            <button
                              type="button"
                              disabled
                              title="该数据来源无需从采集端同步"
                              style={{ ...btnSecondary, ...btnDisabled }}
                            >
                              同步
                            </button>
                          );
                        }
                        if (st === 'syncing') {
                          return (
                            <button type="button" disabled title="该数据正在同步中" style={{ ...btnSecondary, ...btnDisabled }}>
                              同步中
                            </button>
                          );
                        }
                        if (synced) {
                          return (
                            <button type="button" disabled title="该数据已同步" style={{ ...btnSecondary, ...btnDisabled }}>
                              同步
                            </button>
                          );
                        }
                        return (
                          <button
                            type="button"
                            onClick={() => onSync(dataset)}
                            style={{ ...btnSecondary }}
                            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#f9fafb'; }}
                            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                          >
                            同步
                          </button>
                        );
                      })()
                    )}
                        </>
                      );
                    })()}
                    {onDelete && (
                      <button
                        onClick={() => onDelete(dataset.id)}
                        style={{
                          padding: '4px 12px',
                          backgroundColor: 'transparent',
                          border: '1px solid #fecaca',
                          borderRadius: '4px',
                          color: '#dc2626',
                          fontSize: '12px',
                          cursor: 'pointer',
                          transition: 'background-color 0.15s',
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#fef2f2'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                      >
                        {t('common.delete')}
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
