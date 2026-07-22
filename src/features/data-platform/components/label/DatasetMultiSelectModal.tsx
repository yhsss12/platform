'use client';

import { useState, useEffect, useMemo } from 'react';
import { getDataAssets, isDataAssetSynced, type DataAssetItem } from '@/features/data-platform/api/dataAssetsApi';
import { useI18n } from '@/components/common/I18nProvider';

interface DatasetMultiSelectModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (selectedIds: number[]) => void;
  initialSelectedIds?: number[];
  project?: string;
  lockProject?: boolean;
  syncedOnly?: boolean;
  /** 传入时交给 GET /api/data-assets?format=…，并在前端再过滤一层 */
  format?: string;
  /** 列表为空时的提示（例如仅 MCAP 场景） */
  emptyListMessage?: string;
}

export default function DatasetMultiSelectModal({
  open,
  onClose,
  onConfirm,
  initialSelectedIds = [],
  project,
  lockProject = false,
  syncedOnly = false,
  format,
  emptyListMessage,
}: DatasetMultiSelectModalProps) {
  const { t } = useI18n();
  const [datasets, setDatasets] = useState<DataAssetItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set(initialSelectedIds));
  const [keyword, setKeyword] = useState('');
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pageSize] = useState(50);
  // 从当前展示的数据集中提取项目列表
  const availableProjects = useMemo(() => {
    const projects = Array.from(
      new Set(
        datasets
          .map((d) => d.project_name ?? d.project_id ?? (d as { project?: string }).project)
          .filter((p): p is string => p != null && String(p).trim() !== '')
      )
    ).sort();
    return projects;
  }, [datasets]);

  // 加载数据集列表
  useEffect(() => {
    if (open) {
      loadDatasets();
    }
  }, [open, page, keyword, selectedProject, lockProject, project, syncedOnly, format]);

  // 初始化选中状态
  useEffect(() => {
    if (open && initialSelectedIds.length > 0) {
      setSelectedIds(new Set(initialSelectedIds));
    }
  }, [open, initialSelectedIds]);

  useEffect(() => {
    if (!open) return;
    if (lockProject) {
      setSelectedProject(String(project || ''));
      setPage(1);
    }
  }, [open, lockProject, project]);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const params: any = {
        page,
        page_size: pageSize,
      };
      if (keyword) {
        params.keyword = keyword;
      }
      const projectFilter = lockProject ? String(project || '').trim() : String(selectedProject || '').trim();
      if (projectFilter) params.project = projectFilter;
      const fmt = (format || '').trim().toLowerCase();
      if (fmt) params.format = fmt;

      const res = await getDataAssets(params);
      if (res.ok && res.data) {
        let items = syncedOnly ? res.data.items.filter((d) => isDataAssetSynced(d)) : res.data.items;
        if (fmt) {
          items = items.filter((d) => (d.format || '').trim().toLowerCase() === fmt);
        }
        setDatasets(items);
        setTotal(res.data.total);
      }
    } catch (error) {
      console.error('加载数据资产失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 当前页全选状态
  const currentPageAllSelected = useMemo(() => {
    if (datasets.length === 0) return false;
    return datasets.every((d) => selectedIds.has(d.id));
  }, [datasets, selectedIds]);

  // 当前页部分选中状态
  const currentPageSomeSelected = useMemo(() => {
    if (datasets.length === 0) return false;
    const selectedCount = datasets.filter((d) => selectedIds.has(d.id)).length;
    return selectedCount > 0 && selectedCount < datasets.length;
  }, [datasets, selectedIds]);

  // 切换单个选择
  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  // 当前页全选/全不选
  const toggleSelectCurrentPage = () => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (currentPageAllSelected) {
        // 全不选：移除当前页所有项
        datasets.forEach((d) => newSet.delete(d.id));
      } else {
        // 全选：添加当前页所有项
        datasets.forEach((d) => newSet.add(d.id));
      }
      return newSet;
    });
  };

  // 重置筛选
  const handleReset = () => {
    setKeyword('');
    setSelectedProject(lockProject ? String(project || '') : '');
    setPage(1);
  };

  // 确认选择
  const handleConfirm = () => {
    onConfirm(Array.from(selectedIds));
    onClose();
  };

  // 总页数
  const totalPages = Math.ceil(total / pageSize);

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          width: '900px',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div
          style={{
            padding: '20px 24px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <h3
            style={{
              fontSize: '18px',
              fontWeight: '600',
              color: '#111827',
              margin: 0,
            }}
          >
            {t('datasetPicker.title')}
          </h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#6b7280',
              fontSize: '20px',
              cursor: 'pointer',
              padding: '4px',
              lineHeight: 1,
              width: '24px',
              height: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '4px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
              e.currentTarget.style.color = '#111827';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = '#6b7280';
            }}
          >
            ✕
          </button>
        </div>

        {/* 筛选条 */}
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid #e5e7eb',
            backgroundColor: '#f9fafb',
            display: 'flex',
            gap: '12px',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          {/* 项目筛选 */}
          {!lockProject && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: '#374151', fontWeight: '500', whiteSpace: 'nowrap' }}>
                {t('datasetPicker.projectFilter')}：
              </label>
              <select
                value={selectedProject}
                onChange={(e) => {
                  setSelectedProject(e.target.value);
                  setPage(1);
                }}
                style={{
                  padding: '6px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  backgroundColor: '#ffffff',
                  color: '#111827',
                  outline: 'none',
                  cursor: 'pointer',
                  minWidth: '150px',
                }}
              >
                <option value="">{t('datasetPicker.allProjects')}</option>
                {availableProjects.map((project, index) => (
                  <option key={project ? `${project}-${index}` : `empty-${index}`} value={project}>
                    {project}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* 文件名搜索 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: '200px' }}>
            <label style={{ fontSize: '14px', color: '#374151', fontWeight: '500', whiteSpace: 'nowrap' }}>
              {t('common.search')}：
            </label>
            <input
              type="text"
              placeholder={t('datasetPicker.searchProjectPlaceholder')}
              value={keyword}
              onChange={(e) => {
                setKeyword(e.target.value);
                if (!lockProject) setSelectedProject('');
                setPage(1);
              }}
              style={{
                flex: 1,
                padding: '6px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = '#2563eb';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = '#d1d5db';
              }}
            />
          </div>

          {/* 重置按钮 */}
          <button
            onClick={handleReset}
            style={{
              padding: '6px 16px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
              e.currentTarget.style.borderColor = '#9ca3af';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#ffffff';
              e.currentTarget.style.borderColor = '#d1d5db';
            }}
          >
            {t('common.reset')}
          </button>
        </div>

        {/* 操作栏 */}
        <div
          style={{
            padding: '12px 24px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            backgroundColor: '#ffffff',
          }}
        >
          <input
            type="checkbox"
            checked={currentPageAllSelected}
            onChange={toggleSelectCurrentPage}
            style={{
              width: '16px',
              height: '16px',
              cursor: 'pointer',
            }}
            ref={(input) => {
              if (input) {
                input.indeterminate = currentPageSomeSelected;
              }
            }}
          />
          <span style={{ fontSize: '14px', color: '#374151', fontWeight: '500' }}>
            全选
          </span>
          <div style={{ marginLeft: 'auto', fontSize: '13px', color: '#6b7280' }}>
            {t('datasetPicker.selectedCount', { n: selectedIds.size })}
          </div>
        </div>

        {/* 表格 */}
        <div style={{ flex: 1, overflow: 'auto', minHeight: '300px' }}>
          {loading ? (
            <div style={{ padding: '60px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
              加载中...
            </div>
          ) : datasets.length === 0 ? (
            <div style={{ padding: '60px', textAlign: 'center', color: '#6b7280', fontSize: '14px' }}>
              {emptyListMessage || t('datasetPicker.empty')}
            </div>
          ) : (
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
              }}
            >
              <thead
                style={{
                  position: 'sticky',
                  top: 0,
                  backgroundColor: '#f9fafb',
                  zIndex: 10,
                  borderBottom: '2px solid #e5e7eb',
                }}
              >
                <tr>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'left',
                      fontSize: '13px',
                      fontWeight: '600',
                      color: '#374151',
                      width: '50px',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={currentPageAllSelected}
                      onChange={toggleSelectCurrentPage}
                      style={{
                        width: '16px',
                        height: '16px',
                        cursor: 'pointer',
                      }}
                      ref={(input) => {
                        if (input) {
                          input.indeterminate = currentPageSomeSelected;
                        }
                      }}
                    />
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'left',
                      fontSize: '13px',
                      fontWeight: '600',
                      color: '#374151',
                    }}
                  >
                    文件名
                  </th>
                  <th
                    style={{
                      padding: '12px 16px',
                      textAlign: 'left',
                      fontSize: '13px',
                      fontWeight: '600',
                      color: '#374151',
                      width: '200px',
                    }}
                  >
                    项目
                  </th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => {
                  const isSelected = selectedIds.has(dataset.id);
                  return (
                    <tr
                      key={dataset.id}
                      style={{
                        borderBottom: '1px solid #f3f4f6',
                        cursor: 'pointer',
                        transition: 'background-color 0.15s',
                        backgroundColor: isSelected ? '#eff6ff' : '#ffffff',
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) {
                          e.currentTarget.style.backgroundColor = '#f9fafb';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) {
                          e.currentTarget.style.backgroundColor = '#ffffff';
                        }
                      }}
                      onClick={() => toggleSelect(dataset.id)}
                    >
                      <td style={{ padding: '12px 16px' }}>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelect(dataset.id)}
                          onClick={(e) => e.stopPropagation()}
                          style={{
                            width: '16px',
                            height: '16px',
                            cursor: 'pointer',
                          }}
                        />
                      </td>
                      <td
                        style={{
                          padding: '12px 16px',
                          fontSize: '14px',
                          color: '#111827',
                          fontWeight: '500',
                        }}
                      >
                        {dataset.filename ?? (dataset as { name?: string }).name ?? '-'}
                      </td>
                      <td
                        style={{
                          padding: '12px 16px',
                          fontSize: '14px',
                          color: '#6b7280',
                        }}
                      >
                        {dataset.project_name ?? dataset.project_id ?? (dataset as { project?: string }).project ?? '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* 分页 */}
        {totalPages > 1 && (
          <div
            style={{
              padding: '16px 24px',
              borderTop: '1px solid #e5e7eb',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              backgroundColor: '#f9fafb',
            }}
          >
            <div style={{ fontSize: '13px', color: '#6b7280' }}>
              共 {total} 条，第 {page} / {totalPages} 页
            </div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                style={{
                  padding: '6px 12px',
                  backgroundColor: page === 1 ? '#f3f4f6' : '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  color: page === 1 ? '#9ca3af' : '#374151',
                  fontSize: '13px',
                  cursor: page === 1 ? 'not-allowed' : 'pointer',
                }}
              >
                上一页
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                style={{
                  padding: '6px 12px',
                  backgroundColor: page === totalPages ? '#f3f4f6' : '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  color: page === totalPages ? '#9ca3af' : '#374151',
                  fontSize: '13px',
                  cursor: page === totalPages ? 'not-allowed' : 'pointer',
                }}
              >
                下一页
              </button>
            </div>
          </div>
        )}

        {/* 底部按钮 */}
        <div
          style={{
            padding: '16px 24px',
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '12px',
            backgroundColor: '#ffffff',
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#ffffff';
            }}
          >
            {t('common.cancel')}
          </button>
          <button
            onClick={handleConfirm}
            disabled={selectedIds.size === 0}
            style={{
              padding: '10px 20px',
              backgroundColor: selectedIds.size === 0 ? '#9ca3af' : '#2563eb',
              border: 'none',
              borderRadius: '6px',
              color: '#ffffff',
              fontSize: '14px',
              fontWeight: '500',
              cursor: selectedIds.size === 0 ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              if (selectedIds.size > 0) {
                e.currentTarget.style.backgroundColor = '#1d4ed8';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedIds.size > 0) {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }
            }}
          >
            {t('datasetPicker.confirm')}
          </button>
        </div>
      </div>
    </div>
  );
}
