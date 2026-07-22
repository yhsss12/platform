'use client';

import { useEffect, useState } from 'react';
import type { Project } from '@/lib/projects/types';
import { useI18n } from '@/components/common/I18nProvider';
import { isoDateToYmdSlashDisplay, parseYmdSlashOrDashToIso } from '@/utils/format';

export interface DataFilters {
  keyword?: string;
  project?: string;
  format?: string;
  source?: string;
  task_name?: string;
  created_from?: string;
  created_to?: string;
}

export interface TaskOption {
  value: string;
  label: string;
}

interface FiltersBarProps {
  filters: DataFilters;
  onFilterChange: (filters: Partial<DataFilters>) => void;
  onReset: () => void;
  projectList?: Project[];
  /** 任务选项，由项目/数据类型/数据来源滚动更新 */
  taskOptions?: TaskOption[];
}

export default function FiltersBar({ filters, onFilterChange, onReset, projectList = [], taskOptions = [] }: FiltersBarProps) {
  const { t } = useI18n();
  /** 不用原生 type="date"，避免区域设置下出现 yyyy/mm/日 等中英混排与日历图标 */
  const [createdFromText, setCreatedFromText] = useState('');
  const [createdToText, setCreatedToText] = useState('');

  useEffect(() => {
    setCreatedFromText(filters.created_from ? isoDateToYmdSlashDisplay(filters.created_from) : '');
    setCreatedToText(filters.created_to ? isoDateToYmdSlashDisplay(filters.created_to) : '');
  }, [filters.created_from, filters.created_to]);

  const commitCreatedFrom = () => {
    const r = parseYmdSlashOrDashToIso(createdFromText);
    if (r === false) {
      setCreatedFromText(filters.created_from ? isoDateToYmdSlashDisplay(filters.created_from) : '');
      return;
    }
    onFilterChange({ created_from: r === undefined ? '' : r });
    setCreatedFromText(r ? isoDateToYmdSlashDisplay(r) : '');
  };

  const commitCreatedTo = () => {
    const r = parseYmdSlashOrDashToIso(createdToText);
    if (r === false) {
      setCreatedToText(filters.created_to ? isoDateToYmdSlashDisplay(filters.created_to) : '');
      return;
    }
    onFilterChange({ created_to: r === undefined ? '' : r });
    setCreatedToText(r ? isoDateToYmdSlashDisplay(r) : '');
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        flexWrap: 'wrap',
        flex: 1,
      }}
    >
      {/* 关键词搜索（与采集/任务筛选一致，带放大镜图标） */}
      <div style={{ position: 'relative', width: '280px' }}>
        <input
          type="text"
          placeholder={t('common.searchFileName')}
          value={filters.keyword || ''}
          onChange={(e) => onFilterChange({ keyword: e.target.value })}
          style={{
            width: '100%',
            padding: '8px 12px 8px 36px',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            color: '#111827',
            fontSize: '14px',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
        <svg
          style={{
            position: 'absolute',
            left: '12px',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '16px',
            height: '16px',
            fill: '#6b7280',
          }}
          viewBox="0 0 24 24"
        >
          <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
      </div>

      {/* 所属项目筛选 */}
      <select
        value={filters.project || ''}
        onChange={(e) => onFilterChange({ project: e.target.value || undefined })}
        style={{
          padding: '8px 12px',
          backgroundColor: '#ffffff',
          border: '1px solid #d1d5db',
          borderRadius: '6px',
          color: '#111827',
          fontSize: '14px',
          outline: 'none',
          minWidth: '140px',
          cursor: 'pointer',
        }}
      >
        <option value="">{t('dataPage.project')}</option>
        {projectList.map((p) => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>

      {/* 数据格式筛选 */}
      <select
        value={filters.format || ''}
        onChange={(e) => onFilterChange({ format: e.target.value || undefined })}
        style={{
          padding: '8px 12px',
          backgroundColor: '#ffffff',
          border: '1px solid #d1d5db',
          borderRadius: '6px',
          color: '#111827',
          fontSize: '14px',
          outline: 'none',
          minWidth: '140px',
          cursor: 'pointer',
        }}
      >
        <option value="">{t('dataPage.format')}</option>
        <option value="hdf5">HDF5</option>
        <option value="mcap">MCAP</option>
        <option value="lerobot">LeRobot</option>
      </select>

      {/* 来源筛选 */}
      <select
        value={filters.source || ''}
        onChange={(e) => onFilterChange({ source: e.target.value || undefined })}
        style={{
          padding: '8px 12px',
          backgroundColor: '#ffffff',
          border: '1px solid #d1d5db',
          borderRadius: '6px',
          color: '#111827',
          fontSize: '14px',
          outline: 'none',
          minWidth: '140px',
          cursor: 'pointer',
        }}
      >
        <option value="">{t('dataPage.source')}</option>
        <option value="import">{t('dataPage.sourceImport')}</option>
        <option value="collect">{t('dataPage.sourceCollect')}</option>
        <option value="label">{t('dataPage.sourceLabel')}</option>
        <option value="convert">{t('dataPage.sourceConvert')}</option>
      </select>

      {/* 任务筛选（选项随项目/数据类型/数据来源滚动更新） */}
      <select
        value={filters.task_name || ''}
        onChange={(e) => onFilterChange({ task_name: e.target.value || undefined })}
        style={{
          padding: '8px 12px',
          backgroundColor: '#ffffff',
          border: '1px solid #d1d5db',
          borderRadius: '6px',
          color: '#111827',
          fontSize: '14px',
          outline: 'none',
          minWidth: '140px',
          cursor: 'pointer',
        }}
      >
        <option value="">{t('dataPage.task')}</option>
        {taskOptions.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>

      {/* 按创建/入库日期（便于按日批量清理） */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, color: '#6b7280', whiteSpace: 'nowrap' }} title={t('dataPage.filterCreatedDateHint')}>
          {t('dataPage.filterCreatedDate')}
        </span>
        <input
          type="text"
          inputMode="numeric"
          placeholder={t('dataPage.filterDateInputPlaceholder')}
          value={createdFromText}
          onChange={(e) => setCreatedFromText(e.target.value)}
          onBlur={commitCreatedFrom}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
          title={t('dataPage.filterCreatedFrom')}
          autoComplete="off"
          spellCheck={false}
          style={{
            padding: '6px 10px',
            width: '132px',
            boxSizing: 'border-box',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            color: '#111827',
            fontSize: '13px',
            outline: 'none',
            fontVariantNumeric: 'tabular-nums',
          }}
        />
        <span style={{ fontSize: 13, color: '#9ca3af' }}>—</span>
        <input
          type="text"
          inputMode="numeric"
          placeholder={t('dataPage.filterDateInputPlaceholder')}
          value={createdToText}
          onChange={(e) => setCreatedToText(e.target.value)}
          onBlur={commitCreatedTo}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
          title={t('dataPage.filterCreatedTo')}
          autoComplete="off"
          spellCheck={false}
          style={{
            padding: '6px 10px',
            width: '132px',
            boxSizing: 'border-box',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            color: '#111827',
            fontSize: '13px',
            outline: 'none',
            fontVariantNumeric: 'tabular-nums',
          }}
        />
      </div>

      {/* 重置按钮 */}
      <button
        onClick={onReset}
        style={{
          padding: '8px 16px',
          backgroundColor: '#ffffff',
          border: '1px solid #d1d5db',
          borderRadius: '6px',
          color: '#374151',
          fontSize: '14px',
          cursor: 'pointer',
          outline: 'none',
          transition: 'all 0.2s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#f9fafb';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = '#ffffff';
        }}
      >
        {t('common.reset')}
      </button>
    </div>
  );
}
