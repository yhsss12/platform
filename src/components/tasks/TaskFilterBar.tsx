'use client';

import React from 'react';
import { useI18n } from '@/components/common/I18nProvider';

export interface TaskFilterOption {
  key: string;
  value: string;
  placeholder: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

interface TaskFilterBarProps {
  searchPlaceholder?: string;
  searchValue: string;
  onSearchChange: (value: string) => void;
  filters: TaskFilterOption[];
  onReset: () => void;
  rightAction?: React.ReactNode;
  /** 渲染在筛选下拉与重置按钮之间（如日期范围） */
  inlineExtras?: React.ReactNode;
}

/**
 * 统一任务列表筛选栏（搜索 + 多个下拉 + 重置 + 右侧主按钮）。
 * 用于采集 / 标注 / 转换任务列表，保证 UI 与布局一致。
 */
export default function TaskFilterBar({
  searchPlaceholder = '请输入任务名称',
  searchValue,
  onSearchChange,
  filters,
  onReset,
  rightAction,
  inlineExtras,
}: TaskFilterBarProps) {
  const { t } = useI18n();
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        flexWrap: 'wrap',
      }}
    >
      {/* 左侧：搜索 + 下拉 + 重置 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          flexWrap: 'wrap',
          flex: 1,
        }}
      >
        {/* 搜索框（带放大镜图标） */}
        <div style={{ position: 'relative', width: 280 }}>
          <input
            type="text"
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) => onSearchChange(e.target.value)}
            style={{
              width: '100%',
              padding: '8px 12px 8px 36px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              color: '#111827',
              fontSize: 14,
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
          <svg
            style={{
              position: 'absolute',
              left: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              width: 16,
              height: 16,
              fill: '#6b7280',
            }}
            viewBox="0 0 24 24"
          >
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
          </svg>
        </div>

        {/* 动态下拉筛选项 */}
        {filters.map((f) => (
          <select
            key={f.key}
            value={f.value}
            onChange={(e) => f.onChange(e.target.value)}
            style={{
              padding: '8px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              color: '#111827',
              fontSize: 14,
              outline: 'none',
              minWidth: 100,
              cursor: 'pointer',
            }}
          >
            <option value="">{f.placeholder}</option>
            {f.options.map((opt) => (
              <option key={opt.value || opt.label} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        ))}

        {inlineExtras}

        <button
          type="button"
          onClick={onReset}
          style={{
            padding: '8px 16px',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: 6,
            color: '#374151',
            fontSize: 14,
            cursor: 'pointer',
            outline: 'none',
            transition: 'all 0.2s',
          }}
        >
          {t('common.reset')}
        </button>
      </div>

      {/* 右侧主按钮 */}
      {rightAction && (
        <div style={{ flexShrink: 0 }}>
          {rightAction}
        </div>
      )}
    </div>
  );
}

