'use client';

import { ChevronUp, ChevronLeft, ChevronRight } from 'lucide-react';
import { useI18n } from '@/components/common/I18nProvider';

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];

export interface BatchActionItem {
  key: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  /** 危险操作（如删除）使用红色描边 */
  danger?: boolean;
}

export interface ListFooterBarProps {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  /** 当前勾选条数，用于批量操作区展示与禁用 */
  selectedCount?: number;
  /** 批量操作按钮 */
  batchActions?: BatchActionItem[];
  showBackToTop?: boolean;
  className?: string;
  /** inline=仅顶部分隔线（与表格同卡片）；standalone=独立成块 */
  variant?: 'inline' | 'standalone';
  /** 列表请求进行中：隐藏「共 0 条」并禁用分页 */
  loading?: boolean;
}

/**
 * 列表页统一底栏：分页信息 + 批量操作 + 回到顶部。
 * 平台主内容区由 layout 的 <main> 滚动，回到顶部对 main 做 smooth scroll。
 */
export default function ListFooterBar({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  selectedCount = 0,
  batchActions = [],
  showBackToTop = true,
  className,
  variant = 'inline',
  loading = false,
}: ListFooterBarProps) {
  const { t } = useI18n();
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const start = loading || total === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = loading || total === 0 ? 0 : Math.min(page * pageSize, total);

  const scrollToTop = () => {
    const main = document.querySelector('main');
    if (main) {
      main.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const isStandalone = variant === 'standalone';
  const btnBase = {
    display: 'inline-flex' as const,
    alignItems: 'center',
    gap: '6px',
    padding: '8px 14px',
    fontSize: '14px',
    border: '1px solid #e5e7eb',
    borderRadius: '6px',
    cursor: 'pointer' as const,
    transition: 'background-color 0.2s, border-color 0.2s',
  };

  return (
    <footer
      className={className}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        minHeight: 60,
        padding: '0 24px',
        background: '#fff',
        borderTop: '1px solid #e5e7eb',
        flexShrink: 0,
        ...(isStandalone
          ? { marginTop: 12, borderRadius: 8, border: '1px solid #e5e7eb' }
          : {}),
      }}
    >
      {/* 左侧：分页 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 14, color: '#6b7280', whiteSpace: 'nowrap' }}>{t('dataPage.paginationPerPage')}</span>
          <select
            value={pageSize}
            disabled={loading}
            onChange={(e) => {
              const size = Number(e.target.value);
              onPageSizeChange(size);
              onPageChange(1);
            }}
            style={{
              padding: '6px 10px',
              fontSize: 14,
              color: '#374151',
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              backgroundColor: '#fff',
              cursor: 'pointer',
            }}
          >
            {PAGE_SIZE_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
        <span style={{ fontSize: 14, color: '#6b7280' }}>
          {loading ? '加载中…' : t('dataPage.paginationRange', { start, end, total })}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            type="button"
            disabled={loading || page <= 1}
            onClick={() => onPageChange(page - 1)}
            style={{
              ...btnBase,
              backgroundColor: page <= 1 ? '#f9fafb' : '#fff',
              color: page <= 1 ? '#9ca3af' : '#374151',
              cursor: page <= 1 ? 'not-allowed' : 'pointer',
              padding: '8px 10px',
            }}
          >
            <ChevronLeft size={18} />
          </button>
          <button
            type="button"
            disabled={loading || page >= totalPages}
            onClick={() => onPageChange(page + 1)}
            style={{
              ...btnBase,
              backgroundColor: loading || page >= totalPages ? '#f9fafb' : '#fff',
              color: loading || page >= totalPages ? '#9ca3af' : '#374151',
              cursor: loading || page >= totalPages ? 'not-allowed' : 'pointer',
              padding: '8px 10px',
            }}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>

      {/* 右侧：批量操作 + 回到顶部 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {batchActions.map((action) => (
          <button
            key={action.key}
            type="button"
            disabled={action.disabled ?? selectedCount === 0}
            onClick={action.onClick}
            style={{
              ...btnBase,
              backgroundColor: 'transparent',
              color: action.danger ? '#dc2626' : '#374151',
              borderColor: action.danger ? '#fecaca' : '#e5e7eb',
              cursor: action.disabled ?? selectedCount === 0 ? 'not-allowed' : 'pointer',
              opacity: action.disabled ?? selectedCount === 0 ? 0.6 : 1,
            }}
            onMouseEnter={(e) => {
              if (action.disabled || selectedCount === 0) return;
              e.currentTarget.style.backgroundColor = action.danger ? '#fef2f2' : '#f9fafb';
              e.currentTarget.style.borderColor = action.danger ? '#f87171' : '#d1d5db';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = action.danger ? '#fecaca' : '#e5e7eb';
            }}
          >
            {action.label}
          </button>
        ))}
        {showBackToTop && (
          <button
            type="button"
            onClick={scrollToTop}
            style={{
              ...btnBase,
              backgroundColor: 'transparent',
              color: '#374151',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
              e.currentTarget.style.borderColor = '#d1d5db';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }}
          >
            <ChevronUp size={16} style={{ flexShrink: 0 }} />
            {t('common.backToTop')}
          </button>
        )}
      </div>
    </footer>
  );
}
