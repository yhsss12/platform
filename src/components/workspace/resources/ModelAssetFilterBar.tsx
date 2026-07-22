'use client';

import { useI18n } from '@/components/common/I18nProvider';

const controlStyle: React.CSSProperties = {
  height: 36,
  padding: '0 12px',
  backgroundColor: '#ffffff',
  border: '1px solid #d1d5db',
  borderRadius: 6,
  color: '#111827',
  fontSize: 14,
  outline: 'none',
  boxSizing: 'border-box',
};

export interface ModelAssetFilterOption {
  key: string;
  value: string;
  placeholder: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

export function ModelAssetFilterBar({
  searchValue,
  onSearchChange,
  searchPlaceholder = '搜索模型名称、训练任务或数据集',
  filters,
  onReset,
}: {
  searchValue: string;
  onSearchChange: (value: string) => void;
  searchPlaceholder?: string;
  filters: ModelAssetFilterOption[];
  onReset: () => void;
}) {
  const { t } = useI18n();

  return (
    <>
      <style>{`
        .model-asset-filter-row {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        @media (min-width: 1120px) {
          .model-asset-filter-row {
            flex-wrap: nowrap;
          }
        }
        .model-asset-filter-search {
          position: relative;
          flex: 0 1 340px;
          min-width: 240px;
          max-width: 360px;
        }
        .model-asset-filter-select {
          flex: 0 1 200px;
          min-width: 160px;
          max-width: 220px;
        }
        .model-asset-filter-reset {
          flex-shrink: 0;
          margin-left: auto;
        }
        @media (max-width: 1119px) {
          .model-asset-filter-reset {
            margin-left: 0;
          }
        }
      `}</style>

      <div className="model-asset-filter-row">
        <div className="model-asset-filter-search">
          <input
            type="text"
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) => onSearchChange(e.target.value)}
            style={{
              ...controlStyle,
              width: '100%',
              paddingLeft: 36,
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
              fill: '#9ca3af',
              pointerEvents: 'none',
            }}
            viewBox="0 0 24 24"
            aria-hidden
          >
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
          </svg>
        </div>

        {filters.map((filter) => (
          <select
            key={filter.key}
            className="model-asset-filter-select"
            value={filter.value}
            onChange={(e) => filter.onChange(e.target.value)}
            style={{
              ...controlStyle,
              width: '100%',
              cursor: 'pointer',
            }}
          >
            <option value="">{filter.placeholder}</option>
            {filter.options.map((opt) => (
              <option key={opt.value || opt.label} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        ))}

        <button
          type="button"
          className="model-asset-filter-reset"
          onClick={onReset}
          style={{
            height: 36,
            padding: '0 12px',
            border: 'none',
            borderRadius: 6,
            background: 'none',
            color: '#6b7280',
            fontSize: 13,
            fontWeight: 500,
            cursor: 'pointer',
          }}
        >
          {t('common.reset')}
        </button>
      </div>
    </>
  );
}
