'use client';

import { useEffect, useMemo, useState } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceFormFieldClassName,
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import {
  workspaceTableRowHoverHandlers,
  workspaceTableStyle,
  workspaceTdStyle,
  workspaceTdTimeStyle,
  workspaceThStyle,
  WorkspaceTableWrap,
} from '@/components/workspace/workspaceTableStyles';
import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import {
  DEFAULT_DATASET_PICKER_FILTER,
  datasetPickerFilterWithColumn,
  filterTrainingDatasets,
  formatDatasetDisplayName,
  formatDatasetTableCreatedAt,
  getDatasetFilterColumns,
  getDatasetFilterValues,
  getFilteredDatasetSummary,
  isDatasetPickerFilterActive,
  paginateTrainingDatasetOptions,
  resetDatasetPickerFilter,
  resolveTrainingDatasetCountLabel,
  resolveTrainingDatasetFormatLabel,
  resolveTrainingDatasetTaskLabel,
  toggleTrainingDatasetDraftSelection,
  TRAINING_DATASET_PICKER_EMPTY_TEXT,
  TRAINING_DATASET_PICKER_FILTERED_EMPTY_TEXT,
  TRAINING_DATASET_PICKER_PAGE_SIZE,
  type DatasetPickerFilterColumn,
  type DatasetPickerSingleFilter,
  type TrainingDatasetPickerMeta,
  validateTrainingDatasetSelection,
} from '@/lib/workspace/trainingDatasetPicker';
import { DATASET_MERGE_INCOMPATIBLE_HINT, isDatasetCompatibleWithSelection } from '@/lib/workspace/trainingDatasetCompat';
import './datasetPickerDialog.css';

export type DatasetPickerDialogProps = {
  open: boolean;
  options: TrainingDatasetOption[];
  selectedIds: string[];
  multiple?: boolean;
  datasetMetaById?: Record<string, TrainingDatasetPickerMeta>;
  onConfirm: (datasets: TrainingDatasetOption[]) => void;
  onCancel: () => void;
};

export function DatasetPickerDialog({
  open,
  options,
  selectedIds,
  multiple = true,
  datasetMetaById = {},
  onConfirm,
  onCancel,
}: DatasetPickerDialogProps) {
  const [keyword, setKeyword] = useState('');
  const [filter, setFilter] = useState<DatasetPickerSingleFilter>(DEFAULT_DATASET_PICKER_FILTER);
  const [page, setPage] = useState(1);
  const [draftIds, setDraftIds] = useState<string[]>([]);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const rowHover = workspaceTableRowHoverHandlers();

  useEffect(() => {
    if (!open) return;
    setKeyword('');
    setFilter(resetDatasetPickerFilter());
    setPage(1);
    setDraftIds([...selectedIds]);
    setSelectionError(null);
  }, [open, selectedIds]);

  useEffect(() => {
    setPage(1);
  }, [keyword, filter]);

  const filterValueOptions = useMemo(
    () => getDatasetFilterValues(options, filter.column, datasetMetaById),
    [options, filter.column, datasetMetaById]
  );

  const filteredOptions = useMemo(
    () => filterTrainingDatasets(options, keyword, filter, datasetMetaById),
    [options, keyword, filter, datasetMetaById]
  );

  const filterSummary = useMemo(
    () => getFilteredDatasetSummary(options.length, filteredOptions.length),
    [options.length, filteredOptions.length]
  );

  const { items: pagedOptions, totalPages, page: safePage } = useMemo(
    () => paginateTrainingDatasetOptions(filteredOptions, page, TRAINING_DATASET_PICKER_PAGE_SIZE),
    [filteredOptions, page]
  );

  const toggleDraft = (datasetId: string) => {
    const result = toggleTrainingDatasetDraftSelection(draftIds, datasetId, options, multiple);
    setDraftIds(result.nextIds);
    setSelectionError(
      result.error === 'DATASET_MERGE_INCOMPATIBLE' ? DATASET_MERGE_INCOMPATIBLE_HINT : null
    );
  };

  const handleConfirm = () => {
    const validation = validateTrainingDatasetSelection(draftIds, options);
    if (!validation.ok) {
      setSelectionError(DATASET_MERGE_INCOMPATIBLE_HINT);
      return;
    }
    const selected = draftIds
      .map((id) => options.find((item) => item.id === id))
      .filter((item): item is TrainingDatasetOption => Boolean(item));
    onConfirm(selected);
  };

  const handleResetFilters = () => {
    setKeyword('');
    setFilter(resetDatasetPickerFilter());
    setPage(1);
  };

  const showPagination = filteredOptions.length > TRAINING_DATASET_PICKER_PAGE_SIZE;
  const hasActiveFilters = keyword.trim().length > 0 || isDatasetPickerFilterActive(filter);
  const tableColumnCount = 6;

  return (
    <WorkspaceCenteredModal
      open={open}
      title="选择数据集"
      titleId="dataset-picker-dialog-title"
      width={1000}
      zIndex={1600}
      onClose={onCancel}
      footer={
        <div className="ws-dataset-picker-footer">
          <span className="ws-dataset-picker-footer-left">已选 {draftIds.length} 个数据集</span>
          {showPagination ? (
            <div className="ws-dataset-picker-footer-center">
              <span>
                第 {safePage} / {totalPages} 页，共 {filteredOptions.length} 项
              </span>
              <SecondaryButton onClick={() => setPage((prev) => Math.max(1, prev - 1))} disabled={safePage <= 1}>
                上一页
              </SecondaryButton>
              <SecondaryButton
                onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
                disabled={safePage >= totalPages}
              >
                下一页
              </SecondaryButton>
            </div>
          ) : (
            <div className="ws-dataset-picker-footer-center" />
          )}
          <div className="ws-dataset-picker-footer-actions">
            <SecondaryButton onClick={onCancel}>取消</SecondaryButton>
            <PrimaryButton onClick={handleConfirm} disabled={draftIds.length === 0}>
              确认
            </PrimaryButton>
          </div>
        </div>
      }
    >
      <div className="ws-dataset-picker-search">
        <label style={workspaceModalFieldLabel} htmlFor="dataset-picker-search">
          搜索数据集
        </label>
        <input
          id="dataset-picker-search"
          type="search"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          placeholder="按名称、任务或 ID 搜索"
          className={workspaceFormFieldClassName}
          style={workspaceModalSelectStyle}
        />
      </div>

      <div className="ws-dataset-picker-filters">
        <span className="ws-dataset-picker-filters-label">筛选</span>
        <select
          className="ws-dataset-picker-filter-select ws-dataset-picker-filter-select-column"
          value={filter.column}
          onChange={(e) => {
            const column = e.target.value as DatasetPickerFilterColumn;
            setFilter(datasetPickerFilterWithColumn(column));
            setPage(1);
          }}
          aria-label="筛选列"
        >
          {getDatasetFilterColumns().map((item) => (
            <option key={item.value} value={item.value}>
              筛选列: {item.label}
            </option>
          ))}
        </select>
        <select
          className="ws-dataset-picker-filter-select ws-dataset-picker-filter-select-value"
          value={filter.value}
          onChange={(e) => {
            setFilter((prev) => ({ ...prev, value: e.target.value }));
            setPage(1);
          }}
          disabled={filter.column === 'none'}
          aria-label="筛选值"
        >
          {filterValueOptions.map((item) => (
            <option key={item.value} value={item.value}>
              筛选值: {item.label}
            </option>
          ))}
        </select>
        <SecondaryButton onClick={handleResetFilters} disabled={!hasActiveFilters}>
          重置
        </SecondaryButton>
      </div>

      {filterSummary ? <p className="ws-dataset-picker-filter-summary">{filterSummary}</p> : null}

      <div className="ws-dataset-picker-table-wrap">
        <WorkspaceTableWrap>
          <table style={workspaceTableStyle} className="ws-dataset-picker-table-training">
            <thead>
              <tr>
                <th className="ws-dataset-picker-col-select" style={{ ...workspaceThStyle, textAlign: 'center' }}>
                  {multiple ? '选择' : ''}
                </th>
                <th className="ws-dataset-picker-col-name" style={workspaceThStyle}>数据集名称</th>
                <th className="ws-dataset-picker-col-task" style={workspaceThStyle}>任务名称</th>
                <th className="ws-dataset-picker-col-format" style={workspaceThStyle}>数据格式</th>
                <th className="ws-dataset-picker-col-count" style={workspaceThStyle}>数据数量</th>
                <th className="ws-dataset-picker-col-time" style={workspaceThStyle}>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {options.length === 0 ? (
                <tr>
                  <td colSpan={tableColumnCount} style={workspaceTdStyle}>
                    <div className="ws-dataset-picker-empty">{TRAINING_DATASET_PICKER_EMPTY_TEXT}</div>
                  </td>
                </tr>
              ) : filteredOptions.length === 0 ? (
                <tr>
                  <td colSpan={tableColumnCount} style={workspaceTdStyle}>
                    <div className="ws-dataset-picker-empty">{TRAINING_DATASET_PICKER_FILTERED_EMPTY_TEXT}</div>
                  </td>
                </tr>
              ) : (
                pagedOptions.map((option) => {
                  const checked = draftIds.includes(option.id);
                  const compatible =
                    draftIds.length === 0 ||
                    checked ||
                    isDatasetCompatibleWithSelection(option, draftIds, options);
                  const meta = datasetMetaById[option.id];
                  const disabled = !compatible && !checked;
                  const displayName = formatDatasetDisplayName(option, meta);

                  return (
                    <tr
                      key={option.id}
                      style={{
                        transition: 'background-color 0.15s',
                        cursor: disabled ? 'not-allowed' : 'pointer',
                        opacity: disabled ? 0.45 : 1,
                        backgroundColor: checked ? '#eff6ff' : undefined,
                      }}
                      title={disabled ? DATASET_MERGE_INCOMPATIBLE_HINT : displayName}
                      onClick={() => {
                        if (!disabled) toggleDraft(option.id);
                      }}
                      {...(disabled ? {} : rowHover)}
                    >
                      <td className="ws-dataset-picker-col-select" style={{ ...workspaceTdStyle, textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                        <input
                          type={multiple ? 'checkbox' : 'radio'}
                          name="training-dataset-picker"
                          checked={checked}
                          disabled={disabled}
                          onChange={() => toggleDraft(option.id)}
                          aria-label={`选择 ${displayName}`}
                        />
                      </td>
                      <td className="ws-dataset-picker-col-name" style={{ ...workspaceTdStyle, wordBreak: 'break-all' }}>{displayName}</td>
                      <td className="ws-dataset-picker-col-task" style={workspaceTdStyle}>{resolveTrainingDatasetTaskLabel(option)}</td>
                      <td className="ws-dataset-picker-col-format" style={workspaceTdStyle}>{resolveTrainingDatasetFormatLabel(option)}</td>
                      <td className="ws-dataset-picker-col-count" style={workspaceTdStyle}>{resolveTrainingDatasetCountLabel(option)}</td>
                      <td className="ws-dataset-picker-col-time" style={workspaceTdTimeStyle}>
                        {formatDatasetTableCreatedAt(meta?.createdAt ?? option.createdAt)}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </WorkspaceTableWrap>
      </div>

      {selectionError ? <p className="ws-dataset-picker-error">{selectionError}</p> : null}
    </WorkspaceCenteredModal>
  );
}
