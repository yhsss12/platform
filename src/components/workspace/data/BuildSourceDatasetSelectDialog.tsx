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
  BUILD_SOURCE_DATASET_PICKER_EMPTY_TEXT,
  BUILD_SOURCE_DATASET_PICKER_FILTERED_EMPTY_TEXT,
  BUILD_SOURCE_DATASET_PICKER_PAGE_SIZE,
  filterBuildSourceDatasets,
  filterBuildSourceDatasetsByKeyword,
  filterBuildSourceDatasetsByTask,
  formatBuildSourceDatasetCreatedAt,
  getBuildSourceTaskFilterOptions,
  paginateBuildSourceDatasets,
  resolveBuildSourceDatasetDisplayName,
} from '@/lib/workspace/buildSourceDatasetPicker';
import {
  resolveDatasetCountText,
  resolveDatasetSizeText,
} from '@/lib/workspace/datasetDisplay';
import { resolveDatasetFormatLabel, resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';
import type { Dataset } from '@/types/benchmark';
import './buildSourceDatasetSelectDialog.css';

export type BuildSourceDatasetSelectDialogProps = {
  open: boolean;
  datasets: Dataset[];
  selectedDatasetId?: string | null;
  onConfirm: (dataset: Dataset) => void;
  onCancel: () => void;
};

export function BuildSourceDatasetSelectDialog({
  open,
  datasets,
  selectedDatasetId,
  onConfirm,
  onCancel,
}: BuildSourceDatasetSelectDialogProps) {
  const [keyword, setKeyword] = useState('');
  const [taskFilter, setTaskFilter] = useState('all');
  const [page, setPage] = useState(1);
  const [draftId, setDraftId] = useState<string | null>(null);

  const sourceOptions = useMemo(() => filterBuildSourceDatasets(datasets), [datasets]);

  useEffect(() => {
    if (!open) return;
    setKeyword('');
    setTaskFilter('all');
    setPage(1);
    setDraftId(selectedDatasetId ?? null);
  }, [open, selectedDatasetId]);

  useEffect(() => {
    setPage(1);
  }, [keyword, taskFilter]);

  const taskFilterOptions = useMemo(
    () => getBuildSourceTaskFilterOptions(sourceOptions),
    [sourceOptions]
  );

  const filteredOptions = useMemo(() => {
    const searched = filterBuildSourceDatasetsByKeyword(sourceOptions, keyword);
    return filterBuildSourceDatasetsByTask(searched, taskFilter);
  }, [sourceOptions, keyword, taskFilter]);

  const filterSummary =
    sourceOptions.length > 0 && filteredOptions.length !== sourceOptions.length
      ? `已筛选 ${filteredOptions.length} / 共 ${sourceOptions.length} 个数据集`
      : null;

  const { items: pagedOptions, totalPages, page: safePage } = useMemo(
    () => paginateBuildSourceDatasets(filteredOptions, page, BUILD_SOURCE_DATASET_PICKER_PAGE_SIZE),
    [filteredOptions, page]
  );

  const showPagination = filteredOptions.length > BUILD_SOURCE_DATASET_PICKER_PAGE_SIZE;
  const hasActiveFilters = keyword.trim().length > 0 || taskFilter !== 'all';

  const handleConfirm = () => {
    if (!draftId) return;
    const selected = sourceOptions.find((item) => item.id === draftId);
    if (selected) onConfirm(selected);
  };

  const handleResetFilters = () => {
    setKeyword('');
    setTaskFilter('all');
    setPage(1);
  };

  return (
    <WorkspaceCenteredModal
      open={open}
      title="选择源数据集"
      titleId="build-source-dataset-picker-title"
      width={960}
      zIndex={1650}
      onClose={onCancel}
      footer={
        <div className="ws-dataset-picker-footer">
          <span className="ws-dataset-picker-footer-left">
            {draftId ? '已选择 1 个数据集' : '未选择数据集'}
          </span>
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
            <PrimaryButton onClick={handleConfirm} disabled={!draftId}>
              确认
            </PrimaryButton>
          </div>
        </div>
      }
    >
      <p style={{ margin: '0 0 12px', fontSize: 13, color: '#64748b', lineHeight: 1.55 }}>
        仅显示已导入的 HDF5 数据集。
      </p>

      <div className="ws-dataset-picker-search">
        <label style={workspaceModalFieldLabel} htmlFor="build-source-dataset-search">
          搜索数据集
        </label>
        <input
          id="build-source-dataset-search"
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
          className="ws-dataset-picker-filter-select ws-dataset-picker-filter-select-value"
          value={taskFilter}
          onChange={(e) => setTaskFilter(e.target.value)}
          aria-label="任务名称筛选"
        >
          {taskFilterOptions.map((item) => (
            <option key={item.value} value={item.value}>
              任务：{item.label}
            </option>
          ))}
        </select>
        <SecondaryButton onClick={handleResetFilters} disabled={!hasActiveFilters}>
          重置
        </SecondaryButton>
      </div>

      {filterSummary ? <p className="ws-dataset-picker-filter-summary">{filterSummary}</p> : null}

      <div className="ws-dataset-picker-table-wrap">
        <table className="ws-dataset-picker-table ws-build-source-picker-table">
          <thead>
            <tr>
              <th className="ws-dataset-picker-col-select">选择</th>
              <th className="ws-dataset-picker-col-name">数据集名称</th>
              <th className="ws-build-source-picker-col-task">任务名称</th>
              <th className="ws-dataset-picker-col-format">数据格式</th>
              <th className="ws-build-source-picker-col-count">数据数量</th>
              <th className="ws-build-source-picker-col-size">数据大小</th>
              <th className="ws-dataset-picker-col-date">创建时间</th>
            </tr>
          </thead>
          <tbody>
            {sourceOptions.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="ws-dataset-picker-empty">{BUILD_SOURCE_DATASET_PICKER_EMPTY_TEXT}</div>
                </td>
              </tr>
            ) : filteredOptions.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="ws-dataset-picker-empty">{BUILD_SOURCE_DATASET_PICKER_FILTERED_EMPTY_TEXT}</div>
                </td>
              </tr>
            ) : (
              pagedOptions.map((dataset) => {
                const checked = draftId === dataset.id;
                const rowClassName = [
                  'ws-dataset-picker-row',
                  checked ? 'ws-dataset-picker-row-selected' : '',
                ]
                  .filter(Boolean)
                  .join(' ');

                return (
                  <tr
                    key={dataset.id}
                    className={rowClassName}
                    onClick={() => setDraftId(dataset.id)}
                  >
                    <td className="ws-dataset-picker-col-select">
                      <input
                        type="radio"
                        name="build-source-dataset-picker"
                        checked={checked}
                        onChange={() => setDraftId(dataset.id)}
                        onClick={(event) => event.stopPropagation()}
                        aria-label={`选择 ${resolveBuildSourceDatasetDisplayName(dataset)}`}
                      />
                    </td>
                    <td className="ws-dataset-picker-col-name">
                      <span className="ws-dataset-picker-name" title={resolveBuildSourceDatasetDisplayName(dataset)}>
                        {resolveBuildSourceDatasetDisplayName(dataset)}
                      </span>
                    </td>
                    <td className="ws-build-source-picker-col-task">{resolveDatasetSourceTaskLabel(dataset)}</td>
                    <td className="ws-dataset-picker-col-format">{resolveDatasetFormatLabel(dataset)}</td>
                    <td className="ws-build-source-picker-col-count">{resolveDatasetCountText(dataset)}</td>
                    <td className="ws-build-source-picker-col-size">{resolveDatasetSizeText(dataset)}</td>
                    <td className="ws-dataset-picker-col-date">
                      {formatBuildSourceDatasetCreatedAt(dataset.createdAt)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </WorkspaceCenteredModal>
  );
}
