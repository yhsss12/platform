'use client';

import Link from 'next/link';
import type { Dataset } from '@/types/benchmark';
import {
  resolveDatasetFormatLabel,
  resolveDatasetSourceTaskLabel,
} from '@/lib/workspace/taskTemplateMapping';
import {
  resolveDatasetSourceLabel,
  resolveDatasetCountText,
  resolveDatasetSizeText,
} from '@/lib/workspace/datasetDisplay';
import { normalizeDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import { shouldShowDatasetTrainingLink } from '@/lib/workspace/datasetTrainingAccess';
import {
  isImportedWorkspaceDataset,
  normalizeImportedDatasetStatus,
  shouldShowImportedDatasetBuildActionInList,
} from '@/lib/workspace/datasetImportWorkflow';
import {
  workspaceBtnLink,
  workspaceBtnDanger,
  workspaceTableRowHoverHandlers,
  workspaceTableStyle,
  workspaceTdStyle,
  workspaceTdTimeStyle,
  workspaceThStyle,
  WorkspaceActionBar,
  WorkspaceRowCheckbox,
  WorkspaceTableEmptyCell,
  WorkspaceTableHeaderCheckbox,
  WorkspaceTableWrap,
} from '@/components/workspace/workspaceTableStyles';
import { resolveUnifiedDatasetReplayHref } from '@/lib/workspace/datasetTableActions';

function formatCreatedAt(value: string): string {
  if (!value) return '—';
  try {
    const d = new Date(value);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString('zh-CN', { hour12: false });
    }
  } catch {
    /* ignore */
  }
  return value.slice(0, 19).replace('T', ' ');
}

function taskLabel(dataset: Dataset): string {
  return resolveDatasetSourceTaskLabel(dataset);
}

function shouldShowDatasetReplayInList(dataset: Dataset, replayHref: string | null | undefined): replayHref is string {
  if (!replayHref) return false;
  if (!isImportedWorkspaceDataset(dataset)) return true;
  const status = normalizeImportedDatasetStatus(dataset.status);
  return status !== 'failed' && status !== 'needs_build' && status !== 'needs_mapping';
}

interface WorkspaceDatasetTableProps {
  datasets: Dataset[];
  selectedIds: Set<string>;
  onToggleRow: (id: string) => void;
  onToggleSelectAll: () => void;
  allPageSelected: boolean;
  onOpenDetail: (dataset: Dataset) => void;
  onDelete?: (dataset: Dataset) => void;
  onBuild?: (dataset: Dataset) => void;
  emptyMessage?: string;
}

export function WorkspaceDatasetTable({
  datasets,
  selectedIds,
  onToggleRow,
  onToggleSelectAll,
  allPageSelected,
  onOpenDetail,
  onDelete,
  onBuild,
  emptyMessage = '暂无数据集，请通过数据生成登记仿真数据集。',
}: WorkspaceDatasetTableProps) {
  const colCount = 9;
  const rowHover = workspaceTableRowHoverHandlers();

  return (
    <WorkspaceTableWrap>
      <table style={workspaceTableStyle}>
        <thead>
          <tr>
            <th style={{ ...workspaceThStyle, width: 40, textAlign: 'center' }}>
              <WorkspaceTableHeaderCheckbox
                checked={datasets.length > 0 && allPageSelected}
                onChange={onToggleSelectAll}
                disabled={datasets.length === 0}
              />
            </th>
            {[
              '数据集名称',
              '任务名称',
              '数据来源',
              '数据格式',
              '数据数量',
              '数据大小',
              '创建时间',
              '操作',
            ].map((h) => (
              <th key={h} style={workspaceThStyle}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {datasets.length === 0 ? (
            <WorkspaceTableEmptyCell colSpan={colCount} message={emptyMessage} />
          ) : (
            datasets.map((dataset) => {
              const replayHref = resolveUnifiedDatasetReplayHref(dataset);
              const showTraining = shouldShowDatasetTrainingLink(dataset);
              const showReplay = shouldShowDatasetReplayInList(dataset, replayHref);

              return (
                <tr key={dataset.id} style={{ transition: 'background-color 0.15s' }} {...rowHover}>
                  <td style={{ ...workspaceTdStyle, textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceRowCheckbox
                      checked={selectedIds.has(dataset.id)}
                      onChange={() => onToggleRow(dataset.id)}
                    />
                  </td>
                  <td style={{ ...workspaceTdStyle, wordBreak: 'break-all' }}>
                    {normalizeDatasetDisplayName({
                      displayName: dataset.displayName,
                      name: dataset.name,
                      taskType: dataset.taskType,
                      createdAt: dataset.createdAt,
                      sourceJobId: dataset.sourceJobId,
                    })}
                  </td>
                  <td style={workspaceTdStyle}>{taskLabel(dataset)}</td>
                  <td style={workspaceTdStyle}>{resolveDatasetSourceLabel(dataset)}</td>
                  <td style={workspaceTdStyle}>{resolveDatasetFormatLabel(dataset)}</td>
                  <td style={workspaceTdStyle}>{resolveDatasetCountText(dataset)}</td>
                  <td style={workspaceTdStyle}>{resolveDatasetSizeText(dataset)}</td>
                  <td style={workspaceTdTimeStyle}>{formatCreatedAt(dataset.createdAt)}</td>
                  <td style={workspaceTdStyle} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceActionBar>
                      <button type="button" style={workspaceBtnLink} onClick={() => onOpenDetail(dataset)}>
                        详情
                      </button>
                      {showReplay ? (
                        <Link href={replayHref} style={workspaceBtnLink}>
                          回放
                        </Link>
                      ) : null}
                      {showTraining ? (
                        <Link
                          href={`/workspace/training?dataset=${encodeURIComponent(dataset.id)}`}
                          style={workspaceBtnLink}
                        >
                          训练
                        </Link>
                      ) : null}
                      {shouldShowImportedDatasetBuildActionInList(dataset) && onBuild ? (
                        <button type="button" style={workspaceBtnLink} onClick={() => onBuild(dataset)}>
                          构建
                        </button>
                      ) : null}
                      {onDelete ? (
                        <button type="button" style={workspaceBtnDanger} onClick={() => onDelete(dataset)}>
                          删除
                        </button>
                      ) : null}
                    </WorkspaceActionBar>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </WorkspaceTableWrap>
  );
}
