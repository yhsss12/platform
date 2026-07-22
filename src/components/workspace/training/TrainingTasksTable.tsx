'use client';

import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import {
  formatTrainingDatasetTitle,
  formatTrainingTaskTitle,
} from '@/lib/mock/workspaceTrainingMock';
import { TrainingStatusCell } from '@/components/workspace/training/TrainingStatusCell';
import { formatTrainingRecipeLabel } from '@/lib/workspace/trainingRecipe';
import {
  workspaceTableRowHoverHandlers,
  workspaceTableStyle,
  workspaceTdEllipsisStyle,
  workspaceTdMiddleStyle,
  workspaceTdTimeStyle,
  workspaceThStyle,
  WorkspaceActionBar,
  WorkspaceActionLink,
  WorkspaceRowCheckbox,
  WorkspaceTableEmptyCell,
  WorkspaceTableHeaderCheckbox,
  WorkspaceTableWrap,
} from '@/components/workspace/workspaceTableStyles';

export function TrainingTasksTable({
  rows,
  dataCenterItems = [],
  loading = false,
  emptyMessage,
  selectedIds,
  onToggleRow,
  onToggleSelectAll,
  allPageSelected,
  onDetail,
  onDelete,
}: {
  rows: TrainingTaskRow[];
  dataCenterItems?: WorkspaceDataItem[];
  loading?: boolean;
  emptyMessage?: string;
  selectedIds: Set<string>;
  onToggleRow: (id: string) => void;
  onToggleSelectAll: () => void;
  allPageSelected: boolean;
  onDetail: (row: TrainingTaskRow) => void;
  onDelete: (row: TrainingTaskRow) => void;
}) {
  const colCount = 7;
  const rowHover = workspaceTableRowHoverHandlers();

  return (
    <WorkspaceTableWrap>
      <table style={{ ...workspaceTableStyle, tableLayout: 'fixed' }}>
        <colgroup>
          <col style={{ width: 40 }} />
          <col style={{ width: '26%' }} />
          <col style={{ width: '20%' }} />
          <col style={{ width: '16%' }} />
          <col style={{ width: 108 }} />
          <col style={{ width: '13%' }} />
          <col style={{ width: '13%' }} />
        </colgroup>
        <thead>
          <tr>
            <th style={{ ...workspaceThStyle, width: 40, textAlign: 'center', verticalAlign: 'middle' }}>
              <WorkspaceTableHeaderCheckbox
                checked={rows.length > 0 && allPageSelected}
                onChange={onToggleSelectAll}
                disabled={rows.length === 0}
              />
            </th>
            {['训练任务', '数据集', '模型类型', '状态', '创建时间', '操作'].map((label) => (
              <th key={label} style={{ ...workspaceThStyle, verticalAlign: 'middle' }}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <WorkspaceTableEmptyCell
              colSpan={colCount}
              message={emptyMessage ?? (loading ? '加载训练任务…' : '暂无训练任务')}
            />
          ) : (
            rows.map((row) => {
              const taskTitle = formatTrainingTaskTitle(row);
              const datasetTitle = formatTrainingDatasetTitle(row, dataCenterItems);
              const recipeLabel = formatTrainingRecipeLabel(row.trainingBackend, row.modelType);

              return (
                <tr key={row.id} style={{ transition: 'background-color 0.15s' }} {...rowHover}>
                  <td
                    style={{ ...workspaceTdMiddleStyle, textAlign: 'center' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <WorkspaceRowCheckbox
                      checked={selectedIds.has(row.id)}
                      onChange={() => onToggleRow(row.id)}
                    />
                  </td>
                  <td style={workspaceTdEllipsisStyle} title={taskTitle}>
                    {taskTitle}
                  </td>
                  <td style={workspaceTdEllipsisStyle} title={datasetTitle}>
                    {datasetTitle}
                  </td>
                  <td style={workspaceTdEllipsisStyle} title={recipeLabel}>
                    {recipeLabel}
                  </td>
                  <td
                    style={{
                      ...workspaceTdMiddleStyle,
                      minWidth: 108,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    <TrainingStatusCell row={row} />
                  </td>
                  <td style={{ ...workspaceTdTimeStyle, verticalAlign: 'middle' }}>{row.createdAt}</td>
                  <td style={workspaceTdMiddleStyle} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceActionBar>
                      <WorkspaceActionLink label="详情" onClick={() => onDetail(row)} />
                      <WorkspaceActionLink
                        label="删除"
                        variant="danger"
                        onClick={() => onDelete(row)}
                      />
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
