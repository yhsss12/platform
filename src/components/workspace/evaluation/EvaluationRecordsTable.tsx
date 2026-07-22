'use client';

import { buildCableThreadingReplayHref } from '@/lib/workspace/cableThreading';
import {
  buildDualArmEvalReplayHref,
  buildDualArmEvalReportHref,
  isDualArmEvalRow,
} from '@/lib/workspace/dualArmEvaluation';
import {
  buildIsaacEvalReplayHref,
  buildIsaacEvalReportHref,
  isIsaacEvalRow,
} from '@/lib/workspace/isaacBlockStacking';
import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import {
  formatEvaluationListType,
  formatEvaluationTaskListName,
} from '@/lib/mock/workspaceEvaluationRecordsMock';
import { getEvaluationRowDeleteKey, getEvaluationRowJobId } from '@/lib/workspace/evaluationJobId';
import { EvaluationStatusCell } from '@/components/workspace/evaluation/EvaluationStatusCell';
import {
  workspaceTableRowHoverHandlers,
  workspaceTableStyle,
  workspaceTdStyle,
  workspaceTdTimeStyle,
  workspaceThStyle,
  WorkspaceActionBar,
  WorkspaceActionLink,
  WorkspaceRowCheckbox,
  WorkspaceTableEmptyCell,
  WorkspaceTableHeaderCheckbox,
  WorkspaceTableWrap,
} from '@/components/workspace/workspaceTableStyles';

export const EVALUATION_ROUTES = {
  replay: '/workspace/replay',
} as const;

function replayHref(row: EvaluationTaskRow): string {
  const evalJobId =
    getEvaluationRowJobId(row) ||
    (row.id?.startsWith('ct_eval_') ? row.id : '') ||
    (row.evalJobId?.startsWith('ct_eval_') ? row.evalJobId : '');
  if (row.taskType === 'cable_threading' || evalJobId.startsWith('ct_eval_')) {
    return buildCableThreadingReplayHref({ evalId: evalJobId });
  }
  if (isDualArmEvalRow(row)) {
    return buildDualArmEvalReplayHref({ evalJobId });
  }
  if (isIsaacEvalRow(row)) {
    return buildIsaacEvalReplayHref({ evalJobId });
  }
  return `${EVALUATION_ROUTES.replay}?replayType=evaluation&evalId=${encodeURIComponent(evalJobId)}`;
}

function reportHref(row: EvaluationTaskRow): string {
  const evalJobId = getEvaluationRowJobId(row);
  if (isDualArmEvalRow(row)) {
    return buildDualArmEvalReportHref({ evalJobId });
  }
  if (isIsaacEvalRow(row)) {
    return buildIsaacEvalReportHref({ evalJobId });
  }
  return `/workspace/evaluation/report?evalId=${encodeURIComponent(evalJobId)}`;
}

interface EvaluationTasksTableProps {
  rows: EvaluationTaskRow[];
  loading?: boolean;
  emptyMessage?: string;
  selectedIds: Set<string>;
  onToggleRow: (row: EvaluationTaskRow) => void;
  onToggleSelectAll: () => void;
  allPageSelected: boolean;
  onDetail: (row: EvaluationTaskRow) => void;
  onDelete: (row: EvaluationTaskRow) => void;
}

function TaskRowActions({
  row,
  onDetail,
  onDelete,
}: {
  row: EvaluationTaskRow;
  onDetail: (row: EvaluationTaskRow) => void;
  onDelete: (row: EvaluationTaskRow) => void;
}) {
  const isDatasetEval = formatEvaluationListType(row) === '数据集评测';

  return (
    <WorkspaceActionBar>
      <WorkspaceActionLink label="详情" onClick={() => onDetail(row)} />
      <WorkspaceActionLink
        label="回放"
        href={replayHref(row)}
        disabled={isDatasetEval}
        title={isDatasetEval ? '数据集评测不支持回放' : undefined}
      />
      <WorkspaceActionLink label="报告" href={reportHref(row)} />
      <WorkspaceActionLink label="删除" variant="danger" onClick={() => onDelete(row)} />
    </WorkspaceActionBar>
  );
}

export function EvaluationTasksTable({
  rows,
  loading = false,
  emptyMessage,
  selectedIds,
  onToggleRow,
  onToggleSelectAll,
  allPageSelected,
  onDetail,
  onDelete,
}: EvaluationTasksTableProps) {
  const colCount = 8;
  const rowHover = workspaceTableRowHoverHandlers();

  return (
    <WorkspaceTableWrap>
      <table style={{ ...workspaceTableStyle, minWidth: 880 }}>
        <thead>
          <tr>
            <th style={{ ...workspaceThStyle, width: 40, textAlign: 'center' }}>
              <WorkspaceTableHeaderCheckbox
                checked={rows.length > 0 && allPageSelected}
                onChange={onToggleSelectAll}
                disabled={rows.length === 0}
              />
            </th>
            {[
              '评测任务名称',
              '评测类型',
              '关联任务',
              '成功率',
              '状态',
              '创建时间',
              '操作',
            ].map((h) => (
              <th key={h} style={workspaceThStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <WorkspaceTableEmptyCell
              colSpan={colCount}
              message={emptyMessage ?? (loading ? '加载评测任务…' : '暂无评测任务')}
            />
          ) : (
            rows.map((row) => {
              const taskName = formatEvaluationTaskListName(row);
              const evalType = formatEvaluationListType(row);
              const evalJobId = getEvaluationRowJobId(row);
              const deleteKey = getEvaluationRowDeleteKey(row);
              return (
                <tr key={deleteKey || evalJobId || row.id} style={{ transition: 'background-color 0.15s' }} {...rowHover}>
                  <td style={{ ...workspaceTdStyle, textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceRowCheckbox
                      checked={deleteKey ? selectedIds.has(deleteKey) : false}
                      onChange={() => onToggleRow(row)}
                      disabled={!deleteKey}
                    />
                  </td>
                  <td style={{ ...workspaceTdStyle, wordBreak: 'break-all' }} title={taskName}>
                    {taskName}
                  </td>
                  <td style={workspaceTdStyle}>{evalType}</td>
                  <td style={workspaceTdStyle} title={row.relatedTask}>
                    {row.relatedTask}
                  </td>
                  <td
                    style={{
                      ...workspaceTdStyle,
                      color: row.successStats?.available ? undefined : '#94a3b8',
                      textAlign: 'center',
                    }}
                    title={row.successStats?.available ? undefined : row.successStats?.reason}
                  >
                    {row.successStats?.display ?? '-/-'}
                  </td>
                  <td style={workspaceTdStyle}>
                    <EvaluationStatusCell row={row} />
                  </td>
                  <td style={workspaceTdTimeStyle}>{row.createdAt}</td>
                  <td style={workspaceTdStyle} onClick={(e) => e.stopPropagation()}>
                    <TaskRowActions row={row} onDetail={onDetail} onDelete={onDelete} />
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

/** @deprecated 使用 EvaluationTasksTable */
export const EvaluationRecordsTable = EvaluationTasksTable;
