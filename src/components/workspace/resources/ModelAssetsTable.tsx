'use client';

import type { ModelAsset } from '@/types/benchmark';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import {
  formatModelAssetRecipeLabel,
  resolveModelAssetColumnLabel,
  resolveModelAssetSourceLabel,
  resolveModelAssetTaskDatasetColumnLabel,
} from '@/lib/workspace/modelAssetDisplay';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
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

const tdEllipsis = workspaceTdEllipsisStyle;

export function ModelAssetsTable({
  rows,
  trainingByJobId,
  loading,
  emptyMessage,
  selectedIds,
  onToggleRow,
  onToggleSelectAll,
  allPageSelected,
  onDetail,
  onDelete,
}: {
  rows: ModelAsset[];
  trainingByJobId: Map<string, TrainingTaskRow>;
  loading?: boolean;
  emptyMessage?: string;
  selectedIds: Set<string>;
  onToggleRow: (id: string) => void;
  onToggleSelectAll: () => void;
  allPageSelected: boolean;
  onDetail: (asset: ModelAsset) => void;
  onDelete?: (asset: ModelAsset) => void;
}) {
  const colCount = 7;
  const rowHover = workspaceTableRowHoverHandlers();

  return (
    <WorkspaceTableWrap>
      <table style={{ ...workspaceTableStyle, tableLayout: 'fixed' }}>
        <colgroup>
          <col style={{ width: 40 }} />
          <col style={{ width: '24%' }} />
          <col style={{ width: '20%' }} />
          <col style={{ width: '14%' }} />
          <col style={{ width: '10%' }} />
          <col style={{ width: '12%' }} />
          <col style={{ width: '14%' }} />
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
            {['模型资产', '数据集 / 适用任务', '模型信息', '来源', '创建时间', '操作'].map((label) => (
              <th key={label} style={workspaceThStyle}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <WorkspaceTableEmptyCell
              colSpan={colCount}
              message={emptyMessage ?? (loading ? '加载模型资产…' : '暂无模型资产')}
            />
          ) : (
            rows.map((asset) => {
              const trainingRow = trainingByJobId.get(asset.sourceTrainingJobId);
              const assetLabel = resolveModelAssetColumnLabel(asset, trainingRow);
              const taskDataset = resolveModelAssetTaskDatasetColumnLabel(asset, trainingRow);
              const recipe = formatModelAssetRecipeLabel(asset);
              const sourceLabel = resolveModelAssetSourceLabel(asset);

              return (
                <tr
                  key={asset.id}
                  style={{ transition: 'background-color 0.15s', cursor: 'pointer' }}
                  {...rowHover}
                  onClick={() => onDetail(asset)}
                >
                  <td
                    style={{ ...workspaceTdMiddleStyle, textAlign: 'center' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <WorkspaceRowCheckbox
                      checked={selectedIds.has(asset.id)}
                      onChange={() => onToggleRow(asset.id)}
                    />
                  </td>
                  <td style={tdEllipsis} title={assetLabel}>
                    {assetLabel}
                  </td>
                  <td style={tdEllipsis} title={taskDataset}>
                    {taskDataset}
                  </td>
                  <td style={tdEllipsis} title={recipe}>
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: 6,
                        fontSize: 12,
                        fontWeight: 500,
                        backgroundColor: '#eff6ff',
                        color: '#1d4ed8',
                        maxWidth: '100%',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        verticalAlign: 'middle',
                      }}
                    >
                      {recipe}
                    </span>
                  </td>
                  <td style={tdEllipsis} title={sourceLabel}>
                    {sourceLabel}
                  </td>
                  <td style={workspaceTdTimeStyle}>
                    {formatDateTimeMinuteYmdSlash(asset.createdAt)}
                  </td>
                  <td style={workspaceTdMiddleStyle} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceActionBar>
                      <WorkspaceActionLink label="详情" onClick={() => onDetail(asset)} />
                      {onDelete ? (
                        <WorkspaceActionLink
                          label="删除"
                          variant="danger"
                          onClick={() => onDelete(asset)}
                        />
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
