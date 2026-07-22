'use client';

import Link from 'next/link';
import {
  formatListDataScale,
  formatListDatasetBuildStatus,
  formatListDisplayName,
  formatListStatusLabel,
  formatListTypeLabel,
  listDatasetBuildBadgeStatus,
  listStatusBadgeStatus,
  normalizeDataItem,
  type WorkspaceDataItem,
} from '@/lib/mock/workspaceDataMock';
import {
  getWorkspaceDataActions,
  type WorkspaceDataAction,
} from '@/lib/workspace/workspaceDataActions';
import { StatusBadge } from '@/components/workspace/workspaceUi';
import {
  workspaceBtnDanger,
  workspaceBtnLink,
  workspaceBtnPrimary,
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

function ActionButton({ action }: { action: WorkspaceDataAction }) {
  const style =
    action.variant === 'primary'
      ? workspaceBtnPrimary
      : action.variant === 'danger'
        ? workspaceBtnDanger
        : workspaceBtnLink;
  if (action.href) {
    return (
      <Link
        href={action.href}
        style={{ ...style, color: action.variant === 'primary' ? '#fff' : style.color }}
      >
        {action.label}
      </Link>
    );
  }
  return (
    <button type="button" style={style} onClick={action.onClick}>
      {action.label}
    </button>
  );
}

function RowActionsBar({
  item,
  onOpenDetail,
  onBuildDataset,
  onDelete,
}: {
  item: WorkspaceDataItem;
  onOpenDetail: (item: WorkspaceDataItem) => void;
  onBuildDataset: (item: WorkspaceDataItem) => void;
  onDelete?: (item: WorkspaceDataItem) => void;
}) {
  const actions = getWorkspaceDataActions(item, {
    onOpenDetail,
    onBuildDataset,
    onDelete,
  });

  return (
    <WorkspaceActionBar>
      {actions.map((action) => (
        <ActionButton key={action.key} action={action} />
      ))}
    </WorkspaceActionBar>
  );
}

interface WorkspaceDataTableProps {
  items: WorkspaceDataItem[];
  selectedIds: Set<string>;
  onToggleRow: (id: string) => void;
  onToggleSelectAll: () => void;
  allPageSelected: boolean;
  onOpenDetail: (item: WorkspaceDataItem) => void;
  onBuildDataset: (item: WorkspaceDataItem) => void;
  onExport: (item: WorkspaceDataItem) => void;
  onRegenerate?: (item: WorkspaceDataItem) => void;
  onDelete?: (item: WorkspaceDataItem) => void;
  emptyMessage?: string;
}

export function WorkspaceDataTable({
  items,
  selectedIds,
  onToggleRow,
  onToggleSelectAll,
  allPageSelected,
  onOpenDetail,
  onBuildDataset,
  onExport: _onExport,
  onRegenerate: _onRegenerate,
  onDelete,
  emptyMessage = '暂无匹配数据，请调整筛选条件',
}: WorkspaceDataTableProps) {
  const colCount = 9;
  const rowHover = workspaceTableRowHoverHandlers();

  return (
    <WorkspaceTableWrap>
      <table style={workspaceTableStyle}>
        <thead>
          <tr>
            <th style={{ ...workspaceThStyle, width: 40, textAlign: 'center' }}>
              <WorkspaceTableHeaderCheckbox
                checked={items.length > 0 && allPageSelected}
                onChange={onToggleSelectAll}
                disabled={items.length === 0}
              />
            </th>
            {['名称', '类型', '任务模板', '状态', '数据规模', '数据集状态', '创建时间', '操作'].map((h) => (
              <th key={h} style={workspaceThStyle}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.length === 0 ? (
            <WorkspaceTableEmptyCell colSpan={colCount} message={emptyMessage} />
          ) : (
            items.map((raw) => {
              const item = normalizeDataItem(raw);
              return (
                <tr key={item.id} style={{ transition: 'background-color 0.15s' }} {...rowHover}>
                  <td style={{ ...workspaceTdStyle, textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                    <WorkspaceRowCheckbox
                      checked={selectedIds.has(item.id)}
                      onChange={() => onToggleRow(item.id)}
                    />
                  </td>
                  <td style={{ ...workspaceTdStyle, wordBreak: 'break-all' }}>{formatListDisplayName(item)}</td>
                  <td style={workspaceTdStyle}>{formatListTypeLabel(item)}</td>
                  <td style={workspaceTdStyle}>{item.taskName}</td>
                  <td style={workspaceTdStyle}>
                    <StatusBadge
                      status={listStatusBadgeStatus(item)}
                      label={formatListStatusLabel(item)}
                    />
                  </td>
                  <td style={workspaceTdStyle}>{formatListDataScale(item)}</td>
                  <td style={workspaceTdStyle}>
                    <StatusBadge
                      status={listDatasetBuildBadgeStatus(item)}
                      label={formatListDatasetBuildStatus(item)}
                    />
                  </td>
                  <td style={workspaceTdTimeStyle}>{item.generatedAt}</td>
                  <td style={workspaceTdStyle} onClick={(e) => e.stopPropagation()}>
                    <RowActionsBar
                      item={item}
                      onOpenDetail={onOpenDetail}
                      onBuildDataset={onBuildDataset}
                      onDelete={onDelete}
                    />
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
