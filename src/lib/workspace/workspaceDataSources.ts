import {
  normalizeDataItem,
  workspaceDataItemsMock,
  type WorkspaceDataItem,
} from '@/lib/mock/workspaceDataMock';
import {
  evaluationTasksMock,
  type EvaluationTaskRow,
} from '@/lib/mock/workspaceEvaluationRecordsMock';
import { listExtraDataItems, listExtraEvaluationTasks } from '@/lib/mock/workspaceMockFlowStore';
import { shouldShowWorkspaceDemo } from '@/lib/workspace/workspaceDemoConfig';

/** 数据中心 / 训练选集：真实 session 记录 + 可选演示数据 */
export function listWorkspaceDataItemsForUi(): WorkspaceDataItem[] {
  const extra = listExtraDataItems().map(normalizeDataItem);
  if (shouldShowWorkspaceDemo()) {
    return [...workspaceDataItemsMock, ...extra];
  }
  return extra;
}

/** 评测报告 / 回放：可选合并演示评测记录 */
export function listWorkspaceEvaluationTasksForUi(): EvaluationTaskRow[] {
  const extra = listExtraEvaluationTasks();
  if (shouldShowWorkspaceDemo()) {
    return [...evaluationTasksMock, ...extra];
  }
  return extra;
}
