/**
 * 项目数据资产查询（data-assets API）
 */
import {
  getDataAssets,
  getDatasetCountByProject as apiGetDatasetCountByProject,
  type DataAssetItem,
} from '@/features/data-platform/api/dataAssetsApi';

// 后端限制 page_size <= 500，这里使用安全上限
const LIST_PAGE_SIZE = 500;

export async function fetchProjectDatasets(
  projectId: string,
  projectName?: string
): Promise<{ items: DataAssetItem[]; usedFallback: boolean; rawResp?: unknown }> {
  const r1 = await getDataAssets({ project: projectId, page_size: LIST_PAGE_SIZE });
  const items1 = r1.ok && r1.data ? r1.data.items : [];
  if (items1.length > 0) {
    return { items: items1, usedFallback: false, rawResp: r1 };
  }
  if (projectName && projectName.trim()) {
    const r2 = await getDataAssets({ project: projectName, page_size: LIST_PAGE_SIZE });
    const items2 = r2.ok && r2.data ? r2.data.items : [];
    if (items2.length > 0) {
      return { items: items2, usedFallback: true, rawResp: r2 };
    }
  }
  return { items: [], usedFallback: false, rawResp: r1 };
}

export async function getProjectDatasetCount(
  projectId: string,
  projectName?: string
): Promise<number> {
  return apiGetDatasetCountByProject(projectId, projectName);
}

export async function getProjectDatasets(
  projectId: string,
  projectName?: string
): Promise<DataAssetItem[]> {
  const { items } = await fetchProjectDatasets(projectId, projectName);
  return items;
}
