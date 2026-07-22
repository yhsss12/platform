/**
 * HDF5 数据集 API
 */
import { apiGet, type ApiResponse } from './client';

/** 数据资产来源：本地导入 / 采集 / 标注 / 转换 */
export type DatasetSource = 'local' | 'collect' | 'label' | 'convert';

export interface HDF5Dataset {
  id: number;
  name: string;
  /** 所属项目 ID（与项目管理对接），展示时映射为 project.name */
  project: string | null;
  task: string | null;
  device: string | null;
  /** 来源：local=本地, collect=采集, label=标注, convert=转换；缺失/null 视为 local */
  source?: DatasetSource | null;
  created_at: string;
  file_size_bytes: number;
  duration_sec: number | null;
  format: string;
  storage_type: string;
  storage_uri: string;
  qc_status: string;
  label_status: string;
  assign_status: string;
}

/** 来源枚举显示：缺失/null → 本地 */
export function getSourceDisplay(source: DatasetSource | null | undefined): string {
  if (source === 'collect') return '采集';
  if (source === 'label') return '标注';
  if (source === 'convert') return '转换';
  return '本地';
}

export interface DatasetListResponse {
  items: HDF5Dataset[];
  total: number;
  page: number;
  page_size: number;
}

export interface DatasetQueryParams {
  keyword?: string;
  project?: string;
  /** 数据格式：hdf5 / mcap / lerobot */
  format?: string;
  page?: number;
  page_size?: number;
}

/**
 * 获取数据集列表
 */
export async function getDatasets(params: DatasetQueryParams = {}): Promise<ApiResponse<DatasetListResponse>> {
  const queryParams = new URLSearchParams();
  if (params.keyword) queryParams.append('keyword', params.keyword);
  if (params.project) queryParams.append('project', params.project);
  if (params.format) queryParams.append('format', params.format);
  if (params.page) queryParams.append('page', params.page.toString());
  if (params.page_size) queryParams.append('page_size', params.page_size.toString());
  
  const query = queryParams.toString();
  return apiGet<DatasetListResponse>(`/api/hdf5-datasets${query ? `?${query}` : ''}`);
}


