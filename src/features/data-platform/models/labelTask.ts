export type LabelTask = {
  id: string;              // Task ID，前端显示用，例如 "0001"
  backendTaskId?: string; // 后端真实的task_id（用于API调用），例如 "dec33e26"
  /** 稳定序号（1 起连续），用于列表展示与排序迁移 */
  taskNo?: number;
  name: string;            // 任务名称
  datasetDir: string;      // 数据集目录路径（服务器路径）
  datasetIds?: number[];   // 数据集 ID 列表（创建时从数据资产选择）
  dataCount?: number;      // 数据数量（可选，列表展示"100条"）
  deviceType?: string;    // 已废弃，保留兼容
  projectId?: string;      // 所属项目 ID（列表展示项目名称）
  labeler?: string;        // 标注员（可选，列表展示，空则显示 —）
  reviewer?: string;       // 审核员（可选，列表展示，空则显示 —）
  collector: string;       // 采集员（表字段；创建/编辑表单使用；列表不展示该列）
  createdAt: string;       // ISO string
  updatedAt: string;       // ISO string
  /** 是否已完成（标注完成） */
  completed?: boolean;
  verified?: boolean;
};

