/** Phase 1.5 概览页 mock 数据 */

export interface QuickStartItem {
  id: string;
  title: string;
  description: string;
  href: string;
  icon?: string;
}

export interface RecommendedCombo {
  id: string;
  taskTemplate: string;
  scene: string;
  robot: string;
  policy: string;
  metricHint: string;
}

export interface RecentDataTask {
  id: string;
  name: string;
  category: string;
  taskName: string;
  source: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  lastRunAt: string;
}

export interface LatestEvaluation {
  id: string;
  taskName: string;
  checkpoint: string;
  successRate: number;
  avgDurationSec: number;
  collisionCount: number;
  failureSummary: string;
  evaluatedAt: string;
}

export interface ResourceStatusItem {
  label: string;
  used: number;
  total: number;
  unit: string;
}

export const quickStartItems: QuickStartItem[] = [
  {
    id: 'qs1',
    title: '进入数据中心',
    description: '生成 demonstration 数据、构建训练/评测数据集，导入与导出标准格式',
    href: '/workspace/data',
  },
  {
    id: 'qs2',
    title: '进入训练中心',
    description: '选择训练数据集与模型类型，启动训练并生成 checkpoint',
    href: '/workspace/training',
  },
  {
    id: 'qs3',
    title: '进入评测中心',
    description: '加载 checkpoint 执行 rollout，对比 Benchmark 成功率与失败案例',
    href: '/workspace/evaluation',
  },
  {
    id: 'qs4',
    title: '进入资源中心',
    description: '维护任务模板、场景、操作对象、机器人、模型类型与评测指标',
    href: '/workspace/resources',
  },
];

export const recommendedCombos: RecommendedCombo[] = [
  {
    id: 'rc1',
    taskTemplate: '线缆穿杆',
    scene: '桌面双杆穿线工位',
    robot: 'Panda',
    policy: 'scripted',
    metricHint: '50 demos · MuJoCo · 50 轮',
  },
  {
    id: 'rc2',
    taskTemplate: '线缆整理',
    scene: '双臂桌面线缆整理工位',
    robot: 'Dual FR3',
    policy: 'episode_stability',
    metricHint: '20 demos · 稳定性评测',
  },
  {
    id: 'rc3',
    taskTemplate: '螺母装配',
    scene: 'NutAssembly 工位',
    robot: 'Panda 单臂机械臂',
    policy: 'MimicGen',
    metricHint: 'MimicGen · 训练数据集',
  },
];

export const recentDataTasks: RecentDataTask[] = [
  {
    id: 'data-cable-001',
    name: 'cable-threading-demo-50',
    category: '原始 demonstration 数据',
    taskName: '线缆穿杆',
    source: 'MuJoCo 生成',
    status: 'completed',
    lastRunAt: '2026-06-01 14:22',
  },
  {
    id: 'data-dual-arm-001',
    name: 'dual-arm-cable-train-20',
    category: '训练数据集',
    taskName: '线缆整理',
    source: 'MuJoCo 生成',
    status: 'completed',
    lastRunAt: '2026-06-01 11:05',
  },
  {
    id: 'data-nut-001',
    name: 'nut-assembly-mimicgen-20',
    category: '训练数据集',
    taskName: '螺母装配',
    source: 'MuJoCo 生成',
    status: 'completed',
    lastRunAt: '2026-06-02 09:00',
  },
];

/** @deprecated 使用 recentDataTasks */
export const recentSimulations = recentDataTasks;

export const latestEvaluations: LatestEvaluation[] = [
  {
    id: 'ev-cable-001',
    taskName: '线缆穿杆',
    checkpoint: '—',
    successRate: 88.0,
    avgDurationSec: 42.1,
    collisionCount: 1,
    failureSummary: '穿线末端偏移 2 次',
    evaluatedAt: '2026-06-01 14:30',
  },
  {
    id: 'ev-dual-arm-001',
    taskName: '线缆整理',
    checkpoint: '—',
    successRate: 75.0,
    avgDurationSec: 310.5,
    collisionCount: 0,
    failureSummary: '拉伸未达阈值 3 次',
    evaluatedAt: '2026-06-01 11:12',
  },
];

export const platformResources: ResourceStatusItem[] = [
  { label: '训练算力节点', used: 6, total: 10, unit: '节点' },
  { label: 'GPU 训练池', used: 72, total: 128, unit: 'GB' },
  { label: '训练数据集', used: 18, total: 30, unit: '套' },
  { label: 'checkpoint 槽位', used: 22, total: 40, unit: '个' },
];

export const overviewKpis = {
  demonstrationTotal: 48,
  trainingRunning: 3,
  evaluationDone: 412,
  checkpointCount: 22,
};

/** @deprecated */
export const overviewKpisLegacy = {
  simulationTotal: 156,
  simulationRunning: 4,
  evaluationDone: 412,
  onlineNodes: 8,
};
