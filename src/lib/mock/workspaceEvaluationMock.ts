/** 评测中心 — Benchmark + 过程评测 */

export type EvaluationTypeFilter =
  | ''
  | 'task_level'
  | 'process'
  | 'policy_compare'
  | 'failure';

export type BenchmarkEvalStatus = 'completed' | 'running' | 'pending' | 'failed';

export interface EvaluationSummary {
  evaluatedTasks: number;
  avgSuccessRate: number;
  avgDurationSec: number;
  failureCaseCount: number;
  processEvaluableCount: number;
}

export interface BenchmarkResultRow {
  id: string;
  taskName: string;
  dataVolume: string;
  evalBackend: string;
  evalRounds: number;
  modelType: string;
  checkpoint: string;
  successRate: number;
  avgDurationSec: number;
  collisionCount: number;
  failureSummary: string;
  status: BenchmarkEvalStatus;
  /** @deprecated 保留兼容 */
  domain?: string;
  scene?: string;
  robot?: string;
  policy?: string;
  dataName?: string;
  trajectoryErrorMm?: number;
  evaluationType?: 'task_level' | 'process' | 'policy_compare' | 'failure';
}

export interface CurvePoint {
  t: number;
  value: number;
}

export interface ProcessStage {
  id: string;
  name: string;
  status: 'completed' | 'running' | 'pending' | 'risk';
  frameRange?: string;
}

export interface FailureNode {
  id: string;
  label: string;
  severity: 'low' | 'medium' | 'high';
  frame?: string;
  description: string;
}

export interface ProcessEvaluation {
  id: string;
  dataName: string;
  taskName: string;
  domain: string;
  scene: string;
  robot: string;
  policy: string;
  taskDescription: string;
  successCondition: string;
  inputs: string[];
  sampleFps: number;
  frameWisePrediction: boolean;
  detectFailureNodes: boolean;
  currentFrame: number;
  totalFrames: number;
  timestamp: string;
  progressPercent: number;
  successProbability: number;
  failureRisk: '低' | '中' | '高';
  trajectoryQuality: string;
  progressCurve: CurvePoint[];
  successCurve: CurvePoint[];
  stages: ProcessStage[];
  failureNodes: FailureNode[];
  anomalies: string[];
}

export interface FailureCaseRow {
  id: string;
  taskName: string;
  failedStage: string;
  reason: string;
  dataName: string;
}

export const evaluationSummary: EvaluationSummary = {
  evaluatedTasks: 24,
  avgSuccessRate: 88.5,
  avgDurationSec: 52.3,
  failureCaseCount: 7,
  processEvaluableCount: 18,
};

export interface BenchmarkComparisonGroup {
  key: string;
  taskName: string;
  dataVolume: string;
  evalBackend: string;
  evalRounds: number;
  entries: { modelType: string; modelVersion: string; successRate: number }[];
  maxSuccessRate: number;
}

export function formatBenchmarkSampleCount(value: string): string {
  if (/条/.test(value)) return value;
  const match = value.match(/(\d+)\s*demos?/i);
  if (match) return `${match[1]} 条`;
  return value;
}

export function buildBenchmarkComparisonGroups(rows: BenchmarkResultRow[]): BenchmarkComparisonGroup[] {
  const groups = new Map<string, BenchmarkResultRow[]>();

  for (const row of rows) {
    const volume = formatBenchmarkSampleCount(row.dataVolume);
    const key = `${row.taskName}|${volume}|${row.evalBackend}|${row.evalRounds}`;
    const list = groups.get(key) ?? [];
    list.push(row);
    groups.set(key, list);
  }

  return Array.from(groups.entries())
    .filter(([, entries]) => entries.length >= 2)
    .map(([key, entries]) => {
      const sorted = [...entries].sort((a, b) => b.successRate - a.successRate);
      const first = sorted[0];
      return {
        key,
        taskName: first.taskName,
        dataVolume: formatBenchmarkSampleCount(first.dataVolume),
        evalBackend: first.evalBackend,
        evalRounds: first.evalRounds,
        entries: sorted.map((e) => ({
          modelType: e.modelType,
          modelVersion: e.checkpoint,
          successRate: e.successRate,
        })),
        maxSuccessRate: sorted[0].successRate,
      };
    })
    .sort((a, b) => b.maxSuccessRate - a.maxSuccessRate);
}

export function countBenchmarkCompareTasks(rows: BenchmarkResultRow[]): number {
  return new Set(rows.map((r) => r.taskName)).size;
}

export const benchmarkResults: BenchmarkResultRow[] = [
  {
    id: 'bench-cable-001',
    taskName: '线缆穿杆',
    dataVolume: '50 条',
    evalBackend: 'MuJoCo',
    evalRounds: 50,
    modelType: 'scripted',
    checkpoint: '—',
    successRate: 88.0,
    avgDurationSec: 42.1,
    collisionCount: 1,
    failureSummary: '穿线末端偏移 2 次',
    status: 'completed',
    dataName: 'cable-threading-demo-50',
    evaluationType: 'policy_compare',
  },
  {
    id: 'bench-dual-arm-001',
    taskName: '线缆整理',
    dataVolume: '20 条',
    evalBackend: 'MuJoCo',
    evalRounds: 20,
    modelType: 'episode_stability',
    checkpoint: '—',
    successRate: 75.0,
    avgDurationSec: 310.5,
    collisionCount: 0,
    failureSummary: '拉伸未达阈值 3 次',
    status: 'completed',
    dataName: 'dual-arm-cable-train-20',
    evaluationType: 'task_level',
  },
];

const mockProgressCurve: CurvePoint[] = [
  { t: 0, value: 0.05 },
  { t: 10, value: 0.12 },
  { t: 20, value: 0.22 },
  { t: 30, value: 0.35 },
  { t: 40, value: 0.48 },
  { t: 50, value: 0.55 },
  { t: 60, value: 0.62 },
  { t: 68, value: 0.68 },
  { t: 75, value: 0.68 },
  { t: 85, value: 0.72 },
  { t: 100, value: 0.78 },
];

const mockSuccessCurve: CurvePoint[] = [
  { t: 0, value: 0.42 },
  { t: 10, value: 0.48 },
  { t: 20, value: 0.55 },
  { t: 30, value: 0.61 },
  { t: 40, value: 0.68 },
  { t: 50, value: 0.74 },
  { t: 60, value: 0.78 },
  { t: 68, value: 0.82 },
  { t: 75, value: 0.8 },
  { t: 85, value: 0.83 },
  { t: 100, value: 0.85 },
];

export const processEvaluationDefault: ProcessEvaluation = {
  id: 'proc-eval-cable-001',
  dataName: 'cable-threading-traj-001',
  taskName: '线缆穿杆',
  domain: '线缆操作',
  scene: '桌面双杆穿线工位',
  robot: 'Panda',
  policy: 'scripted',
  taskDescription: '固定线缆一端，拖拽另一端穿过双杆间隙，验证柔性对象穿线过程。',
  successCondition: '线缆末端穿过双杆间隙并落到目标区域，锚点稳定，无明显碰撞。',
  inputs: ['轨迹数据', '仿真视频', '状态日志'],
  sampleFps: 30,
  frameWisePrediction: true,
  detectFailureNodes: true,
  currentFrame: 4024,
  totalFrames: 5890,
  timestamp: '00:02:14.133',
  progressPercent: 68,
  successProbability: 0.82,
  failureRisk: '低',
  trajectoryQuality: 'A-',
  progressCurve: mockProgressCurve,
  successCurve: mockSuccessCurve,
  stages: [
    { id: 'st1', name: '场景初始化', status: 'completed', frameRange: '0–120' },
    { id: 'st2', name: '目标识别', status: 'completed', frameRange: '121–890' },
    { id: 'st3', name: '抓取电批', status: 'completed', frameRange: '891–1520' },
    { id: 'st4', name: '拧紧第一颗螺丝', status: 'running', frameRange: '1521–4024' },
    { id: 'st5', name: '拧紧第二颗螺丝', status: 'pending', frameRange: '—' },
    { id: 'st6', name: '质量检测', status: 'pending' },
  ],
  failureNodes: [
    {
      id: 'fn1',
      label: '第二颗螺丝姿态偏差',
      severity: 'medium',
      frame: '帧 4180–4250',
      description: '插入角度偏离 2.3°，成功概率短暂下降至 0.71',
    },
    {
      id: 'fn2',
      label: '目标识别延迟',
      severity: 'low',
      frame: '帧 640–720',
      description: '视觉置信度低于阈值 0.85 持续 1.2s',
    },
    {
      id: 'fn3',
      label: '末端接触不稳定',
      severity: 'low',
      frame: '帧 2890–2950',
      description: '力矩波动超出平稳段基线 15%',
    },
  ],
  anomalies: ['帧 4180–4250 进度预测回退 3%', '帧 2890 接触力尖峰'],
};

/** 过程评测详情页完整数据（Robometer 风格四宫格） */
export type ProcessEvaluationDetail = ProcessEvaluation;

export const processEvaluationDetail: ProcessEvaluationDetail = processEvaluationDefault;

/** 按数据名称获取过程评测详情 */
export function getProcessEvaluationDetail(dataName?: string | null): ProcessEvaluationDetail {
  if (!dataName || dataName === processEvaluationDefault.dataName) {
    return processEvaluationDefault;
  }
  const row = benchmarkResults.find((r) => r.dataName === dataName);
  if (!row) return processEvaluationDefault;
  return {
    ...processEvaluationDefault,
    id: `proc-detail-${row.id}`,
    dataName: row.dataName ?? processEvaluationDefault.dataName,
    taskName: row.taskName,
    domain: row.domain ?? '精密制造',
    scene: row.scene ?? '精密装配工位',
    robot: row.robot ?? '双臂协作机器人',
    policy: row.policy ?? row.modelType,
    progressPercent: Math.min(95, Math.round(row.successRate * 0.75)),
    successProbability: row.successRate / 100,
    failureRisk: row.successRate < 80 ? '中' : '低',
    trajectoryQuality: row.successRate >= 90 ? 'A-' : row.successRate >= 80 ? 'B+' : 'B',
  };
}

export const progressCurve = mockProgressCurve;
export const successCurve = mockSuccessCurve;
export const failureNodes = processEvaluationDefault.failureNodes;

export const failureCases: FailureCaseRow[] = [
  {
    id: 'fc-001',
    taskName: '线缆穿杆',
    failedStage: '穿线阶段',
    reason: '线缆末端未穿过双杆间隙',
    dataName: 'cable-threading-traj-008',
  },
  {
    id: 'fc-002',
    taskName: '线缆整理',
    failedStage: '拉伸阶段',
    reason: '双臂拉伸未达目标跨度',
    dataName: 'dual-arm-cable-traj-005',
  },
  {
    id: 'fc-003',
    taskName: '螺母装配',
    failedStage: '插入阶段',
    reason: '螺母与孔位对准失败',
    dataName: 'nut-assembly-traj-003',
  },
];

export const evaluationTypeFilterOptions: { value: EvaluationTypeFilter; label: string }[] = [
  { value: '', label: '全部' },
  { value: 'task_level', label: '任务级评测' },
  { value: 'process', label: '过程评测' },
  { value: 'policy_compare', label: '策略对比' },
  { value: 'failure', label: '失败案例' },
];

export const benchmarkStatusLabel: Record<BenchmarkEvalStatus, string> = {
  completed: '已完成',
  running: '评测中',
  pending: '待评测',
  failed: '失败',
};

export const workspaceEvaluationDomainOptions = [
  '精密制造',
  '线缆操作',
  '生化实验',
  '柔性装配',
  '通用操作',
] as const;

export const workspaceEvaluationPolicyOptions = [
  'ACT Policy',
  'Diffusion Policy',
  'VLA Policy',
  '规则控制策略',
] as const;

export function benchmarkStatusBadge(
  status: BenchmarkEvalStatus
): 'active' | 'running' | 'draft' | 'failed' {
  switch (status) {
    case 'completed':
      return 'active';
    case 'running':
      return 'running';
    case 'pending':
      return 'draft';
    case 'failed':
      return 'failed';
  }
}
