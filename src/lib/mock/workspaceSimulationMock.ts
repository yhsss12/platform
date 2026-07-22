/** 仿真中心 mock 数据 — 仿真运行控制台（不接后端） */

export type SimulationRunStatus = 'running' | 'paused' | 'idle' | 'completed' | 'failed';
export type SimulationStepStatus = 'completed' | 'running' | 'pending' | 'error';
export type EventLogStatus = 'info' | 'success' | 'warning' | 'error';

export interface CurrentSimulation {
  id: string;
  taskName: string;
  scene: string;
  robot: string;
  policy: string;
  status: SimulationRunStatus;
  runDuration: string;
  progressPercent: number;
  currentStepLabel: string;
  engine: string;
  simTime: string;
  frame: number;
  objectsInScene: string[];
}

export interface SimulationStep {
  id: string;
  name: string;
  status: SimulationStepStatus;
}

export interface RobotStatusItem {
  name: string;
  state: string;
  detail: string;
  endEffector?: string;
}

export interface ObjectStatusItem {
  name: string;
  state: string;
  detail?: string;
}

export interface NodeStatusItem {
  name: string;
  state: string;
  cpu: string;
  gpu: string;
  latency: string;
}

export interface ProcessPrediction {
  progressPercent: number;
  successProbability: number;
  failureRisk: '低' | '中' | '高';
  currentPhase: string;
  estimatedRemaining: string;
}

export interface SimulationEventLog {
  id: string;
  time: string;
  type: string;
  content: string;
  status: EventLogStatus;
}

export interface RecentSimulationRun {
  id: string;
  taskName: string;
  scene: string;
  policy: string;
  status: SimulationRunStatus | 'completed' | 'failed';
  runTime: string;
}

export const currentSimulation: CurrentSimulation = {
  id: 'sim-run-cable-001',
  taskName: '线缆穿杆',
  scene: '桌面双杆穿线工位',
  robot: 'Panda',
  policy: 'scripted',
  status: 'running',
  runDuration: '00:04:38',
  progressPercent: 68,
  currentStepLabel: '穿线中',
  engine: 'MuJoCo',
  simTime: '00:02:14.6',
  frame: 4024,
  objectsInScene: ['线缆', '双杆', '桌面', '目标区域'],
};

export const simulationSteps: SimulationStep[] = [
  { id: 's1', name: '场景初始化', status: 'completed' },
  { id: 's2', name: '目标识别', status: 'completed' },
  { id: 's3', name: '抓取电批', status: 'completed' },
  { id: 's4', name: '拧紧第一颗螺丝', status: 'running' },
  { id: 's5', name: '拧紧第二颗螺丝', status: 'pending' },
  { id: 's6', name: '质量检测', status: 'pending' },
  { id: 's7', name: '完成', status: 'pending' },
];

export const dataGenerationSteps: SimulationStep[] = [
  { id: 'dg1', name: '初始化', status: 'completed' },
  { id: 'dg2', name: '加载场景', status: 'completed' },
  { id: 'dg3', name: '采集运行轨迹', status: 'running' },
  { id: 'dg4', name: '保存轨迹', status: 'pending' },
  { id: 'dg5', name: '保存视频', status: 'pending' },
  { id: 'dg6', name: '写入数据中心', status: 'pending' },
  { id: 'dg7', name: '完成', status: 'pending' },
];

export const evaluationConsoleSteps: SimulationStep[] = [
  { id: 'e1', name: '初始化', status: 'completed' },
  { id: 'e2', name: '识别', status: 'completed' },
  { id: 'e3', name: '抓取电批', status: 'completed' },
  { id: 'e4', name: '拧紧第一颗螺丝', status: 'running' },
  { id: 'e5', name: '拧紧第二颗螺丝', status: 'pending' },
  { id: 'e6', name: '质检', status: 'pending' },
  { id: 'e7', name: '完成', status: 'pending' },
];

export const robotStatus: RobotStatusItem[] = [
  {
    name: '左臂',
    state: '执行中',
    detail: 'J1–J6 跟踪正常 · 力矩 0.42 N·m',
    endEffector: '持握电批',
  },
  {
    name: '右臂',
    state: '协同待命',
    detail: '视觉辅助定位 · 就绪',
    endEffector: '空闲',
  },
];

export const objectStatus: ObjectStatusItem[] = [
  { name: '螺丝', state: '已识别', detail: '两颗待拧螺钉已定位' },
  { name: '工件', state: '已定位', detail: '装夹于工位' },
  { name: '电批', state: '已就绪', detail: '转速 320 RPM' },
  { name: '夹具', state: '已锁定', detail: '虎钳固定' },
];

export const nodeStatus: NodeStatusItem[] = [
  {
    name: 'sim-node-01',
    state: '运行中',
    cpu: '62%',
    gpu: '78%',
    latency: '12 ms',
  },
  {
    name: 'policy-worker',
    state: '推理中',
    cpu: '45%',
    gpu: '91%',
    latency: '8 ms',
  },
];

export const processPrediction: ProcessPrediction = {
  progressPercent: 68,
  successProbability: 0.82,
  failureRisk: '低',
  currentPhase: '拧紧第一颗螺丝',
  estimatedRemaining: '约 1 分 20 秒',
};

export const eventLogs: SimulationEventLog[] = [
  {
    id: 'e1',
    time: '10:21:03',
    type: '场景',
    content: '场景加载完成',
    status: 'success',
  },
  {
    id: 'e2',
    time: '10:21:08',
    type: '机器人',
    content: '双臂机器人初始化完成',
    status: 'success',
  },
  {
    id: 'e3',
    time: '10:21:15',
    type: '策略',
    content: 'ACT Policy 加载完成',
    status: 'success',
  },
  {
    id: 'e4',
    time: '10:21:23',
    type: '感知',
    content: '第一颗螺丝定位完成',
    status: 'info',
  },
  {
    id: 'e5',
    time: '10:21:41',
    type: '操作',
    content: '第一颗螺丝拧紧完成',
    status: 'success',
  },
  {
    id: 'e6',
    time: '10:21:52',
    type: '操作',
    content: '开始拧紧第一颗螺丝（精细阶段）',
    status: 'info',
  },
  {
    id: 'e7',
    time: '10:22:01',
    type: '评测',
    content: '过程评测：成功概率 0.82，失败风险低',
    status: 'info',
  },
];

export const recentSimulationRuns: RecentSimulationRun[] = [
  {
    id: 'sim-run-cable-001',
    taskName: '线缆穿杆',
    scene: '桌面双杆穿线工位',
    policy: 'scripted',
    status: 'completed',
    runTime: '2026-06-01 09:12 · 4m 12s',
  },
  {
    id: 'sim-run-dual-arm-001',
    taskName: '线缆整理',
    scene: '双臂桌面线缆整理工位',
    policy: '感知驱动操控',
    status: 'completed',
    runTime: '2026-05-31 16:40 · 12m 05s',
  },
  {
    id: 'sim-run-nut-001',
    taskName: '螺母装配',
    scene: 'NutAssembly 工位',
    policy: 'MimicGen',
    status: 'failed',
    runTime: '2026-05-31 11:22 · 8m 48s',
  },
];

export const simulationStatusLabel: Record<SimulationRunStatus, string> = {
  running: '运行中',
  paused: '已暂停',
  idle: '空闲',
  completed: '已完成',
  failed: '失败',
};

export const stepStatusLabel: Record<SimulationStepStatus, string> = {
  completed: '已完成',
  running: '运行中',
  pending: '等待中',
  error: '异常',
};
