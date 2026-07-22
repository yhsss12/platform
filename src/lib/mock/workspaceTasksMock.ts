/** 任务中心 mock 数据 — 围绕任务定义、场景、操作对象、机器人、策略与评测组织 */

export type WorkspaceTaskDomain =
  | '精密制造'
  | '线缆操作'
  | '生化实验'
  | '柔性装配'
  | '通用操作';

export type WorkspaceTaskStatus =
  | 'simulatable'
  | 'pending_config'
  | 'evaluated'
  | 'running';

export interface WorkspaceTask {
  id: string;
  name: string;
  domain: WorkspaceTaskDomain;
  type: string;
  status: WorkspaceTaskStatus;
  description: string;
  goal: string;
  scene: string;
  initialState: string;
  objects: string[];
  robot: string;
  policy: string;
  metrics: string[];
  averageSteps: number | null;
  estimatedDuration: string;
  dataStatus: string;
  trajectoryCount: number;
  evaluationStatus: string;
  successRate: string | null;
  lastRunTime: string;
  creator: string;
  tags: string[];
  /** 真实后端 API 任务标识（如 cable_threading） */
  backendTaskType?: string;
}

/** @deprecated 保留类型别名，避免外部引用断裂 */
export type WorkspaceTaskRow = WorkspaceTask;

export const workspaceTaskDomainOptions: WorkspaceTaskDomain[] = [
  '精密制造',
  '线缆操作',
  '生化实验',
  '柔性装配',
  '通用操作',
];

export const workspaceTaskTypeOptions = [
  '拧紧',
  '装夹',
  '上下料',
  '插接',
  '移液',
  '抓取放置',
  '分拣',
  'AGV协同',
  '固定',
  '转移',
  '装配',
] as const;

export const workspaceTaskPolicyOptions = [
  'ACT Policy',
  'Diffusion Policy',
  'VLA Policy',
  '规则控制策略',
] as const;

export const workspaceTaskRobotOptions = [
  '双臂协作机器人',
  '六轴工业机械臂',
  '七轴协作机械臂',
  '桌面协作机械臂',
  'AGV + 机械臂',
] as const;

export const workspaceTaskStatusFilterOptions: {
  value: '' | WorkspaceTaskStatus;
  label: string;
}[] = [
  { value: '', label: '全部' },
  { value: 'simulatable', label: '可仿真' },
  { value: 'pending_config', label: '待配置' },
  { value: 'running', label: '运行中' },
  { value: 'evaluated', label: '已评测' },
];

export const workspaceTaskDomainTabs: { key: 'all' | WorkspaceTaskDomain; label: string }[] = [
  { key: 'all', label: '全部' },
  ...workspaceTaskDomainOptions.map((d) => ({ key: d, label: d })),
];

export const workspaceTaskStatusLabel: Record<WorkspaceTaskStatus, string> = {
  simulatable: '可仿真',
  pending_config: '待配置',
  evaluated: '已评测',
  running: '运行中',
};

export const workspaceTasksMock: WorkspaceTask[] = [
  {
    id: 'task-008',
    name: '线缆穿杆',
    domain: '线缆操作',
    type: '穿线',
    status: 'simulatable',
    description:
      '固定线缆一端，拖拽另一端穿过两根桌面柱子之间的间隙，用于验证柔性对象操作中的数据生成、策略评测、回放与报告流程。',
    goal: '将线缆末端穿过双杆间隙并落到目标区域，满足穿线、直线度与落桌稳定性等终态条件。',
    scene: '桌面双杆穿线工位',
    initialState: '线缆平铺于桌面，一端固定于锚点，机械臂位于工作空间上方待命。',
    objects: ['线缆', '双杆', '桌面', '目标区域'],
    robot: 'Panda',
    policy: 'scripted',
    metrics: ['成功率', '穿线完成度', '直线度误差', '锚点稳定性'],
    averageSteps: 150,
    estimatedDuration: '约 2.5 min',
    dataStatus: '已接入后端',
    trajectoryCount: 0,
    evaluationStatus: '支持策略评测',
    successRate: null,
    lastRunTime: '—',
    creator: '平台',
    tags: ['MuJoCo', '柔性对象', '线缆操作'],
    backendTaskType: 'cable_threading',
  },
  {
    id: 'task-dual-arm-cable',
    name: '线缆整理',
    domain: '线缆操作',
    type: '柔性操控',
    status: 'simulatable',
    description:
      '基于双 Franka FR3 机械臂和视觉感知模块，对桌面杂乱线缆进行感知、抓取、拉伸和放置，实现双臂协作线缆整理。',
    goal: '完成 stacked DLO 线缆的感知识别、双臂抓取、拉伸整理与安全释放，输出 episode 过程视频与结果 JSON。',
    scene: '双臂桌面线缆整理工位',
    initialState: '多根柔性线缆杂乱堆叠于桌面，双臂机器人位于工作空间两侧待命。',
    objects: ['杂乱柔性线缆', '桌面', '双臂工位'],
    robot: 'Dual Franka FR3',
    policy: '感知驱动操控',
    metrics: [
      'episode 成功',
      '双臂接触',
      '拉伸达成',
      '垂度 / 跨度',
    ],
    averageSteps: null,
    estimatedDuration: '约 5–15 min',
    dataStatus: '已完成后端验证',
    trajectoryCount: 0,
    evaluationStatus: '支持稳定性 / 模型评测',
    successRate: null,
    lastRunTime: '—',
    creator: '平台',
    tags: ['MuJoCo', '双臂协作', '柔性线缆', '感知驱动'],
    backendTaskType: 'dual_arm_cable_manipulation',
  },
  {
    id: 'task-nut-assembly',
    name: '螺母装配',
    domain: '精密制造',
    type: '装配',
    status: 'simulatable',
    description: '基于 MimicGen 的螺母装配数据生成与训练评测任务。',
    goal: '完成螺母抓取、对准与装配，输出可训练 HDF5 数据集。',
    scene: 'NutAssembly 工位',
    initialState: '螺母与装配孔位于桌面工位，机械臂待命。',
    objects: ['螺母', '装配孔', '桌面'],
    robot: 'Panda 单臂机械臂',
    policy: 'MimicGen',
    metrics: ['成功率', '抓取次数', '插入成功率'],
    averageSteps: null,
    estimatedDuration: '约 3–10 min',
    dataStatus: '已接入后端',
    trajectoryCount: 0,
    evaluationStatus: '支持训练与评测',
    successRate: null,
    lastRunTime: '—',
    creator: '平台',
    tags: ['MuJoCo', 'MimicGen', '螺母装配'],
    backendTaskType: 'nut_assembly',
  },
  {
    id: 'task-010',
    name: '抓取放置任务',
    domain: '通用操作',
    type: '抓取放置',
    status: 'evaluated',
    description: '从杂乱料盘中识别并抓取目标物体，放置于指定区域。',
    goal: '抓取目标物体并准确放置于标记区域中心。',
    scene: '料盘分拣工位',
    initialState: '料盘内有多类物体，目标物体位置随机，放置区空置。',
    objects: ['目标物体', '干扰物体', '料盘', '放置区'],
    robot: '六轴工业机械臂',
    policy: 'ACT Policy',
    metrics: ['成功率', '抓取精度', '放置误差'],
    averageSteps: 52,
    estimatedDuration: '约 1.0 min',
    dataStatus: '56 条轨迹',
    trajectoryCount: 56,
    evaluationStatus: '已完成 80 次评测',
    successRate: '93.8%',
    lastRunTime: '2026-05-29 17:22',
    creator: '郑凯',
    tags: ['抓取放置', '通用操作', 'ACT'],
  },
  {
    id: 'task-011',
    name: '分拣任务',
    domain: '通用操作',
    type: '分拣',
    status: 'pending_config',
    description: '根据视觉类别将传送带上的物体分拣至对应料箱。',
    goal: '将三类物体分别投入正确料箱，分拣准确率达标。',
    scene: '传送带分拣线',
    initialState: '传送带运行，三类物体混合，三个料箱已就位。',
    objects: ['传送带', '混合物体', '分类料箱'],
    robot: '并联分拣机器人',
    policy: '规则控制策略',
    metrics: ['成功率', '分类准确率', '节拍时间'],
    averageSteps: null,
    estimatedDuration: '约 0.8 min/件',
    dataStatus: '未生成',
    trajectoryCount: 0,
    evaluationStatus: '未评测',
    successRate: null,
    lastRunTime: '—',
    creator: '黄莉',
    tags: ['分拣', '视觉', '通用操作'],
  },
  {
    id: 'task-012',
    name: 'AGV 协同搬运',
    domain: '精密制造',
    type: 'AGV协同',
    status: 'simulatable',
    description: 'AGV 与机械臂协同完成跨工位物料转运。',
    goal: 'AGV 送达后机械臂完成取放，全程无干涉。',
    scene: 'AGV 协同工位',
    initialState: 'AGV 停靠接驳点，料箱在 AGV 上，目标工位空闲。',
    objects: ['料箱', '接驳台', 'AGV 车体'],
    robot: 'AGV + 机械臂',
    policy: 'VLA Policy',
    metrics: ['成功率', '协同节拍', '路径偏差'],
    averageSteps: 88,
    estimatedDuration: '约 4.0 min',
    dataStatus: '6 条轨迹',
    trajectoryCount: 6,
    evaluationStatus: '待评测',
    successRate: null,
    lastRunTime: '2026-05-27 13:40',
    creator: '马超',
    tags: ['AGV协同', '精密制造', 'VLA'],
  },
];

export function taskStatusBadgeStatus(
  status: WorkspaceTaskStatus
): 'active' | 'draft' | 'running' | 'completed' | 'paused' {
  switch (status) {
    case 'simulatable':
      return 'active';
    case 'pending_config':
      return 'draft';
    case 'evaluated':
      return 'completed';
    case 'running':
      return 'running';
  }
}

export function workspaceTaskSummaryStats(tasks: WorkspaceTask[]) {
  const total = tasks.length;
  const simulatable = tasks.filter(
    (t) =>
      t.status === 'simulatable' ||
      t.status === 'evaluated' ||
      t.status === 'running'
  ).length;
  const withData = tasks.filter((t) => t.trajectoryCount > 0).length;
  const evaluated = tasks.filter((t) => t.status === 'evaluated').length;
  return { total, simulatable, withData, evaluated };
}

export function formatWorkspaceTaskObjects(objects: string[], max = 3): string {
  if (objects.length === 0) return '—';
  const shown = objects.slice(0, max).join('、');
  if (objects.length > max) return `${shown} 等${objects.length}项`;
  return shown;
}

/** 表格「数据」列文案 */
export function formatWorkspaceDataCell(task: WorkspaceTask): string {
  if (task.trajectoryCount > 0) {
    return `${task.trajectoryCount} 条轨迹`;
  }
  if (task.dataStatus.includes('生成中')) return '生成中';
  if (task.dataStatus.includes('已导出')) return '已导出';
  if (task.dataStatus.includes('未') || task.trajectoryCount === 0) return '未生成';
  return task.dataStatus;
}

/** 表格「评测」列文案 */
export function formatWorkspaceEvalCell(task: WorkspaceTask): string {
  if (task.successRate) return task.successRate;
  if (
    task.status === 'running' &&
    (task.evaluationStatus.includes('进行') || task.evaluationStatus.includes('运行'))
  ) {
    return '运行中';
  }
  if (task.evaluationStatus.includes('待评测')) return '待评测';
  if (
    !task.evaluationStatus ||
    task.evaluationStatus.includes('未评测')
  ) {
    return '未评测';
  }
  return task.evaluationStatus;
}

export function isWorkspaceDataMuted(text: string): boolean {
  return text === '未生成';
}

export function isWorkspaceEvalMuted(text: string): boolean {
  return text === '未评测' || text === '待评测';
}

export function isWorkspaceEvalSuccess(text: string): boolean {
  return text.includes('%');
}

/** @deprecated 保留供旧引用；新表格请用 formatWorkspaceDataCell / formatWorkspaceEvalCell */
export function formatWorkspaceDataEval(task: WorkspaceTask): {
  dataLine: string;
  evalLine: string;
} {
  return {
    dataLine: formatWorkspaceDataCell(task),
    evalLine: formatWorkspaceEvalCell(task),
  };
}
