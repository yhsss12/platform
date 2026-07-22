/** Phase 1.5 工作台 mock 数据 */

export type MockRowStatus = 'draft' | 'running' | 'completed' | 'failed' | 'active' | 'archived';

const statusLabel: Record<MockRowStatus, string> = {
  draft: '草稿',
  running: '进行中',
  completed: '已完成',
  failed: '失败',
  active: '可用',
  archived: '已归档',
};

export function mockStatusLabel(status: MockRowStatus): string {
  return statusLabel[status] ?? status;
}

/** 任务生成 — 选择器选项 */
export const taskGenTemplates = [
  { id: 'tpl-cable-threading', name: '线缆穿杆', desc: 'MuJoCo 柔性穿线数据生成' },
  { id: 'tpl-nut-assembly', name: '螺母装配', desc: 'MimicGen 装配数据生成' },
  { id: 'tpl-dual-arm-cable', name: '线缆整理', desc: '双臂协作线缆整理' },
  { id: 'tpl-stack-cube', name: '物块堆叠', desc: 'Isaac Lab Franka 方块堆叠' },
];

export const taskGenScenes = [
  { id: 'sc-screw', name: '精密装配工位', desc: '双工位拧紧与检测' },
  { id: 'sc-vise', name: '虎钳装夹场景', desc: '含扰动与夹具碰撞体' },
  { id: 'sc-transfer', name: '工位搬运场景', desc: '输送线与工位对接' },
];

export const taskGenRobots = [
  { id: 'rb-dual', name: '双臂协作机器人', desc: '14-DOF 双臂 + 手爪' },
  { id: 'rb-arm6', name: '六轴工业机械臂', desc: '6-DOF + 力矩传感器' },
  { id: 'rb-agv', name: '复合移动机器人（AGV）', desc: '全向底盘 + 单臂' },
];

export const taskGenPolicies = [
  { id: 'pol-act', name: 'ACT Policy', desc: '动作分块 Transformer 策略' },
  { id: 'pol-diff', name: 'Diffusion Policy', desc: '扩散模型轨迹生成' },
  { id: 'pol-rule', name: '规则控制策略', desc: '状态机 + 轨迹规划' },
];

export const taskGenMetrics = [
  { id: 'mt-assembly', name: '装配成功率指标集', desc: '成功率、节拍、扭矩' },
  { id: 'mt-motion', name: '运动精度指标集', desc: '轨迹误差、碰撞次数' },
  { id: 'mt-logistics', name: '物流搬运指标集', desc: '完成率、到站精度' },
];

/** 资源库条目 */
export interface ResourceItem {
  id: string;
  name: string;
  category: string;
  version: string;
  status: MockRowStatus;
  updatedAt: string;
  description: string;
  tags: string[];
}

export const taskTemplateResources: ResourceItem[] = [
  {
    id: 'tpl-cable-threading',
    name: '线缆穿杆',
    category: '线缆操作',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-28',
    description: 'MuJoCo 单臂线缆穿杆任务模板，支持数据生成与策略评测。',
    tags: ['MuJoCo', '穿线'],
  },
  {
    id: 'tpl-dual-arm-cable',
    name: '线缆整理',
    category: '线缆操作',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-22',
    description: '双臂协作线缆整理，支持 episode 生成与稳定性评测。',
    tags: ['MuJoCo', '双臂'],
  },
  {
    id: 'tpl-nut-assembly',
    name: '螺母装配',
    category: '装配',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-15',
    description: 'MimicGen 螺母装配数据生成任务。',
    tags: ['MuJoCo', 'MimicGen'],
  },
];

export const sceneResources: ResourceItem[] = [
  {
    id: 'sc-01',
    name: '精密装配工位',
    category: '装配单元',
    version: 'v3.0',
    status: 'active',
    updatedAt: '2026-05-25',
    description: '双工位拧紧布局，含工装与扭矩检测占位。',
    tags: ['工位', 'USD'],
  },
  {
    id: 'sc-02',
    name: '虎钳装夹场景',
    category: '装夹单元',
    version: 'v2.1',
    status: 'active',
    updatedAt: '2026-05-20',
    description: '虎钳、挡块与工件碰撞体，支持扰动参数。',
    tags: ['装夹'],
  },
  {
    id: 'sc-03',
    name: '工位搬运场景',
    category: '物流',
    version: 'v1.5',
    status: 'active',
    updatedAt: '2026-05-18',
    description: '输送线、缓存位与 AGV 停靠点。',
    tags: ['搬运', '输送'],
  },
];

export const assetResources: ResourceItem[] = [
  {
    id: 'as-pm-01',
    name: '螺丝',
    category: '精密制造',
    version: 'v1.2',
    status: 'active',
    updatedAt: '2026-05-28',
    description: 'M6 内六角螺丝与垫片组合，用于拧紧与装配任务。',
    tags: ['精密制造', '紧固件'],
  },
  {
    id: 'as-pm-01b',
    name: '工件',
    category: '精密制造',
    version: 'v1.1',
    status: 'active',
    updatedAt: '2026-05-27',
    description: '待装配铝合金工件，用于拧紧、装夹与检测任务。',
    tags: ['精密制造', '工件'],
  },
  {
    id: 'as-pm-01c',
    name: '电批',
    category: '精密制造',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-27',
    description: '电动拧紧工具，用于螺丝拧紧任务中的扭矩与姿态控制。',
    tags: ['精密制造', '工具'],
  },
  {
    id: 'as-pm-02',
    name: '毛料',
    category: '精密制造',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-26',
    description: '待加工铝合金毛料块，支持装夹与切削仿真。',
    tags: ['精密制造', '工件'],
  },
  {
    id: 'as-pm-03',
    name: '加工件',
    category: '精密制造',
    version: 'v2.1',
    status: 'active',
    updatedAt: '2026-05-24',
    description: '已完成粗加工的零件模型，用于检测与二次装夹。',
    tags: ['精密制造', '工件'],
  },
  {
    id: 'as-pm-04',
    name: '夹具',
    category: '精密制造',
    version: 'v3.0',
    status: 'active',
    updatedAt: '2026-05-22',
    description: '虎钳式夹具与定位块，支持工件固定与碰撞检测。',
    tags: ['精密制造', '夹具'],
  },
  {
    id: 'as-pm-05',
    name: '刀具',
    category: '精密制造',
    version: 'v1.5',
    status: 'active',
    updatedAt: '2026-05-20',
    description: '立铣刀与钻头模型，用于加工路径与接触仿真。',
    tags: ['精密制造', '工具'],
  },
  {
    id: 'as-cb-01',
    name: '线缆',
    category: '线缆操作',
    version: 'v2.0',
    status: 'active',
    updatedAt: '2026-05-19',
    description: '柔性线缆段，支持插拔、布线与张紧操作仿真。',
    tags: ['线缆操作', '线缆'],
  },
  {
    id: 'as-cb-02',
    name: '接插件',
    category: '线缆操作',
    version: 'v1.8',
    status: 'active',
    updatedAt: '2026-05-18',
    description: 'RJ45 / USB-C 接插件模型，用于插拔对准任务。',
    tags: ['线缆操作', '接插件'],
  },
  {
    id: 'as-cb-03',
    name: '端子',
    category: '线缆操作',
    version: 'v1.1',
    status: 'active',
    updatedAt: '2026-05-17',
    description: '压接端子与线鼻，用于线束装配与检测。',
    tags: ['线缆操作', '端子'],
  },
  {
    id: 'as-cb-04',
    name: '扎带',
    category: '线缆操作',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-16',
    description: '尼龙扎带与束线工具，用于线束整理任务。',
    tags: ['线缆操作', '耗材'],
  },
  {
    id: 'as-cb-05',
    name: '线束固定夹',
    category: '线缆操作',
    version: 'v1.2',
    status: 'active',
    updatedAt: '2026-05-15',
    description: '线束卡扣与导轨固定夹，用于布线固定仿真。',
    tags: ['线缆操作', '夹具'],
  },
  {
    id: 'as-bio-01',
    name: '试管',
    category: '生化实验',
    version: 'v2.0',
    status: 'active',
    updatedAt: '2026-05-14',
    description: '15 mL 离心管与试管架，用于移液与转移任务。',
    tags: ['生化实验', '容器'],
  },
  {
    id: 'as-bio-02',
    name: '移液枪',
    category: '生化实验',
    version: 'v3.1',
    status: 'active',
    updatedAt: '2026-05-13',
    description: '单通道移液枪与枪头，支持吸液与分液操作。',
    tags: ['生化实验', '工具'],
  },
  {
    id: 'as-bio-03',
    name: '孔板',
    category: '生化实验',
    version: 'v1.5',
    status: 'active',
    updatedAt: '2026-05-12',
    description: '96 孔板模型，用于高通量加样与检测布局。',
    tags: ['生化实验', '容器'],
  },
  {
    id: 'as-bio-04',
    name: '培养皿',
    category: '生化实验',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-11',
    description: '培养皿与盖，用于细胞培养相关操作仿真。',
    tags: ['生化实验', '容器'],
  },
  {
    id: 'as-bio-05',
    name: '试剂瓶',
    category: '生化实验',
    version: 'v1.2',
    status: 'active',
    updatedAt: '2026-05-10',
    description: '试剂瓶与标签，支持开盖、倾倒与移液来源配置。',
    tags: ['生化实验', '容器'],
  },
  {
    id: 'as-gen-01',
    name: '托盘',
    category: '通用操作',
    version: 'v2.0',
    status: 'active',
    updatedAt: '2026-05-09',
    description: '料盘与托盘，用于上下料与分拣任务。',
    tags: ['通用操作', '容器'],
  },
  {
    id: 'as-gen-02',
    name: '工具',
    category: '通用操作',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-08',
    description: '通用操作工具集（扳手、螺丝刀等）占位模型。',
    tags: ['通用操作', '工具'],
  },
  {
    id: 'as-gen-03',
    name: '容器',
    category: '通用操作',
    version: 'v1.1',
    status: 'active',
    updatedAt: '2026-05-07',
    description: '通用料盒与收纳容器，适用于多类搬运任务。',
    tags: ['通用操作', '容器'],
  },
  {
    id: 'as-gen-04',
    name: '标定块',
    category: '通用操作',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-06',
    description: '手眼标定与相机标定用标准块，用于精度验证场景。',
    tags: ['通用操作', '标定'],
  },
];

export const robotResources: ResourceItem[] = [
  {
    id: 'rb-01',
    name: '双臂协作机器人',
    category: '协作臂',
    version: 'v4.2',
    status: 'active',
    updatedAt: '2026-05-08',
    description: '14-DOF 双臂，带平行夹爪与腕部力传感。',
    tags: ['双臂', '14-DOF'],
  },
  {
    id: 'rb-02',
    name: '六轴工业机械臂',
    category: '工业臂',
    version: 'v3.1',
    status: 'active',
    updatedAt: '2026-05-07',
    description: '标准六轴臂，支持力控与轨迹跟踪评测。',
    tags: ['6-DOF'],
  },
  {
    id: 'rb-03',
    name: '复合移动机器人（AGV）',
    category: '移动操作',
    version: 'v2.0',
    status: 'active',
    updatedAt: '2026-05-05',
    description: '全向 AGV 底盘 + 单臂，用于上下料仿真。',
    tags: ['AGV', '移动操作'],
  },
];

export const policyResources: ResourceItem[] = [
  {
    id: 'pl-isaac-stack-scripted-expert',
    name: '物块堆叠脚本专家策略',
    category: '脚本专家策略',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-06-17',
    description:
      '基于状态机和 IK 控制生成堆叠轨迹，已通过 Isaac Lab 冒烟验证。适用任务：物块堆叠。',
    tags: ['Isaac Lab', '物块堆叠', '已接入', '脚本专家策略'],
  },
  {
    id: 'pl-ckpt-01',
    name: 'ckpt-screw-act-50-e80',
    category: 'ACT Policy',
    version: 'e80',
    status: 'active',
    updatedAt: '2026-06-01',
    description: '线缆穿杆 · cable-threading-demo-50 · 50 条 · 训练中心产出',
    tags: ['ACT', '可用于评测', '模型版本'],
  },
  {
    id: 'pl-ckpt-02',
    name: 'ckpt-screw-dp3-50-e100',
    category: 'DP3',
    version: 'e100',
    status: 'active',
    updatedAt: '2026-06-02',
    description: '线缆整理 · dual-arm-cable-train-20 · 20 条 · 训练中心产出',
    tags: ['Diffusion Policy', '可用于评测', '模型版本'],
  },
  {
    id: 'pl-01',
    name: 'ACT Policy（模板）',
    category: '学习策略',
    version: 'v1.4',
    status: 'active',
    updatedAt: '2026-05-29',
    description: 'ACT 策略架构模板，训练后生成模型版本供评测使用。',
    tags: ['ACT', '模仿学习'],
  },
  {
    id: 'pl-02',
    name: 'Diffusion Policy（模板）',
    category: '学习策略',
    version: 'v0.9',
    status: 'active',
    updatedAt: '2026-05-27',
    description: '扩散策略架构模板，适用于装夹与精细操作场景。',
    tags: ['扩散模型'],
  },
  {
    id: 'pl-03',
    name: 'OpenVLA（模板）',
    category: 'VLA 策略',
    version: 'v0.3',
    status: 'active',
    updatedAt: '2026-05-25',
    description: '视觉-语言-动作策略模板，适用于移液等语义操作任务。',
    tags: ['VLA', 'OpenVLA'],
  },
  {
    id: 'pl-04',
    name: '规则控制策略',
    category: '基线策略',
    version: 'v2.0',
    status: 'active',
    updatedAt: '2026-05-20',
    description: 'AGV 上下料与工位对接的规则基线，无需训练模型版本。',
    tags: ['规则', '基线'],
  },
];

export const metricResources: ResourceItem[] = [
  {
    id: 'mt-01',
    name: '装配成功率指标集',
    category: '装配评测',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-01',
    description: '成功率、平均节拍、拧紧扭矩达标率。',
    tags: ['成功率', '节拍'],
  },
  {
    id: 'mt-02',
    name: '运动精度指标集',
    category: '运动评测',
    version: 'v1.0',
    status: 'active',
    updatedAt: '2026-05-01',
    description: '轨迹误差、碰撞次数、末端位姿偏差。',
    tags: ['轨迹', '碰撞'],
  },
];

/** 仿真运行控制台 */
export const simulationConsole = {
  currentTask: {
    id: 'sim-2041',
    name: '双次拧螺丝 — 批次 A',
    scene: '精密装配工位',
    robot: '双臂协作机器人',
    policy: 'ACT Policy',
    status: 'running' as const,
    progress: 62,
    step: '第二次拧紧 · 步骤 3/5',
  },
  steps: [
    { name: '接近螺栓', done: true },
    { name: '第一次拧紧', done: true },
    { name: '第二次拧紧', done: false, current: true },
    { name: '扭矩检测', done: false },
    { name: '退刀与复位', done: false },
  ],
  robots: [
    { name: '左臂', state: '运动中', joint: 'J1–J6 跟踪正常' },
    { name: '右臂', state: '等待', joint: '待命' },
  ],
  objects: [
    { name: '螺栓 M6', state: '已拧紧 ×1' },
    { name: '工装板', state: '固定' },
    { name: '扭矩传感器', state: '采样中' },
  ],
  logs: [
    '[14:22:01] 仿真环境初始化完成',
    '[14:22:05] ACT Policy 加载成功',
    '[14:22:18] 第一次拧紧完成，扭矩 1.2 N·m',
    '[14:22:35] 开始第二次拧紧',
  ],
};

/** 评测分析 */
export const evaluationBenchmark = {
  summary: {
    successRate: 88.5,
    avgTimeSec: 52.3,
    collisionCount: 1.8,
    trajectoryErrorMm: 3.2,
  },
  policyCompare: [
    { policy: 'ACT Policy', success: 91.2, avgTime: 48.6, collisions: 2 },
    { policy: 'Diffusion Policy', success: 96.5, avgTime: 32.1, collisions: 0 },
    { policy: '规则控制策略', success: 68.0, avgTime: 125.4, collisions: 5 },
  ],
  failures: [
    { task: '工位上下料 — 夜班回归', reason: 'AGV 到站超差导致抓取失败', count: 3 },
    { task: '双次拧螺丝 — 批次 A', reason: '第二次拧紧扭矩未达标', count: 2 },
  ],
};

/** 数据回放 */
export interface ReplayTimelinePoint {
  t: string;
  label: string;
}

export interface ReplaySensorSummary {
  torquePeak: string;
  visualConfidence: string;
  endEffectorError: string;
  collisionCount: string;
}

export interface ReplaySession {
  id: string;
  /** 关联评测任务 ID，用于回放页 deep link */
  evalId?: string;
  taskName: string;
  runNumber: string;
  scene: string;
  robot: string;
  modelVersion: string;
  modelType?: string;
  evalBackend: string;
  evalRounds?: number;
  duration: string;
  status: 'completed' | 'failed';
  successRate?: number;
  failureReason?: string;
  failureStage?: string;
  timeline: ReplayTimelinePoint[];
  sensors: ReplaySensorSummary;
  logs: string[];
}

export function resolveReplaySessionIdByEvalId(evalId: string | null | undefined): string | null {
  if (!evalId) return null;
  const match = replaySessions.find((s) => s.evalId === evalId);
  return match?.id ?? null;
}

export const replaySessions: ReplaySession[] = [
  {
    id: 'rp-cable-001',
    evalId: 'eval-cable-001',
    taskName: '线缆穿杆',
    runNumber: 'Run 12',
    scene: '桌面双杆穿线工位',
    robot: 'Panda',
    modelVersion: '—',
    modelType: 'scripted',
    evalBackend: 'MuJoCo',
    evalRounds: 50,
    duration: '3m12s',
    status: 'completed',
    successRate: 88.0,
    timeline: [
      { t: '00:00', label: '环境重置' },
      { t: '00:12', label: '抓取线缆' },
      { t: '01:05', label: '穿线中' },
      { t: '02:40', label: '落桌稳定' },
      { t: '03:12', label: '任务完成' },
    ],
    sensors: {
      torquePeak: '—',
      visualConfidence: '0.91',
      endEffectorError: '3.2 mm',
      collisionCount: '1',
    },
    logs: [
      '[00:00:00] 仿真环境加载完成',
      '[00:00:12] 专家策略开始执行',
      '[00:01:05] 穿线阶段进行中',
      '[00:02:40] 落桌稳定',
      '[00:03:12] 任务完成，成功率 88.0%',
    ],
  },
  {
    id: 'rp-dual-arm-001',
    evalId: 'eval-dual-arm-001',
    taskName: '线缆整理',
    runNumber: 'Run 3',
    scene: '双臂桌面线缆整理工位',
    robot: 'Dual FR3',
    modelVersion: '—',
    modelType: 'episode_stability',
    evalBackend: 'MuJoCo',
    evalRounds: 20,
    duration: '12m05s',
    status: 'completed',
    successRate: 75.0,
    timeline: [
      { t: '00:00', label: '环境重置' },
      { t: '02:00', label: '双臂抓取' },
      { t: '06:30', label: '拉伸整理' },
      { t: '10:45', label: '释放放置' },
      { t: '12:05', label: '任务完成' },
    ],
    sensors: {
      torquePeak: '—',
      visualConfidence: '0.88',
      endEffectorError: '—',
      collisionCount: '0',
    },
    logs: [
      '[00:00:00] 双臂 episode 开始',
      '[00:02:00] 双臂接触线缆',
      '[00:06:30] 拉伸达到目标跨度',
      '[00:12:05] episode 完成',
    ],
  },
];

/** @deprecated 使用 ReplaySession.timeline */
export const replayTimeline = replaySessions[0].timeline;

/** 实验记录 */
export interface ExperimentBatch {
  id: string;
  name: string;
  scene: string;
  policy: string;
  rounds: string;
  successRate: number;
  creator: string;
  createdAt: string;
  status: MockRowStatus;
}

export const experimentBatches: ExperimentBatch[] = [
  {
    id: 'exp-301',
    name: '拧螺丝策略对比 — 六月批次',
    scene: '精密装配工位',
    policy: 'ACT vs Diffusion',
    rounds: '20 / 20',
    successRate: 89.5,
    creator: '张工',
    createdAt: '2026-05-28 09:00',
    status: 'completed',
  },
  {
    id: 'exp-302',
    name: '装夹扰动扫描',
    scene: '虎钳装夹场景',
    policy: 'Diffusion Policy',
    rounds: '8 / 15',
    successRate: 75.0,
    creator: '李工',
    createdAt: '2026-06-01 08:30',
    status: 'running',
  },
  {
    id: 'exp-303',
    name: 'AGV 上下料回归',
    scene: '工位搬运场景',
    policy: '规则控制策略',
    rounds: '30 / 30',
    successRate: 68.0,
    creator: '王工',
    createdAt: '2026-05-25 14:00',
    status: 'completed',
  },
];
