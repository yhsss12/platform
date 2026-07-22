/** 新建任务向导 — 表单选项（mock，不接后端） */

import {
  workspaceTaskDomainOptions,
  workspaceTaskPolicyOptions,
} from '@/lib/mock/workspaceTasksMock';

export const createTaskDomainOptions = [...workspaceTaskDomainOptions];

export const createTaskTypeOptions = [
  '拧紧',
  '装夹',
  '上下料',
  '插接',
  '移液',
  '抓取放置',
  '分拣',
  'AGV协同',
] as const;

export const createTaskSceneOptions = [
  '精密装配工位',
  '线缆插接工作台',
  '生化实验台',
  '柔性装配单元',
  '通用操作台',
] as const;

export const createTaskObjectOptions = [
  '螺丝',
  '工件',
  '夹具',
  '电批',
  '线缆',
  '接插件',
  '端子',
  '试管',
  '移液枪',
  '孔板',
  '托盘',
  '容器',
] as const;

export const createTaskRobotOptions = [
  '双臂协作机器人',
  '六轴工业机械臂',
  'AGV + 机械臂',
  '单臂机器人',
] as const;

export const createTaskPolicyOptions = [...workspaceTaskPolicyOptions];

export const createTaskControlModeOptions = [
  '策略模型推理',
  '规则控制',
  '遥操作回放',
  '混合控制',
] as const;

export const createTaskMetricOptions = [
  '任务成功率',
  '平均耗时',
  '碰撞次数',
  '轨迹误差',
  '完成质量',
  '对象状态变化',
  '异常终止次数',
] as const;

export const createTaskDataTypeOptions = [
  '轨迹数据',
  '图像数据',
  '状态数据',
  '动作数据',
  '运行日志',
] as const;

export const createTaskExportFormatOptions = ['HDF5', 'MCAP', 'JSON', 'CSV', '自定义格式'] as const;

export const CREATE_TASK_DEFAULTS = {
  name: '',
  domain: '精密制造',
  type: '拧紧',
  description: '面向双臂机器人完成两颗螺丝的定位、拧紧与复检。',
  goal: '完成目标对象的定位、操作与结果确认，并满足设定成功条件。',
  scene: '精密装配工位',
  objects: ['螺丝', '工件', '夹具', '电批'] as string[],
  initialState: '工件固定于夹具中，操作对象位于指定初始位置，机器人处于待执行姿态。',
  environmentDisturbance: '',
  robot: '双臂协作机器人',
  policy: 'ACT Policy',
  controlMode: '策略模型推理',
  runRounds: 10,
  metrics: ['任务成功率', '平均耗时', '碰撞次数'] as string[],
  successCondition:
    '所有目标对象完成指定操作，机器人无严重碰撞，最终状态满足任务约束。',
  generateData: true,
  dataTypes: ['轨迹数据', '图像数据', '状态数据'] as string[],
  exportFormat: 'HDF5',
};

export type CreateTaskFormState = typeof CREATE_TASK_DEFAULTS;

export const CREATE_TASK_STEPS = [
  { id: 1, title: '基本信息', hint: '定义任务名称、领域、类型与目标' },
  { id: 2, title: '场景与对象', hint: '选择工位场景、操作对象与初始状态' },
  { id: 3, title: '机器人与策略', hint: '配置执行平台、策略模型与控制方式' },
  { id: 4, title: '评测与数据', hint: '设置评测指标、成功条件与数据生成选项' },
  { id: 5, title: '确认配置', hint: '核对任务配置后保存' },
] as const;
