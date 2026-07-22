/** 平台任务用户可见展示名（不改 taskType / templateId / jobId 前缀） */

import { FRANKA_STACK_CUBE_PRODUCT_NAME, isFrankStackCubeProductTask } from '@/lib/workspace/isaacStackCubeProduct';

export { FRANKA_STACK_CUBE_PRODUCT_NAME } from '@/lib/workspace/isaacStackCubeProduct';

export const NUT_ASSEMBLY_DISPLAY_NAME = '螺母装配';
export const NUT_ASSEMBLY_ENGLISH_NAME = 'Nut Assembly';
export const CABLE_THREADING_DISPLAY_NAME = '线缆穿杆';
export const DUAL_ARM_CABLE_DISPLAY_NAME = '线缆整理';
/** @deprecated 使用 FRANKA_STACK_CUBE_PRODUCT_NAME */
export const ISAAC_BLOCK_STACKING_DISPLAY_NAME = FRANKA_STACK_CUBE_PRODUCT_NAME;
export const ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME = FRANKA_STACK_CUBE_PRODUCT_NAME;
export const ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME = 'Franka 物体搬运';

export const TASK_TYPE_DISPLAY_NAMES: Record<string, string> = {
  cable_threading: CABLE_THREADING_DISPLAY_NAME,
  dual_arm_cable_manipulation: DUAL_ARM_CABLE_DISPLAY_NAME,
  block_stacking: FRANKA_STACK_CUBE_PRODUCT_NAME,
  isaac_block_stacking: FRANKA_STACK_CUBE_PRODUCT_NAME,
  isaaclab_franka_stack_cube: FRANKA_STACK_CUBE_PRODUCT_NAME,
  stacking: FRANKA_STACK_CUBE_PRODUCT_NAME,
  isaacsim_franka_pick_place: ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
  pick_and_place: ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
  nut_assembly: NUT_ASSEMBLY_DISPLAY_NAME,
};

export const TASK_TEMPLATE_DISPLAY_NAMES: Record<string, string> = {
  cable_threading_single_arm: CABLE_THREADING_DISPLAY_NAME,
  dual_arm_cable_manipulation: DUAL_ARM_CABLE_DISPLAY_NAME,
  isaac_block_stacking: FRANKA_STACK_CUBE_PRODUCT_NAME,
  isaaclab_franka_stack_cube: FRANKA_STACK_CUBE_PRODUCT_NAME,
  isaacsim_franka_pick_place: ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
  task_cable_threading_v1: CABLE_THREADING_DISPLAY_NAME,
  task_dual_arm_cable_manipulation_v1: DUAL_ARM_CABLE_DISPLAY_NAME,
  task_isaac_block_stacking_v1: FRANKA_STACK_CUBE_PRODUCT_NAME,
  task_isaaclab_franka_stack_cube_v1: FRANKA_STACK_CUBE_PRODUCT_NAME,
  task_isaacsim_franka_pick_place_v1: ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
  nut_assembly_single_arm: NUT_ASSEMBLY_DISPLAY_NAME,
  task_nut_assembly_v1: NUT_ASSEMBLY_DISPLAY_NAME,
};

const LEGACY_TASK_DISPLAY_NAMES: Record<string, string> = {
  单臂线缆穿杆: CABLE_THREADING_DISPLAY_NAME,
  单臂线缆穿杆任务: CABLE_THREADING_DISPLAY_NAME,
  单臂线缆: CABLE_THREADING_DISPLAY_NAME,
  双臂线缆操控: DUAL_ARM_CABLE_DISPLAY_NAME,
  双臂线缆操控任务: DUAL_ARM_CABLE_DISPLAY_NAME,
  双臂线缆: DUAL_ARM_CABLE_DISPLAY_NAME,
  线缆操控: DUAL_ARM_CABLE_DISPLAY_NAME,
  线缆整理: DUAL_ARM_CABLE_DISPLAY_NAME,
  线缆穿杆任务: CABLE_THREADING_DISPLAY_NAME,
  '线缆穿杆（单臂）': CABLE_THREADING_DISPLAY_NAME,
  物块堆叠: FRANKA_STACK_CUBE_PRODUCT_NAME,
  物块堆叠任务: FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Franka Stack Cube': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Isaac Lab Franka Stack Cube': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Franka 物块堆叠': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Franka 方块堆叠': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Stack Cube': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'stack cube': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'Block Stacking': FRANKA_STACK_CUBE_PRODUCT_NAME,
  'block stacking': FRANKA_STACK_CUBE_PRODUCT_NAME,
};

export const LEGACY_CABLE_THREADING_LABELS = new Set([
  '线缆穿杆（单臂）',
  '线缆穿杆任务',
  '单臂线缆穿杆',
  '单臂线缆穿杆任务',
  CABLE_THREADING_DISPLAY_NAME,
]);

export function normalizeTaskDisplayName(name: string | null | undefined): string {
  if (!name?.trim()) return '';
  const trimmed = name.trim();
  if (LEGACY_TASK_DISPLAY_NAMES[trimmed]) {
    return LEGACY_TASK_DISPLAY_NAMES[trimmed];
  }
  if (isFrankStackCubeProductTask(trimmed)) {
    return FRANKA_STACK_CUBE_PRODUCT_NAME;
  }
  if (trimmed.includes('单臂线缆穿杆')) {
    return trimmed.replace(/单臂线缆穿杆/g, CABLE_THREADING_DISPLAY_NAME);
  }
  if (trimmed.includes('双臂线缆操控')) {
    return trimmed.replace(/双臂线缆操控/g, DUAL_ARM_CABLE_DISPLAY_NAME);
  }
  if (trimmed.includes('线缆操控')) {
    return trimmed.replace(/线缆操控/g, DUAL_ARM_CABLE_DISPLAY_NAME);
  }
  return trimmed;
}

export function getTaskDisplayName(taskType: string | null | undefined): string {
  if (!taskType?.trim()) return '—';
  const key = taskType.trim();
  return TASK_TYPE_DISPLAY_NAMES[key] ?? (normalizeTaskDisplayName(key) || key);
}

export function getTaskTemplateDisplayName(templateId: string | null | undefined): string | null {
  if (!templateId?.trim()) return null;
  const key = templateId.trim();
  if (TASK_TEMPLATE_DISPLAY_NAMES[key]) {
    return TASK_TEMPLATE_DISPLAY_NAMES[key];
  }
  const normalized = normalizeTaskDisplayName(key);
  return normalized || null;
}

export function buildReplayPageTitle(taskName: string): string {
  const normalized = normalizeTaskDisplayName(taskName);
  if (normalized.endsWith('仿真场景回放')) {
    return normalized;
  }
  return `${normalized}仿真场景回放`;
}

export function matchesCableThreadingDisplayName(taskName: string | null | undefined): boolean {
  if (!taskName?.trim()) return false;
  const value = taskName.trim();
  return (
    value === CABLE_THREADING_DISPLAY_NAME ||
    LEGACY_CABLE_THREADING_LABELS.has(value) ||
    value.includes('单臂线缆穿杆') ||
    value.includes('线缆穿杆')
  );
}

export function matchesDualArmCableDisplayName(taskName: string | null | undefined): boolean {
  if (!taskName?.trim()) return false;
  const value = taskName.trim();
  return (
    value === DUAL_ARM_CABLE_DISPLAY_NAME ||
    value === '双臂线缆操控' ||
    value === '双臂线缆操控任务' ||
    value === '线缆操控' ||
    value.includes('双臂线缆操控') ||
    value.includes('线缆操控') ||
    value.includes('线缆整理')
  );
}

export function matchesNutAssemblyDisplayName(taskName: string | null | undefined): boolean {
  if (!taskName?.trim()) return false;
  const value = taskName.trim();
  return (
    value === NUT_ASSEMBLY_DISPLAY_NAME ||
    value === NUT_ASSEMBLY_ENGLISH_NAME ||
    value === 'Nut Assembly' ||
    value.includes('螺母装配')
  );
}
