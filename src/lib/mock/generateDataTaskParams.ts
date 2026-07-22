import {
  CABLE_THREADING_DEFAULTS,
  CABLE_THREADING_DIFFICULTIES,
  CABLE_THREADING_TASK_NAME,
  isCableThreadingTask,
} from '@/lib/workspace/cableThreading';
import {
  DUAL_ARM_CABLE_DEFAULTS,
  DUAL_ARM_CABLE_RELEASE_MODES,
  DUAL_ARM_CABLE_STRETCH_MODES,
  isDualArmCableTask,
} from '@/lib/workspace/dualArmCable';
import {
  isNutAssemblyTask,
  NUT_ASSEMBLY_DEFAULTS,
} from '@/lib/workspace/nutAssembly';

export type GenerateDataTaskParamKind = 'select' | 'number' | 'text';

export interface GenerateDataSelectOption {
  value: string;
  label: string;
}

export const CABLE_OBJECT_MODEL_OPTIONS: GenerateDataSelectOption[] = [
  { value: 'composite_cable', label: '复合线缆模型（推荐）' },
  { value: 'composite_soft', label: '软体复合模型' },
  { value: 'rmb', label: 'RMB 线缆模型' },
  { value: 'flex', label: 'Flex 线缆模型（实验）' },
];

export interface GenerateDataTaskParamField {
  id: string;
  label: string;
  kind: GenerateDataTaskParamKind;
  options?: readonly string[];
  selectOptions?: readonly GenerateDataSelectOption[];
  defaultValue: string | number;
  min?: number;
  max?: number;
  fullWidth?: boolean;
}

export function getGenerateDataTaskParamFields(template: string): GenerateDataTaskParamField[] {
  if (isCableThreadingTask(template)) {
    return [
      {
        id: 'cable_type',
        label: '操作对象模型',
        kind: 'select',
        selectOptions: CABLE_OBJECT_MODEL_OPTIONS,
        defaultValue: CABLE_THREADING_DEFAULTS.cableModel,
      },
      {
        id: 'task_difficulty',
        label: '任务难度',
        kind: 'select',
        options: CABLE_THREADING_DIFFICULTIES,
        defaultValue: CABLE_THREADING_DEFAULTS.difficulty,
      },
      {
        id: 'max_steps',
        label: '最大步数',
        kind: 'number',
        defaultValue: CABLE_THREADING_DEFAULTS.horizon,
        min: 100,
        max: 1000,
      },
    ];
  }

  if (isDualArmCableTask(template)) {
    return [
      {
        id: 'max_cables',
        label: '线缆数量',
        kind: 'number',
        defaultValue: DUAL_ARM_CABLE_DEFAULTS.maxCables,
        min: 1,
        max: 8,
      },
      {
        id: 'stretch_mode',
        label: '拉伸模式',
        kind: 'select',
        selectOptions: DUAL_ARM_CABLE_STRETCH_MODES,
        defaultValue: DUAL_ARM_CABLE_DEFAULTS.stretchMode,
      },
      {
        id: 'release_mode',
        label: '释放模式',
        kind: 'select',
        selectOptions: DUAL_ARM_CABLE_RELEASE_MODES,
        defaultValue: DUAL_ARM_CABLE_DEFAULTS.releaseMode,
      },
    ];
  }

  if (isNutAssemblyTask(template)) {
    return [
      {
        id: 'horizon',
        label: '最大步数',
        kind: 'number',
        defaultValue: NUT_ASSEMBLY_DEFAULTS.horizon,
        min: 50,
        max: 1000,
        fullWidth: true,
      },
    ];
  }

  return [];
}

export function defaultTaskParamValues(template: string): Record<string, string | number> {
  const values: Record<string, string | number> = {};
  for (const field of getGenerateDataTaskParamFields(template)) {
    values[field.id] = field.defaultValue;
  }
  return values;
}

export function cableThreadingFieldsFromTaskParams(
  params: Record<string, string | number>
): {
  cableThreadingCableModel: string;
  cableThreadingDifficulty: string;
  cableThreadingHorizon: number;
} {
  return {
    cableThreadingCableModel: String(params.cable_type ?? CABLE_THREADING_DEFAULTS.cableModel),
    cableThreadingDifficulty: String(params.task_difficulty ?? CABLE_THREADING_DEFAULTS.difficulty),
    cableThreadingHorizon: Number(params.max_steps ?? CABLE_THREADING_DEFAULTS.horizon),
  };
}

export function nutAssemblyFieldsFromTaskParams(
  params: Record<string, string | number>
): {
  nutAssemblyEnvName: string;
  nutAssemblyHorizon: number;
} {
  return {
    nutAssemblyEnvName: NUT_ASSEMBLY_DEFAULTS.envName,
    nutAssemblyHorizon: Number(params.horizon ?? NUT_ASSEMBLY_DEFAULTS.horizon),
  };
}

export function generateDefaultDataName(template: string): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
  return `${template}数据_${stamp}`;
}

export { CABLE_THREADING_TASK_NAME };
