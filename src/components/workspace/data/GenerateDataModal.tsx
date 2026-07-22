'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import {
  nutAssemblyFieldsFromTaskParams,
  cableThreadingFieldsFromTaskParams,
  defaultTaskParamValues,
  generateDefaultDataName,
  getGenerateDataTaskParamFields,
  type GenerateDataTaskParamField,
} from '@/lib/mock/generateDataTaskParams';
import {
  generateDataSimEnvironmentUiOptions,
} from '@/lib/mock/workspaceDataMock';
import {
  physicsProxyModelOptions,
  physicsProxyModeLabel,
  type PhysicsProxyMode,
} from '@/lib/mock/physicsProxiesMock';
import {
  CABLE_THREADING_DEFAULTS,
  CABLE_THREADING_ROBOTS,
  isCableThreadingTask,
} from '@/lib/workspace/cableThreading';
import {
  DUAL_ARM_CABLE_DEFAULTS,
  dualArmCableFieldsFromTaskParams,
  isDualArmCableTask,
} from '@/lib/workspace/dualArmCable';
import {
  isNutAssemblyTask,
  NUT_ASSEMBLY_DEFAULTS,
  NUT_ASSEMBLY_DEFAULT_ROBOT,
  NUT_ASSEMBLY_ROBOT_OPTIONS,
} from '@/lib/workspace/nutAssembly';
import {
  getNutAssemblyMimicgenEnvStatus,
  getNutAssemblyPinnModelStatus,
  type NutAssemblyMimicgenEnvStatus,
  type NutAssemblyPinnModelStatus,
} from '@/lib/api/nutAssemblyClient';
import {
  defaultNutAssemblyPathParams,
  type NutAssemblyPathParamDefaults,
} from '@/lib/workspace/generateDataTaskParams';
import {
  filterNutAssemblySourceDemoDatasets,
  resolveNutAssemblyEffectiveSourceDemoDatasetId,
  resolveNutAssemblySourceDemoPath,
} from '@/lib/workspace/nutAssemblySeedDatasets';
import {
  NutAssemblyGenerateForm,
  useNutAssemblyGenerateValidation,
} from '@/components/workspace/data/NutAssemblyGenerateForm';
import { getIsaacLabRuntimeStatus, type IsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';
import { isIsaacBlockStackingTask } from '@/lib/workspace/isaacBlockStacking';
import { isFrankStackCubeProductTask } from '@/lib/workspace/isaacStackCubeProduct';
import {
  ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS,
  isIsaacLabFrankaStackCubeTask,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import {
  ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS,
  isIsaacSimFrankaPickPlaceTask,
} from '@/lib/workspace/isaacsimFrankaPickPlace';
import {
  filterIsaacSeedDatasets,
} from '@/lib/workspace/isaacSeedDatasets';
import {
  fetchGenerateDataTemplateOptions,
  GENERATE_DATA_TEMPLATE_EMPTY_HINT,
  getGenerateDataTemplateOptions,
} from '@/lib/workspace/generateDataTemplateOptions';
import {
  formatGenerateDataTemplateOptionLabel,
  isDatasetGenerationEnabled,
  resolveDefaultGenerationPath,
  resolveTaskTemplateCapabilities,
} from '@/lib/workspace/taskTemplateCapabilities';
import { CABLE_THREADING_DISPLAY_NAME } from '@/lib/workspace/taskDisplayNames';
import { IsaacStackingGenerateForm, type IsaacStackingGenerationMode } from '@/components/workspace/data/IsaacStackingGenerateForm';
import type { Dataset } from '@/types/benchmark';

import type { GenerateDataPayload, GenerateDataPurpose } from '@/lib/workspace/generateDataPayloadTypes';
export type { GenerateDataPayload, GenerateDataPurpose } from '@/lib/workspace/generateDataPayloadTypes';

const DATA_PURPOSE_OPTIONS: GenerateDataPurpose[] = ['训练数据', '评测数据', '训练与评测'];
const ROBOT_OPTIONS = CABLE_THREADING_ROBOTS;

const DATA_OUTPUT_CARDS = [
  {
    id: 'trajectory',
    title: '轨迹数据',
    description: 'NPZ / 状态 / 动作 / 奖励',
  },
  {
    id: 'image',
    title: '图像数据',
    description: 'HDF5 / 相机图像 / 观测序列',
  },
  {
    id: 'results',
    title: '运行结果',
    description: 'CSV / 成功率 / episode 结果',
  },
  {
    id: 'failures',
    title: '失败记录',
    description: 'failures.json / 失败 episode',
  },
  {
    id: 'processVideo',
    title: '过程视频',
    description: 'MP4 / 数据生成过程回放',
  },
] as const;

function FormSection({
  title,
  first = false,
  children,
}: {
  title: string;
  first?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginTop: first ? 0 : 20 }}>
      <div style={workspaceModalSectionLabel}>{title}</div>
      {children}
    </div>
  );
}

function FieldGrid({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>{children}</div>
  );
}

function TaskParameterFields({
  fields,
  values,
  onChange,
  disabled,
}: {
  fields: GenerateDataTaskParamField[];
  values: Record<string, string | number>;
  onChange: (id: string, value: string | number) => void;
  disabled?: boolean;
}) {
  return (
    <FieldGrid>
      {fields.map((field) => (
        <div key={field.id}>
          <label style={workspaceModalFieldLabel}>{field.label}</label>
          {field.kind === 'select' ? (
            <select
              style={workspaceModalSelectStyle}
              value={String(values[field.id] ?? field.defaultValue)}
              onChange={(e) => onChange(field.id, e.target.value)}
              disabled={disabled}
            >
              {field.selectOptions
                ? field.selectOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))
                : (field.options ?? []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
            </select>
          ) : field.kind === 'number' ? (
            <input
              type="number"
              min={field.min}
              max={field.max}
              value={Number(values[field.id] ?? field.defaultValue)}
              onChange={(e) => onChange(field.id, Number(e.target.value) || Number(field.defaultValue))}
              style={workspaceModalSelectStyle}
              disabled={disabled}
            />
          ) : (
            <input
              type="text"
              value={String(values[field.id] ?? field.defaultValue)}
              onChange={(e) => onChange(field.id, e.target.value)}
              style={workspaceModalSelectStyle}
              disabled={disabled}
            />
          )}
        </div>
      ))}
    </FieldGrid>
  );
}

function DataOutputCard({
  title,
  description,
  hint,
  checked,
  onChange,
  disabled,
}: {
  title: string;
  description: string;
  hint?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label
      style={{
        display: 'flex',
        gap: 10,
        padding: '12px 14px',
        borderRadius: 10,
        border: checked ? '1px solid #bfdbfe' : '1px solid #e5e7eb',
        backgroundColor: checked ? '#f8fbff' : '#fff',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.7 : 1,
        alignItems: 'flex-start',
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        style={{ marginTop: 3, flexShrink: 0 }}
      />
      <span style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#111827' }}>{title}</span>
        <span style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>{description}</span>
        {hint ? (
          <span style={{ fontSize: 11, color: '#9ca3af', lineHeight: 1.45 }}>{hint}</span>
        ) : null}
      </span>
    </label>
  );
}

export function GenerateDataModal({
  open,
  onClose,
  onSubmit,
  initialTemplate,
  submitting = false,
  submittingMessage,
  onImportIsaacDemo,
  onViewIsaacTaskTemplate,
  isaacSeedDatasets = [],
  preferredSeedDatasetId,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (payload: GenerateDataPayload) => void | Promise<void>;
  initialTemplate?: string;
  submitting?: boolean;
  submittingMessage?: string;
  onImportIsaacDemo?: () => void;
  onViewIsaacTaskTemplate?: () => void;
  isaacSeedDatasets?: Dataset[];
  preferredSeedDatasetId?: string | null;
}) {
  const [templateOptions, setTemplateOptions] = useState<string[]>(() => getGenerateDataTemplateOptions());
  const [templateOptionsLoading, setTemplateOptionsLoading] = useState(false);
  const [template, setTemplate] = useState<string>(() => getGenerateDataTemplateOptions()[0] ?? CABLE_THREADING_DISPLAY_NAME);
  const [simBackend, setSimBackend] = useState<string>('MuJoCo');
  const [robot, setRobot] = useState<string>(CABLE_THREADING_DEFAULTS.robot);
  const [episodes, setEpisodes] = useState<number>(50);
  const [seed, setSeed] = useState(0);
  const [dataPurpose, setDataPurpose] = useState<GenerateDataPurpose>('训练与评测');
  const [saveTrajectory, setSaveTrajectory] = useState(true);
  const [saveRunResults, setSaveRunResults] = useState(true);
  const [saveFailureRecords, setSaveFailureRecords] = useState(true);
  const [saveImageData, setSaveImageData] = useState(true);
  const [saveProcessVideo, setSaveProcessVideo] = useState(true);
  const [outputName, setOutputName] = useState('');
  const outputNameTouchedRef = useRef(false);
  const [taskParams, setTaskParams] = useState<Record<string, string | number>>(() =>
    defaultTaskParamValues(getGenerateDataTemplateOptions()[0] ?? CABLE_THREADING_DISPLAY_NAME)
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [physicsProxyMode, setPhysicsProxyMode] = useState<PhysicsProxyMode>('off');
  const [physicsProxyModel, setPhysicsProxyModel] = useState<string>(physicsProxyModelOptions[0]);
  const [physicsProxyErrorThreshold, setPhysicsProxyErrorThreshold] = useState(5);
  const [physicsProxyReviewRatio, setPhysicsProxyReviewRatio] = useState(10);
  const [isaacRuntime, setIsaacRuntime] = useState<IsaacLabRuntimeStatus | null>(null);
  const [isaacRuntimeLoading, setIsaacRuntimeLoading] = useState(false);
  const [isaacGenerationMode, setIsaacGenerationMode] = useState<IsaacStackingGenerationMode>('expert_policy');
  const [isaacSelectedSeedId, setIsaacSelectedSeedId] = useState('');
  const [isaacManualSeedPath, setIsaacManualSeedPath] = useState('');
  const [isaacAdvancedOpen, setIsaacAdvancedOpen] = useState(false);
  const [isaacNumDemos, setIsaacNumDemos] = useState(10);
  const [isaacHeadless, setIsaacHeadless] = useState(true);
  const [isaacEnableCameras, setIsaacEnableCameras] = useState(true);
  const [isaacParallelNumEnvs, setIsaacParallelNumEnvs] = useState(1);
  const [isaacsimHeadless, setIsaacsimHeadless] = useState<boolean>(
    ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.headless
  );
  const [isaaclabHeadless, setIsaaclabHeadless] = useState<boolean>(
    ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.headless
  );
  const [nutAssemblyPathParams, setNutAssemblyPathParams] = useState<NutAssemblyPathParamDefaults>(
    () => defaultNutAssemblyPathParams()
  );
  const [nutAssemblyEnvStatus, setNutAssemblyEnvStatus] = useState<NutAssemblyMimicgenEnvStatus | null>(
    null
  );
  const [nutAssemblyRobot, setNutAssemblyRobot] = useState<string>(NUT_ASSEMBLY_DEFAULT_ROBOT);
  const [nutAssemblyPinnRepairEnabled, setNutAssemblyPinnRepairEnabled] = useState(false);
  const [nutAssemblyPinnModelStatus, setNutAssemblyPinnModelStatus] =
    useState<NutAssemblyPinnModelStatus | null>(null);
  const [nutAssemblyPinnSettingsOpen, setNutAssemblyPinnSettingsOpen] = useState(false);

  const cableThreadingMode = isCableThreadingTask(template);
  const dualArmCableMode = isDualArmCableTask(template);
  const nutAssemblyMode = isNutAssemblyTask(template);
  const isaacBlockStackingMode =
    isIsaacBlockStackingTask(template) || isFrankStackCubeProductTask(template);
  const isaaclabFrankaStackCubeMode =
    isIsaacLabFrankaStackCubeTask(template) && !isaacBlockStackingMode;
  const isaacsimFrankaPickPlaceMode = isIsaacSimFrankaPickPlaceTask(template);
  const availableIsaacSeedDatasets = useMemo(
    () => filterIsaacSeedDatasets(isaacSeedDatasets),
    [isaacSeedDatasets]
  );
  const availableNutAssemblySourceDatasets = useMemo(
    () => filterNutAssemblySourceDemoDatasets(isaacSeedDatasets),
    [isaacSeedDatasets]
  );
  const templateCapabilities = useMemo(() => resolveTaskTemplateCapabilities(template), [template]);
  const showGenerationForm = isDatasetGenerationEnabled(template);
  const visibleTemplateOptions = templateOptions;
  const hasTemplateOptions = visibleTemplateOptions.length > 0;
  const taskParamFields = useMemo(() => getGenerateDataTaskParamFields(template), [template]);

  useEffect(() => {
    if (!open || !nutAssemblyMode) return;
    let cancelled = false;
    void Promise.all([getNutAssemblyMimicgenEnvStatus(false), getNutAssemblyPinnModelStatus()])
      .then(([envStatus, pinnStatus]) => {
        if (cancelled) return;
        setNutAssemblyEnvStatus(envStatus);
        setNutAssemblyPinnModelStatus(pinnStatus);
      })
      .catch(() => {
        if (!cancelled) {
          setNutAssemblyEnvStatus({ overallOk: false, error: '无法读取 MimicGen 环境状态' });
          setNutAssemblyPinnModelStatus({ available: false, error: '无法读取物理增强模型状态' });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open, nutAssemblyMode]);

  const applyTemplateDefaults = (nextTemplate: string, preserveOutputName: boolean) => {
    setTemplate(nextTemplate);
    setSimBackend(
      isFrankStackCubeProductTask(nextTemplate)
        ? 'Isaac Lab'
        : isIsaacSimFrankaPickPlaceTask(nextTemplate)
          ? 'Isaac Sim'
          : 'MuJoCo'
    );
    setRobot(CABLE_THREADING_DEFAULTS.robot);
    setEpisodes(
      isIsaacSimFrankaPickPlaceTask(nextTemplate)
        ? ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.episodes
        : isDualArmCableTask(nextTemplate)
          ? 1
          : isNutAssemblyTask(nextTemplate)
            ? NUT_ASSEMBLY_DEFAULTS.episodes
          : isCableThreadingTask(nextTemplate)
            ? CABLE_THREADING_DEFAULTS.generateEpisodes
            : 50
    );
    setSeed(isDualArmCableTask(nextTemplate) ? DUAL_ARM_CABLE_DEFAULTS.seed : 0);
    setRobot(
      isDualArmCableTask(nextTemplate) ? DUAL_ARM_CABLE_DEFAULTS.robot : CABLE_THREADING_DEFAULTS.robot
    );
    setDataPurpose('训练与评测');
    setSaveTrajectory(true);
    setSaveRunResults(true);
    setSaveFailureRecords(true);
    setSaveImageData(true);
    setSaveProcessVideo(true);
    if (!preserveOutputName) {
      setOutputName(generateDefaultDataName(nextTemplate));
    }
    setTaskParams(defaultTaskParamValues(nextTemplate));
    setAdvancedOpen(false);
    setDebugOpen(false);
    setPhysicsProxyMode('off');
    setPhysicsProxyModel(physicsProxyModelOptions[0]);
    setPhysicsProxyErrorThreshold(5);
    setPhysicsProxyReviewRatio(10);
    setIsaacGenerationMode('expert_policy');
    if (isNutAssemblyTask(nextTemplate)) {
      const defaultPath = resolveDefaultGenerationPath(nextTemplate);
      setNutAssemblyPathParams({
        ...defaultNutAssemblyPathParams(),
        generationPath: defaultPath ?? defaultNutAssemblyPathParams().generationPath,
      });
      setNutAssemblyRobot(NUT_ASSEMBLY_DEFAULT_ROBOT);
      setNutAssemblyPinnRepairEnabled(false);
      setNutAssemblyPinnSettingsOpen(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setTemplateOptionsLoading(true);
    void fetchGenerateDataTemplateOptions()
      .then((options) => {
        if (cancelled) return;
        setTemplateOptions(options);
        const preferred =
          initialTemplate && options.includes(initialTemplate) ? initialTemplate : options[0];
        outputNameTouchedRef.current = false;
        if (preferred) {
          applyTemplateDefaults(preferred, false);
        }
      })
      .finally(() => {
        if (!cancelled) setTemplateOptionsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, initialTemplate]);

  useEffect(() => {
    if (!hasTemplateOptions) return;
    if (!visibleTemplateOptions.includes(template)) {
      applyTemplateDefaults(visibleTemplateOptions[0] ?? CABLE_THREADING_DISPLAY_NAME, false);
    }
  }, [template, visibleTemplateOptions, hasTemplateOptions]);

  useEffect(() => {
    if (cableThreadingMode || dualArmCableMode) {
      setSimBackend('MuJoCo');
    }
    if (isaacBlockStackingMode) {
      setSimBackend('Isaac Lab');
    }
    if (isaaclabFrankaStackCubeMode) {
      setSimBackend('Isaac Lab');
      setEpisodes(ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.episodes);
      setSeed(ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.seed);
      setSaveTrajectory(ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.saveTrajectory);
      setSaveProcessVideo(ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.saveVideo);
      setIsaaclabHeadless(ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS.headless);
      return;
    }
    if (isaacsimFrankaPickPlaceMode) {
      setSimBackend('Isaac Sim');
      setEpisodes(ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.episodes);
      setSeed(ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.seed);
      setSaveTrajectory(ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.saveTrajectory);
      setSaveProcessVideo(ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.saveVideo);
      setIsaacsimHeadless(ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS.headless);
    }
    if (dualArmCableMode) {
      setSaveImageData(false);
      setSaveTrajectory(false);
      setSaveProcessVideo(true);
      setSaveRunResults(true);
      setSaveFailureRecords(true);
    }
  }, [cableThreadingMode, dualArmCableMode, isaacBlockStackingMode, isaaclabFrankaStackCubeMode, isaacsimFrankaPickPlaceMode]);

  useEffect(() => {
    if (!open || !isaacBlockStackingMode) {
      setIsaacRuntime(null);
      setIsaacRuntimeLoading(false);
      return;
    }
    let cancelled = false;
    setIsaacRuntimeLoading(true);
    void getIsaacLabRuntimeStatus()
      .then((data) => {
        if (!cancelled) setIsaacRuntime(data);
      })
      .catch(() => {
        if (!cancelled) setIsaacRuntime(null);
      })
      .finally(() => {
        if (!cancelled) setIsaacRuntimeLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, isaacBlockStackingMode]);

  useEffect(() => {
    if (!open || !preferredSeedDatasetId) return;
    const exists = availableIsaacSeedDatasets.some((row) => row.id === preferredSeedDatasetId);
    if (exists) {
      setIsaacSelectedSeedId(preferredSeedDatasetId);
      setIsaacAdvancedOpen(true);
    }
  }, [open, preferredSeedDatasetId, availableIsaacSeedDatasets]);

  const physicsProxyEnabled = physicsProxyMode !== 'off';

  const resolveOutputName = () => {
    const trimmed = outputName.trim();
    if (trimmed) return trimmed;
    return generateDefaultDataName(template);
  };

  const buildPayload = (launch: 'save' | 'start'): GenerateDataPayload => {
    const cableFields = cableThreadingMode ? cableThreadingFieldsFromTaskParams(taskParams) : null;
    const dualArmFields = dualArmCableMode ? dualArmCableFieldsFromTaskParams(taskParams) : null;
    const nutFields = nutAssemblyMode ? nutAssemblyFieldsFromTaskParams(taskParams) : null;
    const nutSourceDemoDatasetId = nutAssemblyMode
      ? resolveNutAssemblyEffectiveSourceDemoDatasetId(
          nutAssemblyPathParams.generationPath,
          nutAssemblyPathParams.sourceDemoDatasetId
        )
      : '';
    const nutSourceDemoPath =
      nutAssemblyMode &&
      nutAssemblyPathParams.generationPath !== 'demo_augmentation' &&
      nutAssemblyPathParams.generationPath === 'expert_seed_then_augmentation' &&
      nutAssemblyPathParams.useExistingSeedDataset
        ? resolveNutAssemblySourceDemoPath(
            availableNutAssemblySourceDatasets,
            nutAssemblyPathParams.sourceDemoDatasetId
          )
        : null;
    const nutEpisodes =
      nutAssemblyPathParams.generationPath === 'expert_policy'
        ? nutAssemblyPathParams.generationCount
        : nutAssemblyPathParams.targetCount;
    return {
      template,
      simBackend,
      taskConfig: String(taskParams.config_profile ?? 'default'),
      episodes: nutAssemblyMode ? nutEpisodes : episodes,
      seed,
      dataPurpose,
      saveVideo: !cableThreadingMode && !dualArmCableMode && (saveImageData || saveProcessVideo),
      saveTrajectory,
      saveStateLog: saveFailureRecords,
      saveStructuredData: saveRunResults,
      saveImageData,
      saveProcessVideo,
      outputName: resolveOutputName(),
      launch,
      physicsProxyMode,
      physicsProxyModel: physicsProxyEnabled ? physicsProxyModel : null,
      physicsProxyErrorThreshold,
      physicsProxyReviewRatio,
      cableThreadingRobot: cableThreadingMode ? robot : undefined,
      cableThreadingCableModel: cableFields?.cableThreadingCableModel,
      cableThreadingDifficulty: cableFields?.cableThreadingDifficulty,
      cableThreadingHorizon: cableFields?.cableThreadingHorizon,
      cableThreadingSaveHdf5: cableThreadingMode ? saveImageData : undefined,
      cableThreadingSaveProcessVideo: cableThreadingMode ? saveProcessVideo : undefined,
      dualArmMaxCables: dualArmFields?.dualArmMaxCables,
      dualArmStretchMode: dualArmFields?.dualArmStretchMode,
      dualArmReleaseMode: dualArmFields?.dualArmReleaseMode,
      dualArmRecord: dualArmCableMode ? true : undefined,
      dualArmHeadless: dualArmCableMode ? true : undefined,
      isaacGenerationMode: isaacBlockStackingMode ? isaacGenerationMode : undefined,
      isaacSeedDatasetId:
        isaacBlockStackingMode && isaacGenerationMode === 'mimic_auto' && isaacSelectedSeedId
          ? isaacSelectedSeedId
          : undefined,
      isaacSeedDatasetFile:
        isaacBlockStackingMode &&
        isaacGenerationMode === 'mimic_auto' &&
        !isaacSelectedSeedId &&
        isaacManualSeedPath.trim()
          ? isaacManualSeedPath.trim()
          : undefined,
      isaacNumDemos: isaacBlockStackingMode ? isaacNumDemos : undefined,
      isaacHeadless: isaacBlockStackingMode ? isaacHeadless : undefined,
      isaacEnableCameras: isaacBlockStackingMode ? isaacEnableCameras : undefined,
      isaacParallelNumEnvs: isaacBlockStackingMode ? isaacParallelNumEnvs : undefined,
      isaaclabHeadless: isaaclabFrankaStackCubeMode ? isaaclabHeadless : undefined,
      isaacsimHeadless: isaacsimFrankaPickPlaceMode ? isaacsimHeadless : undefined,
      nutAssemblyEnvName: nutFields?.nutAssemblyEnvName,
      nutAssemblyHorizon: nutAssemblyMode ? nutAssemblyPathParams.maxSteps : nutFields?.nutAssemblyHorizon,
      nutAssemblyRenderVideo: nutAssemblyMode ? saveProcessVideo : undefined,
      generationPath: nutAssemblyMode ? nutAssemblyPathParams.generationPath : undefined,
      generationCount: nutAssemblyMode ? nutAssemblyPathParams.generationCount : undefined,
      maxSteps: nutAssemblyMode ? nutAssemblyPathParams.maxSteps : undefined,
      expertPolicy: nutAssemblyMode ? nutAssemblyPathParams.expertPolicy : undefined,
      successFilter: nutAssemblyMode ? nutAssemblyPathParams.successFilter : undefined,
      keepFailedTrajectories: nutAssemblyMode
        ? nutAssemblyPathParams.keepFailedTrajectories
        : undefined,
      sourceDemoDatasetId: nutAssemblyMode ? nutSourceDemoDatasetId || undefined : undefined,
      augmentationAlgorithm: nutAssemblyMode
        ? nutAssemblyPathParams.augmentationAlgorithm
        : undefined,
      targetCount: nutAssemblyMode ? nutAssemblyPathParams.targetCount : undefined,
      seedGenerationCount: nutAssemblyMode ? nutAssemblyPathParams.seedGenerationCount : undefined,
      seedKeepCount: nutAssemblyMode ? nutAssemblyPathParams.seedKeepCount : undefined,
      autoSelectBestSeeds: nutAssemblyMode ? nutAssemblyPathParams.autoSelectBestSeeds : undefined,
      replayValidation: nutAssemblyMode ? nutAssemblyPathParams.replayValidation : undefined,
      useExistingSeedDataset: nutAssemblyMode
        ? nutAssemblyPathParams.useExistingSeedDataset
        : undefined,
      enablePinnRepair: nutAssemblyMode ? nutAssemblyPinnRepairEnabled : undefined,
      outputFormat: nutAssemblyMode ? templateCapabilities?.outputFormat ?? 'HDF5' : undefined,
      nutAssemblyPinnRepairEnabled: nutAssemblyMode ? nutAssemblyPinnRepairEnabled : undefined,
      nutAssemblySourceDemoPath: nutAssemblyMode ? nutSourceDemoPath ?? undefined : undefined,
      nutAssemblyRobot: nutAssemblyMode ? nutAssemblyRobot : undefined,
    };
  };

  const handleTemplateChange = (nextTemplate: string) => {
    applyTemplateDefaults(nextTemplate, outputNameTouchedRef.current);
  };

  const handleTaskParamChange = (id: string, value: string | number) => {
    setTaskParams((prev) => ({ ...prev, [id]: value }));
  };

  const visibleOutputCards = dualArmCableMode
    ? ([
        {
          id: 'processVideo' as const,
          title: '过程视频',
          description: 'MP4 / episode 过程回放',
        },
        {
          id: 'results' as const,
          title: '运行结果',
          description: 'episode_result.json / episode_manifest.json',
        },
        {
          id: 'failures' as const,
          title: '运行日志',
          description: 'run.log / perception 日志',
        },
        {
          id: 'trajectory' as const,
          title: '感知结果',
          description: 'grasp JSON / frame PNG / depth NPY',
        },
      ] as const)
    : DATA_OUTPUT_CARDS;

  const outputCheckedMap: Record<string, boolean> = {
    trajectory: dualArmCableMode ? true : saveTrajectory,
    image: saveImageData,
    results: saveRunResults,
    failures: dualArmCableMode ? true : saveFailureRecords,
    processVideo: saveProcessVideo,
  };

  const outputChangeMap: Record<string, (v: boolean) => void> = {
    trajectory: setSaveTrajectory,
    image: setSaveImageData,
    results: setSaveRunResults,
    failures: setSaveFailureRecords,
    processVideo: setSaveProcessVideo,
  };

  const cableModelInternal = cableThreadingMode
    ? String(taskParams.cable_type ?? CABLE_THREADING_DEFAULTS.cableModel)
    : null;

  const showMuJoCoGenerationForm =
    showGenerationForm && !isaacBlockStackingMode && !isaaclabFrankaStackCubeMode && !isaacsimFrankaPickPlaceMode;
  const isaacMimicReady = Boolean(isaacRuntime?.stackCubeGenerationReady);
  const isaacScriptedReady = Boolean(isaacRuntime?.scriptedExpertReady);
  const isaacGenerationReady =
    isaacGenerationMode !== 'mimic_auto' ? isaacScriptedReady : isaacMimicReady;
  const resolvedDatasetName = resolveOutputName().trim();

  const isaacMissingRequirements = useMemo(() => {
    if (!isaacBlockStackingMode) return [] as string[];
    const missing: string[] = [];
    if (!isaacRuntimeLoading && !isaacGenerationReady) {
      missing.push(
        isaacGenerationMode !== 'mimic_auto'
          ? 'Isaac 专家策略运行环境未就绪'
          : 'Isaac 生成环境未就绪'
      );
    }
    if (!resolvedDatasetName) {
      missing.push('缺少数据集名称');
    }
    if (isaacNumDemos <= 0) {
      missing.push('生成轮次须大于 0');
    }
    return missing;
  }, [
    isaacBlockStackingMode,
    isaacGenerationMode,
    isaacGenerationReady,
    isaacNumDemos,
    isaacRuntimeLoading,
    resolvedDatasetName,
  ]);

  const isaacStartDisabled =
    submitting || isaacRuntimeLoading || isaacMissingRequirements.length > 0;

  const nutAssemblyValidationError = useNutAssemblyGenerateValidation({
    pathParams: nutAssemblyPathParams,
    datasetName: resolvedDatasetName,
    enablePinnRepair: nutAssemblyPinnRepairEnabled,
    capabilities: templateCapabilities,
  });
  const nutAssemblyPinnBlocked =
    nutAssemblyMode &&
    nutAssemblyPinnRepairEnabled &&
    !nutAssemblyPinnModelStatus?.available &&
    (nutAssemblyPathParams.generationPath === 'demo_augmentation' ||
      nutAssemblyPathParams.generationPath === 'expert_seed_then_augmentation');
  const nutAssemblyStartDisabled =
    submitting || Boolean(nutAssemblyValidationError) || nutAssemblyPinnBlocked;
  const nutAssemblyStartHint =
    nutAssemblyValidationError ??
    (nutAssemblyPinnBlocked ? '未检测到物理增强模型，请先完成模型配置。' : null);

  return (
    <WorkspaceCenteredModal
      open={open}
      title="生成任务数据"
      titleId="generate-data-title"
      width={720}
      onClose={submitting ? () => {} : onClose}
      footer={
        isaacBlockStackingMode ? (
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <SecondaryButton onClick={() => { if (!submitting) onClose(); }}>
              取消
            </SecondaryButton>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {onViewIsaacTaskTemplate ? (
                <SecondaryButton
                  onClick={() => {
                    if (submitting) return;
                    onViewIsaacTaskTemplate();
                  }}
                >
                  查看任务模板详情
                </SecondaryButton>
              ) : null}
              <PrimaryButton
                disabled={isaacStartDisabled || !hasTemplateOptions || templateOptionsLoading}
                onClick={() => {
                  void onSubmit(buildPayload('start'));
                }}
              >
                {submitting
                  ? '正在启动…'
                  : !isaacGenerationReady && !isaacRuntimeLoading
                    ? isaacGenerationMode !== 'mimic_auto'
                      ? 'Isaac 专家策略运行环境未就绪'
                      : 'Isaac 生成环境未就绪'
                    : '生成 Isaac 数据'}
              </PrimaryButton>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <SecondaryButton onClick={() => { if (!submitting) onClose(); }}>
              取消
            </SecondaryButton>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <SecondaryButton
                disabled={!hasTemplateOptions || templateOptionsLoading || submitting}
                onClick={() => {
                  if (submitting) return;
                  void onSubmit(buildPayload('save'));
                }}
              >
                保存任务
              </SecondaryButton>
              <PrimaryButton
                disabled={
                  !hasTemplateOptions ||
                  templateOptionsLoading ||
                  (nutAssemblyMode ? nutAssemblyStartDisabled : submitting)
                }
                onClick={() => {
                  void onSubmit(buildPayload('start'));
                }}
              >
                {submitting
                  ? '正在启动…'
                  : nutAssemblyMode && nutAssemblyStartHint
                    ? nutAssemblyStartHint
                    : nutAssemblyMode
                      ? '开始生成'
                      : '启动生成'}
              </PrimaryButton>
            </div>
          </div>
        )
      }
    >
      {templateOptionsLoading ? (
        <p style={{ margin: 0, fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
          正在加载可生成数据的任务模板…
        </p>
      ) : !hasTemplateOptions ? (
        <div
          style={{
            padding: '32px 16px',
            textAlign: 'center',
            color: '#6b7280',
            fontSize: 14,
            lineHeight: 1.6,
            borderRadius: 8,
            border: '1px dashed #e5e7eb',
            backgroundColor: '#f9fafb',
          }}
        >
          {GENERATE_DATA_TEMPLATE_EMPTY_HINT}
        </div>
      ) : (
        <>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
        {nutAssemblyMode
          ? '基于任务模板自动生成训练数据，完成后登记到数据中心。'
          : isaacBlockStackingMode
            ? isaacGenerationMode !== 'mimic_auto'
              ? '基于物块堆叠任务的专家策略自动生成高质量轨迹，并完成 HDF5 录制、质量检测与回放生成。'
              : '通过 Mimic 示范扩增基于 seed HDF5 Demo 自动生成物块堆叠数据集（实验模式），适合已有高质量示范时使用，完成后自动登记到数据中心。'
            : isaaclabFrankaStackCubeMode
              ? '基于 Isaac Lab 的 Franka 方块堆叠任务，使用 Mimic 专家策略生成示教数据；运行依赖 Isaac Lab 环境，完成后自动登记到数据中心。'
              : isaacsimFrankaPickPlaceMode
                ? '基于 Isaac Sim 官方 FrankaPickPlace controller 生成 episode manifest、metrics 与回放视频，完成后自动登记到数据中心。'
                : '平台将基于当前任务模板生成任务运行数据，用于后续数据处理、模型训练和策略评测。'}
      </p>

      {submitting && submittingMessage ? (
        <div
          style={{
            marginBottom: 16,
            padding: '12px 14px',
            borderRadius: 8,
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            color: '#1d4ed8',
            fontSize: 13,
            lineHeight: 1.55,
          }}
        >
          {submittingMessage}
        </div>
      ) : null}

      {isaacBlockStackingMode ? (
        <>
          <FormSection title="基础信息" first>
            <FieldGrid>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>任务模板</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={template}
                  onChange={(e) => handleTemplateChange(e.target.value)}
                  disabled={submitting}
                >
                  {visibleTemplateOptions.map((option) => (
                    <option key={option} value={option}>
                      {formatGenerateDataTemplateOptionLabel(option)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>数据集名称</label>
                <input
                  type="text"
                  value={outputName}
                  placeholder={generateDefaultDataName(template)}
                  onChange={(e) => {
                    outputNameTouchedRef.current = true;
                    setOutputName(e.target.value);
                  }}
                  style={workspaceModalSelectStyle}
                  disabled={submitting}
                />
              </div>
            </FieldGrid>
          </FormSection>
          <IsaacStackingGenerateForm
            generationMode={isaacGenerationMode}
            onGenerationModeChange={setIsaacGenerationMode}
            seedDatasets={availableIsaacSeedDatasets}
            selectedSeedDatasetId={isaacSelectedSeedId}
            onSelectedSeedDatasetIdChange={setIsaacSelectedSeedId}
            manualSeedPath={isaacManualSeedPath}
            onManualSeedPathChange={setIsaacManualSeedPath}
            advancedOpen={isaacAdvancedOpen}
            onAdvancedOpenChange={setIsaacAdvancedOpen}
            onImportCustomSeed={onImportIsaacDemo}
            numDemos={isaacNumDemos}
            onNumDemosChange={setIsaacNumDemos}
            seed={seed}
            onSeedChange={setSeed}
            headless={isaacHeadless}
            onHeadlessChange={setIsaacHeadless}
            enableCameras={isaacEnableCameras}
            onEnableCamerasChange={setIsaacEnableCameras}
            parallelNumEnvs={isaacParallelNumEnvs}
            onParallelNumEnvsChange={setIsaacParallelNumEnvs}
            runtime={isaacRuntime}
            runtimeLoading={isaacRuntimeLoading}
            disabled={submitting}
          />
          {isaacMissingRequirements.length > 0 ? (
            <p style={{ margin: '12px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
              {isaacMissingRequirements.join(' · ')}
            </p>
          ) : null}
        </>
      ) : isaaclabFrankaStackCubeMode ? (
        <>
          <FormSection title="基础信息" first>
            <FieldGrid>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>任务模板</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={template}
                  onChange={(e) => handleTemplateChange(e.target.value)}
                  disabled={submitting}
                >
                  {visibleTemplateOptions.map((option) => (
                    <option key={option} value={option}>
                      {formatGenerateDataTemplateOptionLabel(option)}
                    </option>
                  ))}
                </select>
              </div>
            </FieldGrid>
          </FormSection>
          <FormSection title="仿真配置">
            <FieldGrid>
              <div>
                <label style={workspaceModalFieldLabel}>仿真后端</label>
                <input type="text" readOnly value="Isaac Lab" style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb' }} disabled />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>专家策略</label>
                <input type="text" readOnly value="Mimic + Seed Demonstration" style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb' }} disabled />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>Episode 数量</label>
                <input type="number" min={1} max={3} value={episodes} onChange={(e) => setEpisodes(Math.min(3, Math.max(1, Number(e.target.value) || 1)))} style={workspaceModalSelectStyle} disabled={submitting} />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>随机种子</label>
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} style={workspaceModalSelectStyle} disabled={submitting} />
              </div>
            </FieldGrid>
          </FormSection>
          <FormSection title="输出选项">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <DataOutputCard title="保存视频" description="MP4 / episode 回放视频" checked={saveProcessVideo} onChange={setSaveProcessVideo} disabled={submitting} />
              <DataOutputCard title="保存轨迹" description="HDF5 + trajectory manifest" checked={saveTrajectory} onChange={setSaveTrajectory} disabled={submitting} />
              <DataOutputCard title="Headless" description="无界面运行 Isaac Lab" checked={isaaclabHeadless} onChange={setIsaaclabHeadless} disabled={submitting} />
            </div>
          </FormSection>
        </>
      ) : isaacsimFrankaPickPlaceMode ? (
        <>
          <FormSection title="基础信息" first>
            <FieldGrid>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>任务模板</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={template}
                  onChange={(e) => handleTemplateChange(e.target.value)}
                  disabled={submitting}
                >
                  {visibleTemplateOptions.map((option) => (
                    <option key={option} value={option}>
                      {formatGenerateDataTemplateOptionLabel(option)}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>数据名称</label>
                <input
                  type="text"
                  value={outputName}
                  placeholder={generateDefaultDataName(template)}
                  onChange={(e) => {
                    outputNameTouchedRef.current = true;
                    setOutputName(e.target.value);
                  }}
                  style={workspaceModalSelectStyle}
                  disabled={submitting}
                />
              </div>
            </FieldGrid>
          </FormSection>
          <FormSection title="仿真配置">
            <FieldGrid>
              <div>
                <label style={workspaceModalFieldLabel}>仿真后端</label>
                <input
                  type="text"
                  readOnly
                  value="Isaac Sim"
                  style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb' }}
                  disabled
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>机器人</label>
                <input
                  type="text"
                  readOnly
                  value="Franka Panda"
                  style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb' }}
                  disabled
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>Episode 数量</label>
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={episodes}
                  onChange={(e) => setEpisodes(Math.min(5, Math.max(1, Number(e.target.value) || 1)))}
                  style={workspaceModalSelectStyle}
                  disabled={submitting}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>随机种子</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || 0)}
                  style={workspaceModalSelectStyle}
                  disabled={submitting}
                />
              </div>
            </FieldGrid>
          </FormSection>
          <FormSection title="输出选项">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <DataOutputCard
                title="保存视频"
                description="MP4 / episode 回放视频"
                checked={saveProcessVideo}
                onChange={setSaveProcessVideo}
                disabled={submitting}
              />
              <DataOutputCard
                title="保存轨迹"
                description="trajectory.json / episode manifest"
                checked={saveTrajectory}
                onChange={setSaveTrajectory}
                disabled={submitting}
              />
              <DataOutputCard
                title="Headless 仿真"
                description="无界面运行（推荐）"
                hint="关闭后使用 GUI 模式（需本机 Isaac Sim 显示环境）"
                checked={isaacsimHeadless}
                onChange={setIsaacsimHeadless}
                disabled={submitting}
              />
            </div>
          </FormSection>
        </>
      ) : nutAssemblyMode ? (
        <>
          <FormSection title="基本信息" first>
            <FieldGrid>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>任务模板</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={template}
                  onChange={(e) => handleTemplateChange(e.target.value)}
                  disabled={submitting}
                >
                  {visibleTemplateOptions.map((option) => (
                    <option key={option} value={option}>
                      {formatGenerateDataTemplateOptionLabel(option)}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>数据名称</label>
                <input
                  type="text"
                  value={outputName}
                  placeholder={generateDefaultDataName(template)}
                  onChange={(e) => {
                    outputNameTouchedRef.current = true;
                    setOutputName(e.target.value);
                  }}
                  style={workspaceModalSelectStyle}
                  disabled={submitting}
                />
              </div>
            </FieldGrid>
          </FormSection>

          <NutAssemblyGenerateForm
            templateLabel={template}
            capabilities={templateCapabilities}
            pathParams={nutAssemblyPathParams}
            onPathParamsChange={(patch) =>
              setNutAssemblyPathParams((prev) => ({ ...prev, ...patch }))
            }
            datasetName={resolvedDatasetName}
            robot={nutAssemblyRobot}
            onRobotChange={setNutAssemblyRobot}
            robotOptions={NUT_ASSEMBLY_ROBOT_OPTIONS}
            enablePinnRepair={nutAssemblyPinnRepairEnabled}
            onEnablePinnRepairChange={setNutAssemblyPinnRepairEnabled}
            pinnSettingsOpen={nutAssemblyPinnSettingsOpen}
            onPinnSettingsOpenChange={setNutAssemblyPinnSettingsOpen}
            pinnModelStatus={nutAssemblyPinnModelStatus}
            mimicgenEnvStatus={nutAssemblyEnvStatus}
            saveProcessVideo={saveProcessVideo}
            onSaveProcessVideoChange={setSaveProcessVideo}
            workspaceDatasets={availableNutAssemblySourceDatasets}
            disabled={submitting}
          />

          {nutAssemblyStartHint && !submitting ? (
            <p style={{ margin: '12px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
              {nutAssemblyStartHint}
            </p>
          ) : null}
        </>
      ) : (
        <>
      <FormSection title="基础信息" first>
        <FieldGrid>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={workspaceModalFieldLabel}>任务模板</label>
            <select
              style={workspaceModalSelectStyle}
              value={template}
              onChange={(e) => handleTemplateChange(e.target.value)}
              disabled={submitting}
            >
              {visibleTemplateOptions.map((option) => (
                <option key={option} value={option}>
                  {formatGenerateDataTemplateOptionLabel(option)}
                </option>
              ))}
            </select>
          </div>
        </FieldGrid>
      </FormSection>

      {showMuJoCoGenerationForm ? (
        <>
      <FieldGrid>
          <div>
            <label style={workspaceModalFieldLabel}>数据名称</label>
            <input
              type="text"
              value={outputName}
              placeholder={generateDefaultDataName(template)}
              onChange={(e) => {
                outputNameTouchedRef.current = true;
                setOutputName(e.target.value);
              }}
              style={workspaceModalSelectStyle}
              disabled={submitting}
            />
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>数据用途</label>
            <select
              style={workspaceModalSelectStyle}
              value={dataPurpose}
              onChange={(e) => setDataPurpose(e.target.value as GenerateDataPurpose)}
              disabled={submitting}
            >
              {DATA_PURPOSE_OPTIONS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </FieldGrid>

      <FormSection title="仿真配置">
        <FieldGrid>
          <div>
            <label style={workspaceModalFieldLabel}>仿真环境</label>
            {isaacBlockStackingMode ? (
              <input
                type="text"
                readOnly
                value="Isaac Lab"
                style={{
                  ...workspaceModalSelectStyle,
                  backgroundColor: '#f9fafb',
                  cursor: 'default',
                }}
                disabled
              />
            ) : (
              <select
                style={workspaceModalSelectStyle}
                value={simBackend}
                onChange={(e) => setSimBackend(e.target.value)}
                disabled={submitting || cableThreadingMode || dualArmCableMode}
              >
                {generateDataSimEnvironmentUiOptions
                  .filter((env) => env !== 'Isaac Sim')
                  .map((env) => (
                    <option key={env} value={env}>
                      {env}
                    </option>
                  ))}
              </select>
            )}
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>机器人</label>
            <select
              style={workspaceModalSelectStyle}
              value={dualArmCableMode ? DUAL_ARM_CABLE_DEFAULTS.robot : robot}
              onChange={(e) => setRobot(e.target.value)}
              disabled={submitting || dualArmCableMode}
            >
              {(dualArmCableMode ? [DUAL_ARM_CABLE_DEFAULTS.robot] : ROBOT_OPTIONS).map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>采集轮次</label>
            <input
              type="number"
              min={1}
              max={200}
              value={dualArmCableMode ? 1 : episodes}
              onChange={(e) => setEpisodes(Number(e.target.value) || 1)}
              style={workspaceModalSelectStyle}
              disabled={submitting || dualArmCableMode}
            />
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>随机种子</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              style={workspaceModalSelectStyle}
              disabled={submitting}
            />
          </div>
        </FieldGrid>
      </FormSection>

      <FormSection title="任务参数">
        <TaskParameterFields
          fields={taskParamFields}
          values={taskParams}
          onChange={handleTaskParamChange}
          disabled={submitting}
        />
      </FormSection>

      <FormSection title="数据输出">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {visibleOutputCards.map((card) => (
            <DataOutputCard
              key={card.id}
              title={card.title}
              description={card.description}
              checked={outputCheckedMap[card.id]}
              onChange={outputChangeMap[card.id]}
              disabled={submitting || (dualArmCableMode && card.id === 'trajectory')}
            />
          ))}
        </div>
      </FormSection>

      <button
        type="button"
        onClick={() => setAdvancedOpen((v) => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          marginTop: 20,
          padding: 0,
          border: 'none',
          background: 'none',
          fontSize: 13,
          fontWeight: 600,
          color: '#374151',
          cursor: 'pointer',
        }}
      >
        <span style={{ fontSize: 10, color: '#9ca3af' }}>{advancedOpen ? '▼' : '▶'}</span>
        高级设置
      </button>

      {advancedOpen ? (
        <div
          style={{
            marginTop: 12,
            padding: '14px 16px',
            borderRadius: 8,
            border: '1px solid #e5e7eb',
            backgroundColor: '#f9fafb',
          }}
        >
          <p style={{ margin: '0 0 14px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
            减少采集轮次或关闭图像数据可缩短演示等待时间。
          </p>
          <p style={{ margin: '0 0 14px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
            物理代理加速用于在数据生成过程中快速预测局部接触、形变或材料响应，降低高保真仿真的重复计算成本。
          </p>
          <FieldGrid>
            <div>
              <label style={workspaceModalFieldLabel}>物理代理加速</label>
              <select
                style={workspaceModalSelectStyle}
                value={physicsProxyMode}
                onChange={(e) => setPhysicsProxyMode(e.target.value as PhysicsProxyMode)}
                disabled={submitting}
              >
                <option value="off">{physicsProxyModeLabel('off')}</option>
                <option value="pinn">{physicsProxyModeLabel('pinn')}</option>
                <option value="hybrid">{physicsProxyModeLabel('hybrid')}</option>
              </select>
            </div>
            {physicsProxyEnabled ? (
              <>
                <div>
                  <label style={workspaceModalFieldLabel}>代理模型</label>
                  <select
                    style={workspaceModalSelectStyle}
                    value={physicsProxyModel}
                    onChange={(e) => setPhysicsProxyModel(e.target.value)}
                    disabled={submitting}
                  >
                    {physicsProxyModelOptions.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={workspaceModalFieldLabel}>误差阈值</label>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={physicsProxyErrorThreshold}
                    onChange={(e) => setPhysicsProxyErrorThreshold(Number(e.target.value) || 5)}
                    style={workspaceModalSelectStyle}
                    disabled={submitting}
                  />
                  <span style={{ fontSize: 11, color: '#9ca3af', marginLeft: 4 }}>%</span>
                </div>
                <div>
                  <label style={workspaceModalFieldLabel}>高保真复核比例</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={physicsProxyReviewRatio}
                    onChange={(e) => setPhysicsProxyReviewRatio(Number(e.target.value) || 10)}
                    style={workspaceModalSelectStyle}
                    disabled={submitting}
                  />
                  <span style={{ fontSize: 11, color: '#9ca3af', marginLeft: 4 }}>%</span>
                </div>
              </>
            ) : null}
          </FieldGrid>

          <button
            type="button"
            onClick={() => setDebugOpen((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
              marginTop: 16,
              padding: 0,
              border: 'none',
              background: 'none',
              fontSize: 12,
              fontWeight: 600,
              color: '#6b7280',
              cursor: 'pointer',
            }}
          >
            <span>内部调试信息</span>
            <span style={{ fontSize: 11, fontWeight: 400 }}>{debugOpen ? '收起' : '展开'}</span>
          </button>
          {debugOpen ? (
            <div
              style={{
                marginTop: 10,
                padding: '10px 12px',
                borderRadius: 6,
                border: '1px solid #e5e7eb',
                backgroundColor: '#fff',
                fontSize: 12,
                color: '#374151',
                lineHeight: 1.6,
              }}
            >
              {dualArmCableMode ? (
                <>
                  <div>backendTaskType: dual_arm_cable_manipulation</div>
                  <div>感知模块: Mask2Former（内部）</div>
                  <div>运行入口: platform_runner.py（内部）</div>
                  <div>statusUrl: /api/workspace/dual-arm-cable/jobs/{'{jobId}'}/status</div>
                  <div>frameUrl: /api/workspace/dual-arm-cable/jobs/{'{jobId}'}/frame</div>
                </>
              ) : cableThreadingMode ? (
                <>
                  <div>backendTaskType: cable_threading</div>
                  <div>cableModel: {cableModelInternal}</div>
                  <div>saveHdf5: {String(saveImageData)}</div>
                  <div>saveProcessVideo: {String(saveProcessVideo)}</div>
                  <div>实现包: CableThreadingMVP</div>
                  <div>采集策略: expert</div>
                  <div>运行入口: run.py expert</div>
                  <div>statusUrl: /api/workspace/cable-threading/jobs/{'{backendJobId}'}/status</div>
                  <div>frameUrl: /api/workspace/cable-threading/jobs/{'{backendJobId}'}/frame</div>
                </>
              ) : nutAssemblyMode ? (
                <>
                  <div>taskTemplateId: nut_assembly_single_arm</div>
                  <div>generationPath: {nutAssemblyPathParams.generationPath}</div>
                  <div>sourceDemoDatasetId: {nutAssemblyPathParams.sourceDemoDatasetId || '—'}</div>
                  <div>enablePinnRepair: {nutAssemblyPinnRepairEnabled ? 'true' : 'false'}</div>
                  <div>envName: {NUT_ASSEMBLY_DEFAULTS.envName}</div>
                  <div>statusUrl: /api/workspace/nut-assembly/jobs/{'{jobId}'}/status</div>
                </>
              ) : (
                <div>当前任务模板暂无后端调试字段。</div>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
        </>
      ) : null}
        </>
      )}
        </>
      )}
    </WorkspaceCenteredModal>
  );
}
