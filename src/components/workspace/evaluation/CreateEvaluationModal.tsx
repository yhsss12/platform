'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { listModelAssets, getModelAsset, modelAssetsToCheckpointOptions, isEvaluableModelAsset } from '@/lib/api/modelAssetsClient';
import {
  formatDatasetOptionLabel,
  listEvaluationDatasets,
} from '@/lib/api/datasetsClient';
import {
  listTaskTemplates,
  type TaskTemplateDto,
  type WorkspaceEvaluationMode,
} from '@/lib/api/taskTemplatesClient';
import { listRegistryResources, type RegistryResource } from '@/lib/api/resourceRegistryClient';
import type { Dataset } from '@/types/benchmark';
import {
  formatTaskTemplateDisplayName,
  modelAssetMatchesTaskTemplate,
  resolveTaskTemplateIdFromModelAsset,
  resolveTemplateIdFromLegacyName,
} from '@/lib/workspace/taskTemplateMapping';
import { resolveModelAssetColumnLabel } from '@/lib/workspace/modelAssetDisplay';
import { formatModelEvaluationTaskName } from '@/lib/workspace/evaluationCreateNavigation';
import {
  generateCableThreadingEvalTaskName,
  modelAssetSupportsCableThreadingEvalObs,
  type CableThreadingEvalStrategy,
} from '@/lib/workspace/cableThreading';
import {
  buildDualArmEvalSeeds,
  generateDualArmEvalTaskName,
} from '@/lib/workspace/dualArmEvaluation';
import { generateIsaacBlockStackingEvalTaskName } from '@/lib/workspace/isaacBlockStacking';
import {
  FRANKA_STACK_CUBE_PRODUCT_NAME,
  FRANKA_STACK_CUBE_PRODUCT_SUBTITLE,
  isFrankStackCubeEvalTask,
  isFrankStackCubeInternalTemplateId,
  resolveFrankStackCubeEvaluationTemplateId,
  shouldShowFrankStackCubeInEvalDropdown,
} from '@/lib/workspace/isaacStackCubeProduct';
import {
  clampEpisodes,
  deriveEvaluationConfigFromTask,
  deriveMetricDefinitionsFromTask,
} from '@/lib/workspace/evaluationTaskDerivation';
import { buildProductEvaluationFields } from '@/lib/workspace/evaluationType';
import {
  getModelAssetIncompatibilityMessage,
  getNoCompatibleModelAssetsHint,
  isModelAssetCompatibleWithEvaluationTask,
} from '@/lib/workspace/evaluationModelBackendCompatibility';

export type EvaluationTopType = 'dataset' | 'model';
export type ModelTaskTemplateMode = 'single_task' | 'multi_task';

export const DATASET_EVAL_METRIC_OPTIONS = [
  { id: 'data_quality', label: '数据质量' },
  { id: 'trajectory_integrity', label: '轨迹完整性' },
  { id: 'action_error', label: '动作误差' },
  { id: 'prediction_error', label: 'Prediction Error' },
  { id: 'mse', label: 'MSE' },
  { id: 'success_label_stats', label: 'Success Label 统计' },
] as const;

const summaryBoxStyle: CSSProperties = {
  marginTop: 8,
  padding: '10px 12px',
  borderRadius: 8,
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  fontSize: 13,
  color: '#334155',
  lineHeight: 1.55,
};

function formatEvalObjectOptionLabel(mode: WorkspaceEvaluationMode): string {
  if (mode === 'trained_model_evaluation') return '已训练模型';
  return '专家策略';
}

function resolveExpertEvaluationMode(template: TaskTemplateDto | null): WorkspaceEvaluationMode {
  const modes = template?.supportedEvaluationModes ?? [];
  if (modes.includes('episode_stability')) return 'episode_stability';
  return 'expert_policy_evaluation';
}

function formatEvalTaskOptionLabel(template: TaskTemplateDto): string {
  if (template.id === 'isaaclab_franka_stack_cube' || isFrankStackCubeEvalTask(template.id)) {
    return FRANKA_STACK_CUBE_PRODUCT_NAME;
  }
  return template.name?.trim() || formatTaskTemplateDisplayName(template.id) || template.id;
}

export interface DatasetEvaluationConfigPayload {
  datasetId: string;
  datasetName: string;
  metrics: string[];
}

export interface ModelEvaluationConfigPayload {
  modelPath?: string;
  simConfig: Record<string, unknown>;
  taskTemplate: ModelTaskTemplateMode;
  selectedTaskIds?: string[];
  modelName?: string;
}

export interface CreateEvaluationPayload {
  evaluationType: EvaluationTopType;
  datasetEvaluationConfig?: DatasetEvaluationConfigPayload;
  modelEvaluationConfig?: ModelEvaluationConfigPayload;
  name: string;
  taskName?: string;
  evaluationMode: '策略评测' | 'episode 稳定性评测' | '数据过程评测';
  relatedTask: string;
  taskTemplateId: string;
  evaluationModeApi: WorkspaceEvaluationMode;
  evaluationObject?: string;
  productEvaluationMode?: string;
  evaluationTypeKey?: 'expert_policy' | 'model' | 'dataset';
  evaluationTypeLabel?: '专家策略评测' | '模型评测' | '数据集评测';
  modelAssetId?: string;
  taskConfig: string;
  checkpoint: string;
  evalBackend: string;
  evalRounds: number;
  seed: number;
  saveVideo: boolean;
  generateReport: boolean;
  metrics: string[];
  selectedMetricKeys: string[];
  taskConfigId?: string;
  evaluationConfig?: Record<string, unknown>;
  status: EvaluationTaskRow['status'];
  cableThreadingRobot?: string;
  cableThreadingCableModel?: string;
  cableThreadingDifficulty?: string;
  cableThreadingHorizon?: number;
  cableThreadingPolicy?: string;
  cableThreadingEvalStrategy?: CableThreadingEvalStrategy;
  cableThreadingCheckpointTrainJobId?: string;
  cableThreadingCheckpointPath?: string;
  cableThreadingCheckpointAssetId?: string;
  dualArmCheckpointPath?: string;
  dualArmEvalSeeds?: number[];
  dualArmMaxCables?: number;
  dualArmRecord?: boolean;
  dualArmHeadless?: boolean;
  dualArmStretchMode?: string;
  dualArmReleaseMode?: string;
  isaacHorizon?: number;
  isaacEvalEpisodes?: number;
}

const DATASET_INFO_MESSAGE =
  '注意：仅支持评测已在平台内构建完成的数据集。如需评测自有数据集或真机数据集，请先返回【数据】->【数据构建】模块导入并构建后，再前来选择。';

export function CreateEvaluationModal({
  open,
  onClose,
  onSave,
  onStart,
  onValidationError,
  initialCheckpoint,
  initialCheckpointJobId,
  initialTemplate,
  initialTaskTemplateId,
  initialModelAssetId,
}: {
  open: boolean;
  onClose: () => void;
  onSave: (payload: CreateEvaluationPayload) => void;
  onStart: (payload: CreateEvaluationPayload) => void | Promise<void>;
  onValidationError?: (message: string) => void;
  initialCheckpoint?: string;
  initialCheckpointJobId?: string;
  initialTemplate?: string;
  initialTaskTemplateId?: string;
  initialModelAssetId?: string;
}) {
  const [evaluationTopType, setEvaluationTopType] = useState<EvaluationTopType>('model');
  const [datasetEvalDatasetId, setDatasetEvalDatasetId] = useState('');
  const [datasetEvalMetrics, setDatasetEvalMetrics] = useState<string[]>([]);
  const [datasetSearch, setDatasetSearch] = useState('');

  const [taskTemplates, setTaskTemplates] = useState<TaskTemplateDto[]>([]);
  const [registryTasksById, setRegistryTasksById] = useState<Record<string, RegistryResource>>({});
  const [registryMetrics, setRegistryMetrics] = useState<RegistryResource[]>([]);
  const [evaluationDatasets, setEvaluationDatasets] = useState<Dataset[]>([]);
  const [allModelAssets, setAllModelAssets] = useState<import('@/types/benchmark').ModelAsset[]>([]);
  const [taskTemplateId, setTaskTemplateId] = useState<string>('cable_threading_single_arm');
  const [evaluationModeApi, setEvaluationModeApi] = useState<WorkspaceEvaluationMode>(
    'expert_policy_evaluation'
  );
  const [selectedModelAssetId, setSelectedModelAssetId] = useState('');
  const [taskConfig, setTaskConfig] = useState<string>('default');
  const [evalRounds, setEvalRounds] = useState<number>(10);
  const [seed, setSeed] = useState(0);
  const [saveVideo, setSaveVideo] = useState(true);
  const [generateReport, setGenerateReport] = useState(true);
  const [taskName, setTaskName] = useState('');
  const [hasUserEditedTaskName, setHasUserEditedTaskName] = useState(false);
  const [cableDifficulty, setCableDifficulty] = useState<string>('easy');
  const [cableHorizon, setCableHorizon] = useState<number>(600);
  const [dualArmEpisodes, setDualArmEpisodes] = useState(1);
  const [dualArmBaseSeed, setDualArmBaseSeed] = useState(42);
  const [dualArmMaxCables, setDualArmMaxCables] = useState(1);
  const [dualArmRecord, setDualArmRecord] = useState(true);
  const [dualArmStretchMode, setDualArmStretchMode] = useState<string>('fixed_distance');
  const [dualArmReleaseMode, setDualArmReleaseMode] = useState<string>('three_phase');
  const [isaacEpisodes, setIsaacEpisodes] = useState(1);
  const [isaacHorizon, setIsaacHorizon] = useState(400);
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);
  const [hasUserEditedMetrics, setHasUserEditedMetrics] = useState(false);
  const [prefillAssetError, setPrefillAssetError] = useState<string | null>(null);
  const [modelAssetCompatHint, setModelAssetCompatHint] = useState<string | null>(null);
  const [pinnedAsset, setPinnedAsset] = useState<import('@/types/benchmark').ModelAsset | null>(null);
  const prevTaskTemplateIdRef = useRef<string | null>(null);
  const metricsTaskTemplateRef = useRef<string | null>(null);
  const pinnedModelAssetId = initialModelAssetId?.trim() || null;

  const selectedTemplate = useMemo(
    () => taskTemplates.find((t) => t.id === taskTemplateId) ?? null,
    [taskTemplates, taskTemplateId]
  );

  const cableThreadingMode = taskTemplateId === 'cable_threading_single_arm';
  const dualArmMode = taskTemplateId === 'dual_arm_cable_manipulation';
  const nutAssemblyMode = taskTemplateId === 'nut_assembly_single_arm';
  const isaacBlockStackingMode = isFrankStackCubeEvalTask(taskTemplateId);
  const evaluationTemplateId = useMemo(
    () => resolveFrankStackCubeEvaluationTemplateId(taskTemplateId),
    [taskTemplateId]
  );
  const evaluationTemplate = useMemo(
    () => taskTemplates.find((t) => t.id === evaluationTemplateId) ?? selectedTemplate,
    [taskTemplates, evaluationTemplateId, selectedTemplate]
  );

  const registryTaskResource = useMemo(() => {
    const registryId =
      selectedTemplate?.registryTaskConfigId ??
      (cableThreadingMode ? 'task_cable_threading_v1' : null);
    if (registryId && registryTasksById[registryId]) {
      return registryTasksById[registryId];
    }
    return null;
  }, [selectedTemplate?.registryTaskConfigId, registryTasksById, cableThreadingMode]);

  const evaluationConfig = useMemo(
    () => deriveEvaluationConfigFromTask(selectedTemplate, registryTaskResource),
    [selectedTemplate, registryTaskResource]
  );

  const metricDefinitions = useMemo(
    () => deriveMetricDefinitionsFromTask(selectedTemplate, registryTaskResource, registryMetrics),
    [selectedTemplate, registryTaskResource, registryMetrics]
  );

  const templateLabel =
    selectedTemplate?.name ??
    formatTaskTemplateDisplayName(taskTemplateId) ??
    taskTemplateId;

  const evaluableTaskTemplates = useMemo(
    () =>
      taskTemplates.filter((t) => {
        if (isFrankStackCubeInternalTemplateId(t.id)) return false;
        if (shouldShowFrankStackCubeInEvalDropdown(t)) return true;
        return (t.hasEvaluationRunner ?? (t.supportedEvaluationModes?.length ?? 0) > 0) || t.id === taskTemplateId;
      }),
    [taskTemplates, taskTemplateId]
  );

  const modelCompatibilityContext = useMemo(
    () => ({
      taskTemplateId: evaluationTemplateId,
      taskType: evaluationConfig.taskType,
      evaluationMode: evaluationModeApi,
    }),
    [evaluationTemplateId, evaluationConfig.taskType, evaluationModeApi]
  );

  const filteredModelAssets = useMemo(() => {
    const templateForAssets = evaluationTemplate ?? selectedTemplate;
    if (!templateForAssets) return [];
    let assets = allModelAssets.filter(
      (a) => isEvaluableModelAsset(a) && modelAssetMatchesTaskTemplate(a, templateForAssets)
    );
    if (evaluationModeApi === 'trained_model_evaluation') {
      assets = assets.filter((a) =>
        isModelAssetCompatibleWithEvaluationTask(a, modelCompatibilityContext)
      );
    }
    if (
      templateForAssets?.id === 'cable_threading_single_arm' &&
      evaluationModeApi === 'trained_model_evaluation'
    ) {
      assets = assets.filter((a) => modelAssetSupportsCableThreadingEvalObs(a));
    }
    if (
      pinnedAsset &&
      !assets.some((asset) => asset.id === pinnedAsset.id) &&
      isModelAssetCompatibleWithEvaluationTask(pinnedAsset, modelCompatibilityContext)
    ) {
      assets = [pinnedAsset, ...assets];
    }
    return assets;
  }, [
    allModelAssets,
    selectedTemplate,
    evaluationTemplate,
    evaluationModeApi,
    modelCompatibilityContext,
    pinnedAsset,
  ]);

  const trainingCheckpointOptions = useMemo(() => {
    const options = modelAssetsToCheckpointOptions(filteredModelAssets);
    if (!pinnedModelAssetId) return options;
    if (options.some((item) => item.modelAssetId === pinnedModelAssetId)) return options;
    const asset = pinnedAsset ?? allModelAssets.find((item) => item.id === pinnedModelAssetId);
    if (!asset?.checkpointPath) return options;
    return [
      {
        trainJobId: asset.sourceTrainingJobId,
        modelAssetId: asset.id,
        label: resolveModelAssetColumnLabel(asset),
        ready: true,
        checkpointPath: asset.checkpointPath,
      },
      ...options,
    ];
  }, [filteredModelAssets, pinnedModelAssetId, pinnedAsset, allModelAssets]);

  const filteredEvaluationDatasets = useMemo(() => {
    const q = datasetSearch.trim().toLowerCase();
    if (!q) return evaluationDatasets;
    return evaluationDatasets.filter((d) => formatDatasetOptionLabel(d).toLowerCase().includes(q));
  }, [evaluationDatasets, datasetSearch]);

  const selectedEvaluationDataset = useMemo(
    () => evaluationDatasets.find((d) => d.id === datasetEvalDatasetId) ?? null,
    [evaluationDatasets, datasetEvalDatasetId]
  );

  const dualArmSeeds = useMemo(
    () => buildDualArmEvalSeeds(dualArmEpisodes, dualArmBaseSeed),
    [dualArmEpisodes, dualArmBaseSeed]
  );

  const cableEvalStrategy: CableThreadingEvalStrategy =
    evaluationModeApi === 'trained_model_evaluation' ? 'checkpoint' : 'scripted';

  const cableTrainedPolicyType = useMemo(() => {
    if (cableEvalStrategy !== 'checkpoint') return 'scripted';
    const asset = pinnedAsset ?? allModelAssets.find((item) => item.id === selectedModelAssetId);
    const modelTypeId = String(asset?.modelTypeId ?? '').toLowerCase();
    const modelType = String(asset?.modelType ?? '').toLowerCase();
    const framework = String(asset?.framework ?? '').toLowerCase();
    const trainingBackend = String(asset?.trainingBackend ?? asset?.backendType ?? '').toLowerCase();
    const baseAlgorithm = String(asset?.baseAlgorithm ?? '').toLowerCase();

    if (
      modelType === 'pi0' ||
      trainingBackend === 'pi0' ||
      baseAlgorithm === 'pi0' ||
      modelTypeId === 'pi0' ||
      framework === 'pi0'
    ) {
      return 'pi0';
    }
    if (
      modelType === 'act' ||
      trainingBackend === 'act' ||
      baseAlgorithm === 'act' ||
      modelTypeId === 'act' ||
      framework === 'act'
    ) {
      return 'act';
    }
    if (modelType === 'diffusion_policy' || framework.includes('diffusion')) {
      return 'diffusion_policy';
    }
    return 'robomimic';
  }, [cableEvalStrategy, pinnedAsset, allModelAssets, selectedModelAssetId]);

  const cableCheckpointTrainJobId = useMemo(() => {
    const opt = trainingCheckpointOptions.find((o) => o.modelAssetId === selectedModelAssetId);
    return opt?.trainJobId ?? '';
  }, [trainingCheckpointOptions, selectedModelAssetId]);

  const selectedTrainingCheckpoint = useMemo(
    () => trainingCheckpointOptions.find((item) => item.modelAssetId === selectedModelAssetId) ?? null,
    [trainingCheckpointOptions, selectedModelAssetId]
  );

  const dualArmTrainedMode =
    dualArmMode && evaluationModeApi === 'trained_model_evaluation';

  const isaacTrainedMode =
    isaacBlockStackingMode && evaluationModeApi === 'trained_model_evaluation';

  const dualArmCheckpointUnavailable =
    dualArmTrainedMode &&
    (!selectedTrainingCheckpoint || !selectedTrainingCheckpoint.ready);

  const checkpointUnavailable =
    cableThreadingMode &&
    evaluationModeApi === 'trained_model_evaluation' &&
    (!selectedTrainingCheckpoint || !selectedTrainingCheckpoint.ready);

  const isaacCheckpointUnavailable =
    isaacTrainedMode &&
    (!selectedTrainingCheckpoint || !selectedTrainingCheckpoint.ready);

  const datasetSubmitReady =
    Boolean(datasetEvalDatasetId) && datasetEvalMetrics.length > 0 && evaluationDatasets.length > 0;

  const modelSubmitDisabled =
    !taskTemplateId ||
    (nutAssemblyMode && evaluationModeApi === 'trained_model_evaluation' && !selectedTrainingCheckpoint) ||
    (cableThreadingMode && Boolean(checkpointUnavailable)) ||
    (dualArmMode && Boolean(dualArmCheckpointUnavailable)) ||
    (isaacTrainedMode &&
      (trainingCheckpointOptions.length === 0 || Boolean(isaacCheckpointUnavailable)));

  useEffect(() => {
    if (!open) {
      setPrefillAssetError(null);
      setPinnedAsset(null);
      setHasUserEditedTaskName(false);
      return;
    }
    void listTaskTemplates().then((res) => setTaskTemplates(res.taskTemplates));
    void listEvaluationDatasets()
      .then((res) => setEvaluationDatasets(res.datasets))
      .catch(() => setEvaluationDatasets([]));
    void listModelAssets({ forEvaluation: true }).then((res) => setAllModelAssets(res.modelAssets));
    void listRegistryResources({ assetType: 'task' })
      .then((res) => {
        const map: Record<string, RegistryResource> = {};
        for (const resource of res.resources) {
          map[resource.assetId] = resource;
        }
        setRegistryTasksById(map);
      })
      .catch(() => setRegistryTasksById({}));
    void listRegistryResources({ assetType: 'metric' })
      .then((res) => setRegistryMetrics(res.resources))
      .catch(() => setRegistryMetrics([]));
  }, [open]);

  useEffect(() => {
    if (!open || !pinnedModelAssetId) return;

    let cancelled = false;
    void getModelAsset(pinnedModelAssetId)
      .then((asset) => {
        if (cancelled) return;
        setPinnedAsset(asset);
        setAllModelAssets((prev) =>
          prev.some((item) => item.id === asset.id) ? prev : [asset, ...prev]
        );
        const templateId = resolveTaskTemplateIdFromModelAsset(asset);
        setTaskTemplateId(templateId);
        setEvaluationTopType('model');
        setEvaluationModeApi('trained_model_evaluation');
        setSelectedModelAssetId(asset.id);
        if (!hasUserEditedTaskName) {
          setTaskName(formatModelEvaluationTaskName(resolveModelAssetColumnLabel(asset)));
        }
        setPrefillAssetError(null);
      })
      .catch(() => {
        if (cancelled) return;
        setPrefillAssetError('无法加载所选模型资产，请稍后重试或手动选择其他模型');
        setEvaluationTopType('model');
        setEvaluationModeApi('trained_model_evaluation');
        setSelectedModelAssetId(pinnedModelAssetId);
      });

    return () => {
      cancelled = true;
    };
  }, [open, pinnedModelAssetId, hasUserEditedTaskName]);

  useEffect(() => {
    if (!open) return;
    if (pinnedModelAssetId) {
      setEvaluationTopType('model');
      setEvaluationModeApi('trained_model_evaluation');
      setSelectedModelAssetId(pinnedModelAssetId);
      setPrefillAssetError(null);
      return;
    }
    const fromUrl =
      initialTaskTemplateId ??
      resolveTemplateIdFromLegacyName(initialTemplate ?? '') ??
      'cable_threading_single_arm';
    setEvaluationTopType('model');
    setDatasetEvalDatasetId('');
    setDatasetEvalMetrics([]);
    setDatasetSearch('');
    setTaskTemplateId(fromUrl);
    setTaskConfig('default');
    setTaskName('');
    setHasUserEditedTaskName(false);
    setEvaluationModeApi(
      initialCheckpointJobId || initialModelAssetId
        ? 'trained_model_evaluation'
        : 'expert_policy_evaluation'
    );
    setSelectedModelAssetId(initialModelAssetId ?? initialCheckpoint ?? '');
    setHasUserEditedMetrics(false);
    prevTaskTemplateIdRef.current = null;
    metricsTaskTemplateRef.current = null;
  }, [open, pinnedModelAssetId, initialCheckpointJobId, initialModelAssetId, initialTemplate, initialTaskTemplateId]);

  useEffect(() => {
    if (!open || !selectedTemplate) return;

    const episodesValue = evaluationConfig.episodes;
    const horizonValue = evaluationConfig.horizon;
    const seedValue = evaluationConfig.seed;

    if (isaacBlockStackingMode) {
      setIsaacEpisodes(episodesValue);
      setIsaacHorizon(horizonValue);
    } else if (dualArmMode) {
      setDualArmEpisodes(episodesValue);
      setDualArmBaseSeed(seedValue);
      setDualArmMaxCables(evaluationConfig.maxCables ?? 1);
      setDualArmRecord(evaluationConfig.recordVideo);
      if (evaluationConfig.stretchMode) setDualArmStretchMode(evaluationConfig.stretchMode);
      if (evaluationConfig.releaseMode) setDualArmReleaseMode(evaluationConfig.releaseMode);
    } else if (cableThreadingMode) {
      setEvalRounds(episodesValue);
      setCableHorizon(horizonValue);
      setSeed(seedValue);
      setSaveVideo(evaluationConfig.recordVideo);
      if (evaluationConfig.difficulty) setCableDifficulty(evaluationConfig.difficulty);
    } else {
      setEvalRounds(episodesValue);
      setSeed(seedValue);
      setSaveVideo(evaluationConfig.recordVideo);
    }

    if (evaluationConfig.taskConfigId) {
      setTaskConfig(evaluationConfig.taskConfigId);
    }
  }, [
    open,
    taskTemplateId,
    selectedTemplate,
    evaluationConfig,
    cableThreadingMode,
    dualArmMode,
    isaacBlockStackingMode,
  ]);

  useEffect(() => {
    if (!open || !selectedTemplate) return;

    const taskChanged =
      metricsTaskTemplateRef.current !== null &&
      metricsTaskTemplateRef.current !== taskTemplateId;
    const isFirstBind = metricsTaskTemplateRef.current === null;
    metricsTaskTemplateRef.current = taskTemplateId;

    if ((taskChanged || isFirstBind) && !hasUserEditedMetrics) {
      setSelectedMetricKeys(metricDefinitions.defaultSelectedMetricKeys);
    }
  }, [
    open,
    taskTemplateId,
    selectedTemplate,
    hasUserEditedMetrics,
    metricDefinitions.defaultSelectedMetricKeys,
  ]);

  useEffect(() => {
    if (!open) {
      prevTaskTemplateIdRef.current = null;
      return;
    }
    if (!selectedTemplate) return;

    const taskChanged =
      prevTaskTemplateIdRef.current !== null && prevTaskTemplateIdRef.current !== taskTemplateId;
    const isFirstBind = prevTaskTemplateIdRef.current === null;
    prevTaskTemplateIdRef.current = taskTemplateId;

    const readyOptions = trainingCheckpointOptions.filter((item) => item.ready);
    setSelectedModelAssetId((prev) => {
      if (pinnedModelAssetId) {
        return pinnedModelAssetId;
      }
      if (taskChanged) {
        return readyOptions.length === 1 ? readyOptions[0].modelAssetId : '';
      }
      if (!prev) {
        return readyOptions.length === 1 ? readyOptions[0].modelAssetId : '';
      }
      if (isFirstBind && allModelAssets.length === 0) {
        return prev;
      }
      const stillValid = filteredModelAssets.some((asset) => asset.id === prev);
      return stillValid ? prev : '';
    });
  }, [
    open,
    taskTemplateId,
    selectedTemplate,
    allModelAssets,
    filteredModelAssets,
    trainingCheckpointOptions,
    pinnedModelAssetId,
  ]);

  useEffect(() => {
    if (!open || evaluationModeApi !== 'trained_model_evaluation') {
      setModelAssetCompatHint(null);
      return;
    }
    if (!selectedModelAssetId) {
      setModelAssetCompatHint(null);
      return;
    }
    const asset = pinnedAsset ?? allModelAssets.find((item) => item.id === selectedModelAssetId);
    if (!asset) return;
    if (!isModelAssetCompatibleWithEvaluationTask(asset, modelCompatibilityContext)) {
      if (!pinnedModelAssetId) {
        setSelectedModelAssetId('');
      }
      setModelAssetCompatHint('当前模型资产不兼容所选评测任务，请重新选择。');
      return;
    }
    setModelAssetCompatHint(null);
  }, [
    open,
    evaluationModeApi,
    selectedModelAssetId,
    allModelAssets,
    pinnedAsset,
    modelCompatibilityContext,
    taskTemplateId,
    pinnedModelAssetId,
  ]);

  const validateBeforeStart = (): string | null => {
    if (evaluationTopType === 'dataset') return null;
    if (evaluationModeApi !== 'trained_model_evaluation') return null;
    if (!selectedModelAssetId) {
      return '请选择模型资产';
    }
    const asset = pinnedAsset ?? allModelAssets.find((item) => item.id === selectedModelAssetId);
    if (!asset) {
      return '请选择模型资产';
    }
    if (!isEvaluableModelAsset(asset)) {
      return '所选模型资产不存在或模型文件已丢失，请重新选择可用模型资产。';
    }
    if (!isModelAssetCompatibleWithEvaluationTask(asset, modelCompatibilityContext)) {
      return getModelAssetIncompatibilityMessage(asset, modelCompatibilityContext);
    }
    return null;
  };

  const handleStartClick = (status: EvaluationTaskRow['status']) => {
    const error = validateBeforeStart();
    if (error) {
      onValidationError?.(error);
      return;
    }
    void onStart(buildPayload(status));
  };

  useEffect(() => {
    if (!selectedTemplate) return;
    if (pinnedModelAssetId) {
      if (selectedTemplate.supportedEvaluationModes?.includes('trained_model_evaluation')) {
        setEvaluationModeApi('trained_model_evaluation');
      } else {
        setPrefillAssetError('当前任务类型不支持模型评测');
      }
      return;
    }
    const modes = selectedTemplate.supportedEvaluationModes ?? [];
    if (modes.length > 0 && !modes.includes(evaluationModeApi)) {
      setEvaluationModeApi(modes[0] as WorkspaceEvaluationMode);
    }
  }, [selectedTemplate, evaluationModeApi, pinnedModelAssetId]);

  useEffect(() => {
    if (!open || hasUserEditedTaskName) return;
    if (pinnedModelAssetId) return;
    if (cableThreadingMode) {
      setTaskName(generateCableThreadingEvalTaskName());
    } else if (dualArmMode) {
      setTaskName(generateDualArmEvalTaskName());
    } else if (isaacBlockStackingMode) {
      setTaskName(generateIsaacBlockStackingEvalTaskName());
    }
  }, [
    open,
    hasUserEditedTaskName,
    cableThreadingMode,
    dualArmMode,
    isaacBlockStackingMode,
    taskTemplateId,
    pinnedModelAssetId,
  ]);

  const buildRuntimeConfig = (): Record<string, unknown> => {
    const episodesValue = isaacBlockStackingMode
      ? isaacEpisodes
      : dualArmMode
        ? dualArmEpisodes
        : evalRounds;
    const horizonValue = isaacBlockStackingMode ? isaacHorizon : cableHorizon;
    const seedValue = dualArmMode ? dualArmBaseSeed : seed;
    const recordVideoValue = dualArmMode ? dualArmRecord : saveVideo;

    return {
      ...evaluationConfig.config,
      simulationPlatform: evaluationConfig.simulationPlatform,
      robotType: evaluationConfig.robotType,
      robot: evaluationConfig.config.robotType ?? evaluationConfig.robotType,
      cableModel: evaluationConfig.cableModel,
      difficulty: cableThreadingMode ? cableDifficulty : evaluationConfig.difficulty,
      episodes: episodesValue,
      horizon: horizonValue,
      seed: seedValue,
      recordVideo: recordVideoValue,
      taskConfigId: evaluationConfig.taskConfigId,
      defaultTaskEnv: evaluationConfig.defaultTaskEnv,
      metrics: selectedMetricKeys,
    };
  };

  const buildModelSimConfig = (): Record<string, unknown> => ({
    taskTemplateId: evaluationConfig.taskTemplateId,
    evaluationMode: evaluationModeApi,
    evalBackend: evaluationConfig.simulationPlatform,
    evalRounds: isaacBlockStackingMode ? isaacEpisodes : dualArmMode ? dualArmEpisodes : evalRounds,
    seed: dualArmMode ? dualArmBaseSeed : seed,
    saveVideo: dualArmMode ? dualArmRecord : saveVideo,
    generateReport,
    config: buildRuntimeConfig(),
    metrics: selectedMetricKeys,
    cableThreading: cableThreadingMode
      ? {
          robot: evaluationConfig.config.robotType ?? evaluationConfig.robotType,
          cableModel: evaluationConfig.cableModel,
          difficulty: cableDifficulty,
          horizon: cableHorizon,
          policy: cableEvalStrategy === 'checkpoint' ? 'robomimic' : 'scripted',
        }
      : undefined,
    dualArmCable: dualArmMode
      ? {
          stretchMode: dualArmStretchMode,
          releaseMode: dualArmReleaseMode,
          maxCables: dualArmMaxCables,
          record: dualArmRecord,
          seeds: dualArmSeeds,
        }
      : undefined,
    isaacBlockStacking: isaacBlockStackingMode
      ? {
          taskEnv: evaluationConfig.defaultTaskEnv,
          horizon: isaacHorizon,
          episodes: isaacEpisodes,
        }
      : undefined,
  });

  const buildPayload = (status: EvaluationTaskRow['status']): CreateEvaluationPayload => {
    const generatedDefaultTaskName = isaacBlockStackingMode
      ? generateIsaacBlockStackingEvalTaskName()
      : dualArmMode
        ? generateDualArmEvalTaskName()
        : generateCableThreadingEvalTaskName();
    const normalizedTaskName = taskName.trim() || generatedDefaultTaskName;

    if (process.env.NODE_ENV === 'development') {
      console.debug('[CreateEvaluationModal] submit taskName=', taskName);
      console.debug('[CreateEvaluationModal] payload.taskName=', normalizedTaskName);
    }

    const modelPath = isaacTrainedMode
      ? selectedTrainingCheckpoint?.checkpointPath ?? undefined
      : cableEvalStrategy === 'checkpoint'
        ? selectedTrainingCheckpoint?.checkpointPath ?? undefined
        : dualArmTrainedMode
          ? selectedTrainingCheckpoint?.checkpointPath ?? undefined
          : undefined;

    const runtimeConfig = buildRuntimeConfig();
    const productEvalFields =
      evaluationTopType === 'dataset'
        ? buildProductEvaluationFields({
            evaluationModeApi: 'dataset_evaluation',
            evaluationTopType: 'dataset',
          })
        : buildProductEvaluationFields({
            evaluationModeApi,
            modelAssetId:
              evaluationModeApi === 'trained_model_evaluation' ? selectedModelAssetId || undefined : undefined,
            evaluationTopType: 'model',
          });

    const base: CreateEvaluationPayload = {
      evaluationType: evaluationTopType,
      name: normalizedTaskName,
      taskName: normalizedTaskName,
      evaluationMode:
        productEvalFields.evaluationTypeLabel === '数据集评测'
          ? '数据过程评测'
          : productEvalFields.evaluationTypeLabel === '模型评测'
            ? '策略评测'
            : dualArmMode
              ? 'episode 稳定性评测'
              : '策略评测',
      relatedTask: isaacBlockStackingMode ? FRANKA_STACK_CUBE_PRODUCT_NAME : templateLabel,
      taskTemplateId: evaluationTemplateId,
      evaluationModeApi,
      evaluationObject: productEvalFields.evaluationObject,
      productEvaluationMode: productEvalFields.productEvaluationMode,
      evaluationTypeKey: productEvalFields.evaluationType,
      evaluationTypeLabel: productEvalFields.evaluationTypeLabel,
      modelAssetId:
        evaluationModeApi === 'trained_model_evaluation' ? selectedModelAssetId || undefined : undefined,
      taskConfig: evaluationConfig.taskConfigId ?? taskConfig,
      taskConfigId: evaluationConfig.taskConfigId ?? undefined,
      checkpoint: selectedModelAssetId,
      evalBackend: evaluationConfig.simulationPlatform,
      evalRounds: isaacBlockStackingMode ? isaacEpisodes : dualArmMode ? dualArmEpisodes : evalRounds,
      seed: dualArmMode ? dualArmBaseSeed : seed,
      saveVideo: dualArmMode ? dualArmRecord : saveVideo,
      generateReport,
      metrics: selectedMetricKeys,
      selectedMetricKeys,
      evaluationConfig: runtimeConfig,
      status,
      ...(cableThreadingMode
        ? {
            cableThreadingRobot: String(
              evaluationConfig.config.robotType ?? evaluationConfig.robotType
            ),
            cableThreadingCableModel: evaluationConfig.cableModel ?? undefined,
            cableThreadingDifficulty: cableDifficulty,
            cableThreadingHorizon: cableHorizon,
            cableThreadingPolicy: cableTrainedPolicyType,
            cableThreadingEvalStrategy: cableEvalStrategy,
            cableThreadingCheckpointTrainJobId:
              cableEvalStrategy === 'checkpoint' ? cableCheckpointTrainJobId : undefined,
            cableThreadingCheckpointPath:
              cableEvalStrategy === 'checkpoint'
                ? selectedTrainingCheckpoint?.checkpointPath ?? undefined
                : undefined,
            cableThreadingCheckpointAssetId:
              cableEvalStrategy === 'checkpoint' ? selectedModelAssetId || undefined : undefined,
          }
        : {}),
      ...(dualArmMode
        ? {
            dualArmCheckpointPath:
              dualArmTrainedMode ? selectedTrainingCheckpoint?.checkpointPath ?? undefined : undefined,
            dualArmEvalSeeds: dualArmSeeds,
            dualArmMaxCables: dualArmMaxCables,
            dualArmRecord: dualArmRecord,
            dualArmHeadless: true,
            dualArmStretchMode:
              evaluationModeApi === 'episode_stability' ? dualArmStretchMode : undefined,
            dualArmReleaseMode:
              evaluationModeApi === 'episode_stability' ? dualArmReleaseMode : undefined,
          }
        : {}),
      ...(isaacBlockStackingMode
        ? {
            isaacHorizon: isaacHorizon,
            isaacEvalEpisodes: isaacEpisodes,
          }
        : {}),
    };

    if (evaluationTopType === 'dataset') {
      return {
        ...base,
        datasetEvaluationConfig: {
          datasetId: datasetEvalDatasetId,
          datasetName: selectedEvaluationDataset?.name ?? '',
          metrics: datasetEvalMetrics,
        },
      };
    }

    return {
      ...base,
      modelEvaluationConfig: {
        modelPath,
        simConfig: buildModelSimConfig(),
        taskTemplate: 'single_task',
        modelName: normalizedTaskName,
      },
    };
  };

  const toggleDatasetMetric = (metricId: string) => {
    setDatasetEvalMetrics((prev) =>
      prev.includes(metricId) ? prev.filter((id) => id !== metricId) : [...prev, metricId]
    );
  };

  const toggleModelEvalMetric = (metricKey: string) => {
    setHasUserEditedMetrics(true);
    setSelectedMetricKeys((prev) =>
      prev.includes(metricKey) ? prev.filter((id) => id !== metricKey) : [...prev, metricKey]
    );
  };

  const renderModelTaskNameField = () => (
    <div style={{ marginBottom: 16 }}>
      <label style={workspaceModalFieldLabel}>任务名称</label>
      <input
        type="text"
        value={taskName}
        placeholder="请输入任务名称"
        onChange={(e) => {
          setTaskName(e.target.value);
          setHasUserEditedTaskName(true);
        }}
        style={workspaceModalSelectStyle}
      />
    </div>
  );

  const renderTopTypeSelector = () => (
    <div style={{ marginBottom: 16 }}>
      <div style={workspaceModalSectionLabel}>评测类型</div>
      <div style={{ display: 'flex', flexDirection: 'row', gap: 24, alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, cursor: 'pointer' }}>
          <input
            type="radio"
            name="evaluationTopType"
            checked={evaluationTopType === 'model'}
            onChange={() => setEvaluationTopType('model')}
          />
          模型评测
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, cursor: 'pointer' }}>
          <input
            type="radio"
            name="evaluationTopType"
            checked={evaluationTopType === 'dataset'}
            onChange={() => setEvaluationTopType('dataset')}
          />
          数据集评测
        </label>
      </div>
    </div>
  );

  const renderDatasetForm = () => (
    <>
      <div style={workspaceModalSectionLabel}>数据集评测</div>
      <div style={{ marginBottom: 12 }}>
        <label style={workspaceModalFieldLabel}>选择数据集</label>
        <input
          type="search"
          placeholder="搜索数据集名称或版本"
          value={datasetSearch}
          onChange={(e) => setDatasetSearch(e.target.value)}
          style={{ ...workspaceModalSelectStyle, marginBottom: 8 }}
          disabled={evaluationDatasets.length === 0}
        />
        <select
          style={workspaceModalSelectStyle}
          value={datasetEvalDatasetId}
          onChange={(e) => setDatasetEvalDatasetId(e.target.value)}
          disabled={evaluationDatasets.length === 0}
        >
          {evaluationDatasets.length === 0 ? (
            <option value="">暂无可评测的数据集，请前往数据构建</option>
          ) : (
            <>
              <option value="">请选择数据集</option>
              {filteredEvaluationDatasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {formatDatasetOptionLabel(d)}
                </option>
              ))}
            </>
          )}
        </select>
      </div>

      <div
        style={{
          marginBottom: 16,
          padding: '12px 14px',
          borderRadius: 8,
          border: '1px solid #93c5fd',
          background: '#eff6ff',
          color: '#1e40af',
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        {DATASET_INFO_MESSAGE}
      </div>

      <div>
        <label style={workspaceModalFieldLabel}>评测指标</label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
          {DATASET_EVAL_METRIC_OPTIONS.map((item) => (
            <label
              key={item.id}
              style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}
            >
              <input
                type="checkbox"
                checked={datasetEvalMetrics.includes(item.id)}
                onChange={() => toggleDatasetMetric(item.id)}
              />
              {item.label}
            </label>
          ))}
        </div>
      </div>
    </>
  );

  const supportedEvaluationModes = selectedTemplate?.supportedEvaluationModes ?? [];
  const expertEvaluationModeValue = resolveExpertEvaluationMode(selectedTemplate);
  const showExpertEvalObjectOption = supportedEvaluationModes.some(
    (mode) => mode === 'expert_policy_evaluation' || mode === 'episode_stability'
  );
  const showTrainedEvalObjectOption = supportedEvaluationModes.includes('trained_model_evaluation');

  const showUnifiedEvalParams =
    cableThreadingMode || dualArmMode || isaacBlockStackingMode || nutAssemblyMode;

  const renderReadOnlyTaskBindingCard = () => (
    <div style={{ ...summaryBoxStyle, marginBottom: 12, gridColumn: '1 / -1' }}>
      <div>仿真平台：{evaluationConfig.simulationPlatform}</div>
      <div>机器人类型：{evaluationConfig.robotType}</div>
      {evaluationConfig.cableModelLabel ? (
        <div>线路模型：{evaluationConfig.cableModelLabel}</div>
      ) : null}
      {isaacBlockStackingMode && evaluationConfig.defaultTaskEnv ? (
        <div>任务环境：{evaluationConfig.defaultTaskEnv}</div>
      ) : null}
    </div>
  );

  const renderUnifiedEvaluationParams = () => {
    const episodesValue = isaacBlockStackingMode
      ? isaacEpisodes
      : dualArmMode
        ? dualArmEpisodes
        : evalRounds;
    const episodesMin = evaluationConfig.episodesMin ?? 1;
    const episodesMax = evaluationConfig.episodesMax ?? 100;
    const horizonValue = isaacBlockStackingMode ? isaacHorizon : cableHorizon;
    const horizonMax = evaluationConfig.horizonMax;
    const horizonMin = evaluationConfig.horizonMin;
    const seedValue = dualArmMode ? dualArmBaseSeed : seed;
    const recordChecked = dualArmMode ? dualArmRecord : saveVideo;
    const showDifficulty = cableThreadingMode && Boolean(evaluationConfig.difficulty);

    const renderEpisodesField = () => (
      <div>
        <label style={workspaceModalFieldLabel}>Episodes</label>
        <input
          type="number"
          min={episodesMin}
          max={episodesMax}
          value={episodesValue}
          onChange={(e) => {
            const raw = Number(e.target.value);
            const next = Number.isFinite(raw) ? raw : episodesMin;
            if (next > episodesMax) {
              onValidationError?.(`Episodes 最大支持 ${episodesMax}`);
            }
            const clamped = clampEpisodes(next, episodesMin, episodesMax);
            if (isaacBlockStackingMode) {
              setIsaacEpisodes(clamped);
            } else if (dualArmMode) {
              setDualArmEpisodes(clamped);
            } else {
              setEvalRounds(clamped);
            }
          }}
          style={workspaceModalSelectStyle}
        />
      </div>
    );

    const renderHorizonField = () => (
      <div>
        <label style={workspaceModalFieldLabel}>Horizon</label>
        <input
          type="number"
          min={horizonMin}
          max={horizonMax}
          value={horizonValue}
          onChange={(e) => {
            const next = Number(e.target.value) || horizonMin;
            if (isaacBlockStackingMode) {
              setIsaacHorizon(next);
            } else {
              setCableHorizon(next);
            }
          }}
          style={workspaceModalSelectStyle}
        />
      </div>
    );

    const renderSeedField = () => (
      <div>
        <label style={workspaceModalFieldLabel}>Seed</label>
        <input
          type="number"
          value={seedValue}
          onChange={(e) => {
            const next = Number(e.target.value) || 0;
            if (dualArmMode) {
              setDualArmBaseSeed(next);
            } else {
              setSeed(next);
            }
          }}
          style={workspaceModalSelectStyle}
        />
      </div>
    );

    const renderRecordCheckbox = () => (
      <div style={{ gridColumn: '1 / -1' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
          <input
            type="checkbox"
            checked={recordChecked}
            onChange={(e) => {
              if (dualArmMode) {
                setDualArmRecord(e.target.checked);
              } else {
                setSaveVideo(e.target.checked);
              }
            }}
          />
          是否进行仿真录制
        </label>
      </div>
    );

    return (
      <>
        <div style={{ ...workspaceModalSectionLabel, marginTop: 12 }}>评测参数设置</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
          {isaacBlockStackingMode ? (
            <div style={{ gridColumn: '1 / -1', fontSize: 12, color: '#6b7280', lineHeight: 1.55, marginBottom: 4 }}>
              {FRANKA_STACK_CUBE_PRODUCT_SUBTITLE} · 评测已接入 Isaac Lab rollout，success 来自环境 success_term（当前复用
              isaac_block_stacking adapter）。
            </div>
          ) : null}
          {renderReadOnlyTaskBindingCard()}
          {showDifficulty ? (
            <div>
              <label style={workspaceModalFieldLabel}>难度</label>
              <select
                style={workspaceModalSelectStyle}
                value={cableDifficulty}
                onChange={(e) => setCableDifficulty(e.target.value)}
              >
                {['easy', 'medium', 'hard'].map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          {renderEpisodesField()}
          {cableThreadingMode || isaacBlockStackingMode ? renderHorizonField() : null}
          {renderSeedField()}
          {renderRecordCheckbox()}
        </div>
      </>
    );
  };

  const renderEvaluationMetricSelection = () => {
    if (!showUnifiedEvalParams) return null;
    return (
      <div style={{ marginTop: 12 }}>
        <label style={workspaceModalFieldLabel}>评测指标选择</label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
          {metricDefinitions.availableMetrics.map((item) => (
            <label
              key={item.key}
              style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}
              title={
                item.requiresStepMetrics
                  ? `${item.description ?? ''}（需记录 step 数据）`.trim()
                  : item.description
              }
            >
              <input
                type="checkbox"
                checked={selectedMetricKeys.includes(item.key)}
                onChange={() => toggleModelEvalMetric(item.key)}
              />
              <span>
                {item.label}
                {item.requiresStepMetrics ? (
                  <span style={{ marginLeft: 4, fontSize: 11, color: '#94a3b8' }}>需 step 数据</span>
                ) : null}
              </span>
            </label>
          ))}
        </div>
      </div>
    );
  };

  const renderModelForm = () => (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div>
          <label style={workspaceModalFieldLabel}>关联任务</label>
          <select
            style={workspaceModalSelectStyle}
            value={taskTemplateId}
            onChange={(e) => setTaskTemplateId(e.target.value)}
          >
            {evaluableTaskTemplates.length === 0 ? (
              <option value="">暂无可评测任务</option>
            ) : (
              evaluableTaskTemplates.map((t) => (
                <option key={t.id} value={t.id}>
                  {formatEvalTaskOptionLabel(t)}
                </option>
              ))
            )}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>评测对象</label>
          <select
            style={workspaceModalSelectStyle}
            value={evaluationModeApi}
            onChange={(e) => setEvaluationModeApi(e.target.value as WorkspaceEvaluationMode)}
            disabled={Boolean(pinnedModelAssetId)}
          >
            {showExpertEvalObjectOption ? (
              <option value={expertEvaluationModeValue}>
                {formatEvalObjectOptionLabel(expertEvaluationModeValue)}
              </option>
            ) : null}
            {showTrainedEvalObjectOption ? (
              <option value="trained_model_evaluation">
                {formatEvalObjectOptionLabel('trained_model_evaluation')}
              </option>
            ) : null}
          </select>
          {pinnedModelAssetId ? (
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
              已从训练任务带入模型资产，评测对象为已训练模型。
            </p>
          ) : null}
          {prefillAssetError ? (
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.5 }}>
              {prefillAssetError}
            </p>
          ) : null}
        </div>

        {cableThreadingMode && evaluationModeApi === 'trained_model_evaluation' ? (
          <div>
            <label style={workspaceModalFieldLabel}>模型资产</label>
            <select
              style={workspaceModalSelectStyle}
              value={selectedModelAssetId}
              onChange={(e) => setSelectedModelAssetId(e.target.value)}
              disabled={Boolean(pinnedModelAssetId && !prefillAssetError)}
            >
              <option value="">请选择已训练模型</option>
              {trainingCheckpointOptions.map((item) => (
                <option
                  key={item.modelAssetId}
                  value={item.modelAssetId}
                  disabled={!item.ready}
                >
                  {item.ready ? item.label : `${item.label}（checkpoint 不可用）`}
                </option>
              ))}
            </select>
            {trainingCheckpointOptions.length === 0 ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
                {getNoCompatibleModelAssetsHint(modelCompatibilityContext)}
                当前任务没有可用模型资产，请先完成对应任务的训练，或检查模型文件是否存在。
              </p>
            ) : null}
          </div>
        ) : null}
        {isaacTrainedMode ? (
          <div>
            <label style={workspaceModalFieldLabel}>模型资产</label>
            <select
              style={workspaceModalSelectStyle}
              value={selectedModelAssetId}
              onChange={(e) => setSelectedModelAssetId(e.target.value)}
              disabled={Boolean(pinnedModelAssetId && !prefillAssetError)}
            >
              <option value="">暂无兼容的已训练模型</option>
              {trainingCheckpointOptions.map((item) => (
                <option
                  key={item.modelAssetId}
                  value={item.modelAssetId}
                  disabled={!item.ready}
                >
                  {item.ready ? item.label : `${item.label}（checkpoint 不可用）`}
                </option>
              ))}
            </select>
            {trainingCheckpointOptions.length === 0 ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
                暂无兼容的已训练模型
              </p>
            ) : null}
            {trainingCheckpointOptions.length === 0 ? (
              <p style={{ margin: '4px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.5 }}>
                {getNoCompatibleModelAssetsHint(modelCompatibilityContext)}
              </p>
            ) : null}
            {modelAssetCompatHint ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.5 }}>
                {modelAssetCompatHint}
              </p>
            ) : null}
          </div>
        ) : null}
        {dualArmTrainedMode ? (
          <div>
            <label style={workspaceModalFieldLabel}>模型资产</label>
            <select
              style={workspaceModalSelectStyle}
              value={selectedModelAssetId}
              onChange={(e) => setSelectedModelAssetId(e.target.value)}
              disabled={Boolean(pinnedModelAssetId && !prefillAssetError)}
            >
              <option value="">请选择已训练模型</option>
              {trainingCheckpointOptions.map((item) => (
                <option
                  key={item.modelAssetId}
                  value={item.modelAssetId}
                  disabled={!item.ready}
                >
                  {item.ready ? item.label : `${item.label}（checkpoint 不可用）`}
                </option>
              ))}
            </select>
            {trainingCheckpointOptions.length === 0 ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
                {getNoCompatibleModelAssetsHint(modelCompatibilityContext)}
                当前任务没有可用模型资产，请先完成对应任务的训练，或检查模型文件是否存在。
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      {showUnifiedEvalParams ? renderUnifiedEvaluationParams() : null}
      {renderEvaluationMetricSelection()}
    </>
  );

  return (
    <WorkspaceCenteredModal
      open={open}
      title="新建评测任务"
      titleId="create-eval-task-title"
      width={760}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <SecondaryButton onClick={onClose}>取消</SecondaryButton>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {evaluationTopType === 'dataset' ? (
              <PrimaryButton
                disabled={!datasetSubmitReady}
                onClick={() => void onStart(buildPayload('评测中'))}
              >
                启动评测
              </PrimaryButton>
            ) : cableThreadingMode ? (
              <PrimaryButton
                disabled={modelSubmitDisabled}
                onClick={() => handleStartClick('评测中')}
              >
                启动评测
              </PrimaryButton>
            ) : dualArmMode ? (
              <PrimaryButton
                disabled={modelSubmitDisabled}
                onClick={() => handleStartClick('评测中')}
              >
                启动评测
              </PrimaryButton>
            ) : (
              <>
                <SecondaryButton onClick={() => onSave(buildPayload('待评测'))}>保存任务</SecondaryButton>
                <PrimaryButton
                  disabled={modelSubmitDisabled}
                  onClick={() => handleStartClick('评测中')}
                >
                  启动评测
                </PrimaryButton>
              </>
            )}
          </div>
        </div>
      }
    >
      {renderTopTypeSelector()}
      {evaluationTopType === 'model' ? renderModelTaskNameField() : null}
      {evaluationTopType === 'dataset' ? renderDatasetForm() : renderModelForm()}
    </WorkspaceCenteredModal>
  );
}
