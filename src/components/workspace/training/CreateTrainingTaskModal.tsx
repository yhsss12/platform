'use client';

import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  WorkspaceModalFieldGrid,
  workspaceFormFieldClassName,
  workspaceModalDebugPanelStyle,
  workspaceModalDebugToggleStyle,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { workspaceModalFieldErrorStyle } from '@/components/workspace/training/TrainingAdvancedSettingsSection';
import { TrainingCheckpointSaveSection } from '@/components/workspace/training/TrainingCheckpointSaveSection';
import { getDatasetManifest } from '@/lib/mock/workspaceMockFlowStore';
import { listWorkspaceDataItemsForUi } from '@/lib/workspace/workspaceDataSources';
import type { CreateTrainingTaskInput, TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import {
  buildDefaultTrainingTaskName,
  findTrainingDatasetOption,
  formatTrainingDatasetLabel,
  listMergedTrainingDatasetOptions,
  randomTrainingSeed,
  resolveTrainingSeedInput,
  sanitizeSeedDraftInput,
} from '@/lib/mock/workspaceTrainingMock';
import { resolveTrainingDatasetCardInfo } from '@/lib/workspace/trainingDisplay';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import { listModelAssets, modelAssetsToCheckpointOptions } from '@/lib/api/modelAssetsClient';
import { listAvailableModelTypes } from '@/lib/api/modelTypesClient';
import type { Dataset } from '@/types/benchmark';
import type { ModelAsset } from '@/types/benchmark';
import type { ModelTypeDefinition } from '@/types/modelType';
import {
  DUAL_ARM_TRAINING_BACKEND_PENDING_HINT,
  ISAAC_BLOCK_STACKING_TRAINING_PENDING_HINT,
} from '@/lib/workspace/datasetTrainingAccess';
import {
  isDualArmTrainingBackendPending,
  isDualArmTrainingDatasetOption,
  isIsaacTrainingBackendPending,
  isIsaacTrainingDatasetOption,
} from '@/lib/workspace/resolveTrainingDatasetManifest';
import { getTrainingCapabilities, listTrainingNodes, type TrainingCapabilities } from '@/lib/api/trainingClient';
import {
  apiNodeToDeviceOption,
  DEFAULT_TRAINING_DEVICE,
  formatTrainingNodeStatusLabel,
  TRAINING_DEVICE_FALLBACK_OPTIONS,
  trainingDeviceSubmitParams,
  type TrainingDeviceOption,
  type TrainingDeviceValue,
} from '@/lib/workspace/trainingDevice';
import {
  buildTrainingPretrainedPayload,
  effectivePretrainedTrainingBackend,
  formatInitWeightOptionLines,
  modelAssetHorizonWarning,
  modelAssetMatchesTrainingContext,
  PRETRAINED_STRUCTURE_MISMATCH_HINT,
  resolvePrimaryDatasetSignature,
} from '@/lib/workspace/trainingPretrained';
import {
  modelAssetDpInitCompatible,
  resolveDpInitTargetFromDatasetManifest,
} from '@/lib/workspace/dpInitWeightCompat';
import { InitWeightsSelect } from '@/components/workspace/training/InitWeightsSelect';
import {
  baseAlgorithmLabel,
  downstreamModelTypeForModelType,
  structureConfigSummary,
  trainingBackendForModelType,
} from '@/lib/workspace/modelTypeDisplay';
import { DatasetPickerDialog } from '@/components/workspace/training/DatasetPickerDialog';
import { TrainingNodeSelect } from '@/components/workspace/training/TrainingNodeSelect';
import {
  formatSelectedTrainingDatasetsTriggerLabel,
  type TrainingDatasetPickerMeta,
} from '@/lib/workspace/trainingDatasetPicker';
import { modelTypeSelectOptionLabel } from '@/lib/workspace/modelTypeTrainingCapability';
import {
  filterTrainingDatasetOptionsForModelType,
  isPi0ModelType,
  PI0_NO_DATASET_HINT,
} from '@/lib/workspace/pi0TrainingDatasetFilter';

export type { CreateTrainingTaskInput };

const DEFAULT_PRETRAINED_MODEL_ASSET_ID = '';

const warningHintStyle: CSSProperties = {
  margin: '0 0 12px',
  fontSize: 13,
  color: '#b45309',
  lineHeight: 1.55,
};

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

function DatasetSummaryRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="ws-dataset-summary-card-row">
      <span className="ws-dataset-summary-card-label">{label}</span>
      <span className="ws-dataset-summary-card-value">{value}</span>
    </div>
  );
}

function parsePositiveIntegerInput(raw: string, allowEmpty = false): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return allowEmpty ? null : 1;
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return parsed >= 1 ? parsed : null;
}

function applyModelTypeDefaults(defn: ModelTypeDefinition) {
  const defaults = defn.trainingDefaults ?? {};
  return {
    epochs: Number(defaults.default_epochs ?? 5),
    batchSize: Number(defaults.default_batch_size ?? 16),
    learningRate: Number(defaults.default_learning_rate ?? 0.0001),
    seed:
      defaults.default_seed_strategy === 'fixed'
        ? 1
        : randomTrainingSeed(),
  };
}

function asArray<T>(value: T[] | null | undefined): T[] {
  return Array.isArray(value) ? value : [];
}

export function CreateTrainingTaskModal({
  open,
  onClose,
  onStart,
  initialDataset,
  submitting = false,
}: {
  open: boolean;
  onClose: () => void;
  onStart: (input: CreateTrainingTaskInput) => void | Promise<void>;
  initialDataset?: string;
  submitting?: boolean;
}) {
  const dataCenterItems = useMemo(() => listWorkspaceDataItemsForUi(), [open]);
  const [apiDatasets, setApiDatasets] = useState<Dataset[]>([]);
  const datasetOptions = useMemo(
    () => listMergedTrainingDatasetOptions(dataCenterItems, apiDatasets),
    [dataCenterItems, apiDatasets, open]
  );

  const datasetMetaById = useMemo(() => {
    const map: Record<string, TrainingDatasetPickerMeta> = {};
    for (const item of apiDatasets) {
      map[item.id] = {
        actionSchema: item.actionSchema,
        observationSchema: item.observationSchema,
        createdAt: item.createdAt,
        robotType: item.robotType,
        status: item.status != null ? String(item.status) : null,
      };
    }
    return map;
  }, [apiDatasets]);

  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [datasetPickerDialogOpen, setDatasetPickerDialogOpen] = useState(false);
  const [taskNameDraft, setTaskNameDraft] = useState('');
  const [modelTypeId, setModelTypeId] = useState('');
  const [availableModelTypes, setAvailableModelTypes] = useState<ModelTypeDefinition[]>([]);
  const [modelTypesLoading, setModelTypesLoading] = useState(false);
  const [modelTypesError, setModelTypesError] = useState<string | null>(null);
  const [trainingDevice, setTrainingDevice] = useState<TrainingDeviceValue>(DEFAULT_TRAINING_DEVICE);
  const [trainingNodeOptions, setTrainingNodeOptions] = useState<TrainingDeviceOption[]>(
    TRAINING_DEVICE_FALLBACK_OPTIONS
  );
  const [trainingNodesLoading, setTrainingNodesLoading] = useState(false);
  const [trainingNodesError, setTrainingNodesError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [seedDraft, setSeedDraft] = useState(() => String(randomTrainingSeed()));
  const [saveFinal, setSaveFinal] = useState(true);
  const [saveBest, setSaveBest] = useState(false);
  const [intervalEnabled, setIntervalEnabled] = useState(false);
  const [checkpointIntervalEpochs, setCheckpointIntervalEpochs] = useState(5);
  const [debugInfoExpanded, setDebugInfoExpanded] = useState(false);
  const [pretrainedModelAssetId, setPretrainedModelAssetId] = useState(DEFAULT_PRETRAINED_MODEL_ASSET_ID);
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);
  const [modelAssets, setModelAssets] = useState<ModelAsset[]>([]);
  const [modelAssetsLoading, setModelAssetsLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    void getTrainingCapabilities()
      .then(setCapabilities)
      .catch(() => setCapabilities(null));
    setTrainingNodesLoading(true);
    setTrainingNodesError(null);
    void listTrainingNodes(true)
      .then((response) => {
        const options = (response.nodes ?? []).map(apiNodeToDeviceOption);
        setTrainingNodeOptions(options.length > 0 ? options : TRAINING_DEVICE_FALLBACK_OPTIONS);
        const preferred =
          options.find((item) => item.nodeId === DEFAULT_TRAINING_DEVICE && item.selectable !== false) ??
          options.find((item) => item.selectable !== false) ??
          options[0];
        if (preferred) {
          setTrainingDevice(preferred.value);
        }
      })
      .catch((err) => {
        setTrainingNodeOptions(TRAINING_DEVICE_FALLBACK_OPTIONS);
        const fallbackNode =
          TRAINING_DEVICE_FALLBACK_OPTIONS.find((item) => item.selectable !== false) ??
          TRAINING_DEVICE_FALLBACK_OPTIONS[0];
        if (fallbackNode) {
          setTrainingDevice(fallbackNode.value);
        }
        setTrainingNodesError(err instanceof Error ? err.message : '加载训练节点失败');
      })
      .finally(() => setTrainingNodesLoading(false));
    void listWorkspaceDatasets()
      .then((response) => setApiDatasets(asArray(response.datasets)))
      .catch(() => setApiDatasets([]));
    setModelAssetsLoading(true);
    void listModelAssets()
      .then((response) => setModelAssets(asArray(response.modelAssets)))
      .catch(() => setModelAssets([]))
      .finally(() => setModelAssetsLoading(false));
    setModelTypesLoading(true);
    setModelTypesError(null);
    void listAvailableModelTypes()
      .then((items) => {
        const safeItems = asArray(items);
        setAvailableModelTypes(safeItems);
        const trainable = safeItems.filter((item) => item.trainingReady);
        if (trainable.length > 0) {
          setModelTypeId((prev) =>
            prev && trainable.some((item) => item.modelTypeId === prev) ? prev : trainable[0].modelTypeId
          );
        } else {
          setModelTypeId('');
        }
      })
      .catch((err) => {
        setAvailableModelTypes([]);
        setModelTypeId('');
        setModelTypesError(err instanceof Error ? err.message : '加载模型类型失败');
      })
      .finally(() => setModelTypesLoading(false));
  }, [open]);

  const safeApiDatasets = asArray(apiDatasets);
  const safeAvailableModelTypes = asArray(availableModelTypes);
  const safeModelAssets = asArray(modelAssets);
  const safeTrainingNodeOptions = asArray(trainingNodeOptions);

  const selectedDatasetOptions = useMemo(
    () =>
      selectedDatasets
        .map((id) => findTrainingDatasetOption(id, dataCenterItems, safeApiDatasets))
        .filter((item): item is NonNullable<typeof item> => Boolean(item)),
    [selectedDatasets, dataCenterItems, safeApiDatasets]
  );

  const selectedDatasetOption = selectedDatasetOptions[0];

  const selectedModelType = useMemo(
    () => safeAvailableModelTypes.find((item) => item.modelTypeId === modelTypeId) ?? null,
    [safeAvailableModelTypes, modelTypeId]
  );

  const filteredDatasetOptions = useMemo(
    () => filterTrainingDatasetOptionsForModelType(datasetOptions, selectedModelType, dataCenterItems, safeApiDatasets),
    [datasetOptions, selectedModelType, dataCenterItems, safeApiDatasets]
  );

  const modelTypeTrainingBackend = selectedModelType
    ? trainingBackendForModelType(selectedModelType)
    : 'robomimic_bc';

  const dualArmDatasetSelected = Boolean(
    selectedDatasetOption && isDualArmTrainingDatasetOption(selectedDatasetOption)
  );

  const isaacDatasetSelected = Boolean(
    selectedDatasetOption && isIsaacTrainingDatasetOption(selectedDatasetOption)
  );

  const selectedTrainingBackend =
    isaacDatasetSelected && modelTypeTrainingBackend === 'robomimic_bc'
      ? 'isaac_robomimic_bc'
      : modelTypeTrainingBackend;

  const datasetCardInfo = useMemo(
    () =>
      selectedDatasetOption
        ? resolveTrainingDatasetCardInfo(
            selectedDatasetOption,
            dataCenterItems,
            selectedDatasetOptions
          )
        : null,
    [selectedDatasetOption, selectedDatasetOptions, dataCenterItems]
  );

  const trainingBackendPending = useMemo(() => {
    if (!selectedDatasetOption || !capabilities || !selectedModelType) return false;
    if (selectedTrainingBackend === 'isaac_robomimic_bc') {
      return isIsaacTrainingBackendPending(selectedDatasetOption, capabilities);
    }
    if (selectedTrainingBackend === 'torch_bc') {
      return isDualArmTrainingBackendPending(selectedDatasetOption, capabilities);
    }
    return false;
  }, [selectedDatasetOption, capabilities, selectedModelType, selectedTrainingBackend]);

  const defaultTaskName = useMemo(() => {
    if (!selectedDatasetOption || !selectedModelType) return '';
    return buildDefaultTrainingTaskName({
      datasetName: selectedDatasetOption.datasetName,
      modelLabel: selectedModelType.name,
    });
  }, [selectedDatasetOption, selectedModelType]);

  const datasetSignature = useMemo(
    () => resolvePrimaryDatasetSignature(selectedDatasetOption),
    [selectedDatasetOption]
  );

  const selectedDatasetManifest = useMemo(() => {
    const datasetId = selectedDatasets[0];
    if (!datasetId) return null;
    const manifest = resolveDatasetManifestForTraining(datasetId);
    const apiDataset = safeApiDatasets.find((item) => item.id === datasetId);
    const merged: Record<string, unknown> = {};
    if (manifest && typeof manifest === 'object') {
      Object.assign(merged, manifest as unknown as Record<string, unknown>);
    }
    if (apiDataset?.actionSchema) merged.actionSchema = apiDataset.actionSchema;
    if (apiDataset?.observationSchema) merged.observationSchema = apiDataset.observationSchema;
    return Object.keys(merged).length > 0 ? merged : null;
  }, [selectedDatasets, safeApiDatasets]);

  const effectivePretrainedBackend = useMemo(
    () =>
      effectivePretrainedTrainingBackend(selectedTrainingBackend as never, {
        isDualArm: dualArmDatasetSelected,
        isIsaac: isaacDatasetSelected,
        capabilities,
      }),
    [selectedTrainingBackend, dualArmDatasetSelected, isaacDatasetSelected, capabilities]
  );

  const dpInitTarget = useMemo(() => {
    if (effectivePretrainedBackend !== 'diffusion_policy') return null;
    return resolveDpInitTargetFromDatasetManifest(selectedDatasetManifest, datasetSignature);
  }, [effectivePretrainedBackend, selectedDatasetManifest, datasetSignature]);

  const pretrainedCheckpointOptions = useMemo(() => {
    const compatibleAssets = safeModelAssets.filter((asset) => {
      if (
        !modelAssetMatchesTrainingContext(asset, {
          isDualArm: dualArmDatasetSelected,
          isIsaac: isaacDatasetSelected,
          trainingBackend: effectivePretrainedBackend,
          datasetSignature,
        })
      ) {
        return false;
      }
      if (effectivePretrainedBackend === 'diffusion_policy' && dpInitTarget) {
        return modelAssetDpInitCompatible(asset, dpInitTarget).ok;
      }
      return true;
    });
    return modelAssetsToCheckpointOptions(compatibleAssets).map((item) => {
      const asset = safeModelAssets.find((row) => row.id === item.modelAssetId);
      const lines = asset ? formatInitWeightOptionLines(asset) : null;
      return {
        ...item,
        label: lines ? `${lines.titleLine} · ${lines.subtitleLine}` : item.label,
        titleLine: lines?.titleLine ?? item.label,
        subtitleLine: lines?.subtitleLine,
        title: lines?.title,
      };
    });
  }, [
    safeModelAssets,
    dualArmDatasetSelected,
    isaacDatasetSelected,
    effectivePretrainedBackend,
    datasetSignature,
    dpInitTarget,
  ]);

  const initWeightSelectOptions = useMemo(
    () =>
      pretrainedCheckpointOptions.map((item) => ({
        value: item.modelAssetId,
        titleLine: item.titleLine || item.label,
        subtitleLine: item.subtitleLine,
        title: item.title,
        disabled: !item.ready,
      })),
    [pretrainedCheckpointOptions]
  );

  const selectedPretrainedAsset = useMemo(
    () => safeModelAssets.find((item) => item.id === pretrainedModelAssetId) ?? null,
    [safeModelAssets, pretrainedModelAssetId]
  );

  const pretrainedHorizonWarning = useMemo(() => {
    if (!selectedPretrainedAsset || !datasetSignature) return null;
    return modelAssetHorizonWarning(selectedPretrainedAsset, datasetSignature);
  }, [selectedPretrainedAsset, datasetSignature]);

  const selectedPretrainedOption = useMemo(
    () => pretrainedCheckpointOptions.find((item) => item.modelAssetId === pretrainedModelAssetId),
    [pretrainedCheckpointOptions, pretrainedModelAssetId]
  );

  const handleDatasetPickerConfirm = (datasets: TrainingDatasetOption[]) => {
    setSelectedDatasets(datasets.map((item) => item.id));
    setDatasetPickerDialogOpen(false);
  };

  const datasetTriggerLabel =
    selectedDatasets.length > 0
      ? formatSelectedTrainingDatasetsTriggerLabel(selectedDatasets, datasetOptions)
      : '请选择数据集';

  const pretrainedUnavailable = Boolean(
    pretrainedModelAssetId && (!selectedPretrainedOption || !selectedPretrainedOption.ready)
  );

  useEffect(() => {
    if (!open) return;
    const preferred =
      initialDataset && datasetOptions.some((d) => d.id === initialDataset)
        ? initialDataset
        : datasetOptions[0]?.id ?? '';
    const option = findTrainingDatasetOption(preferred, dataCenterItems, safeApiDatasets);
    const isIsaac = option ? isIsaacTrainingDatasetOption(option) : false;

    setSelectedDatasets(preferred ? [preferred] : []);
    setDatasetPickerDialogOpen(false);
    setTaskNameDraft('');
    setTrainingDevice(DEFAULT_TRAINING_DEVICE);
    setEpochs(isIsaac ? 2 : 5);
    setBatchSize(16);
    setLearningRate(0.0001);
    setSeedDraft(String(randomTrainingSeed()));
    setDebugInfoExpanded(false);
    setPretrainedModelAssetId(DEFAULT_PRETRAINED_MODEL_ASSET_ID);
    setSaveFinal(true);
    setSaveBest(false);
    setIntervalEnabled(false);
    setCheckpointIntervalEpochs(5);
  }, [open, initialDataset, datasetOptions, dataCenterItems, safeApiDatasets]);

  useEffect(() => {
    if (!open || !selectedModelType) return;
    const defaults = applyModelTypeDefaults(selectedModelType);
    if (isPi0ModelType(selectedModelType)) {
      setEpochs(1);
      setBatchSize(2);
    } else {
      setEpochs(defaults.epochs);
      setBatchSize(defaults.batchSize);
    }
    setLearningRate(defaults.learningRate);
    setSeedDraft(String(defaults.seed));
  }, [open, selectedModelType?.modelTypeId]);

  useEffect(() => {
    if (!open || !isPi0ModelType(selectedModelType)) return;
    const allowed = new Set(filteredDatasetOptions.map((item) => item.id));
    setSelectedDatasets((prev) => {
      const next = prev.filter((id) => allowed.has(id));
      if (next.length > 0) return next;
      return filteredDatasetOptions[0]?.id ? [filteredDatasetOptions[0].id] : [];
    });
  }, [open, selectedModelType?.modelTypeId, filteredDatasetOptions]);

  useEffect(() => {
    if (!open) return;
    if (
      pretrainedModelAssetId &&
      !pretrainedCheckpointOptions.some((item) => item.modelAssetId === pretrainedModelAssetId)
    ) {
      setPretrainedModelAssetId(DEFAULT_PRETRAINED_MODEL_ASSET_ID);
    }
  }, [open, pretrainedCheckpointOptions, pretrainedModelAssetId]);

  const payload = (): CreateTrainingTaskInput => {
    const deviceParams = trainingDeviceSubmitParams(trainingDevice, safeTrainingNodeOptions);
    const resolvedSeed = resolveTrainingSeedInput(seedDraft, Number(seedDraft) || randomTrainingSeed());
    const resolvedTaskName = taskNameDraft.trim() || defaultTaskName;
    const primaryDataset = selectedDatasets[0] ?? '';
    const result: CreateTrainingTaskInput = {
      dataset: primaryDataset,
      datasets: selectedDatasets,
      modelTypeId: selectedModelType?.modelTypeId ?? '',
      downstreamModelType: selectedModelType
        ? downstreamModelTypeForModelType(selectedModelType)
        : 'Robomimic',
      trainingBackend: selectedTrainingBackend as CreateTrainingTaskInput['trainingBackend'],
      trainingDevice,
      epochs,
      batchSize,
      learningRate,
      device: deviceParams.device,
      seed: resolvedSeed,
      taskName: resolvedTaskName,
      saveFinal,
      saveBest,
      checkpointIntervalEpochs: intervalEnabled ? checkpointIntervalEpochs : null,
    };

    if (selectedPretrainedOption?.ready) {
      result.pretrained = buildTrainingPretrainedPayload(selectedPretrainedOption, safeModelAssets);
    }

    if (isPi0ModelType(selectedModelType)) {
      result.datasetFormat = 'lerobot';
      result.maxSteps = 10;
      const apiDataset = safeApiDatasets.find((item) => item.id === primaryDataset);
      result.taskInstruction =
        apiDataset?.lerobotTaskInstruction?.trim() ||
        'thread the cable through the pole';
    }

    return result;
  };

  const epochsValid = Number.isInteger(epochs) && epochs >= 1;
  const checkpointIntervalValid = useMemo(() => {
    if (!intervalEnabled) return true;
    return (
      Number.isInteger(checkpointIntervalEpochs) &&
      checkpointIntervalEpochs >= 1 &&
      checkpointIntervalEpochs <= epochs
    );
  }, [intervalEnabled, checkpointIntervalEpochs, epochs]);

  const multiDatasetDpOnlyWarning =
    selectedDatasets.length > 1 && selectedTrainingBackend !== 'diffusion_policy'
      ? '多数据集合并训练当前仅支持 Diffusion Policy 模型类型'
      : null;

  const pi0DatasetEmpty = isPi0ModelType(selectedModelType) && filteredDatasetOptions.length === 0;

  const canStart = Boolean(
    selectedDatasets.length > 0 &&
      selectedDatasetOption &&
      selectedModelType &&
      selectedModelType.trainingReady &&
      !trainingBackendPending &&
      !multiDatasetDpOnlyWarning &&
      !pi0DatasetEmpty &&
      safeAvailableModelTypes.some((item) => item.trainingReady) &&
      !pretrainedUnavailable &&
      epochsValid &&
      checkpointIntervalValid
  );

  return (
    <WorkspaceCenteredModal
      open={open}
      title="新建训练任务"
      titleId="create-training-task-title"
      width={720}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={submitting ? undefined : onClose}>取消</SecondaryButton>
          <PrimaryButton
            onClick={() => void onStart(payload())}
            disabled={!canStart || submitting}
          >
            {submitting ? '提交中…' : '创建训练任务'}
          </PrimaryButton>
        </div>
      }
    >
      <div style={workspaceModalSectionLabel}>数据集</div>
      <div style={{ marginBottom: 10 }}>
        <div className="ws-dataset-selector">
          <button
            type="button"
            className={`ws-dataset-selector-value${selectedDatasets.length === 0 ? ' is-placeholder' : ''}`}
            title={datasetTriggerLabel}
            onClick={() => setDatasetPickerDialogOpen(true)}
          >
            {datasetTriggerLabel}
          </button>
          <button
            type="button"
            className="ws-dataset-selector-action"
            onClick={() => setDatasetPickerDialogOpen(true)}
          >
            选择
          </button>
        </div>
        {multiDatasetDpOnlyWarning ? (
          <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 8 }}>{multiDatasetDpOnlyWarning}</p>
        ) : null}
        {pi0DatasetEmpty ? (
          <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 8 }}>{PI0_NO_DATASET_HINT}</p>
        ) : null}
      </div>

      <DatasetPickerDialog
        open={datasetPickerDialogOpen}
        options={filteredDatasetOptions}
        selectedIds={selectedDatasets}
        multiple
        datasetMetaById={datasetMetaById}
        onConfirm={handleDatasetPickerConfirm}
        onCancel={() => setDatasetPickerDialogOpen(false)}
      />

      <div style={{ marginBottom: 16 }}>
        <label style={workspaceModalFieldLabel}>训练任务名称</label>
        <input
          type="text"
          value={taskNameDraft}
          placeholder={defaultTaskName || '输入训练任务名称'}
          onChange={(e) => setTaskNameDraft(e.target.value)}
          className={workspaceFormFieldClassName}
          style={workspaceModalSelectStyle}
        />
      </div>

      {isaacDatasetSelected ? (
        <div className="ws-dataset-summary-card">
          <DatasetSummaryRow label="数据集" value={selectedDatasetOption?.datasetName ?? '—'} />
          <DatasetSummaryRow label="来源任务" value="物块堆叠" />
          <DatasetSummaryRow label="数据格式" value="HDF5" />
          <DatasetSummaryRow label="Episode 数" value={selectedDatasetOption?.sampleCount ?? '—'} />
        </div>
      ) : datasetCardInfo ? (
        <div className="ws-dataset-summary-card">
          {datasetCardInfo.datasetCount && datasetCardInfo.datasetCount > 1 ? (
            <DatasetSummaryRow label="数据集数量" value={`${datasetCardInfo.datasetCount} 个`} />
          ) : null}
          <DatasetSummaryRow label="来源任务" value={datasetCardInfo.sourceTask} />
          <DatasetSummaryRow
            label={datasetCardInfo.datasetCount && datasetCardInfo.datasetCount > 1 ? '总轨迹数' : '数据规模'}
            value={
              datasetCardInfo.datasetCount && datasetCardInfo.datasetCount > 1
                ? String(datasetCardInfo.totalTrajectories ?? datasetCardInfo.dataScale)
                : datasetCardInfo.dataScale
            }
          />
          <DatasetSummaryRow label="仿真环境" value={datasetCardInfo.simEnvironment} />
          <DatasetSummaryRow label="机器人类型" value={datasetCardInfo.robotType} />
        </div>
      ) : null}

      {trainingBackendPending ? (
        <p style={warningHintStyle}>
          {isaacDatasetSelected
            ? ISAAC_BLOCK_STACKING_TRAINING_PENDING_HINT
            : DUAL_ARM_TRAINING_BACKEND_PENDING_HINT}
        </p>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 4 }}>训练配置</div>
      <WorkspaceModalFieldGrid>
        <div>
          <label style={workspaceModalFieldLabel}>模型类型</label>
          <select
            className={workspaceFormFieldClassName}
            style={workspaceModalSelectStyle}
            value={modelTypeId}
            onChange={(e) => setModelTypeId(e.target.value)}
            disabled={modelTypesLoading || safeAvailableModelTypes.length === 0}
          >
            {modelTypesLoading ? (
              <option value="">加载模型类型…</option>
            ) : safeAvailableModelTypes.length === 0 ? (
              <option value="">暂无可用模型类型</option>
            ) : (
              safeAvailableModelTypes.map((item) => (
                <option
                  key={item.modelTypeId}
                  value={item.modelTypeId}
                  disabled={!item.trainingReady}
                >
                  {modelTypeSelectOptionLabel(item)}
                </option>
              ))
            )}
          </select>
          {selectedModelType && !selectedModelType.trainingReady && selectedModelType.disabledReason ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>
              {selectedModelType.disabledReason}
            </p>
          ) : null}
          {!modelTypesLoading && safeAvailableModelTypes.length === 0 ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>
              请先到「资源 - 模型类型」中新建或启用模型类型。
            </p>
          ) : null}
          {modelTypesError ? (
            <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 8 }}>{modelTypesError}</p>
          ) : null}
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>训练节点</label>
          <TrainingNodeSelect
            options={safeTrainingNodeOptions}
            value={trainingDevice}
            onChange={setTrainingDevice}
            disabled={trainingNodesLoading}
          />
          {trainingNodesLoading ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>正在探测训练节点状态…</p>
          ) : null}
          {trainingNodesError ? (
            <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 8 }}>{trainingNodesError}</p>
          ) : null}
          {!trainingNodesLoading && !trainingNodesError ? (
            (() => {
              const selected = safeTrainingNodeOptions.find(
                (item) => item.value === trainingDevice || item.nodeId === trainingDevice
              );
              if (!selected) return null;
              if (selected.status === 'busy' && selected.message) {
                return (
                  <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>
                    {selected.message}
                  </p>
                );
              }
              if (selected.status === 'unreachable' || selected.status === 'misconfigured') {
                return (
                  <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 8, marginBottom: 0 }}>
                    {selected.message || formatTrainingNodeStatusLabel(selected.status)}
                  </p>
                );
              }
              return null;
            })()
          ) : null}
        </div>
        <div style={{ gridColumn: '1 / -1', minWidth: 0 }}>
          <InitWeightsSelect
            value={pretrainedModelAssetId}
            options={initWeightSelectOptions}
            onChange={setPretrainedModelAssetId}
          />
          {!modelAssetsLoading && initWeightSelectOptions.length === 0 ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>无可用预训练模型</p>
          ) : null}
          {pretrainedModelAssetId && !pretrainedCheckpointOptions.some((item) => item.modelAssetId === pretrainedModelAssetId) ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>
              {PRETRAINED_STRUCTURE_MISMATCH_HINT}
            </p>
          ) : null}
          {pretrainedHorizonWarning ? (
            <p style={{ ...warningHintStyle, marginTop: 8, marginBottom: 0 }}>{pretrainedHorizonWarning}</p>
          ) : null}
          {pretrainedUnavailable ? (
            <p style={{ ...warningHintStyle, marginTop: 8 }}>
              所选 checkpoint 暂不可用，请重新选择或完成训练。
            </p>
          ) : null}
        </div>
      </WorkspaceModalFieldGrid>

      {selectedModelType ? (
        <div style={summaryBoxStyle}>
          <div>基础算法：{baseAlgorithmLabel(selectedModelType.baseAlgorithm)}</div>
          <div>适配器：{selectedModelType.adapterKey}</div>
          <div>结构参数：{structureConfigSummary(selectedModelType)}</div>
          <div style={{ marginTop: 6, fontSize: 12, color: '#64748b' }}>
            模型结构在「资源 - 模型类型」中定义，训练任务中不可修改。
          </div>
        </div>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>训练参数</div>
      <WorkspaceModalFieldGrid>
        <div>
          <label style={workspaceModalFieldLabel}>Epochs</label>
          <input
            type="number"
            min={1}
            step={1}
            value={epochs >= 1 ? epochs : ''}
            onChange={(e) => {
              const next = parsePositiveIntegerInput(e.target.value, true);
              if (next !== null) {
                setEpochs(next);
                return;
              }
              if (e.target.value === '') {
                setEpochs(0);
              }
            }}
            className={workspaceFormFieldClassName}
            style={workspaceModalSelectStyle}
          />
          {!epochsValid ? (
            <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 6 }}>请输入大于等于 1 的正整数</p>
          ) : null}
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Batch Size</label>
          <input
            type="number"
            min={1}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value) || 1)}
            className={workspaceFormFieldClassName}
            style={workspaceModalSelectStyle}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Learning Rate</label>
          <input
            type="number"
            step={0.00001}
            value={learningRate}
            onChange={(e) => setLearningRate(Number(e.target.value))}
            className={workspaceFormFieldClassName}
            style={workspaceModalSelectStyle}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Seed</label>
          <input
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={seedDraft}
            onChange={(e) => setSeedDraft(sanitizeSeedDraftInput(e.target.value))}
            className={workspaceFormFieldClassName}
            style={workspaceModalSelectStyle}
          />
        </div>
      </WorkspaceModalFieldGrid>

      <TrainingCheckpointSaveSection
        backend={selectedTrainingBackend}
        totalEpochs={epochs}
        saveFinal={saveFinal}
        saveBest={saveBest}
        intervalEnabled={intervalEnabled}
        checkpointIntervalEpochs={checkpointIntervalEpochs}
        onSaveFinalChange={setSaveFinal}
        onSaveBestChange={setSaveBest}
        onIntervalEnabledChange={setIntervalEnabled}
        onCheckpointIntervalChange={setCheckpointIntervalEpochs}
      />

      {dualArmDatasetSelected ? (
        <>
          <button
            type="button"
            style={{ ...workspaceModalDebugToggleStyle, marginTop: 16 }}
            onClick={() => setDebugInfoExpanded((prev) => !prev)}
            aria-expanded={debugInfoExpanded}
          >
            <span>内部调试信息</span>
            <span style={{ fontSize: 11, fontWeight: 400 }}>{debugInfoExpanded ? '收起' : '展开'}</span>
          </button>
          {debugInfoExpanded ? (
            <div style={workspaceModalDebugPanelStyle}>
              <div>observationSchema：dual_arm_cable_il_v1</div>
              <div>actionDim：14</div>
              <div>modelTypeId：{selectedModelType?.modelTypeId ?? '—'}</div>
              <div>trainingBackend：{selectedTrainingBackend}</div>
            </div>
          ) : null}
        </>
      ) : null}
    </WorkspaceCenteredModal>
  );
}

export function resolveDatasetManifestForTraining(datasetId: string) {
  return getDatasetManifest(datasetId);
}
