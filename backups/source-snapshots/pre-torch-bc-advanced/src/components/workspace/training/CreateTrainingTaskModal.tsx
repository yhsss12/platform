// Historical snapshot retained from before the advanced torch-BC refactor.
'use client';

import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { getDatasetManifest } from '@/lib/mock/workspaceMockFlowStore';
import { listWorkspaceDataItemsForUi } from '@/lib/workspace/workspaceDataSources';
import type {
  ActTrainingAdvancedParams,
  CreateTrainingTaskInput,
  DiffusionPolicyAdvancedParams,
  RobomimicAdvancedParams,
  TrainingSeedMode,
} from '@/lib/mock/workspaceTrainingMock';
import {
  findTrainingDatasetOption,
  formatTrainingDatasetLabel,
  listMergedTrainingDatasetOptions,
  resolveDatasetDisplayFormat,
} from '@/lib/mock/workspaceTrainingMock';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import type { Dataset } from '@/types/benchmark';
import { DUAL_ARM_TRAINING_BACKEND_PENDING_HINT } from '@/lib/workspace/datasetTrainingAccess';
import {
  isDualArmTrainingBackendPending,
  isDualArmTrainingDatasetOption,
} from '@/lib/workspace/resolveTrainingDatasetManifest';
import {
  getTrainingCapabilities,
  type TrainingBackendRequest,
  type TrainingCapabilities,
} from '@/lib/api/trainingClient';
import {
  recommendDataFormat,
} from '@/lib/workspace/trainingCapabilityUi';
import {
  defaultTrainingRecipeForContext,
  formatTrainingRecipeLabel,
  getTrainingRecipe,
  listAvailableTrainingRecipes,
  recipeToSubmitFields,
  type TrainingRecipeAdvancedFamily,
} from '@/lib/workspace/trainingRecipe';
import {
  DEFAULT_TRAINING_DEVICE,
  TRAINING_DEVICE_OPTIONS,
  trainingDeviceSubmitParams,
  type TrainingDeviceValue,
} from '@/lib/workspace/trainingDevice';

export type { CreateTrainingTaskInput };

type DraftMap = Record<string, string>;
type WeightDecayOption = '1e-4' | '1e-3';
type AdvancedModelType = 'Robomimic' | 'ACT' | 'Diffusion Policy';

function advancedFamilyForRecipe(
  family: TrainingRecipeAdvancedFamily | null | undefined
): AdvancedModelType | null {
  if (family === 'robomimic') return 'Robomimic';
  if (family === 'act') return 'ACT';
  if (family === 'dp') return 'Diffusion Policy';
  return null;
}

const ROBOMIMIC_DEFAULT_PARAMS: RobomimicAdvancedParams = {
  actor_hidden_dims: '512,512',
  l2_regularization: 0,
};

const ACT_DEFAULT_PARAMS: ActTrainingAdvancedParams = {
  chunk_size: 100,
  n_action_steps: 100,
  kl_weight: 10,
  latent_dim: 32,
  hidden_dim: 512,
};

const DP_DEFAULT_PARAMS: DiffusionPolicyAdvancedParams = {
  n_obs_steps: 2,
  horizon: 16,
  n_action_steps: 8,
  num_inference_steps: 20,
  use_ema: true,
  ema_decay: 0.999,
  weight_decay: 1e-4,
  save_best: true,
};

const DEFAULT_PRETRAINED_MODEL_ASSET_ID = '';

interface RobomimicAdvancedFormState {
  drafts: DraftMap;
}

interface DpBooleanState {
  use_ema: boolean;
  save_best: boolean;
}

interface DpAdvancedFormState {
  drafts: DraftMap;
  booleans: DpBooleanState;
  weightDecay: WeightDecayOption;
}

const EMPTY_DRAFTS: DraftMap = {};

function randomTrainingSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647);
}

function resolveNumberDraft(draft: string | undefined, defaultValue: number): number {
  if (draft === undefined || draft.trim() === '') return defaultValue;
  const parsed = Number(draft);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

function resolveRobomimicParams(state: RobomimicAdvancedFormState): RobomimicAdvancedParams {
  const dimsDraft = state.drafts.actor_hidden_dims?.trim();
  return {
    actor_hidden_dims: dimsDraft || ROBOMIMIC_DEFAULT_PARAMS.actor_hidden_dims,
    l2_regularization: resolveNumberDraft(
      state.drafts.l2_regularization,
      ROBOMIMIC_DEFAULT_PARAMS.l2_regularization
    ),
  };
}

function createDefaultRobomimicState(): RobomimicAdvancedFormState {
  return {
    drafts: { ...EMPTY_DRAFTS },
  };
}

function resolveActParams(drafts: DraftMap): ActTrainingAdvancedParams {
  return {
    chunk_size: resolveNumberDraft(drafts.chunk_size, ACT_DEFAULT_PARAMS.chunk_size),
    n_action_steps: resolveNumberDraft(drafts.n_action_steps, ACT_DEFAULT_PARAMS.n_action_steps),
    kl_weight: resolveNumberDraft(drafts.kl_weight, ACT_DEFAULT_PARAMS.kl_weight),
    latent_dim: resolveNumberDraft(drafts.latent_dim, ACT_DEFAULT_PARAMS.latent_dim),
    hidden_dim: resolveNumberDraft(drafts.hidden_dim, ACT_DEFAULT_PARAMS.hidden_dim),
  };
}

function resolveDpParams(state: DpAdvancedFormState): DiffusionPolicyAdvancedParams {
  return {
    n_obs_steps: resolveNumberDraft(state.drafts.n_obs_steps, DP_DEFAULT_PARAMS.n_obs_steps),
    horizon: resolveNumberDraft(state.drafts.horizon, DP_DEFAULT_PARAMS.horizon),
    n_action_steps: resolveNumberDraft(state.drafts.n_action_steps, DP_DEFAULT_PARAMS.n_action_steps),
    num_inference_steps: resolveNumberDraft(
      state.drafts.num_inference_steps,
      DP_DEFAULT_PARAMS.num_inference_steps
    ),
    use_ema: state.booleans.use_ema,
    ema_decay: resolveNumberDraft(state.drafts.ema_decay, DP_DEFAULT_PARAMS.ema_decay),
    weight_decay: state.weightDecay === '1e-3' ? 1e-3 : 1e-4,
    save_best: state.booleans.save_best,
  };
}

function createDefaultDpState(): DpAdvancedFormState {
  return {
    drafts: { ...EMPTY_DRAFTS },
    booleans: {
      use_ema: DP_DEFAULT_PARAMS.use_ema,
      save_best: DP_DEFAULT_PARAMS.save_best,
    },
    weightDecay: '1e-4',
  };
}

function isAdvancedModelType(modelType: string): modelType is AdvancedModelType {
  return modelType === 'Robomimic' || modelType === 'ACT' || modelType === 'Diffusion Policy';
}

const seedSegmentStyle = (active: boolean): CSSProperties => ({
  flex: 1,
  padding: '6px 10px',
  fontSize: 13,
  border: '1px solid #cbd5e1',
  backgroundColor: active ? '#2563eb' : '#fff',
  color: active ? '#fff' : '#334155',
  cursor: 'pointer',
  fontWeight: active ? 600 : 400,
});

const advancedPanelStyle: CSSProperties = {
  marginTop: 10,
  padding: '12px 14px',
  borderRadius: 8,
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
};

const fieldHintStyle: CSSProperties = {
  margin: '4px 0 0',
  fontSize: 12,
  color: '#94a3b8',
  lineHeight: 1.4,
};

const restoreLinkStyle: CSSProperties = {
  border: 'none',
  background: 'none',
  padding: 0,
  fontSize: 12,
  color: '#2563eb',
  cursor: 'pointer',
  textDecoration: 'underline',
};

function PlaceholderNumberInput({
  label,
  defaultValue,
  draft,
  onDraftChange,
  min,
  step,
}: {
  label: string;
  defaultValue: number;
  draft: string | undefined;
  onDraftChange: (value: string) => void;
  min?: number;
  step?: number;
}) {
  return (
    <div>
      <label style={workspaceModalFieldLabel}>{label}</label>
      <input
        type="number"
        min={min}
        step={step}
        value={draft ?? ''}
        placeholder={String(defaultValue)}
        onChange={(e) => onDraftChange(e.target.value)}
        style={workspaceModalSelectStyle}
      />
    </div>
  );
}

function RobomimicAdvancedFields({
  state,
  onDraftChange,
}: {
  state: RobomimicAdvancedFormState;
  onDraftChange: (key: string, value: string) => void;
}) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
      <div>
        <label style={workspaceModalFieldLabel}>Hidden Dims</label>
        <input
          type="text"
          value={state.drafts.actor_hidden_dims ?? ''}
          placeholder={ROBOMIMIC_DEFAULT_PARAMS.actor_hidden_dims}
          onChange={(e) => onDraftChange('actor_hidden_dims', e.target.value)}
          style={workspaceModalSelectStyle}
        />
        <p style={fieldHintStyle}>两层 MLP 宽度，逗号分隔，例如 512,512</p>
      </div>
      <PlaceholderNumberInput
        label="L2 Regularization"
        defaultValue={ROBOMIMIC_DEFAULT_PARAMS.l2_regularization}
        draft={state.drafts.l2_regularization}
        min={0}
        step={0.0001}
        onDraftChange={(value) => onDraftChange('l2_regularization', value)}
      />
    </div>
  );
}

function AdvancedCheckboxField({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
}) {
  return (
    <label
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 8,
        cursor: 'pointer',
        fontSize: 13,
        color: '#334155',
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ marginTop: 3 }}
      />
      <span>
        <span>{label}</span>
        {hint ? (
          <span style={{ display: 'block', marginTop: 4, fontSize: 12, color: '#94a3b8', lineHeight: 1.45 }}>
            {hint}
          </span>
        ) : null}
      </span>
    </label>
  );
}

function ActAdvancedFields({
  drafts,
  onDraftChange,
}: {
  drafts: DraftMap;
  onDraftChange: (key: string, value: string) => void;
}) {
  const fields: {
    key: keyof ActTrainingAdvancedParams;
    label: string;
    defaultValue: number;
    min?: number;
    step?: number;
  }[] = [
    { key: 'chunk_size', label: 'Chunk Size', defaultValue: ACT_DEFAULT_PARAMS.chunk_size, min: 1 },
    { key: 'n_action_steps', label: 'Action Steps', defaultValue: ACT_DEFAULT_PARAMS.n_action_steps, min: 1 },
    { key: 'kl_weight', label: 'KL Weight', defaultValue: ACT_DEFAULT_PARAMS.kl_weight, min: 0, step: 0.1 },
    { key: 'latent_dim', label: 'Latent Dim', defaultValue: ACT_DEFAULT_PARAMS.latent_dim, min: 1 },
    { key: 'hidden_dim', label: 'Hidden Dim', defaultValue: ACT_DEFAULT_PARAMS.hidden_dim, min: 1 },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
      {fields.map((field) => (
        <PlaceholderNumberInput
          key={field.key}
          label={field.label}
          defaultValue={field.defaultValue}
          draft={drafts[field.key]}
          min={field.min}
          step={field.step}
          onDraftChange={(value) => onDraftChange(field.key, value)}
        />
      ))}
    </div>
  );
}

function DiffusionPolicyAdvancedFields({
  state,
  onDraftChange,
  onBooleanChange,
  onWeightDecayChange,
}: {
  state: DpAdvancedFormState;
  onDraftChange: (key: string, value: string) => void;
  onBooleanChange: (key: keyof DpBooleanState, value: boolean) => void;
  onWeightDecayChange: (value: WeightDecayOption) => void;
}) {
  const numberFields: {
    key: 'n_obs_steps' | 'horizon' | 'n_action_steps' | 'num_inference_steps' | 'ema_decay';
    label: string;
    defaultValue: number;
    min?: number;
    step?: number;
  }[] = [
    { key: 'n_obs_steps', label: 'N Obs Steps', defaultValue: DP_DEFAULT_PARAMS.n_obs_steps, min: 1 },
    { key: 'horizon', label: 'Horizon', defaultValue: DP_DEFAULT_PARAMS.horizon, min: 1 },
    { key: 'n_action_steps', label: 'Action Steps', defaultValue: DP_DEFAULT_PARAMS.n_action_steps, min: 1 },
    {
      key: 'num_inference_steps',
      label: 'Inference Steps',
      defaultValue: DP_DEFAULT_PARAMS.num_inference_steps,
      min: 1,
    },
    { key: 'ema_decay', label: 'EMA Decay', defaultValue: DP_DEFAULT_PARAMS.ema_decay, min: 0, step: 0.001 },
  ];

  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
        {numberFields.map((field) => (
          <PlaceholderNumberInput
            key={field.key}
            label={field.label}
            defaultValue={field.defaultValue}
            draft={state.drafts[field.key]}
            min={field.min}
            step={field.step}
            onDraftChange={(value) => onDraftChange(field.key, value)}
          />
        ))}
        <div>
          <label style={workspaceModalFieldLabel}>Weight Decay</label>
          <select
            style={workspaceModalSelectStyle}
            value={state.weightDecay}
            onChange={(e) => onWeightDecayChange(e.target.value as WeightDecayOption)}
          >
            <option value="1e-4">1e-4</option>
            <option value="1e-3">1e-3</option>
          </select>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '12px 16px', marginTop: 12 }}>
        <AdvancedCheckboxField
          label="Use EMA"
          checked={state.booleans.use_ema}
          onChange={(checked) => onBooleanChange('use_ema', checked)}
        />
        <AdvancedCheckboxField
          label="Save Best"
          checked={state.booleans.save_best}
          onChange={(checked) => onBooleanChange('save_best', checked)}
        />
      </div>
    </>
  );
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

  const [dataset, setDataset] = useState<string>(datasetOptions[0]?.id ?? '');
  const [trainingRecipeId, setTrainingRecipeId] = useState<TrainingBackendRequest>('robomimic_bc');
  const [dataFormat, setDataFormat] = useState('HDF5');
  const [trainingDevice, setTrainingDevice] = useState<TrainingDeviceValue>(DEFAULT_TRAINING_DEVICE);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [seedMode, setSeedMode] = useState<TrainingSeedMode>('random');
  const [seed, setSeed] = useState(1);
  const [advancedEnabled, setAdvancedEnabled] = useState(false);
  const [pretrainedModelAssetId, setPretrainedModelAssetId] = useState(DEFAULT_PRETRAINED_MODEL_ASSET_ID);
  const [robomimicState, setRobomimicState] = useState<RobomimicAdvancedFormState>(createDefaultRobomimicState());
  const [actDrafts, setActDrafts] = useState<DraftMap>({ ...EMPTY_DRAFTS });
  const [dpState, setDpState] = useState<DpAdvancedFormState>(createDefaultDpState());
  const [robomimicStateCache, setRobomimicStateCache] = useState<RobomimicAdvancedFormState>(
    createDefaultRobomimicState()
  );
  const [actDraftsCache, setActDraftsCache] = useState<DraftMap>({ ...EMPTY_DRAFTS });
  const [dpStateCache, setDpStateCache] = useState<DpAdvancedFormState>(createDefaultDpState());
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);

  useEffect(() => {
    if (!open) return;
    void getTrainingCapabilities()
      .then(setCapabilities)
      .catch(() => setCapabilities(null));
    void listWorkspaceDatasets()
      .then((response) => setApiDatasets(response.datasets))
      .catch(() => setApiDatasets([]));
  }, [open]);

  const selectedDatasetOption = useMemo(
    () => findTrainingDatasetOption(dataset, dataCenterItems, apiDatasets),
    [dataset, dataCenterItems, apiDatasets]
  );

  const trainingBackendPending = useMemo(
    () =>
      selectedDatasetOption
        ? isDualArmTrainingBackendPending(selectedDatasetOption, capabilities)
        : false,
    [selectedDatasetOption, capabilities]
  );

  const dualArmDatasetSelected = Boolean(
    selectedDatasetOption && isDualArmTrainingDatasetOption(selectedDatasetOption)
  );

  const selectedDevice = useMemo(
    () => TRAINING_DEVICE_OPTIONS.find((item) => item.value === trainingDevice) ?? TRAINING_DEVICE_OPTIONS[0],
    [trainingDevice]
  );

  const availableRecipes = useMemo(
    () =>
      listAvailableTrainingRecipes({
        isDualArm: dualArmDatasetSelected,
        capabilities,
      }),
    [dualArmDatasetSelected, capabilities]
  );

  const selectedRecipe = useMemo(
    () => getTrainingRecipe(trainingRecipeId),
    [trainingRecipeId]
  );

  const datasetFormatLabel = useMemo(
    () => resolveDatasetDisplayFormat(selectedDatasetOption, selectedRecipe?.downstreamModelType ?? 'Robomimic'),
    [selectedDatasetOption, selectedRecipe?.downstreamModelType]
  );

  const advancedModelType = advancedFamilyForRecipe(selectedRecipe?.advancedFamily);
  const showRobomimicAdvanced = selectedRecipe?.advancedFamily === 'robomimic';
  const isActModel = advancedModelType === 'ACT';
  const isDpModel = advancedModelType === 'Diffusion Policy';
  const hasModelAdvancedParams = Boolean(selectedRecipe?.advancedFamily);

  const resetAdvancedForm = () => {
    setRobomimicState(createDefaultRobomimicState());
    setActDrafts({ ...EMPTY_DRAFTS });
    setDpState(createDefaultDpState());
    setRobomimicStateCache(createDefaultRobomimicState());
    setActDraftsCache({ ...EMPTY_DRAFTS });
    setDpStateCache(createDefaultDpState());
  };

  useEffect(() => {
    if (!open) return;
    const preferred =
      initialDataset && datasetOptions.some((d) => d.id === initialDataset)
        ? initialDataset
        : datasetOptions[0]?.id ?? '';
    const option = findTrainingDatasetOption(preferred, dataCenterItems, apiDatasets);
    const isDualArm = option ? isDualArmTrainingDatasetOption(option) : false;
    const defaultRecipe = defaultTrainingRecipeForContext({
      isDualArm,
      capabilities,
    });

    setDataset(preferred);
    setTrainingRecipeId(defaultRecipe);
    setDataFormat(recommendDataFormat(option?.dataFormat));
    setTrainingDevice(DEFAULT_TRAINING_DEVICE);
    setEpochs(5);
    setBatchSize(16);
    setLearningRate(0.0001);
    setSeedMode('random');
    setSeed(randomTrainingSeed());
    setAdvancedEnabled(false);
    setPretrainedModelAssetId(DEFAULT_PRETRAINED_MODEL_ASSET_ID);
    resetAdvancedForm();
  }, [open, initialDataset, datasetOptions, dataCenterItems, apiDatasets, capabilities]);

  useEffect(() => {
    if (!open || !selectedDatasetOption) return;
    setDataFormat(recommendDataFormat(selectedDatasetOption.dataFormat));
  }, [open, selectedDatasetOption?.id, selectedDatasetOption?.dataFormat]);

  useEffect(() => {
    if (!open) return;
    const defaultRecipe = defaultTrainingRecipeForContext({
      isDualArm: dualArmDatasetSelected,
      capabilities,
    });
    if (!availableRecipes.some((recipe) => recipe.id === trainingRecipeId)) {
      setTrainingRecipeId(defaultRecipe);
    }
  }, [open, dualArmDatasetSelected, capabilities, availableRecipes, trainingRecipeId]);

  useEffect(() => {
    if (!open) return;
    if (!hasModelAdvancedParams) {
      setAdvancedEnabled(false);
    }
  }, [open, hasModelAdvancedParams]);

  useEffect(() => {
    if (!open) return;
    if (advancedModelType === 'Robomimic') {
      setRobomimicState({
        drafts: { ...robomimicStateCache.drafts },
      });
      return;
    }
    if (advancedModelType === 'ACT') {
      setActDrafts({ ...actDraftsCache });
      return;
    }
    if (advancedModelType === 'Diffusion Policy') {
      setDpState({
        drafts: { ...dpStateCache.drafts },
        booleans: { ...dpStateCache.booleans },
        weightDecay: dpStateCache.weightDecay,
      });
      return;
    }
    setRobomimicStateCache({
      drafts: { ...robomimicState.drafts },
    });
    setActDraftsCache({ ...actDrafts });
    setDpStateCache({
      drafts: { ...dpState.drafts },
      booleans: { ...dpState.booleans },
      weightDecay: dpState.weightDecay,
    });
  }, [advancedModelType, open]);

  useEffect(() => {
    if (!open || advancedModelType !== 'Robomimic') return;
    setRobomimicStateCache({
      drafts: { ...robomimicState.drafts },
    });
  }, [robomimicState, advancedModelType, open]);

  useEffect(() => {
    if (!open || advancedModelType !== 'ACT') return;
    setActDraftsCache({ ...actDrafts });
  }, [actDrafts, advancedModelType, open]);

  useEffect(() => {
    if (!open || advancedModelType !== 'Diffusion Policy') return;
    setDpStateCache({
      drafts: { ...dpState.drafts },
      booleans: { ...dpState.booleans },
      weightDecay: dpState.weightDecay,
    });
  }, [dpState, advancedModelType, open]);

  const handleSeedModeChange = (mode: TrainingSeedMode) => {
    if (mode === seedMode) return;
    if (mode === 'manual') {
      setSeedMode('manual');
      return;
    }
    setSeedMode('random');
    setSeed(randomTrainingSeed());
  };

  const handleRestoreAdvancedDefaults = () => {
    if (showRobomimicAdvanced) {
      const defaults = createDefaultRobomimicState();
      setRobomimicState(defaults);
      setRobomimicStateCache(defaults);
      return;
    }
    if (isActModel) {
      setActDrafts({ ...EMPTY_DRAFTS });
      setActDraftsCache({ ...EMPTY_DRAFTS });
      return;
    }
    if (isDpModel) {
      const defaults = createDefaultDpState();
      setDpState(defaults);
      setDpStateCache(defaults);
    }
  };

  const payload = (): CreateTrainingTaskInput => {
    const deviceParams = trainingDeviceSubmitParams(trainingDevice);
    const recipeFields = recipeToSubmitFields(trainingRecipeId, {
      isDualArm: dualArmDatasetSelected,
      capabilities,
    });
    const result: CreateTrainingTaskInput = {
      dataset,
      downstreamModelType: recipeFields.downstreamModelType,
      dataFormat,
      trainingBackend: recipeFields.trainingBackend,
      trainingDevice,
      epochs,
      batchSize,
      learningRate,
      device: deviceParams.device,
      seed,
      seedMode,
      advancedEnabled,
      taskName: selectedDatasetOption?.datasetName,
      trainability: recipeFields.trainability,
    };

    if (advancedEnabled && hasModelAdvancedParams) {
      if (showRobomimicAdvanced) {
        result.modelParams = resolveRobomimicParams(robomimicState);
      } else if (isActModel) {
        result.modelParams = resolveActParams(actDrafts);
      } else if (isDpModel) {
        result.modelParams = resolveDpParams(dpState);
      }
    }

    return result;
  };

  const canStart = Boolean(
    dataset && selectedDatasetOption && !trainingBackendPending && availableRecipes.length > 0
  );

  return (
    <WorkspaceCenteredModal
      open={open}
      title="新建训练任务"
      titleId="create-training-task-title"
      width={720}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <span style={{ fontSize: 12, color: '#6b7280', alignSelf: 'center' }}>
            {advancedEnabled
              ? '已启用高级设置。创建后可在训练任务列表中查看进度与结果。'
              : '创建后可在训练任务列表中查看进度与结果。'}
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <SecondaryButton onClick={submitting ? undefined : onClose}>取消</SecondaryButton>
            <PrimaryButton
              onClick={() => void onStart(payload())}
              disabled={!canStart || submitting}
            >
              {submitting ? '提交中…' : '创建训练任务'}
            </PrimaryButton>
          </div>
        </div>
      }
    >
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
        基于已登记数据集 manifest 创建训练任务，用于后续策略评测与模型资产管理。
      </p>

      <div style={workspaceModalSectionLabel}>数据集</div>
      <div style={{ marginBottom: 12 }}>
        <select
          style={workspaceModalSelectStyle}
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
        >
          {datasetOptions.length === 0 ? (
            <option value="">请先在数据中心构建训练数据集</option>
          ) : (
            datasetOptions.map((d) => (
              <option key={d.id} value={d.id}>
                {formatTrainingDatasetLabel(d)}
              </option>
            ))
          )}
        </select>
      </div>

      {selectedDatasetOption ? (
        <div
          style={{
            marginBottom: 16,
            padding: '10px 12px',
            borderRadius: 8,
            backgroundColor: '#f8fafc',
            border: '1px solid #e2e8f0',
            fontSize: 13,
            color: '#334155',
            lineHeight: 1.55,
          }}
        >
          <div>来源任务：{selectedDatasetOption.taskName}</div>
          <div>数据集格式：{datasetFormatLabel}</div>
          <div>成功轨迹：{selectedDatasetOption.sampleCount} 条</div>
          <div>数据规模：{selectedDatasetOption.sampleCount} 条成功轨迹</div>
          {selectedDatasetOption.qualityStatus ? (
            <div>质量状态：{selectedDatasetOption.qualityStatus}</div>
          ) : null}
          {dualArmDatasetSelected ? (
            <>
              <div>observationSchema：dual_arm_cable_il_v1</div>
              <div>actionDim：14</div>
            </>
          ) : null}
          {selectedRecipe ? <div>训练方案：{selectedRecipe.label}</div> : null}
        </div>
      ) : null}

      {trainingBackendPending ? (
        <p style={{ margin: '0 0 12px', fontSize: 13, color: '#b45309', lineHeight: 1.55 }}>
          {DUAL_ARM_TRAINING_BACKEND_PENDING_HINT}
        </p>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 4 }}>训练配置</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={workspaceModalFieldLabel}>训练方案</label>
          <select
            style={workspaceModalSelectStyle}
            value={trainingRecipeId}
            onChange={(e) => setTrainingRecipeId(e.target.value as TrainingBackendRequest)}
          >
            {availableRecipes.map((recipe) => (
              <option key={recipe.id} value={recipe.id}>
                {recipe.label}
                {recipe.trainability === 'placeholder' ? '（待接入）' : ''}
              </option>
            ))}
          </select>
          {selectedRecipe ? (
            <p style={fieldHintStyle}>{selectedRecipe.description}</p>
          ) : (
            <p style={{ ...fieldHintStyle, color: '#b45309' }}>当前数据集暂无可用训练方案。</p>
          )}
          {selectedRecipe?.trainability === 'placeholder' ? (
            <p style={{ ...fieldHintStyle, color: '#b45309' }}>
              该训练方案后端尚未接入，创建后将登记配置，暂不会启动真实训练。
            </p>
          ) : null}
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>数据格式</label>
          <select
            style={workspaceModalSelectStyle}
            value={dataFormat}
            onChange={(e) => setDataFormat(e.target.value)}
          >
            {['HDF5', 'HDF5 + NPZ', 'NPZ', 'LeRobot'].map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>训练节点</label>
          <select
            style={workspaceModalSelectStyle}
            value={trainingDevice}
            onChange={(e) => setTrainingDevice(e.target.value as TrainingDeviceValue)}
          >
            {TRAINING_DEVICE_OPTIONS.map((device) => (
              <option key={device.value} value={device.value}>
                {device.label}
              </option>
            ))}
          </select>
          <p style={{ margin: '6px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
            当前任务将提交至所选训练节点执行。
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
            节点说明：{selectedDevice.description}
          </p>
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={workspaceModalFieldLabel}>预训练模型</label>
          <select
            style={workspaceModalSelectStyle}
            value={pretrainedModelAssetId}
            onChange={(e) => setPretrainedModelAssetId(e.target.value)}
          >
            <option value="">不加载</option>
          </select>
          <p style={fieldHintStyle}>预训练模型选项暂未开放，默认从头训练。</p>
        </div>
      </div>

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>训练参数</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
        <div>
          <label style={workspaceModalFieldLabel}>Epochs</label>
          <input
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value) || 1)}
            style={workspaceModalSelectStyle}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Batch Size</label>
          <input
            type="number"
            min={1}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value) || 1)}
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
            style={workspaceModalSelectStyle}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Seed</label>
          <div style={{ display: 'flex', marginBottom: 8, borderRadius: 6, overflow: 'hidden' }}>
            <button
              type="button"
              style={{ ...seedSegmentStyle(seedMode === 'random'), borderRight: 'none', borderRadius: '6px 0 0 6px' }}
              onClick={() => handleSeedModeChange('random')}
            >
              随机
            </button>
            <button
              type="button"
              style={{ ...seedSegmentStyle(seedMode === 'manual'), borderRadius: '0 6px 6px 0' }}
              onClick={() => handleSeedModeChange('manual')}
            >
              指定
            </button>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              type="number"
              value={seed}
              readOnly={seedMode === 'random'}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              style={{
                ...workspaceModalSelectStyle,
                flex: 1,
                backgroundColor: seedMode === 'random' ? '#f1f5f9' : '#fff',
                cursor: seedMode === 'random' ? 'default' : 'text',
              }}
            />
            {seedMode === 'random' ? (
              <SecondaryButton onClick={() => setSeed(randomTrainingSeed())}>重新随机</SecondaryButton>
            ) : null}
          </div>
          <p style={fieldHintStyle}>
            {seedMode === 'random' ? '随机模式下每次打开弹窗自动生成，可点击重新随机。' : '指定模式下将使用你输入的 Seed 值。'}
          </p>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        {hasModelAdvancedParams ? (
          <>
            <label
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
                cursor: 'pointer',
                fontSize: 14,
                color: '#334155',
              }}
            >
              <input
                type="checkbox"
                checked={advancedEnabled}
                onChange={(e) => setAdvancedEnabled(e.target.checked)}
                style={{ marginTop: 3 }}
              />
              <span>
                <span style={{ fontWeight: 600 }}>启用高级设置</span>
                <span style={{ display: 'block', marginTop: 4, fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                  展开后可调整当前训练方案的额外参数。
                  {selectedRecipe?.trainability === 'placeholder'
                    ? '（训练后端接入前将写入配置，待后续执行生效。）'
                    : null}
                </span>
              </span>
            </label>

            {advancedEnabled ? (
              <div style={advancedPanelStyle}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 12,
                    gap: 8,
                  }}
                >
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>
                    方案参数 · {selectedRecipe?.label ?? formatTrainingRecipeLabel(trainingRecipeId)}
                  </span>
                  {advancedModelType && isAdvancedModelType(advancedModelType) ? (
                    <button type="button" onClick={handleRestoreAdvancedDefaults} style={restoreLinkStyle}>
                      恢复默认
                    </button>
                  ) : null}
                </div>

                {showRobomimicAdvanced ? (
                  <RobomimicAdvancedFields
                    state={robomimicState}
                    onDraftChange={(key, value) =>
                      setRobomimicState((prev) => ({
                        ...prev,
                        drafts: { ...prev.drafts, [key]: value },
                      }))
                    }
                  />
                ) : isActModel ? (
                  <ActAdvancedFields
                    drafts={actDrafts}
                    onDraftChange={(key, value) => setActDrafts((prev) => ({ ...prev, [key]: value }))}
                  />
                ) : isDpModel ? (
                  <DiffusionPolicyAdvancedFields
                    state={dpState}
                    onDraftChange={(key, value) =>
                      setDpState((prev) => ({
                        ...prev,
                        drafts: { ...prev.drafts, [key]: value },
                      }))
                    }
                    onBooleanChange={(key, value) =>
                      setDpState((prev) => ({
                        ...prev,
                        booleans: { ...prev.booleans, [key]: value },
                      }))
                    }
                    onWeightDecayChange={(value) =>
                      setDpState((prev) => ({
                        ...prev,
                        weightDecay: value,
                      }))
                    }
                  />
                ) : null}
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </WorkspaceCenteredModal>
  );
}

export function resolveDatasetManifestForTraining(datasetId: string) {
  return getDatasetManifest(datasetId);
}
