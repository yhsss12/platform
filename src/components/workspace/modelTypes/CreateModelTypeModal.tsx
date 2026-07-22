'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  WorkspaceCenteredModal,
  WorkspaceModalFieldGrid,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  BASE_ALGORITHM_OPTIONS,
  ROBOT_TYPE_OPTIONS,
  SIMULATOR_OPTIONS,
  type CreateModelTypeInput,
} from '@/types/modelType';
import {
  adapterKeyForBaseAlgorithm,
  defaultStructureConfigForAlgorithm,
} from '@/lib/workspace/modelTypeDisplay';

function StructureFields({
  baseAlgorithm,
  config,
  onChange,
}: {
  baseAlgorithm: string;
  config: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const update = (key: string, value: unknown) => onChange({ ...config, [key]: value });

  if (baseAlgorithm === 'robomimic_bc') {
    return (
      <WorkspaceModalFieldGrid>
        <div>
          <label style={workspaceModalFieldLabel}>Hidden Dims</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.actor_hidden_dims ?? '')}
            placeholder="512,512"
            onChange={(e) => update('actor_hidden_dims', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>L2 Regularization</label>
          <input
            type="number"
            min={0}
            step={0.0001}
            style={workspaceModalSelectStyle}
            value={Number(config.l2_regularization ?? 0)}
            onChange={(e) => update('l2_regularization', Number(e.target.value))}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Encoder Type</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.encoder_type ?? 'low_dim')}
            onChange={(e) => update('encoder_type', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Activation</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.activation ?? 'relu')}
            onChange={(e) => update('activation', e.target.value)}
          />
        </div>
      </WorkspaceModalFieldGrid>
    );
  }

  if (baseAlgorithm === 'act') {
    const numberFields: { key: string; label: string; defaultValue: number }[] = [
      { key: 'hidden_dim', label: 'Hidden Dim', defaultValue: 512 },
      { key: 'dim_feedforward', label: 'Dim Feedforward', defaultValue: 2048 },
      { key: 'chunk_size', label: 'Chunk Size', defaultValue: 100 },
      { key: 'n_action_steps', label: 'Action Steps', defaultValue: 100 },
      { key: 'kl_weight', label: 'KL Weight', defaultValue: 10 },
      { key: 'latent_dim', label: 'Latent Dim', defaultValue: 32 },
      { key: 'enc_layers', label: 'Enc Layers', defaultValue: 4 },
      { key: 'dec_layers', label: 'Dec Layers', defaultValue: 4 },
      { key: 'nheads', label: 'N Heads', defaultValue: 8 },
      { key: 'dropout', label: 'Dropout', defaultValue: 0.1 },
    ];
    return (
      <WorkspaceModalFieldGrid>
        {numberFields.map((field) => (
          <div key={field.key}>
            <label style={workspaceModalFieldLabel}>{field.label}</label>
            <input
              type="number"
              style={workspaceModalSelectStyle}
              value={Number(config[field.key] ?? field.defaultValue)}
              onChange={(e) => update(field.key, Number(e.target.value))}
            />
          </div>
        ))}
      </WorkspaceModalFieldGrid>
    );
  }

  if (baseAlgorithm === 'diffusion_policy') {
    return (
      <WorkspaceModalFieldGrid>
        {[
          { key: 'horizon', label: 'Horizon', defaultValue: 16 },
          { key: 'n_obs_steps', label: 'N Obs Steps', defaultValue: 2 },
          { key: 'n_action_steps', label: 'Action Steps', defaultValue: 8 },
          { key: 'num_inference_steps', label: 'Inference Steps', defaultValue: 20 },
        ].map((field) => (
          <div key={field.key}>
            <label style={workspaceModalFieldLabel}>{field.label}</label>
            <input
              type="number"
              min={1}
              style={workspaceModalSelectStyle}
              value={Number(config[field.key] ?? field.defaultValue)}
              onChange={(e) => update(field.key, Number(e.target.value))}
            />
          </div>
        ))}
        <div>
          <label style={workspaceModalFieldLabel}>Weight Decay</label>
          <input
            type="number"
            step={0.0001}
            style={workspaceModalSelectStyle}
            value={Number(config.weight_decay ?? 0.0001)}
            onChange={(e) => update('weight_decay', Number(e.target.value))}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Vision Encoder</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.vision_encoder ?? 'resnet18')}
            onChange={(e) => update('vision_encoder', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Noise Scheduler</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.noise_scheduler ?? 'ddpm')}
            onChange={(e) => update('noise_scheduler', e.target.value)}
          />
        </div>
      </WorkspaceModalFieldGrid>
    );
  }

  if (baseAlgorithm === 'pi0') {
    return (
      <WorkspaceModalFieldGrid>
        {[
          { key: 'context_window', label: 'Context Window', defaultValue: 256 },
          { key: 'action_horizon', label: 'Action Horizon', defaultValue: 16 },
        ].map((field) => (
          <div key={field.key}>
            <label style={workspaceModalFieldLabel}>{field.label}</label>
            <input
              type="number"
              min={1}
              style={workspaceModalSelectStyle}
              value={Number(config[field.key] ?? field.defaultValue)}
              onChange={(e) => update(field.key, Number(e.target.value))}
            />
          </div>
        ))}
        <div>
          <label style={workspaceModalFieldLabel}>Vision Encoder</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.vision_encoder ?? 'siglip')}
            onChange={(e) => update('vision_encoder', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Action Head</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.action_head ?? 'flow_matching')}
            onChange={(e) => update('action_head', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Tokenizer / Processor</label>
          <input
            style={workspaceModalSelectStyle}
            value={String(config.tokenizer_or_processor ?? 'default')}
            onChange={(e) => update('tokenizer_or_processor', e.target.value)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Language Conditioning</label>
          <select
            style={workspaceModalSelectStyle}
            value={String(config.language_conditioning ?? true)}
            onChange={(e) => update('language_conditioning', e.target.value === 'true')}
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </div>
      </WorkspaceModalFieldGrid>
    );
  }

  return null;
}

function slugifyModelTypeId(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120);
}

export function CreateModelTypeModal({
  open,
  onClose,
  onCreated,
  submitting = false,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: () => void | Promise<void>;
  submitting?: boolean;
}) {
  const [name, setName] = useState('');
  const [modelTypeId, setModelTypeId] = useState('');
  const [modelTypeIdTouched, setModelTypeIdTouched] = useState(false);
  const [baseAlgorithm, setBaseAlgorithm] = useState<string>('robomimic_bc');
  const [simulator, setSimulator] = useState('general');
  const [robotType, setRobotType] = useState('general');
  const [tagsDraft, setTagsDraft] = useState('');
  const [description, setDescription] = useState('');
  const [structureConfig, setStructureConfig] = useState<Record<string, unknown>>({});
  const [defaultEpochs, setDefaultEpochs] = useState(5);
  const [defaultBatchSize, setDefaultBatchSize] = useState(16);
  const [defaultLearningRate, setDefaultLearningRate] = useState(0.0001);
  const [defaultSeedStrategy, setDefaultSeedStrategy] = useState('random');
  const [status, setStatus] = useState<'draft' | 'available'>('draft');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setName('');
    setModelTypeId('');
    setModelTypeIdTouched(false);
    setBaseAlgorithm('robomimic_bc');
    setSimulator('general');
    setRobotType('general');
    setTagsDraft('');
    setDescription('');
    setStructureConfig(defaultStructureConfigForAlgorithm('robomimic_bc'));
    setDefaultEpochs(5);
    setDefaultBatchSize(16);
    setDefaultLearningRate(0.0001);
    setDefaultSeedStrategy('random');
    setStatus('draft');
    setError(null);
  }, [open]);

  useEffect(() => {
    if (!open || modelTypeIdTouched) return;
    setModelTypeId(slugifyModelTypeId(name));
  }, [name, open, modelTypeIdTouched]);

  useEffect(() => {
    if (!open) return;
    setStructureConfig(defaultStructureConfigForAlgorithm(baseAlgorithm));
  }, [baseAlgorithm, open]);

  const adapterKey = useMemo(() => adapterKeyForBaseAlgorithm(baseAlgorithm), [baseAlgorithm]);

  const canSubmit = Boolean(name.trim() && baseAlgorithm);

  const handleSubmit = async () => {
    setError(null);
    const payload: CreateModelTypeInput = {
      name: name.trim(),
      modelTypeId: modelTypeId.trim() || undefined,
      baseAlgorithm,
      simulator,
      robotType,
      tags: tagsDraft
        .split(/[,，]/)
        .map((item) => item.trim())
        .filter(Boolean),
      description: description.trim() || undefined,
      structureConfig,
      trainingDefaults: {
        default_epochs: defaultEpochs,
        default_batch_size: defaultBatchSize,
        default_learning_rate: defaultLearningRate,
        default_seed_strategy: defaultSeedStrategy,
      },
      status,
    };
    try {
      const { createModelType } = await import('@/lib/api/modelTypesClient');
      await createModelType(payload);
      await onCreated();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建失败');
    }
  };

  return (
    <WorkspaceCenteredModal
      open={open}
      title="新建模型类型"
      titleId="create-model-type-title"
      width={760}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={submitting ? undefined : onClose}>取消</SecondaryButton>
          <PrimaryButton onClick={() => void handleSubmit()} disabled={!canSubmit || submitting}>
            {submitting ? '提交中…' : '创建模型类型'}
          </PrimaryButton>
        </div>
      }
    >
      <div style={workspaceModalSectionLabel}>基础信息</div>
      <WorkspaceModalFieldGrid>
        <div>
          <label style={workspaceModalFieldLabel}>模型类型名称</label>
          <input
            style={workspaceModalSelectStyle}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="例如 Robomimic BC、自定义 BC"
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>模型标识 modelTypeId</label>
          <input
            style={workspaceModalSelectStyle}
            value={modelTypeId}
            onChange={(e) => {
              setModelTypeIdTouched(true);
              setModelTypeId(e.target.value);
            }}
            placeholder="自动生成，可编辑"
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>基础算法</label>
          <select
            style={workspaceModalSelectStyle}
            value={baseAlgorithm}
            onChange={(e) => setBaseAlgorithm(e.target.value)}
          >
            {BASE_ALGORITHM_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>标准适配层</label>
          <input style={workspaceModalSelectStyle} value={adapterKey} readOnly disabled />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>适用仿真环境</label>
          <select
            style={workspaceModalSelectStyle}
            value={simulator}
            onChange={(e) => setSimulator(e.target.value)}
          >
            {SIMULATOR_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>机器人类型</label>
          <select
            style={workspaceModalSelectStyle}
            value={robotType}
            onChange={(e) => setRobotType(e.target.value)}
          >
            {ROBOT_TYPE_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={workspaceModalFieldLabel}>标签（逗号分隔）</label>
          <input
            style={workspaceModalSelectStyle}
            value={tagsDraft}
            onChange={(e) => setTagsDraft(e.target.value)}
            placeholder="BC, 自定义"
          />
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={workspaceModalFieldLabel}>描述</label>
          <textarea
            style={{ ...workspaceModalSelectStyle, minHeight: 72, resize: 'vertical' }}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
      </WorkspaceModalFieldGrid>

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>模型结构配置</div>
      <StructureFields
        baseAlgorithm={baseAlgorithm}
        config={structureConfig}
        onChange={setStructureConfig}
      />

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>训练默认参数</div>
      <WorkspaceModalFieldGrid>
        <div>
          <label style={workspaceModalFieldLabel}>Default Epochs</label>
          <input
            type="number"
            min={1}
            style={workspaceModalSelectStyle}
            value={defaultEpochs}
            onChange={(e) => setDefaultEpochs(Number(e.target.value) || 1)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Default Batch Size</label>
          <input
            type="number"
            min={1}
            style={workspaceModalSelectStyle}
            value={defaultBatchSize}
            onChange={(e) => setDefaultBatchSize(Number(e.target.value) || 1)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Default Learning Rate</label>
          <input
            type="number"
            step={0.00001}
            style={workspaceModalSelectStyle}
            value={defaultLearningRate}
            onChange={(e) => setDefaultLearningRate(Number(e.target.value))}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>Default Seed Strategy</label>
          <select
            style={workspaceModalSelectStyle}
            value={defaultSeedStrategy}
            onChange={(e) => setDefaultSeedStrategy(e.target.value)}
          >
            <option value="random">random</option>
            <option value="fixed">fixed</option>
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>初始状态</label>
          <select
            style={workspaceModalSelectStyle}
            value={status}
            onChange={(e) => setStatus(e.target.value as 'draft' | 'available')}
          >
            <option value="draft">草稿</option>
            <option value="available">可用</option>
          </select>
        </div>
      </WorkspaceModalFieldGrid>

      {error ? (
        <p style={{ marginTop: 12, fontSize: 13, color: '#dc2626' }}>{error}</p>
      ) : null}
    </WorkspaceCenteredModal>
  );
}
