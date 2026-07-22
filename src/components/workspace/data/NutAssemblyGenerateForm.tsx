'use client';

import { useEffect, useMemo } from 'react';
import {
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import type { NutAssemblyMimicgenEnvStatus, NutAssemblyPinnModelStatus } from '@/lib/api/nutAssemblyClient';
import {
  AUGMENTATION_ALGORITHM_OPTIONS,
  GENERATION_PATH_OPTIONS,
  type AugmentationAlgorithm,
  type GenerationPath,
} from '@/lib/workspace/generateDataTypes';
import {
  NUT_ASSEMBLY_EXPERT_POLICY_OPTIONS,
  validateNutAssemblyGenerateInput,
  type NutAssemblyPathParamDefaults,
} from '@/lib/workspace/generateDataTaskParams';
import {
  filterNutAssemblySourceDemoDatasets,
  formatNutAssemblySourceDemoOptionLabel,
  NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID,
} from '@/lib/workspace/nutAssemblySeedDatasets';
import {
  formatNutAssemblyEnhancementModelDisplayName,
  NUT_ASSEMBLY_PINN_DEFAULTS,
} from '@/lib/workspace/nutAssemblyPhysicsEnhancement';
import {
  isGenerationPathEnabled,
  resolveGenerationPathDisabledReason,
  type TaskTemplateCapabilityProfile,
} from '@/lib/workspace/taskTemplateCapabilities';
import type { Dataset } from '@/types/benchmark';

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

function GenerationPathCard({
  title,
  description,
  selected,
  disabled,
  disabledReason,
  onSelect,
}: {
  title: string;
  description: string;
  selected: boolean;
  disabled?: boolean;
  disabledReason?: string | null;
  onSelect: () => void;
}) {
  return (
    <label
      style={{
        display: 'flex',
        gap: 10,
        padding: '10px 12px',
        borderRadius: 10,
        border: selected ? '1px solid #2563eb' : '1px solid #e5e7eb',
        background: selected ? '#eff6ff' : '#fff',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.55 : 1,
      }}
    >
      <input
        type="radio"
        checked={selected}
        disabled={disabled}
        onChange={onSelect}
        style={{ marginTop: 3, flexShrink: 0 }}
      />
      <span style={{ minWidth: 0 }}>
        <strong style={{ display: 'block', fontSize: 13, color: '#111827' }}>{title}</strong>
        <span style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.45 }}>{description}</span>
        {disabled && disabledReason ? (
          <span style={{ display: 'block', marginTop: 4, fontSize: 11, color: '#b45309' }}>
            {disabledReason}
          </span>
        ) : null}
      </span>
    </label>
  );
}

function NutAssemblyBuiltInDefaultDemoSelect({ disabled }: { disabled?: boolean }) {
  return (
    <select
      style={workspaceModalSelectStyle}
      defaultValue={NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID}
      disabled={disabled}
      aria-label="源示范数据集"
    >
      <option value={NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID}>螺母装配示范数据（默认）</option>
    </select>
  );
}

function SourceDemoDatasetSelect({
  datasets,
  value,
  onChange,
  disabled,
}: {
  datasets: Dataset[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  if (datasets.length === 0) {
    return (
      <div
        style={{
          padding: '12px 14px',
          borderRadius: 8,
          border: '1px dashed #e5e7eb',
          backgroundColor: '#f9fafb',
          fontSize: 12,
          color: '#6b7280',
          lineHeight: 1.55,
        }}
      >
        暂无可用示范数据，请先通过专家策略生成种子数据。
      </div>
    );
  }

  return (
    <select
      style={workspaceModalSelectStyle}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
    >
      <option value="">请选择源示范数据集</option>
      {datasets.map((dataset) => (
        <option key={dataset.id} value={dataset.id}>
          {formatNutAssemblySourceDemoOptionLabel(dataset)}
        </option>
      ))}
    </select>
  );
}

export interface NutAssemblyGenerateFormProps {
  templateLabel: string;
  capabilities: TaskTemplateCapabilityProfile | null;
  pathParams: NutAssemblyPathParamDefaults;
  onPathParamsChange: (patch: Partial<NutAssemblyPathParamDefaults>) => void;
  datasetName: string;
  robot: string;
  onRobotChange: (robot: string) => void;
  robotOptions: ReadonlyArray<{ value: string; label: string }>;
  enablePinnRepair: boolean;
  onEnablePinnRepairChange: (enabled: boolean) => void;
  pinnSettingsOpen: boolean;
  onPinnSettingsOpenChange: (open: boolean) => void;
  pinnModelStatus: NutAssemblyPinnModelStatus | null;
  mimicgenEnvStatus: NutAssemblyMimicgenEnvStatus | null;
  saveProcessVideo: boolean;
  onSaveProcessVideoChange: (value: boolean) => void;
  workspaceDatasets: Dataset[];
  disabled?: boolean;
}

export function useNutAssemblyGenerateValidation(
  props: Pick<
    NutAssemblyGenerateFormProps,
    'pathParams' | 'datasetName' | 'enablePinnRepair' | 'capabilities'
  >
): string | null {
  return useMemo(
    () =>
      validateNutAssemblyGenerateInput({
        datasetName: props.datasetName,
        generationPath: props.pathParams.generationPath,
        sourceDemoDatasetId: props.pathParams.sourceDemoDatasetId,
        seedGenerationCount: props.pathParams.seedGenerationCount,
        seedKeepCount: props.pathParams.seedKeepCount,
        targetCount: props.pathParams.targetCount,
        generationCount: props.pathParams.generationCount,
        maxSteps: props.pathParams.maxSteps,
        enablePinnRepair: props.enablePinnRepair,
        supportsPinnRepair: props.capabilities?.supportsPinnRepair === true,
        useExistingSeedDataset: props.pathParams.useExistingSeedDataset,
        requiresSourceDemo: false,
      }),
    [props.capabilities?.supportsPinnRepair, props.datasetName, props.enablePinnRepair, props.pathParams]
  );
}

export function NutAssemblyGenerateForm({
  templateLabel,
  capabilities,
  pathParams,
  onPathParamsChange,
  datasetName,
  robot,
  onRobotChange,
  robotOptions,
  enablePinnRepair,
  onEnablePinnRepairChange,
  pinnSettingsOpen,
  onPinnSettingsOpenChange,
  pinnModelStatus,
  mimicgenEnvStatus,
  saveProcessVideo,
  onSaveProcessVideoChange,
  workspaceDatasets,
  disabled = false,
}: NutAssemblyGenerateFormProps) {
  const sourceDemoDatasets = useMemo(
    () => filterNutAssemblySourceDemoDatasets(workspaceDatasets),
    [workspaceDatasets]
  );
  const showPinnSection =
    capabilities?.supportsPinnRepair === true &&
    (pathParams.generationPath === 'demo_augmentation' ||
      pathParams.generationPath === 'expert_seed_then_augmentation');
  const showSourceDemo =
    pathParams.generationPath === 'demo_augmentation' ||
    (pathParams.generationPath === 'expert_seed_then_augmentation' &&
      pathParams.useExistingSeedDataset);
  const mimicgenPathSelected =
    pathParams.generationPath === 'demo_augmentation' ||
    pathParams.generationPath === 'expert_seed_then_augmentation';

  useEffect(() => {
    if (pathParams.generationPath !== 'demo_augmentation') return;
    if (pathParams.sourceDemoDatasetId === NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID) return;
    onPathParamsChange({ sourceDemoDatasetId: NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID });
  }, [pathParams.generationPath, pathParams.sourceDemoDatasetId, onPathParamsChange]);

  return (
    <>
      <FormSection title="仿真配置">
        <FieldGrid>
          <div>
            <label style={workspaceModalFieldLabel}>仿真环境</label>
            <input
              type="text"
              readOnly
              value="MuJoCo"
              style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb', cursor: 'default' }}
              disabled
            />
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>机器人</label>
            <select
              style={workspaceModalSelectStyle}
              value={robot}
              onChange={(e) => onRobotChange(e.target.value)}
              disabled={disabled}
            >
              {robotOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </FieldGrid>
      </FormSection>

      <FormSection title="生成路径">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {GENERATION_PATH_OPTIONS.map((option) => {
            const pathEnabled = isGenerationPathEnabled(templateLabel, option.value);
            const disabledReason = resolveGenerationPathDisabledReason(templateLabel, option.value);
            return (
              <GenerationPathCard
                key={option.value}
                title={option.title}
                description={option.description}
                selected={pathParams.generationPath === option.value}
                disabled={disabled || !pathEnabled}
                disabledReason={disabledReason}
                onSelect={() => {
                  if (disabled || !pathEnabled) return;
                  onPathParamsChange({
                    generationPath: option.value,
                    ...(option.value === 'demo_augmentation'
                      ? { sourceDemoDatasetId: NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID }
                      : {}),
                  });
                }}
              />
            );
          })}
        </div>
        {mimicgenPathSelected &&
        mimicgenEnvStatus &&
        !mimicgenEnvStatus.overallOk ? (
          <div
            style={{
              marginTop: 10,
              padding: '10px 12px',
              borderRadius: 10,
              border: '1px solid #fde68a',
              backgroundColor: '#fffbeb',
              fontSize: 12,
              lineHeight: 1.6,
              color: '#92400e',
            }}
          >
            MimicGen 运行环境未就绪，请先完成环境检查。
          </div>
        ) : null}
      </FormSection>

      <FormSection title="路径参数">
        {pathParams.generationPath === 'expert_policy' ? (
          <FieldGrid>
            <div>
              <label style={workspaceModalFieldLabel}>生成条数</label>
              <input
                type="number"
                min={1}
                max={200}
                value={pathParams.generationCount}
                onChange={(e) =>
                  onPathParamsChange({ generationCount: Number(e.target.value) || 1 })
                }
                style={workspaceModalSelectStyle}
                disabled={disabled}
              />
            </div>
            <div>
              <label style={workspaceModalFieldLabel}>最大步数</label>
              <input
                type="number"
                min={50}
                max={1000}
                value={pathParams.maxSteps}
                onChange={(e) => onPathParamsChange({ maxSteps: Number(e.target.value) || 500 })}
                style={workspaceModalSelectStyle}
                disabled={disabled}
              />
            </div>
            <div>
              <label style={workspaceModalFieldLabel}>专家策略</label>
              <select
                style={workspaceModalSelectStyle}
                value={pathParams.expertPolicy}
                onChange={(e) => onPathParamsChange({ expertPolicy: e.target.value })}
                disabled={disabled}
              >
                {NUT_ASSEMBLY_EXPERT_POLICY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label style={workspaceModalFieldLabel}>成功筛选</label>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  minHeight: 40,
                  padding: '0 12px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#fff',
                  fontSize: 13,
                  cursor: disabled ? 'not-allowed' : 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={pathParams.successFilter}
                  disabled={disabled}
                  onChange={(e) => onPathParamsChange({ successFilter: e.target.checked })}
                />
                {pathParams.successFilter ? '开启' : '关闭'}
              </label>
            </div>
            <div>
              <label style={workspaceModalFieldLabel}>失败轨迹保留</label>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  minHeight: 40,
                  padding: '0 12px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#fff',
                  fontSize: 13,
                  cursor: disabled ? 'not-allowed' : 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={pathParams.keepFailedTrajectories}
                  disabled={disabled}
                  onChange={(e) =>
                    onPathParamsChange({ keepFailedTrajectories: e.target.checked })
                  }
                />
                {pathParams.keepFailedTrajectories ? '保留' : '不保留'}
              </label>
            </div>
          </FieldGrid>
        ) : null}

        {pathParams.generationPath === 'demo_augmentation' ? (
          <>
            <div style={{ marginBottom: 12 }}>
              <label style={workspaceModalFieldLabel}>源示范数据集</label>
              <NutAssemblyBuiltInDefaultDemoSelect disabled={disabled} />
            </div>
            <FieldGrid>
              <div>
                <label style={workspaceModalFieldLabel}>扩增算法</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={pathParams.augmentationAlgorithm}
                  onChange={(e) =>
                    onPathParamsChange({
                      augmentationAlgorithm: e.target.value as AugmentationAlgorithm,
                    })
                  }
                  disabled={disabled}
                >
                  {AUGMENTATION_ALGORITHM_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>目标生成条数</label>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={pathParams.targetCount}
                  onChange={(e) => onPathParamsChange({ targetCount: Number(e.target.value) || 1 })}
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>最大步数</label>
                <input
                  type="number"
                  min={50}
                  max={1000}
                  value={pathParams.maxSteps}
                  onChange={(e) => onPathParamsChange({ maxSteps: Number(e.target.value) || 500 })}
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>成功复核</label>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    minHeight: 40,
                    padding: '0 12px',
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    backgroundColor: '#fff',
                    fontSize: 13,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={pathParams.replayValidation}
                    disabled={disabled}
                    onChange={(e) => onPathParamsChange({ replayValidation: e.target.checked })}
                  />
                  {pathParams.replayValidation ? '开启' : '关闭'}
                </label>
              </div>
            </FieldGrid>
          </>
        ) : null}

        {pathParams.generationPath === 'expert_seed_then_augmentation' ? (
          <>
            <FieldGrid>
              <div>
                <label style={workspaceModalFieldLabel}>种子生成条数</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={pathParams.seedGenerationCount}
                  onChange={(e) =>
                    onPathParamsChange({ seedGenerationCount: Number(e.target.value) || 1 })
                  }
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>种子保留条数</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={pathParams.seedKeepCount}
                  onChange={(e) =>
                    onPathParamsChange({ seedKeepCount: Number(e.target.value) || 1 })
                  }
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>扩增目标条数</label>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={pathParams.targetCount}
                  onChange={(e) => onPathParamsChange({ targetCount: Number(e.target.value) || 1 })}
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>扩增算法</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={pathParams.augmentationAlgorithm}
                  onChange={(e) =>
                    onPathParamsChange({
                      augmentationAlgorithm: e.target.value as AugmentationAlgorithm,
                    })
                  }
                  disabled={disabled}
                >
                  {AUGMENTATION_ALGORITHM_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>最大步数</label>
                <input
                  type="number"
                  min={50}
                  max={1000}
                  value={pathParams.maxSteps}
                  onChange={(e) => onPathParamsChange({ maxSteps: Number(e.target.value) || 500 })}
                  style={workspaceModalSelectStyle}
                  disabled={disabled}
                />
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>自动筛选优质种子</label>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    minHeight: 40,
                    padding: '0 12px',
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    backgroundColor: '#fff',
                    fontSize: 13,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={pathParams.autoSelectBestSeeds}
                    disabled={disabled}
                    onChange={(e) =>
                      onPathParamsChange({ autoSelectBestSeeds: e.target.checked })
                    }
                  />
                  {pathParams.autoSelectBestSeeds ? '开启' : '关闭'}
                </label>
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>成功复核</label>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    minHeight: 40,
                    padding: '0 12px',
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    backgroundColor: '#fff',
                    fontSize: 13,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={pathParams.replayValidation}
                    disabled={disabled}
                    onChange={(e) => onPathParamsChange({ replayValidation: e.target.checked })}
                  />
                  {pathParams.replayValidation ? '开启' : '关闭'}
                </label>
              </div>
            </FieldGrid>
            <div style={{ marginTop: 12 }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 13,
                  color: '#374151',
                  cursor: disabled ? 'not-allowed' : 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={pathParams.useExistingSeedDataset}
                  disabled={disabled}
                  onChange={(e) =>
                    onPathParamsChange({
                      useExistingSeedDataset: e.target.checked,
                      sourceDemoDatasetId: e.target.checked ? pathParams.sourceDemoDatasetId : '',
                    })
                  }
                />
                使用已有种子数据集（高级）
              </label>
            </div>
            {showSourceDemo ? (
              <div style={{ marginTop: 12 }}>
                <label style={workspaceModalFieldLabel}>源示范数据集</label>
                <SourceDemoDatasetSelect
                  datasets={sourceDemoDatasets}
                  value={pathParams.sourceDemoDatasetId}
                  onChange={(id) => onPathParamsChange({ sourceDemoDatasetId: id })}
                  disabled={disabled}
                />
              </div>
            ) : null}
          </>
        ) : null}
      </FormSection>

      {showPinnSection ? (
        <FormSection title="数据质量增强">
          <label
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
              fontSize: 13,
              color: '#111827',
              cursor: disabled ? 'not-allowed' : 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={enablePinnRepair}
              disabled={disabled || !pinnModelStatus?.available}
              onChange={(e) => onEnablePinnRepairChange(e.target.checked)}
              style={{ marginTop: 3 }}
            />
            <span>
              <span style={{ fontWeight: 600 }}>物理增强模型</span>
              <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
                对生成过程中的失败轨迹、边界轨迹和低质量轨迹进行修复参数优选，并通过仿真复核后写入最终数据集。
              </div>
            </span>
          </label>
          {!pinnModelStatus?.available ? (
            <div style={{ marginTop: 8, fontSize: 12, color: '#b45309' }}>
              {pinnModelStatus?.error ?? '未检测到物理增强模型，请先完成模型配置。'}
            </div>
          ) : null}
          {enablePinnRepair ? (
            <>
              <button
                type="button"
                onClick={() => onPinnSettingsOpenChange(!pinnSettingsOpen)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginTop: 12,
                  padding: 0,
                  border: 'none',
                  background: 'none',
                  fontSize: 12,
                  fontWeight: 600,
                  color: '#6b7280',
                  cursor: 'pointer',
                }}
              >
                <span style={{ fontSize: 10, color: '#9ca3af' }}>
                  {pinnSettingsOpen ? '▼' : '▶'}
                </span>
                修复参数
              </button>
              {pinnSettingsOpen ? (
                <div
                  style={{
                    marginTop: 10,
                    padding: '12px 14px',
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    backgroundColor: '#f9fafb',
                    fontSize: 13,
                    lineHeight: 1.8,
                    color: '#374151',
                  }}
                >
                  <div>修复阶段：对齐阶段 / 插入阶段</div>
                  <div>
                    修复模型：
                    {formatNutAssemblyEnhancementModelDisplayName(pinnModelStatus?.displayName)}
                  </div>
                  <div>候选轨迹上限：{NUT_ASSEMBLY_PINN_DEFAULTS.maxCandidates}</div>
                  <div>
                    每条轨迹最大修复次数：{NUT_ASSEMBLY_PINN_DEFAULTS.maxRepairAttemptsPerCandidate}
                  </div>
                  <div>
                    对齐误差阈值：{(NUT_ASSEMBLY_PINN_DEFAULTS.xyErrorThreshold * 100).toFixed(1)} cm
                  </div>
                  <div>复核方式：MuJoCo 仿真复核</div>
                </div>
              ) : null}
            </>
          ) : null}
        </FormSection>
      ) : null}

      <FormSection title="输出配置">
        <FieldGrid>
          <div>
            <label style={workspaceModalFieldLabel}>数据格式</label>
            <input
              type="text"
              readOnly
              value={capabilities?.outputFormat ?? 'HDF5'}
              style={{ ...workspaceModalSelectStyle, backgroundColor: '#f9fafb', cursor: 'default' }}
              disabled
            />
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>生成回放视频</label>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                minHeight: 40,
                padding: '0 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#fff',
                fontSize: 13,
                cursor: disabled ? 'not-allowed' : 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={saveProcessVideo}
                disabled={disabled}
                onChange={(e) => onSaveProcessVideoChange(e.target.checked)}
                style={{ margin: 0, accentColor: '#2563eb' }}
              />
              {saveProcessVideo ? '开' : '关'}
            </label>
          </div>
        </FieldGrid>
      </FormSection>
    </>
  );
}

// Re-export nut assembly path helpers for modal consumers
export {
  defaultNutAssemblyPathParams,
  validateNutAssemblyGenerateInput,
  NUT_ASSEMBLY_PATH_DEFAULTS,
} from '@/lib/workspace/generateDataTaskParams';
export type { GenerationPath, AugmentationAlgorithm } from '@/lib/workspace/generateDataTypes';
