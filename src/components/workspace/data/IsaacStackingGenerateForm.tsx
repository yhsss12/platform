'use client';

import {
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import type { IsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';
import { ISAAC_BLOCK_STACKING_DEFAULT_ENV } from '@/lib/workspace/isaacBlockStacking';
import {
  formatIsaacStackCubeIssues,
  formatScriptedExpertIssues,
} from '@/lib/workspace/isaacStackCubeRuntime';
import {
  formatSeedDatasetOptionLabel,
} from '@/lib/workspace/isaacSeedDatasets';
import {
  formatPhysicsBackendLabel,
  formatSimulatorBackendLabel,
} from '@/lib/workspace/taskTemplateCapabilities';
import type { Dataset } from '@/types/benchmark';
import { SecondaryButton } from '@/components/workspace/workspaceUi';

export type IsaacStackingGenerationMode = 'expert_policy' | 'mimic_auto' | 'scripted_expert';

export const ISAAC_EXPERT_POLICY_LABEL = '专家策略生成';

export const ISAAC_EXPERT_POLICY_DESCRIPTION =
  '基于物块堆叠任务的专家策略自动生成高质量轨迹，并完成 HDF5 录制、质量检测与回放生成。';

/** Mimic 实验模式 — 仅高级设置中可选，非默认主流程 */
export const ISAAC_MIMIC_EXPERIMENT_OPTION = {
  mode: 'mimic_auto' as const,
  label: 'Mimic 示范扩增（实验模式）',
  description:
    '基于 seed demonstrations 进行子任务标注与轨迹扩增，仅供高级实验；默认推荐使用专家策略生成。',
};

/** @deprecated */
export const ISAAC_STACKING_GENERATION_MODE_OPTIONS = [
  { mode: 'expert_policy' as const, label: ISAAC_EXPERT_POLICY_LABEL, description: ISAAC_EXPERT_POLICY_DESCRIPTION },
  ISAAC_MIMIC_EXPERIMENT_OPTION,
];

/** @deprecated use ISAAC_EXPERT_POLICY_LABEL */
export const ISAAC_STACKING_GENERATION_MODE_LABELS: Record<IsaacStackingGenerationMode, string> = {
  expert_policy: ISAAC_EXPERT_POLICY_LABEL,
  scripted_expert: ISAAC_EXPERT_POLICY_LABEL,
  mimic_auto: 'Mimic 示范扩增',
};

function ReadonlyField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <label style={workspaceModalFieldLabel}>{label}</label>
      <input
        type="text"
        readOnly
        value={value}
        style={{
          ...workspaceModalSelectStyle,
          backgroundColor: '#f9fafb',
          cursor: 'default',
        }}
      />
    </div>
  );
}

export function IsaacStackingGenerateForm({
  generationMode,
  onGenerationModeChange,
  seedDatasets,
  selectedSeedDatasetId,
  onSelectedSeedDatasetIdChange,
  manualSeedPath,
  onManualSeedPathChange,
  advancedOpen,
  onAdvancedOpenChange,
  onImportCustomSeed,
  numDemos,
  onNumDemosChange,
  seed,
  onSeedChange,
  headless,
  onHeadlessChange,
  enableCameras,
  onEnableCamerasChange,
  parallelNumEnvs,
  onParallelNumEnvsChange,
  runtime,
  runtimeLoading,
  disabled,
}: {
  generationMode: IsaacStackingGenerationMode;
  onGenerationModeChange: (mode: IsaacStackingGenerationMode) => void;
  seedDatasets: Dataset[];
  selectedSeedDatasetId: string;
  onSelectedSeedDatasetIdChange: (id: string) => void;
  manualSeedPath: string;
  onManualSeedPathChange: (path: string) => void;
  advancedOpen: boolean;
  onAdvancedOpenChange: (open: boolean) => void;
  onImportCustomSeed?: () => void;
  numDemos: number;
  onNumDemosChange: (value: number) => void;
  seed: number;
  onSeedChange: (value: number) => void;
  headless: boolean;
  onHeadlessChange: (value: boolean) => void;
  enableCameras: boolean;
  onEnableCamerasChange: (value: boolean) => void;
  parallelNumEnvs: number;
  onParallelNumEnvsChange: (value: number) => void;
  runtime: IsaacLabRuntimeStatus | null;
  runtimeLoading: boolean;
  disabled?: boolean;
}) {
  const isMimicExperiment = generationMode === 'mimic_auto';
  const mimicReady = Boolean(runtime?.stackCubeGenerationReady);
  const expertReady = Boolean(runtime?.scriptedExpertReady);
  const generationReady = isMimicExperiment ? mimicReady : expertReady;
  const issueLabels = isMimicExperiment
    ? formatIsaacStackCubeIssues(runtime?.stackCubeIssueCodes ?? [])
    : formatScriptedExpertIssues(runtime?.scriptedExpertIssueCodes ?? []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {!runtimeLoading && !generationReady ? (
        <div
          style={{
            padding: '10px 12px',
            borderRadius: 8,
            backgroundColor: '#fffbeb',
            border: '1px solid #fde68a',
            fontSize: 12,
            color: '#92400e',
            lineHeight: 1.6,
          }}
        >
          <div>
            {isMimicExperiment
              ? '当前环境尚未完成 Isaac Lab Mimic 生成配置，请联系平台管理员。'
              : 'Isaac 专家策略运行环境未就绪，请联系平台管理员配置 Isaac Lab 运行节点。'}
          </div>
          {issueLabels.length > 0 ? (
            <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
              {issueLabels.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
        <ReadonlyField label="仿真后端" value={formatSimulatorBackendLabel('isaac_lab')} />
        <ReadonlyField label="物理引擎" value={formatPhysicsBackendLabel('physx')} />
        <ReadonlyField label="机器人" value="Franka Panda" />
        <div>
          <label style={workspaceModalFieldLabel}>生成方式</label>
          <select
            style={workspaceModalSelectStyle}
            value={generationMode === 'scripted_expert' ? 'expert_policy' : generationMode}
            onChange={(e) => onGenerationModeChange(e.target.value as IsaacStackingGenerationMode)}
            disabled={disabled}
          >
            <option value="expert_policy">{ISAAC_EXPERT_POLICY_LABEL}</option>
            <option value="mimic_auto">Mimic 示范扩增</option>
          </select>
        </div>
      </div>
      <p style={{ margin: '-8px 0 0', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
        {isMimicExperiment
          ? '使用平台默认或自定义 Seed Demonstration，通过 Isaac Lab Mimic 扩增生成目标轨迹。'
          : ISAAC_EXPERT_POLICY_DESCRIPTION}
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
        <div>
          <label style={workspaceModalFieldLabel}>生成轮次 numDemos</label>
          <input
            type="number"
            min={1}
            max={1000}
            value={numDemos}
            onChange={(e) => onNumDemosChange(Number(e.target.value) || 1)}
            style={workspaceModalSelectStyle}
            disabled={disabled}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>随机种子 seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => onSeedChange(Number(e.target.value) || 0)}
            style={workspaceModalSelectStyle}
            disabled={disabled}
          />
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 13 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={headless}
            onChange={(e) => onHeadlessChange(e.target.checked)}
            disabled={disabled}
          />
          headless
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={enableCameras}
            onChange={(e) => onEnableCamerasChange(e.target.checked)}
            disabled={disabled}
          />
          enable_cameras
        </label>
      </div>

      <div>
        <button
          type="button"
          onClick={() => onAdvancedOpenChange(!advancedOpen)}
          disabled={disabled}
          style={{
            padding: 0,
            border: 'none',
            background: 'none',
            color: '#2563eb',
            fontSize: 12,
            cursor: disabled ? 'not-allowed' : 'pointer',
          }}
        >
          {advancedOpen ? '收起高级设置' : '高级设置'}
        </button>

        {advancedOpen && isMimicExperiment ? (
          <div
            style={{
              marginTop: 12,
              padding: '12px 14px',
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              backgroundColor: '#fafafa',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
            }}
          >
            {isMimicExperiment ? (
              <>
                <p style={{ margin: 0, fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
                  默认使用平台预置物块堆叠 Seed Demo 进行 Mimic 扩增。如需使用自定义 HDF5，可从已导入数据集中选择或填写服务器路径。
                </p>

                {seedDatasets.length > 0 ? (
                  <div>
                    <label style={workspaceModalFieldLabel}>Seed Dataset（可选）</label>
                    <select
                      style={workspaceModalSelectStyle}
                      value={selectedSeedDatasetId}
                      onChange={(e) => onSelectedSeedDatasetIdChange(e.target.value)}
                      disabled={disabled}
                    >
                      <option value="">使用平台默认 Seed Demo</option>
                      {seedDatasets.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>
                          {formatSeedDatasetOptionLabel(dataset)}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : null}

                <div>
                  <label style={workspaceModalFieldLabel}>服务器路径 seedDatasetFile（可选）</label>
                  <input
                    type="text"
                    value={manualSeedPath}
                    onChange={(e) => onManualSeedPathChange(e.target.value)}
                    placeholder="留空则使用平台默认 Seed"
                    style={workspaceModalSelectStyle}
                    disabled={disabled}
                  />
                </div>

                <div>
                  <label style={workspaceModalFieldLabel}>并行环境数 numEnvs（高级）</label>
                  <select
                    style={workspaceModalSelectStyle}
                    value={parallelNumEnvs}
                    onChange={(e) => onParallelNumEnvsChange(Number(e.target.value))}
                    disabled={disabled}
                  >
                    {[1, 4, 8, 16].map((value) => (
                      <option key={value} value={value}>
                        {value}
                        {value === 1 ? '（默认 · 单环境预览）' : ''}
                      </option>
                    ))}
                  </select>
                  <p style={{ margin: '6px 0 0', fontSize: 11, color: '#9ca3af', lineHeight: 1.5 }}>
                    运行控制台始终聚焦 env_0 单环境画面；提高并行数可加速生成，但不会影响预览视角。
                  </p>
                </div>

                <div style={{ fontSize: 12, color: '#6b7280' }}>
                  Default Env:{' '}
                  <span style={{ fontFamily: 'monospace' }}>{ISAAC_BLOCK_STACKING_DEFAULT_ENV}</span>
                </div>

                {onImportCustomSeed ? (
                  <div>
                    <SecondaryButton onClick={disabled ? undefined : onImportCustomSeed}>
                      使用自定义 Seed Demo
                    </SecondaryButton>
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
