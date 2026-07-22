'use client';

import type { IsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';
import {
  formatPhysicsBackendLabel,
  formatSimulatorBackendLabel,
  type TaskTemplateCapabilityProfile,
} from '@/lib/workspace/taskTemplateCapabilities';
import { workspaceModalFieldLabel, workspaceModalSelectStyle } from '@/components/workspace/WorkspaceCenteredModal';

const SUPPORTED_ITEMS = [
  'HDF5 Demo 导入',
  'Demo 回放',
  'stdout / stderr 日志',
  'replay job 状态',
  'replay.mp4（如果可生成）',
] as const;

const PLANNED_ITEMS = [
  '自动生成 HDF5 数据',
  'Robomimic BC 训练',
  'trained model evaluation',
] as const;

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
          color: '#374151',
          cursor: 'default',
        }}
      />
    </div>
  );
}

export function IsaacStackingGenerateCapabilityPanel({
  capabilities,
  runtime,
  runtimeLoading,
}: {
  capabilities: TaskTemplateCapabilityProfile;
  runtime: IsaacLabRuntimeStatus | null;
  runtimeLoading: boolean;
}) {
  const runtimeLabel = runtimeLoading
    ? '检测中…'
    : runtime?.available
      ? '可用'
      : runtime?.configured
        ? '已配置（待验证）'
        : '未配置';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div
        style={{
          padding: '16px 18px',
          borderRadius: 12,
          border: '1px solid #bfdbfe',
          background: 'linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%)',
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 10 }}>
          <span style={{ fontSize: 17, fontWeight: 600, color: '#111827' }}>物块堆叠</span>
          <span
            style={{
              fontSize: 11,
              fontWeight: 500,
              padding: '3px 10px',
              borderRadius: 9999,
              backgroundColor: '#dbeafe',
              border: '1px solid #93c5fd',
              color: '#1d4ed8',
            }}
          >
            Isaac Lab · PhysX · HDF5 回放已接入
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 13, color: '#374151', lineHeight: 1.65 }}>
          当前已支持 Isaac Lab HDF5 Demo 导入与回放验证；自动仿真数据生成尚未接入。请先在数据中心使用「导入」登记
          物块堆叠 HDF5 demo，或配置 Isaac Lab 运行节点后进入后续数据生成接入流程。
        </p>
      </div>

      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>运行状态</div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0 16px',
          }}
        >
          <ReadonlyField label="Isaac Lab Runtime" value={runtimeLabel} />
          <ReadonlyField label="Default Env" value={capabilities.defaultEnv ?? '—'} />
          <ReadonlyField label="Dataset Format" value={capabilities.datasetFormat ?? 'HDF5'} />
          <ReadonlyField
            label="当前能力"
            value="导入 Demo、回放验证"
          />
          <div style={{ gridColumn: '1 / -1' }}>
            <ReadonlyField
              label="待接入能力"
              value="自动数据生成、训练、模型评测"
            />
          </div>
        </div>
      </div>

      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>仿真后端</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
          <ReadonlyField
            label="仿真后端"
            value={formatSimulatorBackendLabel(capabilities.simulatorBackend)}
          />
          <ReadonlyField
            label="物理引擎"
            value={formatPhysicsBackendLabel(capabilities.physicsBackend ?? 'physx')}
          />
          <ReadonlyField label="机器人" value={capabilities.robotLabel} />
        </div>
      </div>

      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>当前已支持</div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
          {SUPPORTED_ITEMS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>待接入</div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#6b7280', lineHeight: 1.7 }}>
          {PLANNED_ITEMS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <p style={{ margin: 0, fontSize: 12, color: '#6b7280', lineHeight: 1.6 }}>
        该任务当前支持导入 Isaac Lab HDF5 Demo 并进行回放验证；自动仿真数据生成将在配置 Isaac Lab 运行节点后接入。
      </p>
    </div>
  );
}
