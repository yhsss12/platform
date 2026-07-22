'use client';

import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
} from '@/components/workspace/WorkspaceCenteredModal';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import type { ModelTypeDefinition } from '@/types/modelType';
import { MODEL_TYPE_STATUS_LABELS } from '@/types/modelType';
import {
  baseAlgorithmLabel,
  robotTypeLabel,
  simulatorLabel,
  structureConfigSummary,
} from '@/lib/workspace/modelTypeDisplay';

export function ModelTypeDetailDrawer({
  open,
  modelType,
  onClose,
  onEdit,
  onDisable,
  onDelete,
}: {
  open: boolean;
  modelType: ModelTypeDefinition | null;
  onClose: () => void;
  onEdit?: () => void;
  onDisable?: () => void;
  onDelete?: () => void;
}) {
  if (!open || !modelType) return null;

  const canEdit = !modelType.isBuiltin && modelType.status !== 'deleted';
  const canDelete = !modelType.isBuiltin;

  return (
    <WorkspaceCenteredModal
      open={open}
      title={modelType.name}
      titleId="model-type-detail-title"
      width={640}
      onClose={onClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, flexWrap: 'wrap' }}>
          {canEdit && onEdit ? <SecondaryButton onClick={onEdit}>编辑</SecondaryButton> : null}
          {modelType.status === 'available' && onDisable ? (
            <SecondaryButton onClick={onDisable}>禁用</SecondaryButton>
          ) : null}
          {canDelete && onDelete ? <SecondaryButton onClick={onDelete}>删除</SecondaryButton> : null}
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
      }
    >
      <div style={workspaceModalSectionLabel}>基础信息</div>
      <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.7 }}>
        <div>模型标识：{modelType.modelTypeId}</div>
        <div>基础算法：{baseAlgorithmLabel(modelType.baseAlgorithm)}</div>
        <div>适配器：{modelType.adapterKey}</div>
        <div>仿真环境：{simulatorLabel(modelType.simulator)}</div>
        <div>机器人类型：{robotTypeLabel(modelType.robotType)}</div>
        <div>状态：{MODEL_TYPE_STATUS_LABELS[modelType.status] ?? modelType.status}</div>
        <div>更新时间：{modelType.updatedAt?.slice(0, 10) ?? '—'}</div>
        {modelType.isBuiltin ? <div>内置模型类型（不建议删除）</div> : null}
      </div>

      {modelType.description ? (
        <>
          <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>描述</div>
          <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{modelType.description}</p>
        </>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>结构参数摘要</div>
      <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>{structureConfigSummary(modelType)}</p>

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>训练默认值</div>
      <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.7 }}>
        <div>Epochs：{String(modelType.trainingDefaults.default_epochs ?? '—')}</div>
        <div>Batch Size：{String(modelType.trainingDefaults.default_batch_size ?? '—')}</div>
        <div>Learning Rate：{String(modelType.trainingDefaults.default_learning_rate ?? '—')}</div>
        <div>Seed Strategy：{String(modelType.trainingDefaults.default_seed_strategy ?? '—')}</div>
      </div>

      {modelType.tags.length > 0 ? (
        <>
          <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>标签</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {modelType.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  fontSize: 12,
                  padding: '2px 8px',
                  borderRadius: 999,
                  background: '#f1f5f9',
                  color: '#475569',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        </>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 16 }}>完整结构配置</div>
      <pre
        style={{
          margin: 0,
          padding: 12,
          borderRadius: 8,
          background: '#f8fafc',
          fontSize: 12,
          overflow: 'auto',
        }}
      >
        {JSON.stringify(modelType.structureConfig, null, 2)}
      </pre>
    </WorkspaceCenteredModal>
  );
}
