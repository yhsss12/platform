'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  WorkspaceCenteredModal,
  WorkspaceModalFieldGrid,
  workspaceFormFieldClassName,
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { workspaceModalFieldErrorStyle } from '@/components/workspace/training/TrainingAdvancedSettingsSection';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import { importPretrainedModelAsset, type ImportModelAssetResponse } from '@/lib/api/modelAssetsClient';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import type { Dataset } from '@/types/benchmark';
import { canOpenDatasetTraining } from '@/lib/workspace/datasetTrainingAccess';
import { resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';
import { ModelAssetFileUploadZone } from '@/components/workspace/resources/ModelAssetFileUploadZone';

const CHECKPOINT_MAX_BYTES = 500 * 1024 * 1024;

const MODEL_TYPE_OPTIONS = [
  { value: 'diffusion_policy', label: 'Diffusion Policy' },
  { value: 'robomimic_bc', label: 'Robomimic BC' },
  { value: 'act', label: 'ACT' },
] as const;

const TASK_TYPE_OPTIONS = [
  { value: 'cable_threading', label: '线缆穿杆' },
  { value: 'dual_arm_cable_manipulation', label: '双臂线缆协作' },
  { value: 'isaac_block_stacking', label: '物块堆叠' },
] as const;

export function ImportPretrainedModelModal({
  open,
  onClose,
  onImported,
}: {
  open: boolean;
  onClose: () => void;
  onImported: (result: ImportModelAssetResponse) => void;
}) {
  const [modelName, setModelName] = useState('');
  const [modelType, setModelType] = useState<string>(MODEL_TYPE_OPTIONS[0].value);
  const [taskType, setTaskType] = useState<string>(TASK_TYPE_OPTIONS[0].value);
  const [datasetId, setDatasetId] = useState('');
  const [note, setNote] = useState('');
  const [checkpointFile, setCheckpointFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setModelName('');
    setModelType(MODEL_TYPE_OPTIONS[0].value);
    setTaskType(TASK_TYPE_OPTIONS[0].value);
    setDatasetId('');
    setNote('');
    setCheckpointFile(null);
    setMetadataFile(null);
    setError(null);
    setFileError(null);
    setLoadingDatasets(true);
    void listWorkspaceDatasets()
      .then((response) => setDatasets(response.datasets.filter((item) => canOpenDatasetTraining(item))))
      .catch(() => setDatasets([]))
      .finally(() => setLoadingDatasets(false));
  }, [open]);

  const datasetOptions = useMemo(
    () =>
      datasets.filter((dataset) => {
        if (taskType === 'cable_threading') {
          return dataset.sourceJobId?.startsWith('ct_gen_') || dataset.taskType === 'cable_threading';
        }
        if (taskType === 'dual_arm_cable_manipulation') {
          return dataset.sourceJobId?.startsWith('dac_gen_') || dataset.taskType === 'dual_arm_cable_manipulation';
        }
        if (taskType === 'isaac_block_stacking') {
          return dataset.taskType === 'isaac_block_stacking' || dataset.simulatorBackend === 'isaac_lab';
        }
        return true;
      }),
    [datasets, taskType]
  );

  useEffect(() => {
    if (!open) return;
    if (datasetId && !datasetOptions.some((item) => item.id === datasetId)) {
      setDatasetId(datasetOptions[0]?.id ?? '');
    } else if (!datasetId && datasetOptions[0]) {
      setDatasetId(datasetOptions[0].id);
    }
  }, [open, datasetId, datasetOptions]);

  const canSubmit = Boolean(modelName.trim() && datasetId && checkpointFile && !submitting);

  const handleCheckpointChange = (file: File | null) => {
    setFileError(null);
    if (file && file.size > CHECKPOINT_MAX_BYTES) {
      setFileError('checkpoint 文件不能超过 500MB');
      setCheckpointFile(null);
      return;
    }
    setCheckpointFile(file);
  };

  const handleSubmit = async () => {
    if (!checkpointFile || !canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('modelName', modelName.trim());
      form.append('modelType', modelType);
      form.append('taskType', taskType);
      form.append('datasetId', datasetId);
      form.append('checkpoint', checkpointFile);
      if (metadataFile) form.append('metadata', metadataFile);
      if (note.trim()) form.append('note', note.trim());
      const result = await importPretrainedModelAsset(form);
      onImported(result);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : '导入失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <WorkspaceCenteredModal
      open={open}
      title="导入预训练模型"
      titleId="import-pretrained-model-title"
      width={760}
      onClose={() => {
        if (!submitting) onClose();
      }}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={submitting ? undefined : onClose}>取消</SecondaryButton>
          <PrimaryButton onClick={() => void handleSubmit()} disabled={!canSubmit}>
            {submitting ? '导入中…' : '确认导入'}
          </PrimaryButton>
        </div>
      }
    >
      <p className="ws-import-modal-desc">
        上传外部 checkpoint，并通过参考数据集校验 observation schema 与 action 维度，校验通过后注册为模型资产。
      </p>

      <div className="ws-form-section" style={{ marginBottom: 16 }}>
        <h3 className="ws-form-section-title">基本信息</h3>
        <WorkspaceModalFieldGrid>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={workspaceModalFieldLabel}>模型名称</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="例如：线缆穿杆预训练 Final"
              className={workspaceFormFieldClassName}
              style={workspaceModalSelectStyle}
            />
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>模型类型</label>
            <select
              className={workspaceFormFieldClassName}
              style={workspaceModalSelectStyle}
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
            >
              {MODEL_TYPE_OPTIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label style={workspaceModalFieldLabel}>适用任务 / 场景</label>
            <select
              className={workspaceFormFieldClassName}
              style={workspaceModalSelectStyle}
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
            >
              {TASK_TYPE_OPTIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={workspaceModalFieldLabel}>参考数据集</label>
            <select
              className={workspaceFormFieldClassName}
              style={workspaceModalSelectStyle}
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={loadingDatasets || datasetOptions.length === 0}
            >
              {loadingDatasets ? (
                <option value="">加载数据集…</option>
              ) : datasetOptions.length === 0 ? (
                <option value="">暂无可用数据集</option>
              ) : (
                datasetOptions.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} · {resolveDatasetSourceTaskLabel(dataset)} · {dataset.episodeCount} 条
                  </option>
                ))
              )}
            </select>
          </div>
        </WorkspaceModalFieldGrid>
      </div>

      <div className="ws-form-section" style={{ marginBottom: 16 }}>
        <h3 className="ws-form-section-title">模型文件</h3>
        <div style={{ marginBottom: 14 }}>
          <label style={workspaceModalFieldLabel}>checkpoint 文件</label>
          <ModelAssetFileUploadZone
            accept=".pt,.pth,.ckpt"
            emptyTitle="选择 checkpoint 文件"
            emptySubtitle="支持 .pt / .pth / .ckpt，最大 500MB"
            file={checkpointFile}
            onFileChange={handleCheckpointChange}
            onInvalidFile={(message) => setFileError(message)}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>metadata / config 文件（可选）</label>
          <ModelAssetFileUploadZone
            accept=".json,.yaml,.yml"
            emptyTitle="选择 metadata / config 文件（可选）"
            emptySubtitle="支持 .json / .yaml / .yml"
            file={metadataFile}
            onFileChange={(file) => {
              setFileError(null);
              setMetadataFile(file);
            }}
            onInvalidFile={(message) => setFileError(message)}
          />
        </div>
      </div>

      <div className="ws-form-section">
        <h3 className="ws-form-section-title">校验与备注</h3>
        <p className="ws-form-hint-card">
          系统将使用参考数据集的 observation schema、action_dim、image_keys 与 low_dim_keys 校验 checkpoint 结构。
        </p>
        <div style={{ marginTop: 14 }}>
          <label style={workspaceModalFieldLabel}>备注</label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={3}
            placeholder="可选：记录来源、版本或用途"
            className={workspaceFormFieldClassName}
            style={{ ...workspaceModalSelectStyle, marginBottom: 0, minHeight: 72 }}
          />
        </div>
      </div>

      {fileError ? <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 12 }}>{fileError}</p> : null}
      {error ? <p style={{ ...workspaceModalFieldErrorStyle, marginTop: 12 }}>{error}</p> : null}
    </WorkspaceCenteredModal>
  );
}
