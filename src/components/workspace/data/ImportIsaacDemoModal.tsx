'use client';

import { useState } from 'react';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import { importIsaacLabDemoDataset } from '@/lib/api/isaacLabClient';
import { buildDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import { ISAAC_BLOCK_STACKING_DEFAULT_ENV } from '@/lib/workspace/isaacBlockStacking';
import type { Dataset } from '@/types/benchmark';

export interface ImportIsaacDemoPayload {
  datasetFile: string;
  displayName: string;
  taskId: string;
}

interface ImportIsaacDemoModalProps {
  open: boolean;
  onClose: () => void;
  onImported: (dataset: Dataset) => void;
}

export function ImportIsaacDemoModal({ open, onClose, onImported }: ImportIsaacDemoModalProps) {
  const [datasetFile, setDatasetFile] = useState('');
  const [displayName, setDisplayName] = useState(() =>
    buildDatasetDisplayName({ taskType: 'block_stacking' })
  );
  const [taskId, setTaskId] = useState(ISAAC_BLOCK_STACKING_DEFAULT_ENV);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClose = () => {
    if (submitting) return;
    setError(null);
    onClose();
  };

  const handleSubmit = async () => {
    const file = datasetFile.trim();
    const name = displayName.trim();
    if (!file || !name) {
      setError('请填写 HDF5 路径与数据集名称');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const response = await importIsaacLabDemoDataset({
        datasetFile: file,
        displayName: name,
        taskId: taskId.trim() || ISAAC_BLOCK_STACKING_DEFAULT_ENV,
      });
      onImported(response.dataset);
      setDatasetFile('');
      setDisplayName(buildDatasetDisplayName({ taskType: 'block_stacking' }));
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
      title="导入 Isaac Lab HDF5 Demo"
      titleId="import-isaac-demo-title"
      onClose={handleClose}
      width={520}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={() => { if (!submitting) handleClose(); }}>
            取消
          </SecondaryButton>
          <PrimaryButton onClick={() => void handleSubmit()} disabled={submitting}>
            {submitting ? '导入中…' : '导入'}
          </PrimaryButton>
        </div>
      }
    >
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.5 }}>
        登记物块堆叠等 Isaac Lab 导出的 HDF5 演示文件。文件必须存在于服务器可访问路径，登记后可在列表中一键回放。
      </p>

      <label style={workspaceModalFieldLabel}>
        HDF5 文件路径
        <input
          style={workspaceModalSelectStyle}
          value={datasetFile}
          onChange={(e) => setDatasetFile(e.target.value)}
          placeholder="/path/to/dataset.hdf5"
          disabled={submitting}
        />
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        数据集名称
        <input
          style={workspaceModalSelectStyle}
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          disabled={submitting}
        />
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        Isaac 任务 ID
        <input
          style={workspaceModalSelectStyle}
          value={taskId}
          onChange={(e) => setTaskId(e.target.value)}
          disabled={submitting}
        />
      </label>

      {error ? (
        <div style={{ marginTop: 12, fontSize: 13, color: '#b91c1c' }}>{error}</div>
      ) : null}
    </WorkspaceCenteredModal>
  );
}
