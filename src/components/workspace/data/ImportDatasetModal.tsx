'use client';

import { useRef, useState } from 'react';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import { importWorkspaceDataset } from '@/lib/api/datasetsClient';
import type { Dataset } from '@/types/benchmark';

const DATA_SOURCE_OPTIONS = [
  { value: 'real_collection', label: '真实采集' },
  { value: 'simulation_export', label: '仿真导出' },
  { value: 'public_dataset', label: '外部公开数据' },
  { value: 'other', label: '其他' },
] as const;

const TASK_TYPE_OPTIONS = [
  { value: 'cable_threading', label: '线缆穿杆' },
  { value: 'dual_arm_cable', label: '线缆整理' },
  { value: 'stack_cube', label: 'Stack Cube' },
  { value: 'custom', label: '自定义' },
] as const;

const ROBOT_TYPE_OPTIONS = [
  { value: 'fr3', label: 'FR3' },
  { value: 'dual_fr3', label: '双臂 FR3' },
  { value: 'realman', label: 'Realman' },
  { value: 'other', label: '其他' },
] as const;

interface ImportDatasetModalProps {
  open: boolean;
  onClose: () => void;
  onImported: (dataset: Dataset) => void;
}

export function ImportDatasetModal({ open, onClose, onImported }: ImportDatasetModalProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState('');
  const [dataSource, setDataSource] = useState<string>(DATA_SOURCE_OPTIONS[0].value);
  const [taskType, setTaskType] = useState<string>(TASK_TYPE_OPTIONS[0].value);
  const [robotType, setRobotType] = useState<string>(ROBOT_TYPE_OPTIONS[0].value);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resetForm = () => {
    setName('');
    setDataSource(DATA_SOURCE_OPTIONS[0].value);
    setTaskType(TASK_TYPE_OPTIONS[0].value);
    setRobotType(ROBOT_TYPE_OPTIONS[0].value);
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleClose = () => {
    if (submitting) return;
    setError(null);
    onClose();
  };

  const handleFileChange = (file: File | null) => {
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.hdf5') && !lower.endsWith('.h5')) {
      setError('仅支持 .hdf5 / .h5 文件');
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }
    setError(null);
    setSelectedFile(file);
    if (!name.trim()) {
      const base = file.name.replace(/\.(hdf5|h5)$/i, '');
      setName(base);
    }
  };

  const handleSubmit = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) {
      setError('请填写数据集名称');
      return;
    }
    if (!selectedFile) {
      setError('请选择 HDF5 文件');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('name', trimmedName);
      form.append('dataSource', dataSource);
      form.append('taskType', taskType);
      form.append('robotType', robotType);
      form.append('file', selectedFile);

      const response = await importWorkspaceDataset(form);
      onImported(response.dataset);
      resetForm();
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
      title="导入数据集"
      titleId="import-dataset-title"
      onClose={handleClose}
      width={560}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={() => { if (!submitting) handleClose(); }}>
            取消
          </SecondaryButton>
          <PrimaryButton onClick={() => void handleSubmit()} disabled={submitting}>
            {submitting ? '上传解析中…' : '开始导入'}
          </PrimaryButton>
        </div>
      }
    >
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.5 }}>
        通过浏览器选择本地 HDF5 文件上传。文件将保存到服务器并由后端自动解析结构、识别训练字段并登记到数据中心。
      </p>

      <label style={workspaceModalFieldLabel}>
        数据集名称
        <input
          style={workspaceModalSelectStyle}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="例如：线缆穿杆真机采集_01"
          disabled={submitting}
        />
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        数据来源
        <select
          style={workspaceModalSelectStyle}
          value={dataSource}
          onChange={(e) => setDataSource(e.target.value)}
          disabled={submitting}
        >
          {DATA_SOURCE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        任务类型
        <select
          style={workspaceModalSelectStyle}
          value={taskType}
          onChange={(e) => setTaskType(e.target.value)}
          disabled={submitting}
        >
          {TASK_TYPE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        机器人类型
        <select
          style={workspaceModalSelectStyle}
          value={robotType}
          onChange={(e) => setRobotType(e.target.value)}
          disabled={submitting}
        >
          {ROBOT_TYPE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </label>

      <label style={{ ...workspaceModalFieldLabel, marginTop: 12 }}>
        HDF5 文件
        <input
          ref={fileInputRef}
          type="file"
          accept=".hdf5,.h5"
          style={workspaceModalSelectStyle}
          disabled={submitting}
          onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
        />
      </label>

      {selectedFile ? (
        <div style={{ marginTop: 8, fontSize: 12, color: '#6b7280' }}>
          已选择：{selectedFile.name}（{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB）
        </div>
      ) : null}

      {error ? (
        <div style={{ marginTop: 12, fontSize: 13, color: '#b91c1c' }}>{error}</div>
      ) : null}
    </WorkspaceCenteredModal>
  );
}
