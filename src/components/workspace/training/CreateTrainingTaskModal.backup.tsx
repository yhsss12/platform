'use client';

import { useEffect, useMemo, useState } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import { getDatasetManifest } from '@/lib/mock/workspaceMockFlowStore';
import { listWorkspaceDataItemsForUi } from '@/lib/workspace/workspaceDataSources';
import type { CreateTrainingTaskInput } from '@/lib/mock/workspaceTrainingMock';
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
  DOWNSTREAM_MODEL_TYPES,
  backendOptionsForDownstream,
  defaultBackendForModelType,
  recommendDataFormat,
  recommendDownstreamModelType,
  resolveTrainingTrainability,
  type DownstreamModelType,
} from '@/lib/workspace/trainingCapabilityUi';
import {
  DEFAULT_TRAINING_DEVICE,
  TRAINING_DEVICE_OPTIONS,
  trainingDeviceSubmitParams,
  type TrainingDeviceValue,
} from '@/lib/workspace/trainingDevice';

export type { CreateTrainingTaskInput };

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
  const [downstreamModelType, setDownstreamModelType] = useState<DownstreamModelType>('Robomimic');
  const [dataFormat, setDataFormat] = useState('HDF5');
  const [trainingBackend, setTrainingBackend] = useState<TrainingBackendRequest>('robomimic_bc');
  const [trainingDevice, setTrainingDevice] = useState<TrainingDeviceValue>(DEFAULT_TRAINING_DEVICE);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [seed, setSeed] = useState(1);
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

  const backendOptions = useMemo(
    () => backendOptionsForDownstream(downstreamModelType, capabilities),
    [downstreamModelType, capabilities]
  );

  const datasetFormatLabel = useMemo(
    () => resolveDatasetDisplayFormat(selectedDatasetOption, downstreamModelType),
    [selectedDatasetOption, downstreamModelType]
  );

  useEffect(() => {
    if (!open) return;
    const preferred =
      initialDataset && datasetOptions.some((d) => d.id === initialDataset)
        ? initialDataset
        : datasetOptions[0]?.id ?? '';
    const option = findTrainingDatasetOption(preferred, dataCenterItems, apiDatasets);
    const recommendedModel = recommendDownstreamModelType(
      capabilities,
      option?.dataFormat,
      option?.modelFormat
    );

    setDataset(preferred);
    setDownstreamModelType(recommendedModel);
    setDataFormat(recommendDataFormat(option?.dataFormat));
    setTrainingBackend(
      defaultBackendForModelType(
        recommendedModel,
        capabilities,
        option ? isDualArmTrainingDatasetOption(option) : false
      )
    );
    setTrainingDevice(DEFAULT_TRAINING_DEVICE);
    setEpochs(5);
    setBatchSize(16);
    setLearningRate(0.0001);
    setSeed(1);
  }, [open, initialDataset, datasetOptions, dataCenterItems, apiDatasets, capabilities]);

  useEffect(() => {
    if (!open || !selectedDatasetOption) return;
    setDataFormat(recommendDataFormat(selectedDatasetOption.dataFormat));
  }, [open, selectedDatasetOption?.id, selectedDatasetOption?.dataFormat]);

  useEffect(() => {
    if (!open) return;
    setTrainingBackend(
      defaultBackendForModelType(
        downstreamModelType,
        capabilities,
        selectedDatasetOption ? isDualArmTrainingDatasetOption(selectedDatasetOption) : false
      )
    );
  }, [downstreamModelType, open, capabilities, selectedDatasetOption]);

  const payload = (): CreateTrainingTaskInput => {
    const deviceParams = trainingDeviceSubmitParams(trainingDevice);
    return {
      dataset,
      downstreamModelType,
      dataFormat,
      trainingBackend,
      trainingDevice,
      epochs,
      batchSize,
      learningRate,
      device: deviceParams.device,
      seed,
      taskName: selectedDatasetOption?.datasetName,
      trainability: resolveTrainingTrainability(
        downstreamModelType,
        capabilities,
        selectedDatasetOption ? isDualArmTrainingDatasetOption(selectedDatasetOption) : false
      ),
    };
  };

  const canStart = Boolean(dataset && selectedDatasetOption && !trainingBackendPending);

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
            创建后可在训练任务列表中查看进度与结果。
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
              <div>训练后端：torch_bc</div>
            </>
          ) : null}
        </div>
      ) : null}

      {trainingBackendPending ? (
        <p style={{ margin: '0 0 12px', fontSize: 13, color: '#b45309', lineHeight: 1.55 }}>
          {DUAL_ARM_TRAINING_BACKEND_PENDING_HINT}
        </p>
      ) : null}

      <div style={{ ...workspaceModalSectionLabel, marginTop: 4 }}>模型与训练配置</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0 16px' }}>
        <div>
          <label style={workspaceModalFieldLabel}>模型类型</label>
          <select
            style={workspaceModalSelectStyle}
            value={downstreamModelType}
            onChange={(e) => setDownstreamModelType(e.target.value as DownstreamModelType)}
          >
            {DOWNSTREAM_MODEL_TYPES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
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
          <label style={workspaceModalFieldLabel}>训练框架</label>
          <select
            style={workspaceModalSelectStyle}
            value={trainingBackend}
            onChange={(e) => setTrainingBackend(e.target.value as TrainingBackendRequest)}
          >
            {backendOptions.map((b) => (
              <option key={b.value} value={b.value} disabled={b.disabled}>
                {b.label}
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
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={workspaceModalSelectStyle}
          />
        </div>
      </div>
    </WorkspaceCenteredModal>
  );
}

export function resolveDatasetManifestForTraining(datasetId: string) {
  return getDatasetManifest(datasetId);
}
