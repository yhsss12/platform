'use client';

import { useEffect, useMemo, useState } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import {
  buildDatasetFromImport,
  getDatasetSchema,
  type DatasetFieldMapping,
  type Hdf5SchemaField,
} from '@/lib/api/datasetsClient';
import { BuildSourceDatasetSelectDialog } from '@/components/workspace/data/BuildSourceDatasetSelectDialog';
import {
  filterBuildSourceDatasets,
  formatBuildSourceDatasetCreatedAt,
  resolveBuildSourceDatasetDisplayName,
} from '@/lib/workspace/buildSourceDatasetPicker';
import {
  resolveDatasetCountText,
  resolveDatasetSizeText,
  resolveDatasetSourceLabel,
} from '@/lib/workspace/datasetDisplay';
import { resolveDatasetFormatLabel } from '@/lib/workspace/taskTemplateMapping';
import '@/components/workspace/workspaceModalForm.css';
import type { Dataset } from '@/types/benchmark';

const AUTO_FIELD_DETECT_HINT = '未能自动识别训练所需字段，请打开高级配置手动指定。';

const TASK_TYPE_OPTIONS = [
  { value: 'cable_threading', label: '线缆穿杆' },
  { value: 'dual_arm_cable', label: '线缆整理' },
  { value: 'stack_cube', label: 'Stack Cube' },
  { value: 'custom', label: '自定义' },
] as const;

const summaryCardStyle: React.CSSProperties = {
  padding: '12px 14px',
  borderRadius: 10,
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  fontSize: 13,
  color: '#1e293b',
  lineHeight: 1.55,
};

const fieldHintStyle: React.CSSProperties = {
  marginTop: 4,
  fontSize: 12,
  color: '#b45309',
  lineHeight: 1.45,
};

const advancedHintStyle: React.CSSProperties = {
  margin: '0 0 10px',
  fontSize: 12,
  color: '#64748b',
  lineHeight: 1.55,
};

const advancedGroupTitleStyle: React.CSSProperties = {
  margin: '12px 0 8px',
  fontSize: 12,
  fontWeight: 600,
  color: '#64748b',
};

type ManualFieldKey = keyof DatasetFieldMapping;

type ManualFieldConfig = {
  key: ManualFieldKey;
  label: string;
  placeholder: string;
};

const PRIMARY_MANUAL_FIELDS: ManualFieldConfig[] = [
  { key: 'action', label: '专家动作', placeholder: '自动识别' },
  { key: 'qpos', label: '机器人状态', placeholder: '自动识别' },
  { key: 'image', label: '视觉图像', placeholder: '自动识别' },
];

const EXTRA_MANUAL_FIELDS: ManualFieldConfig[] = [
  { key: 'qvel', label: '关节速度', placeholder: '自动识别' },
  { key: 'done', label: '结束标记', placeholder: '自动识别' },
];

function formatFieldOption(field: Hdf5SchemaField): string {
  const path = field.path.startsWith('/') ? field.path : `/${field.path}`;
  const dtype = field.dtype ?? 'unknown';
  const shape = Array.isArray(field.shape) ? field.shape.join(', ') : '';
  return `${path}    ${dtype} [${shape}]`;
}

function defaultOutputName(source: Dataset): string {
  const base = source.displayName?.trim() || source.name?.trim() || source.id;
  return `${base}_built`;
}

function collectExistingDatasetNames(allDatasets: Dataset[]): Set<string> {
  const names = new Set<string>();
  for (const dataset of allDatasets) {
    for (const raw of [dataset.displayName, dataset.name]) {
      const trimmed = raw?.trim();
      if (trimmed) names.add(trimmed.toLowerCase());
    }
  }
  return names;
}

function extractManualFieldMapping(fieldMapping: DatasetFieldMapping): DatasetFieldMapping | undefined {
  const manual: DatasetFieldMapping = {};
  let hasManual = false;
  for (const key of ['action', 'qpos', 'image', 'qvel', 'done'] as ManualFieldKey[]) {
    const value = fieldMapping[key]?.trim();
    if (value) {
      manual[key] = value;
      hasManual = true;
    }
  }
  return hasManual ? manual : undefined;
}

function ManualFieldSelect({
  config,
  value,
  schemaFields,
  disabled,
  onChange,
}: {
  config: ManualFieldConfig;
  value: string | null | undefined;
  schemaFields: Hdf5SchemaField[];
  disabled: boolean;
  onChange: (key: ManualFieldKey, value: string) => void;
}) {
  return (
    <div>
      <label style={workspaceModalFieldLabel}>{config.label}</label>
      <select
        style={workspaceModalSelectStyle}
        value={value ?? ''}
        onChange={(e) => onChange(config.key, e.target.value)}
        disabled={disabled}
      >
        <option value="">{config.placeholder}</option>
        {schemaFields.map((field) => (
          <option key={field.path} value={field.path}>
            {formatFieldOption(field)}
          </option>
        ))}
      </select>
    </div>
  );
}

export function BuildDatasetModal({
  open,
  onClose,
  onBuilt,
  datasets,
  preselectedSourceDatasetId,
}: {
  open: boolean;
  onClose: () => void;
  onBuilt: (dataset: Dataset) => void;
  datasets: Dataset[];
  preselectedSourceDatasetId?: string | null;
}) {
  const buildableSources = useMemo(
    () => filterBuildSourceDatasets(datasets),
    [datasets]
  );

  const existingNames = useMemo(() => collectExistingDatasetNames(datasets), [datasets]);

  const [sourceDatasetId, setSourceDatasetId] = useState('');
  const [sourcePickerOpen, setSourcePickerOpen] = useState(false);
  const [outputName, setOutputName] = useState('');
  const [taskType, setTaskType] = useState('custom');
  const [fieldMapping, setFieldMapping] = useState<DatasetFieldMapping>({
    action: '',
    qpos: '',
    image: '',
    qvel: '',
    done: '',
  });
  const [schemaFields, setSchemaFields] = useState<Hdf5SchemaField[]>([]);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const selectedSource = useMemo(
    () => buildableSources.find((d) => d.id === sourceDatasetId) ?? null,
    [buildableSources, sourceDatasetId]
  );

  const trimmedOutputName = outputName.trim();
  const nameEmpty = trimmedOutputName.length === 0;
  const nameDuplicate =
    trimmedOutputName.length > 0 && existingNames.has(trimmedOutputName.toLowerCase());

  const canSubmit = Boolean(selectedSource) && !nameEmpty && !submitting;

  const sourceTriggerLabel = selectedSource
    ? resolveBuildSourceDatasetDisplayName(selectedSource)
    : '请选择已导入的 HDF5 数据集';

  useEffect(() => {
    if (!open) return;
    const preferred =
      preselectedSourceDatasetId &&
      buildableSources.some((d) => d.id === preselectedSourceDatasetId)
        ? preselectedSourceDatasetId
        : '';
    const source = preferred
      ? buildableSources.find((d) => d.id === preferred) ?? null
      : null;
    setSourceDatasetId(source?.id ?? '');
    setOutputName(source ? defaultOutputName(source) : '');
    setTaskType(source?.taskType?.trim() || 'custom');
    setFieldMapping({ action: '', qpos: '', image: '', qvel: '', done: '' });
    setSchemaFields([]);
    setSchemaError(null);
    setSubmitError(null);
    setAdvancedOpen(false);
    setSourcePickerOpen(false);
  }, [open, preselectedSourceDatasetId, buildableSources]);

  const handleSourceConfirm = (dataset: Dataset) => {
    setSourceDatasetId(dataset.id);
    setOutputName(defaultOutputName(dataset));
    setTaskType(dataset.taskType?.trim() || 'custom');
    setSourcePickerOpen(false);
  };

  useEffect(() => {
    if (!open || !advancedOpen || !sourceDatasetId) {
      if (!advancedOpen) setSchemaFields([]);
      return;
    }
    let cancelled = false;
    setSchemaLoading(true);
    setSchemaError(null);
    void getDatasetSchema(sourceDatasetId)
      .then((schema) => {
        if (cancelled) return;
        setSchemaFields(schema.fields ?? []);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setSchemaFields([]);
        setSchemaError(err instanceof Error ? err.message : '读取数据字段失败');
      })
      .finally(() => {
        if (!cancelled) setSchemaLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, advancedOpen, sourceDatasetId]);

  const handleManualFieldChange = (key: ManualFieldKey, value: string) => {
    setFieldMapping((prev) => ({
      ...prev,
      [key]: value.trim(),
    }));
  };

  const handleBuild = async () => {
    if (!selectedSource || !canSubmit) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const manualMapping = extractManualFieldMapping(fieldMapping);
      const response = await buildDatasetFromImport({
        sourceDatasetId: selectedSource.id,
        outputName: trimmedOutputName,
        taskType,
        targetFormat: 'standard_hdf5',
        auto: !manualMapping,
        fieldMapping: manualMapping,
      });
      if (response.dataset) {
        onBuilt(response.dataset);
      }
      onClose();
    } catch (err) {
      const message = err instanceof Error ? err.message : '构建失败';
      setSubmitError(message);
      if (message.includes('未能自动识别') || message.includes('高级配置')) {
        setAdvancedOpen(true);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <WorkspaceCenteredModal
      open={open}
      title="构建数据集"
      titleId="build-dataset-title"
      width={640}
      onClose={() => {
        if (!submitting) onClose();
      }}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <SecondaryButton onClick={onClose} disabled={submitting}>
            取消
          </SecondaryButton>
          <PrimaryButton onClick={() => void handleBuild()} disabled={!canSubmit}>
            {submitting ? '构建中…' : '开始构建'}
          </PrimaryButton>
        </div>
      }
    >
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
        选择已导入的 HDF5 数据，生成可用于训练的数据集。
      </p>

      {buildableSources.length === 0 ? (
        <div style={{ fontSize: 13, color: '#b45309' }}>
          暂无可构建的数据集，请先通过「导入」上传 HDF5 文件。
        </div>
      ) : (
        <>
          <div>
            <label style={workspaceModalFieldLabel}>源数据集</label>
            <div className="ws-dataset-selector">
              <button
                type="button"
                className={`ws-dataset-selector-value${selectedSource ? '' : ' is-placeholder'}`}
                title={sourceTriggerLabel}
                disabled={submitting}
                onClick={() => setSourcePickerOpen(true)}
              >
                {sourceTriggerLabel}
              </button>
              <button
                type="button"
                className="ws-dataset-selector-action"
                disabled={submitting}
                onClick={() => setSourcePickerOpen(true)}
              >
                选择
              </button>
            </div>
            {selectedSource ? (
              <div style={{ ...summaryCardStyle, marginTop: 10 }}>
                <div>
                  <span style={{ color: '#64748b' }}>数据来源：</span>
                  {resolveDatasetSourceLabel(selectedSource)}
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>数据格式：</span>
                  {resolveDatasetFormatLabel(selectedSource)}
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>数据数量：</span>
                  {resolveDatasetCountText(selectedSource)}
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>数据大小：</span>
                  {resolveDatasetSizeText(selectedSource)}
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>创建时间：</span>
                  {formatBuildSourceDatasetCreatedAt(selectedSource.createdAt)}
                </div>
              </div>
            ) : null}
          </div>

          <BuildSourceDatasetSelectDialog
            open={sourcePickerOpen}
            datasets={datasets}
            selectedDatasetId={sourceDatasetId || null}
            onConfirm={handleSourceConfirm}
            onCancel={() => setSourcePickerOpen(false)}
          />

          <div style={{ marginTop: 20 }}>
            <div style={workspaceModalSectionLabel}>构建配置</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={workspaceModalFieldLabel}>构建后名称</label>
                <input
                  type="text"
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  style={workspaceModalSelectStyle}
                  disabled={!selectedSource || submitting}
                  placeholder="请输入构建后的数据集名称"
                />
                {nameEmpty && selectedSource ? (
                  <div style={fieldHintStyle}>名称不能为空</div>
                ) : null}
                {nameDuplicate ? (
                  <div style={fieldHintStyle}>已存在同名数据集，建议修改名称以避免混淆</div>
                ) : null}
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>目标任务</label>
                <select
                  style={workspaceModalSelectStyle}
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value)}
                  disabled={!selectedSource || submitting}
                >
                  {TASK_TYPE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={workspaceModalFieldLabel}>输出格式</label>
                <select style={workspaceModalSelectStyle} value="hdf5" disabled>
                  <option value="hdf5">HDF5</option>
                </select>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 20, borderTop: '1px solid #f3f4f6', paddingTop: 12 }}>
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              disabled={submitting}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                padding: 0,
                border: 'none',
                background: 'none',
                cursor: submitting ? 'not-allowed' : 'pointer',
                fontSize: 13,
                fontWeight: 600,
                color: '#334155',
              }}
            >
              <span>高级配置</span>
              <span style={{ fontWeight: 400, fontSize: 12, color: '#6b7280' }}>
                {advancedOpen ? '收起' : '展开'}
              </span>
            </button>
            {advancedOpen ? (
              <div style={{ marginTop: 10 }}>
                <p style={advancedHintStyle}>
                  系统将自动识别训练字段。识别失败时，可在此手动指定。
                </p>
                {schemaLoading ? (
                  <div style={{ fontSize: 13, color: '#64748b' }}>正在加载可选字段…</div>
                ) : schemaError ? (
                  <div style={{ fontSize: 13, color: '#b45309' }}>{schemaError}</div>
                ) : schemaFields.length === 0 ? (
                  <div style={{ fontSize: 13, color: '#64748b' }}>暂无可选字段</div>
                ) : (
                  <>
                    {PRIMARY_MANUAL_FIELDS.map((config) => (
                      <ManualFieldSelect
                        key={config.key}
                        config={config}
                        value={fieldMapping[config.key]}
                        schemaFields={schemaFields}
                        disabled={!selectedSource || submitting}
                        onChange={handleManualFieldChange}
                      />
                    ))}
                    <div style={advancedGroupTitleStyle}>更多字段</div>
                    {EXTRA_MANUAL_FIELDS.map((config) => (
                      <ManualFieldSelect
                        key={config.key}
                        config={config}
                        value={fieldMapping[config.key]}
                        schemaFields={schemaFields}
                        disabled={!selectedSource || submitting}
                        onChange={handleManualFieldChange}
                      />
                    ))}
                  </>
                )}
              </div>
            ) : null}
          </div>
        </>
      )}

      {submitError ? (
        <div style={{ marginTop: 12, fontSize: 13, color: '#b45309' }}>
          {submitError.includes('未能自动识别') ? AUTO_FIELD_DETECT_HINT : submitError}
        </div>
      ) : null}
    </WorkspaceCenteredModal>
  );
}
