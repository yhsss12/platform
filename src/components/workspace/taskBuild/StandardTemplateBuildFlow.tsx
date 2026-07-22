'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  TaskBuildBackLink,
  TaskBuildField,
  TaskBuildStepper,
} from '@/components/workspace/taskBuild/TaskBuildUi';
import { listTaskTemplates, type TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import { listModelAssets } from '@/lib/api/modelAssetsClient';
import {
  CABLE_MANIPULATION_FAMILY,
  datasetMatchesTaskTemplate,
  EVALUATION_MODE_LABELS,
  modelAssetMatchesTaskTemplate,
} from '@/lib/workspace/taskTemplateMapping';
import {
  makeTaskBuildConfigId,
  saveTaskBuildConfig,
} from '@/lib/workspace/taskBuildConfigStore';
import {
  buildDataGenerateHref,
  buildEvaluationCreateHref,
  buildTrainingCreateHref,
} from '@/lib/workspace/taskBuildNavigation';
import type { Dataset, ModelAsset, TaskBuildConfig } from '@/types/benchmark';

const STEPS = ['任务族', '任务模板', '场景与策略', '数据与模型', '完成'];

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: 14,
  borderRadius: 8,
  border: '1px solid #d1d5db',
  boxSizing: 'border-box',
};

const TASK_FAMILIES: Array<{ id: string; label: string; description: string }> = [
  {
    id: CABLE_MANIPULATION_FAMILY,
    label: '线缆操作任务族',
    description: '单臂线缆穿杆与双臂线缆操控标准模板',
  },
];

export function StandardTemplateBuildFlow() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const presetTemplateId = searchParams.get('taskTemplateId') ?? undefined;

  const [step, setStep] = useState(1);
  const [templates, setTemplates] = useState<TaskTemplateDto[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modelAssets, setModelAssets] = useState<ModelAsset[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [taskFamily, setTaskFamily] = useState(CABLE_MANIPULATION_FAMILY);
  const [selectedTemplateId, setSelectedTemplateId] = useState('');
  const [configName, setConfigName] = useState('');
  const [linkedDatasetId, setLinkedDatasetId] = useState('');
  const [linkedModelAssetId, setLinkedModelAssetId] = useState('');
  const [savedConfig, setSavedConfig] = useState<TaskBuildConfig | null>(null);

  useEffect(() => {
    void (async () => {
      setLoading(true);
      setLoadError(null);
      try {
        const [tplRes, dsRes, maRes] = await Promise.all([
          listTaskTemplates(),
          listWorkspaceDatasets(),
          listModelAssets(),
        ]);
        setTemplates(tplRes.taskTemplates);
        setDatasets(dsRes.datasets);
        setModelAssets(maRes.modelAssets);
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : '加载任务模板失败');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (presetTemplateId && templates.some((t) => t.id === presetTemplateId)) {
      setSelectedTemplateId(presetTemplateId);
      setTaskFamily(CABLE_MANIPULATION_FAMILY);
      setStep(2);
    }
  }, [presetTemplateId, templates]);

  const familyTemplates = useMemo(
    () => templates.filter((t) => t.taskFamily === taskFamily),
    [templates, taskFamily]
  );

  const selectedTemplate = useMemo(
    () => templates.find((t) => t.id === selectedTemplateId) ?? null,
    [templates, selectedTemplateId]
  );

  const matchedDatasets = useMemo(() => {
    if (!selectedTemplate) return [];
    return datasets.filter((d) => datasetMatchesTaskTemplate(d, selectedTemplate));
  }, [datasets, selectedTemplate]);

  const matchedModelAssets = useMemo(() => {
    if (!selectedTemplate) return [];
    return modelAssets.filter((a) => modelAssetMatchesTaskTemplate(a, selectedTemplate));
  }, [modelAssets, selectedTemplate]);

  const handleSaveConfig = useCallback(() => {
    if (!selectedTemplate) return;
    const name =
      configName.trim() ||
      `${selectedTemplate.name}_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`;
    const config: TaskBuildConfig = {
      id: makeTaskBuildConfigId(),
      name,
      taskTemplateId: selectedTemplate.id,
      taskFamily: selectedTemplate.taskFamily,
      simulatorType: selectedTemplate.simulatorType,
      registryTaskConfigId: selectedTemplate.registryTaskConfigId ?? null,
      linkedDatasetId: linkedDatasetId || null,
      linkedModelAssetId: linkedModelAssetId || null,
      supportedEvaluationModes: selectedTemplate.supportedEvaluationModes ?? [],
      createdAt: new Date().toISOString(),
    };
    saveTaskBuildConfig(config);
    setSavedConfig(config);
    setStep(5);
  }, [configName, linkedDatasetId, linkedModelAssetId, selectedTemplate]);

  const canNext = useMemo(() => {
    if (step === 1) return Boolean(taskFamily);
    if (step === 2) return Boolean(selectedTemplateId);
    if (step === 3) return Boolean(selectedTemplate);
    if (step === 4) return true;
    return false;
  }, [step, taskFamily, selectedTemplateId, selectedTemplate]);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="标准任务模板配置"
        subtitle="基于 TaskTemplate registry 生成任务配置，可跳转数据生成、训练或评测。"
        actions={
          <SecondaryButton onClick={() => router.push('/workspace/task-build')}>返回入口</SecondaryButton>
        }
      />

      <TaskBuildBackLink />
      <TaskBuildStepper steps={STEPS} currentStep={step} />

      {loadError ? (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: 8,
            background: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#991b1b',
            fontSize: 13,
            marginBottom: 16,
          }}
        >
          {loadError}
        </div>
      ) : null}

      {loading ? <p style={{ fontSize: 14, color: '#6b7280' }}>加载任务模板…</p> : null}

      {!loading && step === 1 ? (
        <div style={{ display: 'grid', gap: 12, maxWidth: 560 }}>
          {TASK_FAMILIES.map((family) => (
            <button
              key={family.id}
              type="button"
              onClick={() => setTaskFamily(family.id)}
              style={{
                textAlign: 'left',
                padding: '16px 18px',
                borderRadius: 10,
                border: `2px solid ${taskFamily === family.id ? '#2563eb' : '#e5e7eb'}`,
                backgroundColor: taskFamily === family.id ? '#eff6ff' : '#fff',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 15, color: '#111827' }}>{family.label}</div>
              <div style={{ fontSize: 13, color: '#6b7280', marginTop: 4 }}>{family.description}</div>
            </button>
          ))}
        </div>
      ) : null}

      {!loading && step === 2 ? (
        <div style={{ display: 'grid', gap: 10, maxWidth: 640 }}>
          {familyTemplates.length === 0 ? (
            <p style={{ fontSize: 14, color: '#6b7280' }}>该任务族下暂无可用任务模板。</p>
          ) : (
            familyTemplates.map((tpl) => (
              <button
                key={tpl.id}
                type="button"
                onClick={() => setSelectedTemplateId(tpl.id)}
                style={{
                  textAlign: 'left',
                  padding: '14px 16px',
                  borderRadius: 10,
                  border: `2px solid ${selectedTemplateId === tpl.id ? '#2563eb' : '#e5e7eb'}`,
                  backgroundColor: selectedTemplateId === tpl.id ? '#eff6ff' : '#fff',
                  cursor: 'pointer',
                }}
              >
                <div style={{ fontWeight: 600, color: '#111827' }}>{tpl.name}</div>
                <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{tpl.description}</div>
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4, fontFamily: 'monospace' }}>
                  {tpl.id}
                </div>
              </button>
            ))
          )}
        </div>
      ) : null}

      {!loading && step === 3 && selectedTemplate ? (
        <div
          style={{
            maxWidth: 640,
            padding: 20,
            borderRadius: 12,
            border: '1px solid #e5e7eb',
            backgroundColor: '#fafafa',
          }}
        >
          <TaskBuildField label="任务模板">
            <div style={{ fontSize: 14, color: '#111827' }}>{selectedTemplate.name}</div>
          </TaskBuildField>
          <TaskBuildField label="taskFamily">
            <div style={{ fontSize: 14 }}>{selectedTemplate.taskFamily}</div>
          </TaskBuildField>
          <TaskBuildField label="simulatorType">
            <div style={{ fontSize: 14 }}>{selectedTemplate.simulatorType}</div>
          </TaskBuildField>
          <TaskBuildField label="supportedEvaluationModes">
            <div style={{ fontSize: 14 }}>
              {(selectedTemplate.supportedEvaluationModes ?? [])
                .map((m) => EVALUATION_MODE_LABELS[m] ?? m)
                .join('、') || '—'}
            </div>
          </TaskBuildField>
          <TaskBuildField label="supportedPolicyTypes">
            <div style={{ fontSize: 14 }}>
              {selectedTemplate.supportedPolicyTypes?.join('、') || '—'}
            </div>
          </TaskBuildField>
          <TaskBuildField label="可用 Dataset 数量">
            <div style={{ fontSize: 14 }}>{matchedDatasets.length}</div>
          </TaskBuildField>
          <TaskBuildField label="可用 ModelAsset 数量">
            <div style={{ fontSize: 14 }}>{matchedModelAssets.length}</div>
          </TaskBuildField>
        </div>
      ) : null}

      {!loading && step === 4 && selectedTemplate ? (
        <div style={{ maxWidth: 560 }}>
          <TaskBuildField label="配置名称">
            <input
              type="text"
              value={configName}
              placeholder={`${selectedTemplate.name} 配置`}
              onChange={(e) => setConfigName(e.target.value)}
              style={inputStyle}
            />
          </TaskBuildField>
          <TaskBuildField label="关联数据集（可选）">
            <select
              value={linkedDatasetId}
              onChange={(e) => setLinkedDatasetId(e.target.value)}
              style={inputStyle}
            >
              <option value="">不关联</option>
              {matchedDatasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.episodeCount} episodes
                </option>
              ))}
            </select>
            {matchedDatasets.length === 0 ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
                当前模板下暂无 Dataset，可先前往数据中心生成数据。
              </p>
            ) : null}
          </TaskBuildField>
          <TaskBuildField label="关联模型资产（可选）">
            <select
              value={linkedModelAssetId}
              onChange={(e) => setLinkedModelAssetId(e.target.value)}
              style={inputStyle}
            >
              <option value="">不关联</option>
              {matchedModelAssets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
            {matchedModelAssets.length === 0 ? (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#6b7280' }}>
                当前模板下暂无 ModelAsset，可先完成训练任务。
              </p>
            ) : null}
          </TaskBuildField>
        </div>
      ) : null}

      {!loading && step === 5 && savedConfig && selectedTemplate ? (
        <div style={{ maxWidth: 640 }}>
          <div
            style={{
              padding: 16,
              borderRadius: 10,
              backgroundColor: '#f0fdf4',
              border: '1px solid #bbf7d0',
              marginBottom: 20,
            }}
          >
            <div style={{ fontWeight: 600, color: '#166534', marginBottom: 6 }}>任务配置已生成</div>
            <p style={{ margin: 0, fontSize: 14, color: '#15803d', lineHeight: 1.55 }}>
              配置「{savedConfig.name}」已保存至当前会话，可跳转至下游流程继续操作。
            </p>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <Link
              href={buildDataGenerateHref({
                taskTemplateId: savedConfig.taskTemplateId,
                templateName: selectedTemplate.name,
                taskConfigId: savedConfig.registryTaskConfigId ?? undefined,
              })}
              style={{
                padding: '8px 14px',
                fontSize: 13,
                borderRadius: 8,
                backgroundColor: '#2563eb',
                color: '#fff',
                textDecoration: 'none',
              }}
            >
              前往数据生成
            </Link>
            <Link
              href={buildTrainingCreateHref({
                taskTemplateId: savedConfig.taskTemplateId,
                datasetId: savedConfig.linkedDatasetId ?? undefined,
              })}
              style={{
                padding: '8px 14px',
                fontSize: 13,
                borderRadius: 8,
                border: '1px solid #d1d5db',
                color: '#374151',
                textDecoration: 'none',
                backgroundColor: '#fff',
              }}
            >
              前往模型训练
            </Link>
            <Link
              href={buildEvaluationCreateHref({
                taskTemplateId: savedConfig.taskTemplateId,
                templateName: selectedTemplate.name,
                taskConfigId: savedConfig.registryTaskConfigId ?? undefined,
                datasetId: savedConfig.linkedDatasetId ?? undefined,
                modelAssetId: savedConfig.linkedModelAssetId ?? undefined,
              })}
              style={{
                padding: '8px 14px',
                fontSize: 13,
                borderRadius: 8,
                border: '1px solid #d1d5db',
                color: '#374151',
                textDecoration: 'none',
                backgroundColor: '#fff',
              }}
            >
              前往评测创建
            </Link>
            <SecondaryButton onClick={() => router.push('/workspace/resources/task-templates')}>
              查看任务模板库
            </SecondaryButton>
          </div>
        </div>
      ) : null}

      {step < 5 ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 24,
            maxWidth: 640,
          }}
        >
          <SecondaryButton disabled={step <= 1} onClick={() => setStep((s) => Math.max(1, s - 1))}>
            上一步
          </SecondaryButton>
          {step < 4 ? (
            <PrimaryButton disabled={!canNext} onClick={() => setStep((s) => s + 1)}>
              下一步
            </PrimaryButton>
          ) : (
            <PrimaryButton disabled={!selectedTemplate} onClick={handleSaveConfig}>
              生成任务配置
            </PrimaryButton>
          )}
        </div>
      ) : null}
    </ModulePageContainer>
  );
}
