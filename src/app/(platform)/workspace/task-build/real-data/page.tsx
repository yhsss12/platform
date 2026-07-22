'use client';

import { Suspense, useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
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
import {
  makeRealDataImportDraftId,
  saveRealDataImportDraft,
} from '@/lib/workspace/realDataImportDraftStore';
import type { RealDataImportDraft } from '@/types/benchmark';

const STEPS = [
  '数据来源',
  '真机数据',
  '数据结构解析',
  '仿真模板',
  '场景参数',
  '构建草稿',
];

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: 14,
  borderRadius: 8,
  border: '1px solid #d1d5db',
  boxSizing: 'border-box',
};

const DATA_SOURCES = [
  { id: 'local_file', label: '本地文件', description: '选择或登记本地真机采集文件路径' },
  { id: 'platform_import', label: '平台导入记录', description: '引用后续将接入的平台导入索引' },
];

const FORMAT_OPTIONS = ['hdf5', 'npz', 'mcap', 'json', 'unknown'];

function inferFormatFromName(fileName: string): string {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.hdf5') || lower.endsWith('.h5')) return 'hdf5';
  if (lower.endsWith('.npz')) return 'npz';
  if (lower.endsWith('.mcap')) return 'mcap';
  if (lower.endsWith('.json')) return 'json';
  return 'unknown';
}

function RealDataBuildPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const from = searchParams.get('from');

  const [step, setStep] = useState(1);
  const [templates, setTemplates] = useState<TaskTemplateDto[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(true);

  const [draftName, setDraftName] = useState('');
  const [dataSource, setDataSource] = useState('local_file');
  const [sourceFileName, setSourceFileName] = useState('');
  const [dataFormat, setDataFormat] = useState('unknown');
  const [signalsText, setSignalsText] = useState('');
  const [parsedSignals, setParsedSignals] = useState<string[]>([]);
  const [parseMessage, setParseMessage] = useState<string | null>(null);
  const [linkedTaskTemplateId, setLinkedTaskTemplateId] = useState('');
  const [sceneRobot, setSceneRobot] = useState('');
  const [sceneDifficulty, setSceneDifficulty] = useState('');
  const [perturbationAmplitude, setPerturbationAmplitude] = useState('');
  const [savedDraft, setSavedDraft] = useState<RealDataImportDraft | null>(null);

  useEffect(() => {
    void listTaskTemplates()
      .then((res) => setTemplates(res.taskTemplates))
      .finally(() => setLoadingTemplates(false));
  }, []);

  const handleFilePick = useCallback((file: File | null) => {
    if (!file) return;
    setSourceFileName(file.name);
    setDataFormat(inferFormatFromName(file.name));
    if (!draftName.trim()) {
      setDraftName(file.name.replace(/\.[^.]+$/, ''));
    }
  }, [draftName]);

  const handleParseSignals = useCallback(() => {
    const lines = signalsText
      .split(/[\n,;]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      setParseMessage('请填写或补充待解析的信号字段名，每行一个或使用逗号分隔。');
      setParsedSignals([]);
      return;
    }
    setParsedSignals(lines);
    setParseMessage(`已整理 ${lines.length} 个信号字段，请确认后继续。`);
  }, [signalsText]);

  const handleSaveDraft = useCallback(() => {
    const now = new Date().toISOString();
    const draft: RealDataImportDraft = {
      id: makeRealDataImportDraftId(),
      name: draftName.trim() || sourceFileName || '真机数据构建草稿',
      sourceFileName,
      dataFormat,
      parsedSignals,
      linkedTaskTemplateId: linkedTaskTemplateId || null,
      sceneConfig: {
        robot: sceneRobot || undefined,
        difficulty: sceneDifficulty || undefined,
      },
      perturbationConfig: {
        amplitude: perturbationAmplitude ? Number(perturbationAmplitude) : undefined,
      },
      status: 'configured',
      createdAt: now,
      updatedAt: now,
    };
    saveRealDataImportDraft(draft);
    setSavedDraft(draft);
    setStep(6);
  }, [
    draftName,
    sourceFileName,
    dataFormat,
    parsedSignals,
    linkedTaskTemplateId,
    sceneRobot,
    sceneDifficulty,
    perturbationAmplitude,
  ]);

  const backHref = from === 'data' ? '/workspace/data' : '/workspace/task-build';

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="基于真机数据配置仿真任务"
        subtitle="解析数据结构、选择仿真模板并配置场景参数，保存为构建草稿（不写入 Dataset registry）。"
        actions={
          <SecondaryButton onClick={() => router.push(backHref)}>返回</SecondaryButton>
        }
      />

      <TaskBuildBackLink href={backHref} />
      <TaskBuildStepper steps={STEPS} currentStep={step} />

      {step === 1 ? (
        <div style={{ display: 'grid', gap: 10, maxWidth: 560 }}>
          {DATA_SOURCES.map((src) => (
            <button
              key={src.id}
              type="button"
              onClick={() => setDataSource(src.id)}
              style={{
                textAlign: 'left',
                padding: '14px 16px',
                borderRadius: 10,
                border: `2px solid ${dataSource === src.id ? '#2563eb' : '#e5e7eb'}`,
                backgroundColor: dataSource === src.id ? '#eff6ff' : '#fff',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontWeight: 600 }}>{src.label}</div>
              <div style={{ fontSize: 13, color: '#6b7280', marginTop: 4 }}>{src.description}</div>
            </button>
          ))}
        </div>
      ) : null}

      {step === 2 ? (
        <div style={{ maxWidth: 560 }}>
          <TaskBuildField label="草稿名称">
            <input
              type="text"
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder="真机数据构建草稿"
              style={inputStyle}
            />
          </TaskBuildField>
          <TaskBuildField label="真机数据文件">
            <input
              type="file"
              onChange={(e) => handleFilePick(e.target.files?.[0] ?? null)}
              style={{ fontSize: 13 }}
            />
            <input
              type="text"
              value={sourceFileName}
              onChange={(e) => {
                setSourceFileName(e.target.value);
                setDataFormat(inferFormatFromName(e.target.value));
              }}
              placeholder="或输入文件名称"
              style={{ ...inputStyle, marginTop: 8 }}
            />
          </TaskBuildField>
          <TaskBuildField label="数据格式">
            <select
              value={dataFormat}
              onChange={(e) => setDataFormat(e.target.value)}
              style={inputStyle}
            >
              {FORMAT_OPTIONS.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </TaskBuildField>
        </div>
      ) : null}

      {step === 3 ? (
        <div style={{ maxWidth: 560 }}>
          <TaskBuildField label="信号字段（数据结构解析）">
            <textarea
              value={signalsText}
              onChange={(e) => setSignalsText(e.target.value)}
              placeholder="每行一个信号名，例如：joint_pos, joint_vel, gripper_state"
              rows={6}
              style={{ ...inputStyle, fontFamily: 'monospace', fontSize: 13 }}
            />
          </TaskBuildField>
          <SecondaryButton onClick={handleParseSignals}>解析数据结构</SecondaryButton>
          {parseMessage ? (
            <p style={{ margin: '10px 0 0', fontSize: 13, color: '#4b5563' }}>{parseMessage}</p>
          ) : null}
          {parsedSignals.length > 0 ? (
            <ul style={{ margin: '12px 0 0', paddingLeft: 18, fontSize: 13, color: '#374151' }}>
              {parsedSignals.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {step === 4 ? (
        <div style={{ maxWidth: 560 }}>
          {loadingTemplates ? (
            <p style={{ fontSize: 14, color: '#6b7280' }}>加载仿真模板…</p>
          ) : (
            <TaskBuildField label="关联仿真任务模板">
              <select
                value={linkedTaskTemplateId}
                onChange={(e) => setLinkedTaskTemplateId(e.target.value)}
                style={inputStyle}
              >
                <option value="">请选择任务模板</option>
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} ({t.id})
                  </option>
                ))}
              </select>
            </TaskBuildField>
          )}
        </div>
      ) : null}

      {step === 5 ? (
        <div style={{ maxWidth: 560 }}>
          <TaskBuildField label="场景参数 · 机器人">
            <input
              type="text"
              value={sceneRobot}
              onChange={(e) => setSceneRobot(e.target.value)}
              placeholder="例如 Panda"
              style={inputStyle}
            />
          </TaskBuildField>
          <TaskBuildField label="场景参数 · 难度">
            <input
              type="text"
              value={sceneDifficulty}
              onChange={(e) => setSceneDifficulty(e.target.value)}
              placeholder="例如 easy"
              style={inputStyle}
            />
          </TaskBuildField>
          <TaskBuildField label="扰动配置 · 幅度（可选）">
            <input
              type="number"
              value={perturbationAmplitude}
              onChange={(e) => setPerturbationAmplitude(e.target.value)}
              placeholder="数值"
              style={inputStyle}
            />
          </TaskBuildField>
        </div>
      ) : null}

      {step === 6 && savedDraft ? (
        <div
          style={{
            maxWidth: 560,
            padding: 16,
            borderRadius: 10,
            backgroundColor: '#f9fafb',
            border: '1px solid #e5e7eb',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 8 }}>构建草稿已保存</div>
          <p style={{ margin: '0 0 12px', fontSize: 14, color: '#4b5563', lineHeight: 1.55 }}>
            草稿「{savedDraft.name}」已保存至本地（RealDataImportDraft），未写入 Dataset registry。
            后续可迁移至后端 RealDataImport 服务。
          </p>
          <div style={{ fontSize: 12, color: '#6b7280', fontFamily: 'monospace' }}>
            id: {savedDraft.id}
          </div>
          <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
            <SecondaryButton onClick={() => router.push('/workspace/data')}>
              返回数据中心
            </SecondaryButton>
            <PrimaryButton onClick={() => router.push('/workspace/task-build')}>
              返回任务构建
            </PrimaryButton>
          </div>
        </div>
      ) : null}

      {step < 6 ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 24,
            maxWidth: 560,
          }}
        >
          <button
            type="button"
            disabled={step <= 1}
            onClick={() => setStep((s) => s - 1)}
            style={{
              padding: '8px 16px',
              fontSize: 14,
              borderRadius: 6,
              border: '1px solid #d1d5db',
              backgroundColor: '#fff',
              color: '#374151',
              cursor: step <= 1 ? 'not-allowed' : 'pointer',
              opacity: step <= 1 ? 0.5 : 1,
            }}
          >
            上一步
          </button>
          {step < 5 ? (
            <PrimaryButton
              disabled={
                (step === 2 && !sourceFileName.trim()) ||
                (step === 4 && !linkedTaskTemplateId)
              }
              onClick={() => setStep((s) => s + 1)}
            >
              下一步
            </PrimaryButton>
          ) : (
            <PrimaryButton onClick={handleSaveDraft}>保存构建草稿</PrimaryButton>
          )}
        </div>
      ) : null}
    </ModulePageContainer>
  );
}

export default function RealDataBuildPage() {
  return (
    <Suspense fallback={null}>
      <RealDataBuildPageContent />
    </Suspense>
  );
}
