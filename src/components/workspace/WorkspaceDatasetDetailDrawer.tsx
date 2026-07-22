'use client';

import { useEffect, useState } from 'react';
import type { Dataset } from '@/types/benchmark';
import {
  resolveDatasetFormatLabel,
  resolveDatasetSourceTaskLabel,
} from '@/lib/workspace/taskTemplateMapping';
import {
  resolveDatasetSourceLabel,
  resolveDatasetCountText,
  resolveDatasetSimulatorBackendLabel,
} from '@/lib/workspace/datasetDisplay';
import { normalizeDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import {
  datasetTrainingDisabledHint,
  isDualArmCableDataset,
  shouldShowDatasetTrainingLink,
} from '@/lib/workspace/datasetTrainingAccess';
import { shouldShowImportedDatasetBuildAction } from '@/lib/workspace/datasetImportWorkflow';
import {
  formatNutAssemblyDatagenSuccessRate,
  formatNutAssemblyPolicyMode,
  formatNutAssemblySuccessRate,
  isNutAssemblyDataset,
  NUT_ASSEMBLY_TASK_DISPLAY_NAME,
} from '@/lib/workspace/nutAssembly';
import {
  buildDualArmIlExport,
  probeDualArmIlExport,
  type DualArmIlExportProbeResponse,
} from '@/lib/api/dualArmCableClient';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, color: '#111827', lineHeight: 1.5 }}>{children}</div>
    </div>
  );
}

function PathValue({ value }: { value: string }) {
  return (
    <code
      style={{
        display: 'block',
        fontSize: 12,
        padding: '8px 10px',
        background: '#f3f4f6',
        borderRadius: 6,
        wordBreak: 'break-all',
        whiteSpace: 'pre-wrap',
      }}
    >
      {value}
    </code>
  );
}

function formatCreatedAt(value: string): string {
  if (!value) return '—';
  try {
    const d = new Date(value);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString('zh-CN', { hour12: false });
    }
  } catch {
    /* ignore */
  }
  return value.slice(0, 19).replace('T', ' ');
}

export function WorkspaceDatasetDetailDrawer({
  dataset,
  onClose,
  onTrain,
  onBuilt,
  onBuild,
}: {
  dataset: Dataset | null;
  onClose: () => void;
  onTrain?: (dataset: Dataset) => void;
  onBuilt?: () => void;
  onBuild?: (dataset: Dataset) => void;
}) {
  const [ilProbe, setIlProbe] = useState<DualArmIlExportProbeResponse | null>(null);
  const [building, setBuilding] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [dataset, onClose]);

  useEffect(() => {
    if (!dataset || !isDualArmCableDataset(dataset)) {
      setIlProbe(null);
      setBuildError(null);
      return;
    }
    let cancelled = false;
    void probeDualArmIlExport(dataset.sourceJobId)
      .then((probe) => {
        if (!cancelled) setIlProbe(probe);
      })
      .catch(() => {
        if (!cancelled) setIlProbe(null);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  if (!dataset) return null;

  const displayName = normalizeDatasetDisplayName({
    displayName: dataset.displayName,
    name: dataset.name,
    taskType: dataset.taskType,
    createdAt: dataset.createdAt,
    sourceJobId: dataset.sourceJobId,
  });
  const simulatorBackendLabel = resolveDatasetSimulatorBackendLabel(dataset.simulatorBackend);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1600,
        display: 'flex',
        justifyContent: 'flex-end',
        backgroundColor: 'rgba(15, 23, 42, 0.35)',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 'min(520px, 100%)',
          height: '100%',
          backgroundColor: '#fff',
          boxShadow: '-4px 0 24px rgba(0,0,0,0.12)',
          padding: 24,
          overflowY: 'auto',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: 20,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#111827' }}>数据集详情</h3>
            <p style={{ margin: '6px 0 0', fontSize: 13, color: '#6b7280', wordBreak: 'break-all' }}>
              {displayName}
            </p>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <DetailRow label="数据集 ID">{dataset.id}</DetailRow>
        <DetailRow label="任务名称">{resolveDatasetSourceTaskLabel(dataset)}</DetailRow>
        <DetailRow label="来源 Job">
          <PathValue value={dataset.sourceJobId || '—'} />
        </DetailRow>
        <DetailRow label="Manifest 路径">
          <PathValue value={dataset.manifestPath || '—'} />
        </DetailRow>
        <DetailRow label="存储路径">
          <PathValue value={dataset.storagePath || '—'} />
        </DetailRow>
        <DetailRow label="数据来源">{resolveDatasetSourceLabel(dataset)}</DetailRow>
        {simulatorBackendLabel ? (
          <DetailRow label="仿真后端">{simulatorBackendLabel}</DetailRow>
        ) : null}
        <DetailRow label="数据格式">
          {resolveDatasetFormatLabel(dataset)}
          {dataset.datasetFormat &&
          dataset.datasetFormat !== dataset.format &&
          resolveDatasetFormatLabel(dataset) !== dataset.datasetFormat ? (
            <span style={{ marginLeft: 8, fontSize: 12, color: '#6b7280' }}>
              ({dataset.datasetFormat})
            </span>
          ) : null}
        </DetailRow>
        <DetailRow label="数据数量">{resolveDatasetCountText(dataset)}</DetailRow>
        <DetailRow label="状态">{dataset.status}</DetailRow>
        {isDualArmCableDataset(dataset) ? (
          <>
            <DetailRow label="训练适配说明">
              {dataset.trainable
                ? '已构建 HDF5 模仿学习数据集；训练后端待接入。'
                : dataset.format === 'manifest'
                  ? '已生成 episode 过程记录，可用于回放和稳定性评测；尚未生成模仿学习训练数据集。'
                  : '过程记录数据集。'}
            </DetailRow>
            {dataset.trainable != null ? (
              <DetailRow label="可训练">{dataset.trainable ? '是（后端待接入）' : '否'}</DetailRow>
            ) : null}
            {ilProbe ? (
              <DetailRow label="IL 导出探测">
                {ilProbe.exportReady ? '可构建训练集' : ilProbe.reason ?? '不可构建'}
              </DetailRow>
            ) : null}
          </>
        ) : null}
        {isNutAssemblyDataset(dataset) ? (
          <>
            <DetailRow label="任务名称">{NUT_ASSEMBLY_TASK_DISPLAY_NAME}</DetailRow>
            <DetailRow label="生成方式">
              {dataset.dataSourceLabel ??
                (dataset.physicsEnhancementEnabled ? 'MimicGen + PINN' : 'MimicGen 生成')}
            </DetailRow>
            {dataset.physicsEnhancementEnabled ? (
              <DetailRow label="数据增强">PINN 轨迹修复</DetailRow>
            ) : null}
            <DetailRow label="策略模式">
              {formatNutAssemblyPolicyMode(dataset.policyMode)}
            </DetailRow>
            <DetailRow label="示教数据来源">
              {dataset.sourceDemoOrigin ?? '—'}
            </DetailRow>
            {dataset.physicsEnhancementEnabled ? (
              <>
                <DetailRow label="MimicGen 原始 demo 数">
                  {dataset.mimicgenGeneratedDemos ?? dataset.rawDemoCount ?? '—'}
                </DetailRow>
                <DetailRow label="PINN 修复 demo 数">
                  {dataset.repairedDemoCount ?? '—'}
                </DetailRow>
                <DetailRow label="最终 demo 数">
                  {dataset.finalDemoCount ?? dataset.demoCount ?? '—'}
                </DetailRow>
                <DetailRow label="PINN 模型">
                  {dataset.pinnModelId ?? 'NutAssembly-PINN v1'}
                </DetailRow>
                <DetailRow label="修复复核通过率">
                  {dataset.pinnRepairValidationRate != null
                    ? `${Math.round(Number(dataset.pinnRepairValidationRate) * 1000) / 10}%`
                    : '—'}
                </DetailRow>
              </>
            ) : null}
            <DetailRow label="Source Demo 路径">
              {dataset.sourceDemoPath ?? '—'}
            </DetailRow>
            <DetailRow label="Source Demo Hash">
              {dataset.sourceDemoHash ?? '—'}
            </DetailRow>
            <DetailRow label="环境名称">{dataset.envName ?? '—'}</DetailRow>
            <DetailRow label="请求 episode 数">
              {dataset.episodesRequested ?? '—'}
            </DetailRow>
            <DetailRow label="生成 episode 数">
              {dataset.episodesGenerated ?? dataset.demoCount ?? dataset.episodeCount ?? '—'}
            </DetailRow>
            <DetailRow label="Datagen 失败次数">
              {dataset.datagenFailedTrials ?? '—'}
            </DetailRow>
            <DetailRow label="Datagen 成功率">
              {formatNutAssemblyDatagenSuccessRate(dataset)}
            </DetailRow>
            <DetailRow label="任务评测成功率">
              {formatNutAssemblySuccessRate(dataset)}
            </DetailRow>
            <DetailRow label="datagen_info">
              {dataset.hasDatagenInfo ? '已写入' : '无'}
            </DetailRow>
            <DetailRow label="objectPoseKeys">
              {dataset.objectPoseKeys?.length
                ? dataset.objectPoseKeys.join(', ')
                : '—'}
            </DetailRow>
            <DetailRow label="Demo 数">
              {dataset.demoCount ?? dataset.episodeCount ?? '—'}
            </DetailRow>
            <DetailRow label="总步数">{dataset.totalSteps ?? '—'}</DetailRow>
            <DetailRow label="成功 episode">
              {dataset.hasEpisodeMetadata && dataset.successEpisodes != null
                ? String(dataset.successEpisodes)
                : '未标注'}
            </DetailRow>
            <DetailRow label="对象位姿">
              {dataset.hasObjectPoses ? '已写入 datagen_info/object_poses' : '无'}
            </DetailRow>
            <DetailRow label="Episode 元数据">
              {dataset.hasEpisodeMetadata ? '已标注 success_flag / failure_type' : '未标注'}
            </DetailRow>
            <DetailRow label="可训练轨迹">
              {dataset.hasEpisodeMetadata && dataset.validForTrainingEpisodes != null
                ? String(dataset.validForTrainingEpisodes)
                : '未标注'}
            </DetailRow>
            <DetailRow label="训练筛选策略">
              {dataset.trainingFilterMode ?? dataset.defaultTrainingFilterMode ?? 'valid_for_training_only'}
            </DetailRow>
            <DetailRow label="已过滤轨迹">
              {dataset.filteredDemoCount != null ? String(dataset.filteredDemoCount) : '—'}
            </DetailRow>
            <DetailRow label="可直接用于训练">
              {(dataset.validForTrainingEpisodes ?? 0) > 0 || dataset.trainingBuildReady
                ? '是（需构建训练集）'
                : dataset.hasEpisodeMetadata
                  ? '否（暂无可训练成功轨迹）'
                  : '未标注'}
            </DetailRow>
            {dataset.hasStageStatistics ? (
              <>
                <DetailRow label="Grasp 成功">
                  {dataset.graspSuccessEpisodes ?? '—'}
                </DetailRow>
                <DetailRow label="Lift 成功">
                  {dataset.liftSuccessEpisodes ?? '—'}
                </DetailRow>
                <DetailRow label="Insertion 成功">
                  {dataset.insertionSuccessEpisodes ?? '—'}
                </DetailRow>
                <DetailRow label="平均 Grasp 次数">
                  {dataset.averageGraspAttempts ?? '—'}
                </DetailRow>
              </>
            ) : (
              <DetailRow label="阶段统计">未包含</DetailRow>
            )}
          </>
        ) : null}
        <DetailRow label="创建时间">{formatCreatedAt(dataset.createdAt)}</DetailRow>

        <div style={{ display: 'flex', gap: 8, marginTop: 24, flexWrap: 'wrap' }}>
          {onTrain && shouldShowDatasetTrainingLink(dataset) ? (
            <PrimaryButton onClick={() => onTrain(dataset)}>进入训练</PrimaryButton>
          ) : null}
          {!shouldShowDatasetTrainingLink(dataset) && datasetTrainingDisabledHint(dataset) ? (
            <span style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.5, flex: '1 1 100%' }}>
              {datasetTrainingDisabledHint(dataset)}
            </span>
          ) : null}
          {onBuild && shouldShowImportedDatasetBuildAction(dataset) ? (
            <PrimaryButton onClick={() => onBuild(dataset)}>构建数据集</PrimaryButton>
          ) : null}
          {isDualArmCableDataset(dataset) && ilProbe?.exportReady && !dataset.trainable ? (
            <PrimaryButton
              disabled={building}
              onClick={() => {
                setBuilding(true);
                setBuildError(null);
                void buildDualArmIlExport(dataset.sourceJobId)
                  .then(() => {
                    onBuilt?.();
                  })
                  .catch((err: unknown) => {
                    const detail =
                      err && typeof err === 'object' && 'message' in err
                        ? String((err as { message?: string }).message)
                        : '构建失败';
                    setBuildError(detail);
                  })
                  .finally(() => setBuilding(false));
              }}
            >
              {building ? '构建中…' : '构建训练集'}
            </PrimaryButton>
          ) : null}
          {buildError ? (
            <span style={{ fontSize: 13, color: '#b45309', flex: '1 1 100%' }}>{buildError}</span>
          ) : null}
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
      </div>
    </div>
  );
}
