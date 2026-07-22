'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { buildCableThreadingReplayHref } from '@/lib/workspace/cableThreading';
import {
  buildDualArmCableReplayHref,
  formatDualArmMetric,
  releaseModeLabel,
  stretchModeLabel,
} from '@/lib/workspace/dualArmCable';
import { useRouter } from 'next/navigation';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import {
  dataStatusBadgeStatus,
  formatDataCategoryLabel,
  formatDataScale,
  formatDatasetDisplayModel,
  formatDatasetMainFormat,
  formatListDatasetBuildStatus,
  formatListDisplayName,
  formatListStatusLabel,
  formatListTypeLabel,
  formatSampleCountForDrawer,
  getWorkspaceDataFileEntries,
  hasBuiltDataset,
  isDemoDataCategory,
  isGenerateDataRow,
  listStatusBadgeStatus,
  normalizeDataCategory,
  normalizeDataSource,
} from '@/lib/mock/workspaceDataMock';
import {
  canBuildDualArmIlDataset,
  dualArmIlBuildDisabledReason,
  shouldShowDualArmIlBuildAction,
  applyDualArmIlProbeToItem,
} from '@/lib/workspace/dualArmIlExport';
import { probeDualArmIlExport } from '@/lib/api/dualArmCableClient';
import { isDatasetBuildSupported } from '@/lib/workspace/workspaceDataActions';
import { shouldShowWorkspaceDemo } from '@/lib/workspace/workspaceDemoConfig';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { PrimaryButton, SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 460,
  maxWidth: '100vw',
  backgroundColor: '#ffffff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, color: '#111827', lineHeight: 1.5 }}>{children}</div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 12,
        fontWeight: 600,
        color: '#6b7280',
        marginTop: 16,
        marginBottom: 12,
        paddingTop: 12,
        borderTop: '1px solid #f3f4f6',
      }}
    >
      {children}
    </div>
  );
}

function PathValue({ value }: { value: string }) {
  return (
    <span
      style={{
        display: 'block',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 11,
        lineHeight: 1.5,
        wordBreak: 'break-all',
        overflowWrap: 'anywhere',
        maxWidth: '100%',
      }}
    >
      {value}
    </span>
  );
}

function boolLabel(value: boolean | undefined, contents?: string[], keyword?: string): string {
  if (value !== undefined) return value ? '是' : '否';
  if (contents && keyword) return contents.some((c) => c.includes(keyword)) ? '是' : '否';
  return '—';
}

interface WorkspaceDataDetailDrawerProps {
  item: WorkspaceDataItem | null;
  onClose: () => void;
  onExport: (item: WorkspaceDataItem) => void;
  onBuildDataset?: (item: WorkspaceDataItem) => void;
}

export function WorkspaceDataDetailDrawer({
  item: initialItem,
  onClose,
  onExport,
  onBuildDataset,
}: WorkspaceDataDetailDrawerProps) {
  const router = useRouter();
  const [resolvedItem, setResolvedItem] = useState<WorkspaceDataItem | null>(initialItem);

  useEffect(() => {
    setResolvedItem(initialItem);
    if (!initialItem || initialItem.taskType !== 'dual_arm_cable_manipulation') return;
    const jobId = initialItem.jobId ?? initialItem.backendJobId ?? initialItem.sourceJobId ?? initialItem.id;
    if (!jobId?.startsWith('dac_gen_')) return;
    let cancelled = false;
    void probeDualArmIlExport(jobId)
      .then((probe) => {
        if (!cancelled) setResolvedItem(applyDualArmIlProbeToItem(initialItem, probe));
      })
      .catch(() => {
        if (!cancelled) setResolvedItem({ ...initialItem, ilExportProbed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [initialItem]);

  useEffect(() => {
    if (!initialItem) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [initialItem, onClose]);

  if (!resolvedItem) return null;
  const item = resolvedItem;

  const category = normalizeDataCategory(item.dataCategory);
  const source = normalizeDataSource(item.source);
  const fileEntries = getWorkspaceDataFileEntries(item);
  const isDualArm = item.taskType === 'dual_arm_cable_manipulation';
  const sourceJobId =
    item.jobId ??
    item.backendJobId ??
    item.sourceJobId ??
    (item.taskType === 'cable_threading' || isDualArm ? item.simulationId : undefined);
  const persistedJobId = item.jobId ?? item.backendJobId;
  const episodeCount = formatSampleCountForDrawer(item.dataVolume);
  const canBuildHint =
    item.qualityStatus === '可构建' || item.qualityStatus === 'ready'
      ? '是'
      : item.qualityStatus
        ? '否'
        : item.status === 'completed' && isDemoDataCategory(category)
          ? '待确认'
          : '—';

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="workspace-data-drawer-title">
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h2
              id="workspace-data-drawer-title"
              style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827' }}
            >
              数据详情
            </h2>
            <div style={{ marginTop: 8 }}>
              <StatusBadge
                status={listStatusBadgeStatus(item)}
                label={formatListStatusLabel(item)}
              />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <DetailRow label="数据名称">{formatListDisplayName(item)}</DetailRow>
          <DetailRow label="类型">{formatListTypeLabel(item)}</DetailRow>
          <DetailRow label="任务模板">{item.taskName}</DetailRow>
          <DetailRow label="生成状态">
            <StatusBadge
              status={listStatusBadgeStatus(item)}
              label={formatListStatusLabel(item)}
            />
          </DetailRow>
          {isGenerateDataRow(item) ? (
            <DetailRow label="数据集状态">{formatListDatasetBuildStatus(item)}</DetailRow>
          ) : null}
          <DetailRow label="创建时间">{item.generatedAt}</DetailRow>
          <DetailRow label="创建人">{item.creator}</DetailRow>

          <SectionTitle>来源信息</SectionTitle>
          {persistedJobId ? <DetailRow label="jobId"><PathValue value={persistedJobId} /></DetailRow> : null}
          {item.taskType ? <DetailRow label="taskType">{item.taskType}</DetailRow> : null}
          {item.backendJobStatus ? <DetailRow label="jobType">generate</DetailRow> : null}
          {sourceJobId && sourceJobId !== persistedJobId ? (
            <DetailRow label="来源记录">{sourceJobId}</DetailRow>
          ) : null}
          {item.sourceRecordName ? (
            <DetailRow label="来源任务数据">{item.sourceRecordName}</DetailRow>
          ) : (
            <DetailRow label="来源任务">{source}</DetailRow>
          )}
          {item.simBackend ? (
            <DetailRow label="仿真后端">
              {item.simBackend === 'RoboTwin' ? 'MuJoCo' : item.simBackend}
            </DetailRow>
          ) : null}
          {item.robot ? <DetailRow label="机器人">{item.robot}</DetailRow> : null}
          {item.cableModel ? <DetailRow label="对象模型">{item.cableModel}</DetailRow> : null}
          {item.difficulty ? <DetailRow label="难度">{item.difficulty}</DetailRow> : null}
          {isDualArm ? (
            <>
              <DetailRow label="夹爪">Robotiq 2F-85</DetailRow>
              <DetailRow label="随机种子">{String(item.dualArmSeed ?? '—')}</DetailRow>
              <DetailRow label="线缆数量">{String(item.dualArmMaxCables ?? '—')}</DetailRow>
              <DetailRow label="拉伸模式">{stretchModeLabel(item.dualArmStretchMode)}</DetailRow>
              <DetailRow label="释放策略">{releaseModeLabel(item.dualArmReleaseMode)}</DetailRow>
            </>
          ) : null}
          {item.scene ? <DetailRow label="场景">{item.scene}</DetailRow> : null}
          {item.dataPurpose ? <DetailRow label="数据用途">{item.dataPurpose}</DetailRow> : null}
          {item.taskConfig ? <DetailRow label="任务配置">{item.taskConfig}</DetailRow> : null}

          <SectionTitle>数据质量</SectionTitle>
          <DetailRow label="采集轮次">{episodeCount !== '—' ? `${episodeCount} 条` : '—'}</DetailRow>
          {item.successfulEpisodes != null ? (
            <DetailRow label="成功轨迹">
              {item.successfulEpisodes}
              {episodeCount !== '—' ? ` / ${episodeCount}` : ''}
            </DetailRow>
          ) : null}
          {isDualArm ? (
            <>
              <DetailRow label="episode 成功">{formatDualArmMetric(item.dualArmEpisodeSuccess)}</DetailRow>
              <DetailRow label="成功线缆">
                {item.dualArmSucceededCables ?? '—'} / {item.dualArmMaxCables ?? '—'}
              </DetailRow>
              <DetailRow label="left_contact">{formatDualArmMetric(item.dualArmLeftContact)}</DetailRow>
              <DetailRow label="right_contact">{formatDualArmMetric(item.dualArmRightContact)}</DetailRow>
              <DetailRow label="stretch_reached">{formatDualArmMetric(item.dualArmStretchReached)}</DetailRow>
              <DetailRow label="sag_m">{formatDualArmMetric(item.dualArmSagM)}</DetailRow>
              <DetailRow label="span_m">{formatDualArmMetric(item.dualArmSpanM)}</DetailRow>
              <DetailRow label="final_sag_m">{formatDualArmMetric(item.dualArmFinalSagM)}</DetailRow>
              <DetailRow label="final_span_m">{formatDualArmMetric(item.dualArmFinalSpanM)}</DetailRow>
            </>
          ) : null}
          {!isDualArm && item.successRate != null ? (
            <DetailRow label="成功率">
              <span style={{ fontWeight: 600, color: '#059669' }}>{item.successRate}%</span>
            </DetailRow>
          ) : null}
          {item.qualityStatus ? <DetailRow label="质量状态">{item.qualityStatus}</DetailRow> : null}
          {isDemoDataCategory(category) || item.isDatasetAsset ? (
            <DetailRow label="是否可构建">{canBuildHint}</DetailRow>
          ) : null}
          {item.frameOrTrajectoryCount ? (
            <DetailRow label="轨迹统计">{item.frameOrTrajectoryCount}</DetailRow>
          ) : null}
          <DetailRow label="数据规模">{formatDataScale(item)}</DetailRow>

          <SectionTitle>文件清单</SectionTitle>
          {fileEntries.length > 0 ? (
            fileEntries.map((entry) => (
              <DetailRow key={entry.label} label={entry.label}>
                <PathValue value={entry.value} />
              </DetailRow>
            ))
          ) : (
            <DetailRow label="文件">暂无文件路径记录</DetailRow>
          )}

          {isDualArm ? (
            <>
              <SectionTitle>训练适配</SectionTitle>
              <DetailRow label="说明">
                {hasBuiltDataset(item) || item.trainable
                  ? '已构建 HDF5 模仿学习数据集；该训练后端暂未开放。'
                  : '已生成 episode 过程记录，可用于回放和稳定性评测；尚未生成模仿学习训练数据集。'}
              </DetailRow>
              {item.ilExportFailureReason && !hasBuiltDataset(item) ? (
                <DetailRow label="构建状态">{item.ilExportFailureReason}</DetailRow>
              ) : null}
            </>
          ) : null}

          {item.taskType === 'cable_threading' && (item.lerobotPath || item.dataOrganizationFormat === 'LeRobot') ? (
            <>
              <SectionTitle>LeRobot 导出</SectionTitle>
              <DetailRow label="格式">LeRobot</DetailRow>
              {item.lerobotPath ? (
                <DetailRow label="输出路径">
                  <PathValue value={item.lerobotPath} />
                </DetailRow>
              ) : null}
              {item.lerobotTaskInstruction ? (
                <DetailRow label="task instruction">{item.lerobotTaskInstruction}</DetailRow>
              ) : null}
              {item.lerobotStateDim != null ? (
                <DetailRow label="state_dim">{String(item.lerobotStateDim)}</DetailRow>
              ) : null}
              {item.lerobotActionDim != null ? (
                <DetailRow label="action_dim">{String(item.lerobotActionDim)}</DetailRow>
              ) : null}
              {item.pi0Ready != null ? (
                <DetailRow label="pi0Ready">{item.pi0Ready ? '是' : '否'}</DetailRow>
              ) : null}
              {item.pi0Ready === false && item.pi0ReadyReason ? (
                <DetailRow label="pi0Ready 原因">{item.pi0ReadyReason}</DetailRow>
              ) : null}
              {item.lerobotStatsPath ? (
                <DetailRow label="stats 路径">
                  <PathValue value={item.lerobotStatsPath} />
                </DetailRow>
              ) : null}
              {item.lerobotReportPath ? (
                <DetailRow label="report 路径">
                  <PathValue value={item.lerobotReportPath} />
                </DetailRow>
              ) : null}
            </>
          ) : null}

          {item.isDatasetAsset || item.dataCategory === '训练数据集' || hasBuiltDataset(item) ? (
            <>
              <SectionTitle>训练适配</SectionTitle>
              <DetailRow label="下游模型类型">{formatDatasetDisplayModel(item)}</DetailRow>
              <DetailRow label="数据组织格式">{formatDatasetMainFormat(item)}</DetailRow>
              {item.trainingView ? (
                <DetailRow label="训练数据视图">{item.trainingView}</DetailRow>
              ) : null}
              {item.datasetId ? (
                <DetailRow label="datasetId"><PathValue value={item.datasetId} /></DetailRow>
              ) : null}
              {item.datasetManifestPath ? (
                <DetailRow label="datasetManifest"><PathValue value={item.datasetManifestPath} /></DetailRow>
              ) : null}
              <DetailRow label="是否可用于训练">
                {hasBuiltDataset(item) || item.isDatasetAsset || item.dataCategory === '训练数据集'
                  ? '是'
                  : '—'}
              </DetailRow>
              {item.datasetUsage ? (
                <DetailRow label="数据集用途">{item.datasetUsage}</DetailRow>
              ) : null}
            </>
          ) : null}

          <SectionTitle>采集设置</SectionTitle>
          <DetailRow label="保存视频">{boolLabel(item.saveVideo, item.contents, '视频')}</DetailRow>
          <DetailRow label="保存轨迹">{boolLabel(item.saveTrajectory, item.contents, '轨迹')}</DetailRow>
          <DetailRow label="保存状态日志">{boolLabel(item.saveStateLog, item.contents, '状态')}</DetailRow>
          {item.contents?.length ? (
            <DetailRow label="包含内容">{item.contents.join(' · ')}</DetailRow>
          ) : null}
          {item.sampleRate ? <DetailRow label="采样频率">{item.sampleRate}</DetailRow> : null}

          <SectionTitle>内部信息</SectionTitle>
          <DetailRow label="数据 ID">
            <PathValue value={item.id} />
          </DetailRow>
          <DetailRow label="任务 ID">
            <PathValue value={item.taskId} />
          </DetailRow>
          <DetailRow label="内部 run / batch ID">{item.simulationId}</DetailRow>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 20 }}>
            {isGenerateDataRow(item) &&
            onBuildDataset &&
            isDatasetBuildSupported(item) &&
            !isDualArm &&
            shouldShowWorkspaceDemo() ? (
              <PrimaryButton onClick={() => onBuildDataset(item)}>登记索引</PrimaryButton>
            ) : null}
            {isGenerateDataRow(item) && onBuildDataset && isDualArm && shouldShowDualArmIlBuildAction(item) ? (
              canBuildDualArmIlDataset(item) ? (
                <PrimaryButton onClick={() => onBuildDataset(item)}>构建训练集</PrimaryButton>
              ) : (
                <SecondaryButton disabled title={dualArmIlBuildDisabledReason(item)}>
                  构建训练集
                </SecondaryButton>
              )
            ) : null}
            {hasBuiltDataset(item) || item.isDatasetAsset || category === '训练数据集' ? (
              <SecondaryButton
                onClick={() =>
                  router.push(
                    `/workspace/training?dataset=${encodeURIComponent(item.datasetId ?? item.id)}`
                  )
                }
              >
                训练
              </SecondaryButton>
            ) : null}
            {category === '评测数据集' || category === '外部数据' || category === '真实数据' ? (
              <SecondaryButton onClick={() => router.push('/workspace/evaluation?openCreate=1')}>
                评测
              </SecondaryButton>
            ) : null}
            <Link
              href={
                isDualArm && sourceJobId
                  ? buildDualArmCableReplayHref({ jobId: sourceJobId })
                  : item.taskType === 'cable_threading'
                    ? buildCableThreadingReplayHref({
                        jobId: item.sourceJobId ?? item.jobId ?? item.backendJobId ?? item.id,
                      })
                    : '/workspace/replay'
              }
              style={{ textDecoration: 'none' }}
            >
              <SecondaryButton>回放</SecondaryButton>
            </Link>
            <SecondaryButton onClick={() => onExport(item)}>导出</SecondaryButton>
          </div>
        </div>
      </aside>
    </>
  );
}
