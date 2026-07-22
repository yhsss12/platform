'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  getTaskConfig,
  registryStatusLabel,
  type RegistryResource,
  type TaskConfigDetail,
} from '@/lib/api/resourceRegistryClient';
import { getIsaacLabRuntimeStatus, type IsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';
import type { TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import { IsaacLabRuntimeStatusCard } from '@/components/workspace/IsaacLabRuntimeStatusCard';
import { IsaacLabSmokeTestPanel } from '@/components/workspace/IsaacLabSmokeTestPanel';
import { IsaacLabReplayDemoPanel } from '@/components/workspace/IsaacLabReplayDemoPanel';
import { ISAAC_BLOCK_STACKING_TEMPLATE_ID, buildIsaacBlockStackingConsoleHref } from '@/lib/workspace/isaacBlockStacking';
import { buildIsaacLabFrankaStackCubeConsoleHref, ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID } from '@/lib/workspace/isaaclabFrankaStackCube';
import {
  FRANKA_STACK_CUBE_DATA_CAPABILITY_LABEL,
  FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL,
  FRANKA_STACK_CUBE_PRODUCT_NAME,
  FRANKA_STACK_CUBE_PRODUCT_SUBTITLE,
  isFrankStackCubeProductTask,
} from '@/lib/workspace/isaacStackCubeProduct';
import { isValidDataGenJobId, isValidIsaacGenerateJobId } from '@/lib/workspace/backendJobIds';
import { formatIsaacGenerationMode } from '@/lib/workspace/isaacGenerationMode';
import {
  buildTaskTemplateAssetRow,
  statusBadgeStyle,
  trajectoryQualityBadgeStyle,
} from '@/lib/workspace/taskTemplatePresentation';
import { GhostButton, PrimaryButton } from './workspaceUi';

interface TaskConfigDependenciesDrawerProps {
  taskConfigId: string | null;
  taskTemplates?: TaskTemplateDto[];
  onClose: () => void;
  isaacReplayJobId?: string | null;
  isaacGenerateJobId?: string | null;
  onClearIsaacReplayQuery?: () => void;
}

function DetailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: '#374151',
          marginBottom: 10,
          paddingBottom: 6,
          borderBottom: '1px solid #f3f4f6',
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 8, fontSize: 13, lineHeight: 1.55 }}>
      <span style={{ color: '#9ca3af', minWidth: 108, flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#374151', flex: 1 }}>{value}</span>
    </div>
  );
}

function ResourceGroup({
  title,
  items,
}: {
  title: string;
  items: RegistryResource | RegistryResource[] | undefined;
}) {
  const list = Array.isArray(items) ? items : items ? [items] : [];
  if (list.length === 0) {
    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6 }}>{title}</div>
        <div style={{ fontSize: 13, color: '#9ca3af' }}>—</div>
      </div>
    );
  }
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 8 }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {list.map((item) => (
          <div
            key={item.assetId}
            style={{
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              backgroundColor: '#fafafa',
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 500, color: '#111827' }}>{item.name}</div>
            <div style={{ fontSize: 12, color: '#6b7280', fontFamily: 'monospace', marginTop: 4 }}>
              {item.assetId}
            </div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
              {registryStatusLabel(item.status)} · {item.version} · {item.simBackend}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TaskConfigDependenciesDrawer({
  taskConfigId,
  taskTemplates = [],
  onClose,
  isaacReplayJobId = null,
  isaacGenerateJobId = null,
  onClearIsaacReplayQuery,
}: TaskConfigDependenciesDrawerProps) {
  const [detail, setDetail] = useState<TaskConfigDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isaacRuntime, setIsaacRuntime] = useState<IsaacLabRuntimeStatus | null>(null);
  const [isaacRuntimeLoading, setIsaacRuntimeLoading] = useState(false);
  const [isaacRuntimeError, setIsaacRuntimeError] = useState<string | null>(null);
  const [showInternal, setShowInternal] = useState(false);

  const matchedTemplate = useMemo(
    () =>
      taskTemplates.find(
        (item) => item.registryTaskConfigId === taskConfigId || item.id === taskConfigId
      ) ?? null,
    [taskConfigId, taskTemplates]
  );

  const assetRow = useMemo(
    () => (matchedTemplate ? buildTaskTemplateAssetRow(matchedTemplate) : null),
    [matchedTemplate]
  );

  const isFrankStackCubeTemplate =
    matchedTemplate?.id === ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID ||
    matchedTemplate?.id === ISAAC_BLOCK_STACKING_TEMPLATE_ID ||
    isFrankStackCubeProductTask(matchedTemplate?.id);
  const isIsaacTemplate = isFrankStackCubeTemplate;
  const trackingReplayJob = isaacReplayJobId ?? null;
  const trackingGenerateJob = isaacGenerateJobId ?? null;
  const showIsaacReplayPanel =
    isIsaacTemplate && (!isaacRuntimeLoading || Boolean(trackingReplayJob)) && !trackingGenerateJob;
  const showIsaacSmokePanel =
    isIsaacTemplate && !isaacRuntimeLoading && !isaacRuntimeError && !trackingReplayJob && !trackingGenerateJob;

  useEffect(() => {
    if (!taskConfigId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void getTaskConfig(taskConfigId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '加载任务配置失败');
          setDetail(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [taskConfigId]);

  useEffect(() => {
    if (!isIsaacTemplate) {
      setIsaacRuntime(null);
      setIsaacRuntimeError(null);
      return;
    }
    let cancelled = false;
    setIsaacRuntimeLoading(true);
    setIsaacRuntimeError(null);
    void getIsaacLabRuntimeStatus()
      .then((data) => {
        if (!cancelled) setIsaacRuntime(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setIsaacRuntimeError(err instanceof Error ? err.message : '加载 Isaac Lab 运行状态失败');
          setIsaacRuntime(null);
        }
      })
      .finally(() => {
        if (!cancelled) setIsaacRuntimeLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isIsaacTemplate]);

  if (!taskConfigId) return null;

  const resolved = detail?.resolvedResources ?? {};
  const displayName = assetRow?.name ?? matchedTemplate?.name ?? detail?.name ?? taskConfigId;
  const qualityStyle = assetRow
    ? trajectoryQualityBadgeStyle(assetRow.trajectoryQualitySeverity)
    : null;
  const statusStyle = assetRow ? statusBadgeStyle(assetRow.statusKey) : null;
  const generationModeLabel = assetRow?.meta.defaultGenerationMode
    ? formatIsaacGenerationMode(assetRow.meta.defaultGenerationMode)
    : assetRow?.expertStrategyLabel;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1600,
        backgroundColor: 'rgba(17,24,39,0.45)',
        display: 'flex',
        justifyContent: 'flex-end',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 'min(520px, 100vw)',
          height: '100%',
          overflowY: 'auto',
          backgroundColor: '#fff',
          borderLeft: '1px solid #e5e7eb',
          padding: 24,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>任务模板详情</h3>
          <GhostButton onClick={onClose}>关闭</GhostButton>
        </div>

        {assetRow ? (
          <>
            <DetailSection title="基本信息">
              <DetailRow label="任务名称" value={displayName} />
              {isFrankStackCubeTemplate ? (
                <DetailRow label="说明" value={FRANKA_STACK_CUBE_PRODUCT_SUBTITLE} />
              ) : null}
              <DetailRow label="taskType" value={<code style={{ fontSize: 12 }}>{assetRow.taskTypeLabel}</code>} />
              {showInternal ? (
                <DetailRow
                  label="taskTemplateId"
                  value={<code style={{ fontSize: 12 }}>{assetRow.templateId}</code>}
                />
              ) : null}
              <DetailRow label="版本" value={detail?.version ?? 'v1'} />
              <DetailRow label="仿真后端" value={assetRow.simulatorLabel} />
              {assetRow.meta.runtimeBackendLabel ? (
                <DetailRow label="底层运行时" value={assetRow.meta.runtimeBackendLabel} />
              ) : null}
              <DetailRow
                label="状态"
                value={
                  statusStyle ? (
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: 999,
                        fontSize: 11,
                        backgroundColor: statusStyle.backgroundColor,
                        color: statusStyle.color,
                        border: `1px solid ${statusStyle.borderColor}`,
                      }}
                    >
                      {assetRow.statusLabel}
                    </span>
                  ) : (
                    assetRow.statusLabel
                  )
                }
              />
            </DetailSection>

            <DetailSection title="任务说明">
              <p style={{ margin: '0 0 10px', fontSize: 13, color: '#374151', lineHeight: 1.65 }}>
                {assetRow.shortDescription}
              </p>
              <DetailRow label="操作对象" value={assetRow.meta.involvedObjects} />
              <DetailRow label="机器人" value={assetRow.meta.robotLabel} />
              <DetailRow label="场景" value={assetRow.meta.sceneLabel} />
            </DetailSection>

            {isFrankStackCubeTemplate ? (
              <DetailSection title="能力状态">
                <DetailRow label="数据生成" value={FRANKA_STACK_CUBE_DATA_CAPABILITY_LABEL} />
                <DetailRow label="训练" value="已接入 Robomimic BC" />
                <DetailRow label="评测" value={FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL} />
                <DetailRow label="回放" value="已接入数据生成与评测回放" />
                <p style={{ margin: '8px 0 0', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
                  评测依赖 Isaac Lab 运行时，当前复用 isaac_block_stacking adapter；底层 templateId 尚未完全合并。
                </p>
              </DetailSection>
            ) : null}

            <DetailSection title="数据生成能力">
              <DetailRow label="默认专家策略" value={assetRow.expertStrategyLabel} />
              <DetailRow label="generationMode" value={generationModeLabel ?? '—'} />
              <DetailRow
                label="HDF5 导出"
                value={assetRow.supportsDataGeneration ? '支持' : '暂不支持'}
              />
              <DetailRow
                label="自动回放视频"
                value={matchedTemplate?.replayAvailable ? '支持' : '—'}
              />
              <DetailRow label="轨迹质量检测" value={isIsaacTemplate ? '支持' : '部分支持'} />
              <DetailRow label="默认采集轮次" value={assetRow.meta.defaultCollectionRounds ?? '—'} />
              {assetRow.meta.advancedExperimentNote ? (
                <p style={{ margin: '8px 0 0', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
                  {assetRow.meta.advancedExperimentNote}
                </p>
              ) : null}
            </DetailSection>

            <DetailSection title="训练与评测能力">
              <DetailRow label="训练后端" value={assetRow.trainingLabel} />
              <DetailRow label="评测模式" value={assetRow.evaluationLabel} />
              <DetailRow
                label="默认指标"
                value={
                  matchedTemplate?.defaultMetricIds?.length
                    ? matchedTemplate.defaultMetricIds.join('、')
                    : '—'
                }
              />
            </DetailSection>

            <DetailSection title="轨迹质量参考">
              <DetailRow
                label="当前策略质量"
                value={
                  qualityStyle ? (
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: 999,
                        fontSize: 11,
                        backgroundColor: qualityStyle.backgroundColor,
                        color: qualityStyle.color,
                        border: `1px solid ${qualityStyle.borderColor}`,
                      }}
                    >
                      {assetRow.trajectoryQualityLabel}
                    </span>
                  ) : (
                    assetRow.trajectoryQualityLabel
                  )
                }
              />
              <DetailRow
                label="最近数据集"
                value="请在数据中心查看该任务模板下的最新生成记录"
              />
            </DetailSection>
          </>
        ) : null}

        {isIsaacTemplate ? (
          <div style={{ marginBottom: 20 }}>
            {!isaacRuntimeLoading || trackingReplayJob ? (
              <IsaacLabRuntimeStatusCard
                status={isaacRuntime}
                loading={isaacRuntimeLoading}
                error={trackingReplayJob ? null : isaacRuntimeError}
              />
            ) : null}
            {showIsaacSmokePanel ? <IsaacLabSmokeTestPanel runtime={isaacRuntime} /> : null}
            {trackingGenerateJob &&
            (isValidIsaacGenerateJobId(trackingGenerateJob) || isValidDataGenJobId(trackingGenerateJob)) ? (
              <div
                style={{
                  marginTop: 12,
                  padding: 14,
                  borderRadius: 10,
                  border: '1px solid #bfdbfe',
                  backgroundColor: '#eff6ff',
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e3a8a', marginBottom: 8 }}>
                  数据生成任务运行中
                </div>
                <p style={{ margin: '0 0 12px', fontSize: 12, color: '#1d4ed8', lineHeight: 1.55 }}>
                  可在运行控制台查看 Mimic 专家策略数据生成进度与回放状态。
                </p>
                <PrimaryButton
                  onClick={() => {
                    const href = isValidDataGenJobId(trackingGenerateJob)
                      ? buildIsaacLabFrankaStackCubeConsoleHref({ jobId: trackingGenerateJob })
                      : buildIsaacBlockStackingConsoleHref({ jobId: trackingGenerateJob });
                    window.location.href = href;
                  }}
                >
                  前往运行控制台
                </PrimaryButton>
              </div>
            ) : null}
            {showIsaacReplayPanel ? (
              <IsaacLabReplayDemoPanel
                runtime={isaacRuntime}
                autoReplayJobId={trackingReplayJob}
                onReplayJobCleared={onClearIsaacReplayQuery}
              />
            ) : null}
          </div>
        ) : null}

        {loading ? (
          <p style={{ color: '#6b7280', fontSize: 14 }}>正在加载资源配置…</p>
        ) : error && !matchedTemplate ? (
          <p style={{ color: '#b45309', fontSize: 14 }}>{error}</p>
        ) : detail ? (
          <DetailSection title="任务资源配置">
            <ResourceGroup title="机器人" items={resolved.robots as RegistryResource[]} />
            <ResourceGroup title="末端执行器" items={resolved.endEffectors as RegistryResource[]} />
            <ResourceGroup title="场景" items={resolved.scenes as RegistryResource[]} />
            <ResourceGroup title="操作对象" items={resolved.objects as RegistryResource[]} />
            <ResourceGroup title="评测指标" items={resolved.metrics as RegistryResource[]} />
            <ResourceGroup title="策略" items={resolved.policies as RegistryResource[]} />
          </DetailSection>
        ) : !assetRow && !isIsaacTemplate ? (
          <p style={{ color: '#6b7280', fontSize: 14 }}>暂无任务模板详情。</p>
        ) : null}

        {matchedTemplate ? (
          <div style={{ marginTop: 8 }}>
            <button
              type="button"
              onClick={() => setShowInternal((v) => !v)}
              style={{
                padding: 0,
                border: 'none',
                background: 'none',
                color: '#6b7280',
                fontSize: 12,
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
            >
              {showInternal ? '收起内部信息' : '展开内部信息'}
            </button>
            {showInternal ? (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 8,
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                }}
              >
                <DetailRow label="simBackend" value={matchedTemplate.simulatorBackend ?? '—'} />
                <DetailRow
                  label="simulatorBackendLabel"
                  value={matchedTemplate.simulatorBackendLabel ?? '—'}
                />
                <DetailRow label="registryTaskConfigId" value={matchedTemplate.registryTaskConfigId ?? '—'} />
                <DetailRow label="physicsBackend" value={matchedTemplate.physicsBackend ?? '—'} />
                <DetailRow label="defaultEnv" value={matchedTemplate.defaultEnv ?? '—'} />
                <DetailRow label="adapterStatus" value={matchedTemplate.adapterStatus ?? '—'} />
                <DetailRow
                  label="requiresExternalRuntime"
                  value={matchedTemplate.requiresExternalRuntime ? 'true' : 'false'}
                />
                <DetailRow
                  label="defaultMetricIds"
                  value={matchedTemplate.defaultMetricIds?.join(', ') ?? '—'}
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
