'use client';

import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ChangeEvent, type DragEvent } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Upload } from 'lucide-react';
import {
  ModulePageContainer,
} from '@/components/layout/ModulePageLayout';
import {
  boxesToPayload,
  ImageBoxAnnotator,
  type AnnotationBox,
  type BoxMode,
} from '@/components/workspace/sim-assets/ImageBoxAnnotator';
import {
  CutoutLargePreview,
  type CutoutManifestItem,
} from '@/components/workspace/sim-assets/CutoutCandidateGrid';
import { AssetExportDialog } from '@/components/workspace/sim-assets/AssetExportDialog';
import { AssetInteractivePreview } from '@/components/workspace/sim-assets/AssetInteractivePreview';
import { CutoutOverlayPreview } from '@/components/workspace/sim-assets/CutoutOverlayPreview';
import { CutoutSelectDialog } from '@/components/workspace/sim-assets/CutoutSelectDialog';
import { ReconstructProcessGuide } from '@/components/workspace/sim-assets/ReconstructProcessGuide';
import {
  SimAssetBackLink,
  SimAssetToast,
  formControlStyle,
} from '@/components/workspace/sim-assets/simAssetUi';
import {
  GhostButton,
  PlaceholderPanel,
  PrimaryButton,
  SecondaryButton,
  WS,
} from '@/components/workspace/workspaceUi';
import {
  createAssetPipelineJob,
  downloadAssetPipelineFile,
  fetchAssetPipelineFileBlob,
  formatAssetPipelineFileError,
  getAssetPipelineFileUrl,
  getAssetPipelineJob,
  PIPELINE_TERMINAL_STATUSES,
  resolveInputImageRelPath,
  resolveMujocoCollisionObjRelPath,
  resolveMujocoVisualObjRelPath,
  resolveObjectGlbRelPath,
  resolveAssetExportFiles,
  startAssetReconstruction,
  startAssetSegmentation,
  type AssetPipelineJobStatus,
} from '@/lib/api/sam3dAssetPipelineClient';
import type { ReconstructionJobDraft } from '@/types/simAsset';

const POLL_MS = 2000;
const POLLING_STATUSES = new Set(['segmenting', 'reconstructing']);
const WORKBENCH_HEIGHT = 'min(680px, calc(100vh - 220px))';
const WORKBENCH_MIN_HEIGHT = 520;

const UPLOAD_ACCEPT = '.png,.jpg,.jpeg,.webp';
const UPLOAD_ACCEPT_LIST = ['.png', '.jpg', '.jpeg', '.webp'];

function validateUploadExtension(name: string): boolean {
  const ext = name.includes('.') ? `.${name.split('.').pop()?.toLowerCase()}` : '';
  return UPLOAD_ACCEPT_LIST.some((item) => item === ext || item.endsWith(ext));
}

type WorkbenchPhase =
  | 'target_selection'
  | 'segmenting'
  | 'segmented'
  | 'reconstructing'
  | 'reconstructed'
  | 'failed';

function shouldPollJobStatus(status: string): boolean {
  return POLLING_STATUSES.has(status);
}

function resolveWorkbenchPhase(status: string | undefined | null): WorkbenchPhase {
  switch (status) {
    case 'segmenting':
      return 'segmenting';
    case 'segmented':
      return 'segmented';
    case 'reconstructing':
      return 'reconstructing';
    case 'reconstructed':
      return 'reconstructed';
    case 'failed':
      return 'failed';
    default:
      return 'target_selection';
  }
}

function restoreBoxesFromSegmentation(segmentation: Record<string, unknown> | null | undefined): AnnotationBox[] {
  if (!segmentation) return [];
  const positive = (segmentation.positiveBoxes as number[][] | undefined) || [];
  const negative = (segmentation.negativeBoxes as number[][] | undefined) || [];
  const restored: AnnotationBox[] = [];
  positive.forEach((coords, index) => {
    if (!Array.isArray(coords) || coords.length !== 4) return;
    restored.push({
      id: `pos_restored_${index}`,
      type: 'positive',
      x0: Math.round(coords[0]),
      y0: Math.round(coords[1]),
      x1: Math.round(coords[2]),
      y1: Math.round(coords[3]),
    });
  });
  negative.forEach((coords, index) => {
    if (!Array.isArray(coords) || coords.length !== 4) return;
    restored.push({
      id: `neg_restored_${index}`,
      type: 'negative',
      x0: Math.round(coords[0]),
      y0: Math.round(coords[1]),
      x1: Math.round(coords[2]),
      y1: Math.round(coords[3]),
    });
  });
  return restored;
}

interface ManifestItem extends CutoutManifestItem {}

function PreviewEmptyState({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: WORKBENCH_MIN_HEIGHT - 120,
        padding: 32,
        textAlign: 'center',
        color: '#94a3b8',
      }}
    >
      <div style={{ fontSize: 15, fontWeight: 500, color: '#64748b', marginBottom: subtitle ? 8 : 0 }}>
        {title}
      </div>
      {subtitle ? <div style={{ fontSize: 13, lineHeight: 1.6 }}>{subtitle}</div> : null}
    </div>
  );
}

function PreviewLoadingState({ label }: { label: string }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: WORKBENCH_MIN_HEIGHT - 120,
        fontSize: 14,
        color: '#64748b',
      }}
    >
      {label}
    </div>
  );
}

function SegmentedModeToggle({
  mode,
  onChange,
  disabled,
}: {
  mode: BoxMode;
  onChange: (mode: BoxMode) => void;
  disabled?: boolean;
}) {
  const itemStyle = (active: boolean): CSSProperties => ({
    padding: '8px 20px',
    fontSize: 13,
    fontWeight: 500,
    border: 'none',
    background: active ? '#fff' : 'transparent',
    color: active ? '#111827' : '#64748b',
    cursor: disabled ? 'not-allowed' : 'pointer',
    boxShadow: active ? '0 1px 3px rgba(15, 23, 42, 0.08)' : 'none',
    borderRadius: 6,
  });

  return (
    <div
      style={{
        display: 'inline-flex',
        padding: 3,
        borderRadius: 8,
        background: '#f1f5f9',
        border: '1px solid #e2e8f0',
        opacity: disabled ? 0.6 : 1,
      }}
    >
      <button type="button" disabled={disabled} style={itemStyle(mode === 'positive')} onClick={() => onChange('positive')}>
        正框
      </button>
      <button type="button" disabled={disabled} style={itemStyle(mode === 'negative')} onClick={() => onChange('negative')}>
        负框
      </button>
    </div>
  );
}

function PromptCompactInput({
  prompt,
  onPromptChange,
  disabled,
}: {
  prompt: string;
  onPromptChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        minWidth: 0,
        flex: '1 1 200px',
        maxWidth: 300,
      }}
    >
      <label style={{ fontSize: 12, color: '#64748b', whiteSpace: 'nowrap', flexShrink: 0 }}>文本提示</label>
      <input
        type="text"
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        placeholder="可选：输入目标描述"
        disabled={disabled}
        style={{
          ...formControlStyle,
          flex: 1,
          minWidth: 0,
          fontSize: 13,
          padding: '6px 10px',
        }}
      />
    </div>
  );
}

export default function SimAssetReconstructPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialJobId = searchParams.get('jobId');

  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(initialJobId);
  const [jobStatus, setJobStatus] = useState<AssetPipelineJobStatus | null>(null);
  const [polling, setPolling] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [editMaskMode, setEditMaskMode] = useState(false);
  const [boxes, setBoxes] = useState<AnnotationBox[]>([]);
  const [boxMode, setBoxMode] = useState<BoxMode>('positive');
  const [hoverPoint, setHoverPoint] = useState<{ x: number; y: number } | null>(null);
  const [boxesRestored, setBoxesRestored] = useState(false);
  const [confidenceThreshold] = useState(0.05);
  const [selectedCutoutIndex, setSelectedCutoutIndex] = useState<number | undefined>();
  const [draft, setDraft] = useState<ReconstructionJobDraft>({
    name: '',
    inputImage: null,
    prompt: '',
    positiveBoxes: [],
    negativeBoxes: [],
    targetEngine: 'mujoco',
    assetType: 'object',
  });
  const [inputImageObjectUrl, setInputImageObjectUrl] = useState<string | null>(null);
  const [inputImageLoadError, setInputImageLoadError] = useState<string | null>(null);
  const [inputImageLoading, setInputImageLoading] = useState(false);
  const [inputImageNaturalSize, setInputImageNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [inputImageReloadKey, setInputImageReloadKey] = useState(0);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);
  const [cutoutDialogOpen, setCutoutDialogOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);

  const pipelineStatus = jobStatus?.status;
  const workbenchPhase = resolveWorkbenchPhase(pipelineStatus);

  const inputImageRelPath = useMemo(
    () => (jobId ? resolveInputImageRelPath(jobStatus) : 'input/image.png'),
    [jobId, jobStatus]
  );

  const localImagePreviewUrl = useMemo(() => {
    if (!draft.inputImage) return null;
    return URL.createObjectURL(draft.inputImage);
  }, [draft.inputImage]);

  const imageUrl = inputImageObjectUrl ?? localImagePreviewUrl;
  const directApiUrl = jobId ? getAssetPipelineFileUrl(jobId, inputImageRelPath) : null;
  const annotationDisabled =
    busy === 'segment' || workbenchPhase === 'segmenting' || workbenchPhase === 'reconstructing';
  const positiveCount = boxes.filter((box) => box.type === 'positive').length;
  const negativeCount = boxes.filter((box) => box.type === 'negative').length;

  const showAnnotator =
    editMaskMode || workbenchPhase === 'target_selection' || workbenchPhase === 'segmenting';
  const showCutoutOverlay =
    !showAnnotator &&
    (workbenchPhase === 'segmented' ||
      workbenchPhase === 'reconstructing' ||
      workbenchPhase === 'reconstructed' ||
      (workbenchPhase === 'failed' && Boolean(jobStatus?.segmentation)));

  const cutoutItems = useMemo(() => {
    const items = (jobStatus?.segmentation?.items || []) as ManifestItem[];
    return items
      .filter((item) => item.cutoutPath || item.previewPath)
      .sort((a, b) => a.cutoutIndex - b.cutoutIndex);
  }, [jobStatus?.segmentation]);

  const selectedCutoutItem = useMemo(
    () => cutoutItems.find((item) => item.cutoutIndex === selectedCutoutIndex) ?? null,
    [cutoutItems, selectedCutoutIndex]
  );
  const objectGlbRelPath = useMemo(() => resolveObjectGlbRelPath(jobStatus), [jobStatus]);
  const mujocoVisualObjRelPath = useMemo(() => resolveMujocoVisualObjRelPath(jobStatus), [jobStatus]);
  const mujocoCollisionObjRelPath = useMemo(() => resolveMujocoCollisionObjRelPath(jobStatus), [jobStatus]);
  const exportFileCount = useMemo(() => resolveAssetExportFiles(jobStatus).length, [jobStatus]);
  const reconstructBusy = polling || workbenchPhase === 'reconstructing';
  const showOfflineHint = (jobStatus?.error || '').includes('Offline SAM3D dependencies missing');

  useEffect(() => {
    return () => {
      if (localImagePreviewUrl) URL.revokeObjectURL(localImagePreviewUrl);
    };
  }, [localImagePreviewUrl]);

  useEffect(() => {
    if (!jobId) {
      setInputImageObjectUrl(null);
      setInputImageLoadError(null);
      setInputImageLoading(false);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;
    setInputImageLoading(true);
    setInputImageLoadError(null);
    setInputImageNaturalSize(null);

    void fetchAssetPipelineFileBlob(jobId, inputImageRelPath)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setInputImageObjectUrl(objectUrl);
      })
      .catch((err) => {
        if (cancelled) return;
        setInputImageObjectUrl(null);
        setInputImageLoadError(err instanceof Error ? err.message : '原图加载失败');
      })
      .finally(() => {
        if (!cancelled) setInputImageLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId, inputImageRelPath, inputImageReloadKey]);

  useEffect(() => {
    if (!toastMsg) return;
    const timer = window.setTimeout(() => setToastMsg(null), 6000);
    return () => window.clearTimeout(timer);
  }, [toastMsg]);

  const refreshJob = useCallback(async (id: string) => {
    const status = await getAssetPipelineJob(id);
    setJobStatus(status);
    if (status.name) {
      setDraft((prev) => (prev.name ? prev : { ...prev, name: status.name || prev.name }));
    }
    const segPrompt = status.segmentation?.prompt;
    if (typeof segPrompt === 'string' && segPrompt) {
      setDraft((prev) => (prev.prompt ? prev : { ...prev, prompt: segPrompt }));
    }
    return status;
  }, []);

  useEffect(() => {
    if (!initialJobId) return;
    setJobId(initialJobId);
    void (async () => {
      try {
        setBusy('restore');
        const status = await refreshJob(initialJobId);
        if (shouldPollJobStatus(status.status)) setPolling(true);
      } catch (err) {
        setToastMsg(err instanceof Error ? err.message : '恢复任务失败');
      } finally {
        setBusy(null);
      }
    })();
  }, [initialJobId, refreshJob]);

  useEffect(() => {
    if (!jobId || !polling) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const status = await refreshJob(jobId);
        if (cancelled) return;
        if (PIPELINE_TERMINAL_STATUSES.has(status.status)) {
          setPolling(false);
          if (status.status === 'segmented') setEditMaskMode(false);
          if (status.status === 'failed') {
            setToastMsg(status.error || '任务失败，请查看日志。');
          }
        }
      } catch (err) {
        if (!cancelled) {
          setPolling(false);
          setToastMsg(err instanceof Error ? err.message : '轮询失败');
        }
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, polling, refreshJob]);

  useEffect(() => {
    if (!cutoutItems.length) return;

    if (
      selectedCutoutIndex != null &&
      cutoutItems.some((item) => item.cutoutIndex === selectedCutoutIndex)
    ) {
      return;
    }

    const fromRecon = jobStatus?.reconstruction?.cutoutIndex;
    if (
      typeof fromRecon === 'number' &&
      cutoutItems.some((item) => item.cutoutIndex === fromRecon)
    ) {
      setSelectedCutoutIndex(fromRecon);
      return;
    }

    const first = cutoutItems.find((item) => item.selectable !== false) ?? cutoutItems[0];
    setSelectedCutoutIndex(first.cutoutIndex);
  }, [cutoutItems, jobStatus?.reconstruction?.cutoutIndex, selectedCutoutIndex]);

  useEffect(() => {
    if (boxesRestored || !jobStatus?.segmentation) return;
    const restored = restoreBoxesFromSegmentation(jobStatus.segmentation);
    if (restored.length) {
      setBoxes(restored);
      setBoxesRestored(true);
    }
  }, [boxesRestored, jobStatus?.segmentation]);

  const handleUploadFileChange = (file: File | null) => {
    if (!file) {
      setDraft((p) => ({ ...p, inputImage: null }));
      return;
    }
    if (!validateUploadExtension(file.name)) {
      setToastMsg(`文件格式不支持，请选择 ${UPLOAD_ACCEPT_LIST.join(' / ')}`);
      return;
    }
    setDraft((p) => ({ ...p, inputImage: file }));
  };

  const handleUploadInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleUploadFileChange(event.target.files?.[0] ?? null);
    event.target.value = '';
  };

  const handleUploadDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    handleUploadFileChange(event.dataTransfer.files?.[0] ?? null);
  };

  const handleCreateJob = async () => {
    if (!draft.inputImage) return setToastMsg('请先上传图片。');
    if (!draft.name.trim()) return setToastMsg('请填写资产名称。');
    setBusy('create');
    try {
      const result = await createAssetPipelineJob(draft.name.trim(), draft.inputImage);
      setJobId(result.jobId);
      await refreshJob(result.jobId);
      router.replace(
        `/workspace/resources/sim-assets/reconstruct?jobId=${encodeURIComponent(result.jobId)}`
      );
      setToastMsg('图片已上传，请在左侧选择目标对象。');
    } catch (err) {
      setToastMsg(err instanceof Error ? err.message : '创建任务失败');
    } finally {
      setBusy(null);
    }
  };

  const handleRunSegmentation = async () => {
    if (!jobId) return setToastMsg('请先创建任务。');
    const { positiveBoxes, negativeBoxes } = boxesToPayload(boxes);
    if (!draft.prompt.trim() && positiveBoxes.length === 0) {
      return setToastMsg('请填写 prompt 或至少添加一个正框。');
    }
    setBusy('segment');
    try {
      const status = await startAssetSegmentation(jobId, {
        prompt: draft.prompt.trim() || null,
        positiveBoxes,
        negativeBoxes,
        confidenceThreshold,
        textOnly: positiveBoxes.length === 0,
      });
      setJobStatus(status);
      setPolling(true);
      setEditMaskMode(false);
    } catch (err) {
      setToastMsg(err instanceof Error ? err.message : '启动分割失败');
    } finally {
      setBusy(null);
    }
  };

  const handleRunReconstruction = async () => {
    if (!jobId || selectedCutoutIndex == null) return setToastMsg('请选择一个 cutout。');
    setBusy('reconstruct');
    try {
      const status = await startAssetReconstruction(jobId, {
        cutoutIndex: selectedCutoutIndex,
        seed: 42,
      });
      setJobStatus(status);
      setPolling(true);
    } catch (err) {
      setToastMsg(err instanceof Error ? err.message : '启动重建失败');
    } finally {
      setBusy(null);
    }
  };

  const handleDownloadFile = async (relPath: string, filename: string) => {
    if (!jobId) return;
    setDownloadingFile(relPath);
    try {
      await downloadAssetPipelineFile(jobId, relPath, filename);
      setToastMsg('下载已开始');
    } catch (err) {
      setToastMsg(formatAssetPipelineFileError(err, '下载'));
    } finally {
      setDownloadingFile(null);
    }
  };

  const renderRightPreview = () => {
    if (workbenchPhase === 'failed') {
      return (
        <PreviewEmptyState
          title="任务失败"
          subtitle={jobStatus?.error || '请检查日志或重新编辑框选后重试。'}
        />
      );
    }

    if (workbenchPhase === 'target_selection') {
      return (
        <PreviewEmptyState
          title="三维模型将在此显示"
          subtitle="请框选目标并确认分割"
        />
      );
    }

    if (workbenchPhase === 'segmenting') {
      return <PreviewLoadingState label="正在分割目标…" />;
    }

    if (workbenchPhase === 'segmented') {
      return (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            padding: 16,
            boxSizing: 'border-box',
            gap: 12,
          }}
        >
          {selectedCutoutItem ? (
            <>
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                <strong>cutout #{selectedCutoutItem.cutoutIndex}</strong>
                {selectedCutoutItem.score != null ? ` · 得分 ${selectedCutoutItem.score.toFixed(4)}` : ''}
              </div>
              <CutoutLargePreview jobId={jobId!} item={selectedCutoutItem} />
              <SecondaryButton
                onClick={() => setCutoutDialogOpen(true)}
                disabled={!jobId || cutoutItems.length === 0}
              >
                选择 cutout
              </SecondaryButton>
            </>
          ) : (
            <PreviewEmptyState
              title="分割结果将在此显示"
              subtitle={
                cutoutItems.length
                  ? '请选择一个 cutout 作为三维重建输入'
                  : '暂无可用 cutout，请重新编辑框选并分割'
              }
            />
          )}
        </div>
      );
    }

    if (workbenchPhase === 'reconstructing') {
      return <PreviewLoadingState label="正在生成三维模型…" />;
    }

    if (workbenchPhase === 'reconstructed') {
      return (
        <div style={{ height: '100%', padding: 12, boxSizing: 'border-box' }}>
          {jobId && (objectGlbRelPath || mujocoVisualObjRelPath) ? (
            <AssetInteractivePreview
              jobId={jobId}
              objectGlbPath={objectGlbRelPath}
              mujocoVisualObjPath={mujocoVisualObjRelPath}
              mujocoCollisionObjPath={mujocoCollisionObjRelPath}
            />
          ) : (
            <PreviewEmptyState title="三维预览不可用" subtitle="重建文件尚未就绪" />
          )}
        </div>
      );
    }

    return null;
  };

  const renderPrimaryAction = () => {
    if (workbenchPhase === 'target_selection' || editMaskMode) {
      return (
        <PrimaryButton
          onClick={() => void handleRunSegmentation()}
          disabled={!jobId || annotationDisabled}
        >
          {workbenchPhase === 'segmenting' ? '分割中…' : '确认分割'}
        </PrimaryButton>
      );
    }
    if (workbenchPhase === 'segmenting') {
      return (
        <PrimaryButton disabled>
          分割中…
        </PrimaryButton>
      );
    }
    if (workbenchPhase === 'segmented') {
      return (
        <PrimaryButton
          onClick={() => void handleRunReconstruction()}
          disabled={!jobId || selectedCutoutIndex == null || reconstructBusy}
        >
          {reconstructBusy ? '生成中…' : '生成三维模型'}
        </PrimaryButton>
      );
    }
    if (workbenchPhase === 'reconstructing') {
      return (
        <PrimaryButton disabled>
          生成中…
        </PrimaryButton>
      );
    }
    return null;
  };

  return (
    <ModulePageContainer>
      <SimAssetBackLink href="/workspace/resources/sim-assets" label="返回" />

      {!jobId ? (
        <div
          style={{
            ...WS.card,
            display: 'flex',
            minHeight: WORKBENCH_MIN_HEIGHT,
            borderRadius: 12,
            overflow: 'hidden',
            padding: 0,
          }}
        >
          <aside
            style={{
              width: 320,
              flexShrink: 0,
              padding: '24px 20px',
              background: '#f8fafc',
              borderRight: '1px solid #e5e7eb',
              boxSizing: 'border-box',
            }}
          >
            <ReconstructProcessGuide activeStep={0} />
          </aside>

          <section
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '32px 40px',
              boxSizing: 'border-box',
            }}
          >
            <div
              style={{
                width: '100%',
                maxWidth: 520,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 20,
              }}
            >
              <div
                style={{
                  width: '100%',
                  padding: '36px 28px',
                  borderRadius: 16,
                  border: '2px dashed #cbd5e1',
                  background: '#fafbfc',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 12,
                  textAlign: 'center',
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={handleUploadDrop}
              >
                <label
                  htmlFor="reconstruct-upload-input"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '10px 18px',
                    borderRadius: 12,
                    border: '1px solid #cbd5e1',
                    background: '#fff',
                    color: '#0f172a',
                    fontSize: 14,
                    fontWeight: 500,
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(15, 23, 42, 0.04)',
                  }}
                >
                  <Upload size={18} strokeWidth={1.75} color="#2563eb" />
                  <span>{draft.inputImage ? draft.inputImage.name : '上传图片'}</span>
                </label>
                <input
                  id="reconstruct-upload-input"
                  type="file"
                  accept={UPLOAD_ACCEPT}
                  style={{ display: 'none' }}
                  onChange={handleUploadInputChange}
                />
                <div style={{ fontSize: 13, color: '#64748b' }}>支持 PNG / JPG / WEBP</div>
              </div>

              <div style={{ width: '100%', maxWidth: 360 }}>
                <label style={{ display: 'block', fontSize: 12, color: '#64748b', marginBottom: 6 }}>
                  资产名称
                </label>
                <input
                  type="text"
                  value={draft.name}
                  onChange={(e) => setDraft((p) => ({ ...p, name: e.target.value }))}
                  style={{ ...formControlStyle, fontSize: 13 }}
                />
              </div>

              <PrimaryButton onClick={() => void handleCreateJob()} disabled={busy === 'create'}>
                {busy === 'create' ? '创建中…' : '创建重建任务'}
              </PrimaryButton>
            </div>
          </section>
        </div>
      ) : (
        <div
          style={{
            ...WS.card,
            borderRadius: 12,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            height: WORKBENCH_HEIGHT,
            minHeight: WORKBENCH_MIN_HEIGHT,
          }}
        >
          {/* Toolbar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              padding: '10px 16px',
              borderBottom: '1px solid #e5e7eb',
              background: '#fff',
              flexShrink: 0,
              gap: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {showOfflineHint ? (
                <span style={{ fontSize: 11, color: '#b45309' }}>SAM3D 离线依赖缺失</span>
              ) : null}
              {workbenchPhase === 'reconstructed' && jobStatus?.mujocoExport?.status === 'completed' ? (
                <span style={{ fontSize: 12, color: '#64748b' }}>MuJoCo 资产：已完成</span>
              ) : null}
              {workbenchPhase === 'reconstructed' ? (
                <PrimaryButton
                  onClick={() => setExportDialogOpen(true)}
                  disabled={!jobId || exportFileCount === 0}
                >
                  导出{exportFileCount > 0 ? ` (${exportFileCount})` : ''}
                </PrimaryButton>
              ) : null}
            </div>
          </div>

          {/* Main split view */}
          <div
            style={{
              flex: 1,
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
              minHeight: 0,
              overflow: 'hidden',
            }}
          >
            {/* Left: image panel */}
            <section
              style={{
                borderRight: '1px solid #e5e7eb',
                background: '#f8fafc',
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
                overflow: 'hidden',
              }}
            >
              <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: 12 }}>
                {inputImageLoading ? (
                  <PlaceholderPanel label="正在加载原图…" height={360} />
                ) : showAnnotator ? (
                  imageUrl ? (
                    <ImageBoxAnnotator
                      imageUrl={imageUrl}
                      boxes={boxes}
                      mode={boxMode}
                      disabled={annotationDisabled}
                      onBoxesChange={setBoxes}
                      onImageSizeChange={(size) => setInputImageNaturalSize(size)}
                      onImageError={(message) => setInputImageLoadError(message)}
                      onHoverPointChange={setHoverPoint}
                      maxPanelHeight={560}
                    />
                  ) : (
                    <PlaceholderPanel label="等待原图加载" height={360} />
                  )
                ) : showCutoutOverlay && jobId ? (
                  <CutoutOverlayPreview
                    jobId={jobId}
                    originalImagePath={inputImageRelPath}
                    selectedItem={selectedCutoutItem}
                    emptyLabel={
                      cutoutItems.length
                        ? '请选择一个 cutout 作为三维重建输入'
                        : '暂无可用 cutout，请编辑框选后重新分割'
                    }
                  />
                ) : (
                  <PlaceholderPanel label="等待任务加载" height={360} />
                )}

                {inputImageLoadError ? (
                  <div
                    style={{
                      marginTop: 12,
                      padding: 12,
                      borderRadius: 8,
                      background: '#fef2f2',
                      color: '#b91c1c',
                      fontSize: 13,
                    }}
                  >
                    <div>
                      <strong>原图加载失败</strong>
                    </div>
                    <div style={{ marginTop: 6 }}>{inputImageLoadError}</div>
                    {directApiUrl ? (
                      <div style={{ marginTop: 6, fontSize: 11, color: '#6b7280', wordBreak: 'break-all' }}>
                        {directApiUrl}
                      </div>
                    ) : null}
                    <div style={{ marginTop: 10 }}>
                      <SecondaryButton onClick={() => setInputImageReloadKey((key) => key + 1)}>
                        重试加载
                      </SecondaryButton>
                    </div>
                  </div>
                ) : null}
              </div>

              {showAnnotator && inputImageNaturalSize ? (
                <div
                  style={{
                    flexShrink: 0,
                    padding: '8px 12px',
                    borderTop: '1px solid #e5e7eb',
                    background: '#fff',
                    fontSize: 11,
                    color: '#94a3b8',
                    lineHeight: 1.5,
                  }}
                >
                  {inputImageNaturalSize.width} × {inputImageNaturalSize.height} px · 正框 {positiveCount} · 负框{' '}
                  {negativeCount}
                  {hoverPoint ? ` · ${hoverPoint.x}, ${hoverPoint.y}` : ''}
                </div>
              ) : null}
            </section>

            {/* Right: preview panel */}
            <section
              style={{
                background: '#fff',
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
                overflow: 'hidden',
              }}
            >
              <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>{renderRightPreview()}</div>
            </section>
          </div>

          {/* Bottom bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              minHeight: 72,
              padding: '0 16px',
              borderTop: '1px solid #e5e7eb',
              background: '#fff',
              flexShrink: 0,
              gap: 12,
            }}
          >
            {showAnnotator ? (
              <>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  <PromptCompactInput
                    prompt={draft.prompt}
                    onPromptChange={(value) => setDraft((p) => ({ ...p, prompt: value }))}
                    disabled={annotationDisabled}
                  />
                  <SegmentedModeToggle mode={boxMode} onChange={setBoxMode} disabled={annotationDisabled} />
                </div>
                <div style={{ flex: '0 0 auto' }}>{renderPrimaryAction()}</div>
              </>
            ) : (
              <>
                <div style={{ flex: '0 0 auto' }}>
                  {(workbenchPhase === 'segmented' ||
                    workbenchPhase === 'reconstructing' ||
                    workbenchPhase === 'reconstructed' ||
                    workbenchPhase === 'failed') && !editMaskMode ? (
                    <GhostButton
                      onClick={() => setEditMaskMode(true)}
                      disabled={workbenchPhase === 'reconstructing'}
                    >
                      编辑框选
                    </GhostButton>
                  ) : editMaskMode ? (
                    <GhostButton onClick={() => setEditMaskMode(false)}>取消编辑</GhostButton>
                  ) : null}
                </div>
                <div style={{ flex: '0 0 auto' }}>{renderPrimaryAction()}</div>
              </>
            )}
          </div>
        </div>
      )}

      {jobId ? (
        <>
          <CutoutSelectDialog
            open={cutoutDialogOpen}
            jobId={jobId}
            items={cutoutItems}
            selectedCutoutIndex={selectedCutoutIndex}
            onConfirm={(cutoutIndex) => {
              setSelectedCutoutIndex(cutoutIndex);
              setCutoutDialogOpen(false);
            }}
            onCancel={() => setCutoutDialogOpen(false)}
          />
          <AssetExportDialog
            open={exportDialogOpen}
            onClose={() => setExportDialogOpen(false)}
            jobId={jobId}
            jobStatus={jobStatus}
            downloadingFile={downloadingFile}
            onDownload={handleDownloadFile}
          />
        </>
      ) : null}

      <SimAssetToast message={toastMsg} />
    </ModulePageContainer>
  );
}
