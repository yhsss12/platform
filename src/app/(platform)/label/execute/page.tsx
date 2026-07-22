'use client';

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { FrameCacheProvider, useFrameCache } from '@/features/label-runner/context/FrameCacheContext';
import EpisodeViewerLayout from '@/features/label-runner/components/EpisodeViewerLayout';
import RightPanel from '@/features/label-runner/components/RightPanel';
import type { AgentLogEntry } from '@/features/label-runner/models';
import { loadLabelTasks } from '@/features/data-platform/storage/labelTasks';
import { saveAgentLog } from '@/features/label-runner/storage';
import { useI18n } from '@/components/common/I18nProvider';
import { useAuthStore } from '@/store/authStore';
import { canAnnotateLabelTask } from '@/lib/label/labelTaskActorPermissions';
import { getLabelTaskActorContext } from '@/features/label-runner/api/labelApi';
import { isNetworkOrConnectionError, resolveNetworkConnectionMessage } from '@/lib/errors/networkError';

type ViewerState = {
  selectedEpisode: string | null;
  viewportTopics: (string | null)[];
  maxFrame: number;
  taskId: string;
  cameras: string[];
};

const MAX_PRELOAD_CAMERAS = 4;
const RESULT_CACHE_PREFIX = 'label_execute:last_generated:';

function resultCacheKey(taskId: string): string {
  return `${RESULT_CACHE_PREFIX}${taskId}`;
}

function LabelExecuteContent() {
  const searchParams = useSearchParams();
  const backendTaskId = searchParams.get('taskId') || '';
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);
  const [annotateGate, setAnnotateGate] = useState<'loading' | 'ok' | 'denied'>('loading');
  const [actorCtx, setActorCtx] = useState<{ project_id?: string | null } | null>(null);

  const [viewerState, setViewerState] = useState<ViewerState | null>(null);
  const [taskName, setTaskName] = useState('');
  const [agentLogs, setAgentLogs] = useState<AgentLogEntry[]>([]);
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [generatedDescription, setGeneratedDescription] = useState<string>('');
  const [annotationProgress, setAnnotationProgress] = useState(0);
  const [batchAnnotationLoading, setBatchAnnotationLoading] = useState(false);
  const [batchAnnotationResults, setBatchAnnotationResults] = useState<
    Array<{ episode_id: string; path?: string; instruction?: string; error?: string }>
  >([]);
  const [toast, setToast] = useState<{ message: string; isError?: boolean } | null>(null);
  const showToast = useCallback((message: string, isError?: boolean) => {
    setToast({ message, isError });
    setTimeout(() => setToast(null), 2200);
  }, []);

  const agentIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const descriptionUpdateRef = useRef<((desc: string) => void) | null>(null);
  const instructionsRefreshRef = useRef<{ refresh: () => void } | null>(null);
  /** 自动/批量标注写入后端后刷新左侧 episodes（instruction_text、已标注） */
  const episodesRefreshRef = useRef<{ refresh: () => Promise<void> } | null>(null);

  const { preload, clearCache, preloadProgress, isPreloading, preloadError } = useFrameCache();
  const MODEL_BUSY_MSG = 'API 节点忙，请稍后重试';
  const NETWORK_TIMEOUT_MSG = t('feedback.networkConnectionTimeout');
  const normalizeModelBusyMessage = useCallback(
    (raw?: string) => {
      const msg = String(raw || '').trim();
      if (!msg) return t('feedback.requestFailed');
      if (isNetworkOrConnectionError(msg)) {
        return resolveNetworkConnectionMessage(msg, NETWORK_TIMEOUT_MSG);
      }
      const lower = msg.toLowerCase();
      const isBusy =
        (lower.includes('http 503') && lower.includes('openai')) ||
        lower.includes('currently experiencing high demand') ||
        (lower.includes('upstream_error') && lower.includes('503')) ||
        (lower.includes('code') && lower.includes('503'));
      return isBusy ? MODEL_BUSY_MSG : msg;
    },
    [t, NETWORK_TIMEOUT_MSG]
  );
  const isModelBusyMessage = useCallback(
    (raw?: string) => normalizeModelBusyMessage(raw) === MODEL_BUSY_MSG,
    [normalizeModelBusyMessage]
  );
  const isNetworkTimeoutMessage = useCallback(
    (raw?: string) => normalizeModelBusyMessage(raw) === NETWORK_TIMEOUT_MSG,
    [normalizeModelBusyMessage, NETWORK_TIMEOUT_MSG]
  );

  useEffect(() => {
    if (!backendTaskId) {
      setAnnotateGate('denied');
      return;
    }
    let cancelled = false;
    setAnnotateGate('loading');
    (async () => {
      const res = await getLabelTaskActorContext(backendTaskId);
      if (cancelled) return;
      if (!res.ok || !res.data) {
        setAnnotateGate('denied');
        setActorCtx(null);
        return;
      }
      setActorCtx({ project_id: res.data.project_id });
      const ok = canAnnotateLabelTask(authUser, res.data);
      setAnnotateGate(ok ? 'ok' : 'denied');
    })();
    return () => {
      cancelled = true;
    };
  }, [backendTaskId, authUser]);

  useEffect(() => {
    if (!backendTaskId) return;
    const tasks = loadLabelTasks();
    const task = tasks.find((x) => x.backendTaskId === backendTaskId || x.id === backendTaskId);
    if (task) setTaskName(task.name);
  }, [backendTaskId]);

  useEffect(() => {
    const { selectedEpisode, viewportTopics, maxFrame, taskId } = viewerState ?? {};
    if (!selectedEpisode || !taskId || !viewportTopics?.length) return;
    clearCache();
    const cameras = [...new Set(viewportTopics.filter(Boolean) as string[])].slice(0, MAX_PRELOAD_CAMERAS);
    const toLoad = Math.min(maxFrame || 150, 500);
    (async () => {
      try {
        for (const camera of cameras) {
          await preload(selectedEpisode, camera, 0, toLoad, taskId);
        }
      } catch {
        // 预加载失败仅记录后端/控制台，不在前端弹窗
      }
    })();
  }, [viewerState?.selectedEpisode, viewerState?.taskId, viewerState?.maxFrame, viewerState?.viewportTopics, preload, clearCache]);

  useEffect(() => {
    if (!preloadError) return;
    // 预加载错误不再前端提示，保持静默
  }, [preloadError]);

  const handleNewAnnotation = useCallback(() => {
    setAgentLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toTimeString().slice(0, 8),
        message: t('labelAgentPanel.toastNewSession'),
      },
    ]);
  }, [t]);

  const handleSave = useCallback(() => {
    const episode = viewerState?.selectedEpisode;
    if (episode && agentLogs.length > 0 && backendTaskId) {
      const logText = agentLogs.map((log) => `[${log.timestamp}] ${log.message}`).join('\n');
      saveAgentLog(backendTaskId, episode, logText);
    }
  }, [viewerState?.selectedEpisode, agentLogs, backendTaskId]);
  useEffect(() => {
    if (!backendTaskId || typeof window === 'undefined') return;
    try {
      const raw = window.localStorage.getItem(resultCacheKey(backendTaskId));
      if (!raw) return;
      const parsed = JSON.parse(raw) as { description?: string; successMessage?: string };
      if (parsed.description && parsed.description.trim()) {
        setGeneratedDescription(parsed.description);
      }
      if (parsed.successMessage && parsed.successMessage.trim()) {
        showToast(parsed.successMessage, false);
      }
    } catch {
      // ignore malformed cache
    }
  }, [backendTaskId, showToast]);

  const handleStartAgent = useCallback(
    async (cameraName?: string, modelConfig?: { selectedModels: string[]; apiConfig: { apiKey: string; baseUrl: string } }) => {
      const selectedEpisode = viewerState?.selectedEpisode;
      if (!selectedEpisode) return;

      setIsAgentRunning(true);
      setAgentLogs([]);
      setAnnotationProgress(0);

      const modelsToUse = modelConfig?.selectedModels?.length ? modelConfig.selectedModels : [];
      const options =
        modelsToUse.length > 0 && modelConfig?.apiConfig
          ? {
              model: modelsToUse[0],
              openai_api_key: modelConfig.apiConfig.apiKey || undefined,
              openai_base_url: modelConfig.apiConfig.baseUrl || undefined,
            }
          : undefined;

      const addLog = (message: string) => {
        const now = new Date();
        const timestamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
        setAgentLogs((prev) => [...prev, { timestamp, message }]);
      };

      try {
        const { generateAnnotation, getAnnotationStatus } = await import('@/features/label-runner/api/labelApi');
        addLog(t('labelAgentPanel.logStartAutoLabel', { episode: selectedEpisode }));
        if (cameraName) addLog(t('labelAgentPanel.logUseCamera', { camera: cameraName }));
        if (options?.model) addLog(t('labelAgentPanel.logUseModel', { model: options.model }));
        if (backendTaskId) addLog(t('labelAgentPanel.logTaskId', { taskId: backendTaskId }));

        const response = await generateAnnotation(selectedEpisode, cameraName, backendTaskId, options);
        if (!response.ok || !response.data) {
          const errMsg = normalizeModelBusyMessage(response.error);
          addLog(t('labelAgentPanel.logError', { error: errMsg }));
          if (isModelBusyMessage(response.error)) {
            showToast(MODEL_BUSY_MSG);
          } else if (isNetworkTimeoutMessage(response.error)) {
            showToast(NETWORK_TIMEOUT_MSG, true);
          }
          setIsAgentRunning(false);
          return;
        }

        const jobId = response.data.jobId;
        addLog(t('labelAgentPanel.logJobCreated', { jobId }));
        addLog(t('labelAgentPanel.logGeneratingWait'));

        let pollCount = 0;
        const maxPollCount = 300;
        const pollInterval = setInterval(async () => {
          pollCount++;
          if (pollCount > maxPollCount) {
            clearInterval(pollInterval);
            addLog(t('labelAgentPanel.logTimeoutStopped'));
            setIsAgentRunning(false);
            return;
          }
          try {
            const statusResponse = await getAnnotationStatus(jobId);
            if (!statusResponse.ok || !statusResponse.data) {
              clearInterval(pollInterval);
              addLog(t('labelAgentPanel.logQueryStatusFailed', { error: statusResponse.error || t('feedback.error') }));
              setIsAgentRunning(false);
              return;
            }
            const job = statusResponse.data;
            setAnnotationProgress(job.progress ?? 0);
            if (job.status === 'completed') {
              clearInterval(pollInterval);
              if (job.result) {
                addLog(t('labelAgentPanel.logCompleted', { result: job.result }));
                setGeneratedDescription(job.result);
                descriptionUpdateRef.current?.(job.result);
                const { saveInstruction, updateLabelTask } = await import('@/features/label-runner/api/labelApi');
                const saveRes = await saveInstruction(selectedEpisode, job.result, undefined, backendTaskId);
                if (!saveRes.ok) {
                  console.warn('自动保存 instruction 失败:', saveRes.error);
                }
                // 自动标注完成后：任务状态自动置为已完成，校验状态重置为未校验（待审核员人工确认）
                if (backendTaskId) {
                  updateLabelTask(backendTaskId, { completed: true, verified: false }).catch((err) =>
                    console.warn('自动更新任务完成/校验状态失败:', err)
                  );
                }
                instructionsRefreshRef.current?.refresh();
                await episodesRefreshRef.current?.refresh();
                if (!saveRes.ok) {
                  console.warn('自动保存 instruction 失败:', saveRes.error);
                  showToast(saveRes.error || '标注已生成，但保存到任务/数据库失败，请查看控制台', true);
                } else {
                  const successMessage = '标注结果生成成功';
                  showToast(successMessage, false);
                  if (typeof window !== 'undefined') {
                    try {
                      window.localStorage.setItem(
                        resultCacheKey(backendTaskId),
                        JSON.stringify({
                          description: job.result,
                          successMessage,
                          savedAt: Date.now(),
                        })
                      );
                    } catch {
                      // ignore storage failure
                    }
                  }
                }
              } else {
                addLog(t('labelAgentPanel.logCompletedNoResult'));
              }
              setIsAgentRunning(false);
            } else if (job.status === 'failed') {
              clearInterval(pollInterval);
              const failMsg = normalizeModelBusyMessage(job.error);
              addLog(t('labelAgentPanel.logFailed', { error: failMsg }));
              if (isModelBusyMessage(job.error)) {
                showToast(MODEL_BUSY_MSG);
              } else if (isNetworkTimeoutMessage(job.error)) {
                showToast(NETWORK_TIMEOUT_MSG, true);
              }
              setIsAgentRunning(false);
            } else if (job.status === 'cancelled') {
              clearInterval(pollInterval);
              addLog(t('labelAgentPanel.logCanceled'));
              setIsAgentRunning(false);
            }
          } catch (error) {
            clearInterval(pollInterval);
            addLog(
              t('labelAgentPanel.logQueryStatusException', {
                error: normalizeModelBusyMessage(error instanceof Error ? error.message : String(error)),
              })
            );
            const pollErrRaw = error instanceof Error ? error.message : String(error);
            if (isModelBusyMessage(pollErrRaw)) {
              showToast(MODEL_BUSY_MSG);
            } else if (isNetworkTimeoutMessage(pollErrRaw)) {
              showToast(NETWORK_TIMEOUT_MSG, true);
            }
            setIsAgentRunning(false);
          }
        }, 1000);
        agentIntervalRef.current = pollInterval;
      } catch (error) {
        const errRaw = error instanceof Error ? error.message : String(error);
        const errMsg = normalizeModelBusyMessage(errRaw);
        addLog(t('labelAgentPanel.logError', { error: errMsg }));
        if (isModelBusyMessage(errRaw)) {
          showToast(MODEL_BUSY_MSG);
        } else if (isNetworkTimeoutMessage(errRaw)) {
          showToast(NETWORK_TIMEOUT_MSG, true);
        }
        setIsAgentRunning(false);
      }
    },
    [
      viewerState?.selectedEpisode,
      backendTaskId,
      t,
      normalizeModelBusyMessage,
      isModelBusyMessage,
      isNetworkTimeoutMessage,
      showToast,
      NETWORK_TIMEOUT_MSG,
    ]
  );

  const handleStopAgent = useCallback(() => {
    if (agentIntervalRef.current) {
      clearInterval(agentIntervalRef.current);
      agentIntervalRef.current = null;
    }
    setIsAgentRunning(false);
  }, []);

  const handleBatchAnnotation = useCallback(
    async (
      cameraName?: string,
      modelConfig?: { selectedModels: string[]; apiConfig: { apiKey: string; baseUrl: string } }
    ) => {
      if (!backendTaskId) return;
      setBatchAnnotationLoading(true);
      try {
        const { batchGenerateAnnotation, updateLabelTask } = await import('@/features/label-runner/api/labelApi');
        const modelsToUse = modelConfig?.selectedModels?.length ? modelConfig.selectedModels : [];
        const apiOpt = modelConfig?.apiConfig
          ? {
              openai_api_key: modelConfig.apiConfig.apiKey || undefined,
              openai_base_url: modelConfig.apiConfig.baseUrl || undefined,
            }
          : undefined;

        if (modelsToUse.length === 0) {
          const res = await batchGenerateAnnotation(backendTaskId, cameraName);
          if (res.ok && res.data?.results) {
            setBatchAnnotationResults(res.data.results);
            const failed = res.data.results.filter((r: { error?: string }) => r.error);
            const success = res.data.results.filter((r: { error?: string }) => !r.error);
            if (failed.length === 0) {
              showToast(t('labelAgentPanel.batchDoneSuccess', { n: success.length }));
              await updateLabelTask(backendTaskId, { completed: true, verified: false }).catch((err) =>
                console.warn('自动更新任务完成/校验状态失败:', err)
              );
            } else {
              const normalizedReasons = failed
                .map((r: { error?: string }) => normalizeModelBusyMessage(r.error))
                .join('; ');
              const hasOnlyModelBusy = failed.every((r: { error?: string }) => isModelBusyMessage(r.error));
              if (hasOnlyModelBusy) {
                showToast(MODEL_BUSY_MSG);
              }
              showToast(
                t('labelAgentPanel.batchDoneWithFailed', {
                  success: success.length,
                  failed: failed.length,
                  reasons: normalizedReasons,
                }),
                !hasOnlyModelBusy
              );
            }
            await episodesRefreshRef.current?.refresh();
            await instructionsRefreshRef.current?.refresh();
          } else {
            const msg = normalizeModelBusyMessage(res.error || t('labelAgentPanel.batchFailed'));
            if (isModelBusyMessage(res.error)) {
              showToast(MODEL_BUSY_MSG);
            } else if (isNetworkTimeoutMessage(res.error)) {
              showToast(NETWORK_TIMEOUT_MSG, true);
            } else {
              showToast(msg, true);
            }
          }
        } else {
          let lastResults: Array<{ episode_id: string; path?: string; instruction?: string; error?: string }> = [];
          for (const model of modelsToUse) {
            const res = await batchGenerateAnnotation(backendTaskId, cameraName, { model, ...apiOpt });
            if (res.ok && res.data?.results) lastResults = res.data.results;
          }
          setBatchAnnotationResults(lastResults);
          const failed = lastResults.filter((r) => r.error);
          const success = lastResults.filter((r) => !r.error);
          if (failed.length === 0) {
            showToast(t('labelAgentPanel.batchDoneSuccess', { n: success.length }));
            await updateLabelTask(backendTaskId, { completed: true, verified: false }).catch((err) =>
              console.warn('自动更新任务完成/校验状态失败:', err)
            );
          } else {
            const normalizedReasons = failed.map((r) => normalizeModelBusyMessage(r.error)).join('; ');
            const hasOnlyModelBusy = failed.every((r) => isModelBusyMessage(r.error));
            if (hasOnlyModelBusy) {
              showToast(MODEL_BUSY_MSG);
            }
            showToast(
              t('labelAgentPanel.batchDoneWithFailed', {
                success: success.length,
                failed: failed.length,
                reasons: normalizedReasons,
              }),
              !hasOnlyModelBusy
            );
          }
          await episodesRefreshRef.current?.refresh();
          await instructionsRefreshRef.current?.refresh();
        }
      } catch (e) {
        const errText = normalizeModelBusyMessage(e instanceof Error ? e.message : String(e));
        showToast(
          t('labelAgentPanel.logQueryStatusException', {
            error: errText,
          }),
          !isModelBusyMessage(errText)
        );
      } finally {
        setBatchAnnotationLoading(false);
      }
    },
    [
      backendTaskId,
      t,
      showToast,
      normalizeModelBusyMessage,
      isModelBusyMessage,
      isNetworkTimeoutMessage,
      NETWORK_TIMEOUT_MSG,
    ]
  );

  useEffect(() => {
    return () => {
      if (agentIntervalRef.current) clearInterval(agentIntervalRef.current);
    };
  }, []);

  if (!backendTaskId) {
    return (
      <div style={{ padding: 24, color: '#6b7280' }}>
        {t('labelExecutePage.missingTaskId')}
      </div>
    );
  }

  if (annotateGate === 'loading') {
    return (
      <div
        style={{
          height: 'calc(100vh - 60px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#6b7280',
        }}
      >
        {t('common.loading')}
      </div>
    );
  }

  if (annotateGate === 'denied') {
    return (
      <div style={{ padding: 32, maxWidth: 480 }}>
        <p style={{ color: '#374151', fontSize: 15, marginBottom: 16 }}>
          {t('labelTasksPage.annotatePageDenied')}
        </p>
        <Link href="/label/tasks" style={{ color: '#2563eb', fontSize: 14 }}>
          {t('common.back')}
        </Link>
      </div>
    );
  }

  return (
    <>
    <EpisodeViewerLayout
      source={{ type: 'task', taskId: backendTaskId }}
      rightPanel={
        <RightPanel
          selectedEpisode={viewerState?.selectedEpisode ?? null}
          cameras={viewerState?.cameras ?? []}
          agentLogs={agentLogs}
          isAgentRunning={isAgentRunning}
          onStartAgent={handleStartAgent}
          onStopAgent={handleStopAgent}
          taskId={backendTaskId}
          generatedDescription={generatedDescription}
          annotationProgress={annotationProgress}
          onBatchAnnotation={handleBatchAnnotation}
          batchAnnotationLoading={batchAnnotationLoading}
          batchAnnotationResults={batchAnnotationResults}
          apiConfigScope={{
            userId: String((authUser as { id?: string | number } | null)?.id ?? ''),
            projectId: (actorCtx?.project_id || '').trim(),
          }}
          projectId={(actorCtx?.project_id || '').trim()}
          canManageModelConfig={
            (String((authUser as { role?: string } | null)?.role || '').toUpperCase() === 'SUPER_ADMIN') ||
            (String((authUser as { role?: string } | null)?.role || '').toUpperCase() === 'ADMIN')
          }
        />
      }
      missingContextMessage={t('labelExecutePage.missingTaskId')}
      onNewAnnotation={handleNewAnnotation}
      onSave={handleSave}
      taskName={taskName}
      onDescriptionUpdateRef={descriptionUpdateRef}
      instructionsRefreshRef={instructionsRefreshRef}
      episodesRefreshRef={episodesRefreshRef}
      onViewerStateChange={setViewerState}
      isPreloading={isPreloading}
      preloadProgress={preloadProgress}
      t={t}
    />
      {toast && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 18px',
            borderRadius: 10,
            fontSize: 13,
            color: toast.isError ? '#b91c1c' : '#ffffff',
            backgroundColor: toast.isError ? '#fef2f2' : 'rgba(17,24,39,0.92)',
            boxShadow: '0 18px 60px rgba(15,23,42,0.25)',
            zIndex: 1800,
          }}
        >
          {toast.message}
        </div>
      )}
    </>
  );
}

function LabelExecuteFallback() {
  const { t } = useI18n();
  return (
    <div
      style={{
        height: 'calc(100vh - 60px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#6b7280',
      }}
    >
      {t('common.loading')}
    </div>
  );
}

export default function LabelExecutePage() {
  return (
    <FrameCacheProvider>
      <Suspense fallback={<LabelExecuteFallback />}>
        <LabelExecuteContent />
      </Suspense>
    </FrameCacheProvider>
  );
}
