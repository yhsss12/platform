'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { AgentLogEntry } from '../models';
import ModelManageModal from './ModelManageModal';
import { getStoredApiConfig, setStoredApiConfig, type ApiConfig, type ApiConfigScope } from './ApiConfigDialog';
import { getProviderDetail, listProviders } from '../api/llmApi';
import { useI18n } from '@/components/common/I18nProvider';

const MODEL_OPTIONS = [
  { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { id: 'gemini-3-pro-image-preview', label: 'Gemini 3 Pro Image Preview' },
  { id: 'gemini-3-flash-preview', label: 'gemini-3-flash-preview' },
] as const;

const STORAGE_KEY_SELECTED_MODELS = 'label_agent_selected_models';

function getStoredSelectedModels(): string[] {
  if (typeof window === 'undefined') return ['gemini-3-flash-preview'];
  try {
    const raw = localStorage.getItem(STORAGE_KEY_SELECTED_MODELS);
    if (!raw) return ['gemini-3-flash-preview'];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) && arr.every((x) => typeof x === 'string') ? arr : ['gemini-3-flash-preview'];
  } catch {
    return ['gemini-3-flash-preview'];
  }
}

function getStoredSelectedModel(): string {
  const arr = getStoredSelectedModels();
  return arr.length > 0 ? arr[0] : 'gemini-3-flash-preview';
}

function setStoredSelectedModels(ids: string[]) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY_SELECTED_MODELS, JSON.stringify(ids));
}

function setStoredSelectedModel(id: string) {
  setStoredSelectedModels(id ? [id] : []);
}

export interface AgentModelConfig {
  selectedModels: string[];
  apiConfig: ApiConfig;
}

interface RightPanelProps {
  selectedEpisode: string | null;
  cameras?: string[];
  agentLogs: AgentLogEntry[];
  isAgentRunning: boolean;
  onStartAgent: (cameraName?: string, modelConfig?: AgentModelConfig) => void;
  onStopAgent: () => void;
  taskId: string;
  generatedDescription?: string;
  annotationProgress?: number;
  /** 批量自动标注（对整个任务的所有 episode） */
  onBatchAnnotation?: (cameraName?: string, modelConfig?: AgentModelConfig) => Promise<void>;
  batchAnnotationLoading?: boolean;
  /** 批量标注完成后的结果列表 */
  batchAnnotationResults?: Array<{ episode_id: string; path?: string; instruction?: string; error?: string }>;
  /** API 配置作用域：同账号同项目复用，不同项目隔离 */
  apiConfigScope?: ApiConfigScope;
  projectId?: string;
  canManageModelConfig?: boolean;
}

export default function RightPanel({
  selectedEpisode,
  cameras,
  agentLogs,
  isAgentRunning,
  onStartAgent,
  onStopAgent,
  taskId,
  generatedDescription,
  annotationProgress = 0,
  onBatchAnnotation,
  batchAnnotationLoading = false,
  batchAnnotationResults = [],
  apiConfigScope,
  projectId,
  canManageModelConfig = true,
}: RightPanelProps) {
  const { t } = useI18n();
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>(() => getStoredSelectedModel());
  const [apiConfig, setApiConfig] = useState<ApiConfig>(() => getStoredApiConfig(apiConfigScope));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [providerModelOptions, setProviderModelOptions] = useState<Array<{ id: string; label: string }> | null>(null);
  // 用于平滑进度条：后端只有 10% 和 100%，运行中时从 10 缓慢增加到 90，收到 100 时直接显示 100
  const [displayProgress, setDisplayProgress] = useState(0);
  const progressTickerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setStoredSelectedModel(selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    setApiConfig(getStoredApiConfig(apiConfigScope));
  }, [apiConfigScope?.userId, apiConfigScope?.projectId]);

  useEffect(() => {
    let cancelled = false;
    const pid = (projectId || '').trim();
    if (!pid) return;
    (async () => {
      try {
        const pRes = await listProviders(undefined, pid);
        if (!pRes.ok || !pRes.data || pRes.data.length === 0 || cancelled) return;
        const enabledProvider = pRes.data.find((p) => p.is_enabled) || pRes.data[0];
        const dRes = await getProviderDetail(enabledProvider.id, pid);
        if (!dRes.ok || !dRes.data || cancelled) return;
        const models = (dRes.data.models || [])
          .filter((m) => m.is_selected)
          .map((m) => m.name)
          .filter(Boolean);
        if (models.length > 0) {
          setProviderModelOptions(models.map((name) => ({ id: name, label: name })));
          if (!models.includes(selectedModel)) {
            setSelectedModel(models[0]);
            setStoredSelectedModel(models[0]);
          }
        } else {
          setProviderModelOptions([]);
        }
      } catch {
        // keep existing local fallback
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, selectedModel]);

  useEffect(() => {
    if (!isAgentRunning) {
      setDisplayProgress(0);
      if (progressTickerRef.current) {
        clearInterval(progressTickerRef.current);
        progressTickerRef.current = null;
      }
      return;
    }
    if (annotationProgress >= 100) {
      setDisplayProgress(100);
      if (progressTickerRef.current) {
        clearInterval(progressTickerRef.current);
        progressTickerRef.current = null;
      }
      return;
    }
    if (annotationProgress >= 10) {
      setDisplayProgress(annotationProgress);
      // 运行中且后端一直返回 10：每 2 秒增加约 8%，直到 90，避免长时间停在 10%
      if (!progressTickerRef.current) {
        progressTickerRef.current = setInterval(() => {
          setDisplayProgress((p) => {
            if (p >= 90) {
              if (progressTickerRef.current) {
                clearInterval(progressTickerRef.current);
                progressTickerRef.current = null;
              }
              return 90;
            }
            return Math.min(90, p + 8);
          });
        }, 2000);
      }
      return;
    }
    setDisplayProgress(annotationProgress);
  }, [isAgentRunning, annotationProgress]);

  const cameraOptions = useMemo(() => {
    const list = (cameras || []).filter(Boolean);
    return Array.from(new Set(list));
  }, [cameras]);

  const modelOptions = useMemo(() => {
    const base = providerModelOptions ?? [...MODEL_OPTIONS];
    const list = [...base];
    if (selectedModel && !list.some((o) => o.id === selectedModel)) {
      list.push({ id: selectedModel, label: selectedModel });
    }
    return list;
  }, [selectedModel, providerModelOptions]);
  const normalizeBusyError = (raw?: string) => {
    const msg = String(raw || '').trim();
    if (!msg) return '';
    const lower = msg.toLowerCase();
    const isBusy =
      (lower.includes('http 503') && lower.includes('openai')) ||
      lower.includes('currently experiencing high demand') ||
      (lower.includes('upstream_error') && lower.includes('503')) ||
      (lower.includes('code') && lower.includes('503'));
    return isBusy ? 'API 节点忙，请稍后重试' : msg;
  };

  // episode/cameras 变化时，默认选中第一个相机（但不强行覆盖用户已选）
  useEffect(() => {
    if (!selectedEpisode) {
      setSelectedCamera('');
      return;
    }
    if (!selectedCamera && cameraOptions.length > 0) {
      setSelectedCamera(cameraOptions[0]);
    }
  }, [selectedEpisode, cameraOptions, selectedCamera]);

  // 自动滚动到底部
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [agentLogs]);

  // 注意：日志保存由父组件的保存按钮统一处理

  const batchBusy = Boolean(onBatchAnnotation && batchAnnotationLoading);
  const autoLabelDisabled =
    !selectedEpisode || isAgentRunning || cameraOptions.length === 0 || batchBusy;

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#ffffff',
        borderLeft: '1px solid #e5e7eb',
        overflow: 'hidden',
      }}
    >
      {/* Agent 控制卡片 */}
      <div
        style={{
          padding: '16px',
          borderBottom: '1px solid #e5e7eb',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '8px',
          }}
        >
          <div
            style={{
              fontSize: '14px',
              fontWeight: '600',
              color: '#111827',
            }}
          >
            {t('labelAgentPanel.title')}
          </div>
          <button
            type="button"
            onClick={() => {
              if (typeof window !== 'undefined') window.history.back();
            }}
            style={{
              height: '26px',
              padding: '0 10px',
              border: '1px solid #2563eb',
              borderRadius: '6px',
              backgroundColor: '#2563eb',
              color: '#ffffff',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            {t('common.back')}
          </button>
        </div>
        <div
          style={{
            fontSize: '12px',
            color: '#6b7280',
            marginBottom: '12px',
            lineHeight: '1.5',
          }}
        >
          {t('labelAgentPanel.hintSelectDataset')}
        </div>

        {/* 相机选择 */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '12px', color: '#374151', marginBottom: '6px' }}>
            {t('labelAgentPanel.cameraViewLabel')}
          </div>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            disabled={!selectedEpisode || isAgentRunning || cameraOptions.length === 0}
            style={{
              width: '100%',
              height: '32px',
              padding: '0 10px',
              borderRadius: '6px',
              border: '1px solid #e5e7eb',
              backgroundColor: !selectedEpisode || isAgentRunning ? '#f3f4f6' : '#ffffff',
              color: !selectedEpisode || cameraOptions.length === 0 ? '#9ca3af' : '#111827',
              fontSize: '13px',
              outline: 'none',
            }}
          >
            {cameraOptions.length === 0 ? (
            <option value="">{t('labelAgentPanel.noCameraOption')}</option>
            ) : (
              cameraOptions.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))
            )}
          </select>
          {selectedEpisode && cameraOptions.length === 0 && (
            <div style={{ marginTop: '6px', fontSize: '12px', color: '#9ca3af', lineHeight: 1.4 }}>
              {t('labelAgentPanel.hintNeedDoubleClick')}
            </div>
          )}
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            gap: '8px',
            alignItems: 'stretch',
          }}
        >
          <button
            onClick={
              isAgentRunning
                ? onStopAgent
                : () => {
                    const modelConfig: AgentModelConfig = {
                      selectedModels: selectedModel ? [selectedModel] : [],
                      apiConfig,
                    };
                    onStartAgent(selectedCamera || undefined, modelConfig);
                  }
            }
            disabled={autoLabelDisabled}
            style={{
              flex: 1,
              minWidth: 0,
              height: '32px',
              padding: '0 8px',
              backgroundColor: autoLabelDisabled ? '#f3f4f6' : '#2563eb',
              border: 'none',
              borderRadius: '6px',
              color: autoLabelDisabled ? '#9ca3af' : '#ffffff',
              fontSize: '13px',
              cursor: autoLabelDisabled ? 'not-allowed' : 'pointer',
              fontWeight: '500',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
            onMouseEnter={(e) => {
              if (!autoLabelDisabled && !isAgentRunning) {
                e.currentTarget.style.backgroundColor = '#1d4ed8';
              }
            }}
            onMouseLeave={(e) => {
              if (!autoLabelDisabled && !isAgentRunning) {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }
            }}
          >
            {isAgentRunning ? (
              <>
                <span
                  style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: '#9ca3af',
                    animation: 'pulse 1.5s ease-in-out infinite',
                  }}
                />
                {t('labelAgentPanel.running')}
              </>
            ) : (
              t('labelAgentPanel.actionAutoLabelCurrent')
            )}
          </button>

          {onBatchAnnotation && (
            <button
              onClick={async () => {
                const modelConfig: AgentModelConfig = {
                  selectedModels: selectedModel ? [selectedModel] : [],
                  apiConfig,
                };
                await onBatchAnnotation(selectedCamera || undefined, modelConfig);
              }}
              disabled={isAgentRunning || batchAnnotationLoading}
              style={{
                flex: 1,
                minWidth: 0,
                height: '32px',
                padding: '0 8px',
                backgroundColor: isAgentRunning || batchAnnotationLoading ? '#f3f4f6' : '#059669',
                border: 'none',
                borderRadius: '6px',
                color: isAgentRunning || batchAnnotationLoading ? '#9ca3af' : '#ffffff',
                fontSize: '13px',
                cursor: isAgentRunning || batchAnnotationLoading ? 'not-allowed' : 'pointer',
                fontWeight: '500',
              }}
            >
              {batchAnnotationLoading ? t('labelAgentPanel.batchRunning') : t('labelAgentPanel.actionAutoLabelBatch')}
            </button>
          )}
        </div>

        {/* 批量标注结果：可滑动列表 */}
        {batchAnnotationResults.length > 0 && (
          <div
            style={{
              marginTop: '12px',
              border: '1px solid #e5e7eb',
              borderRadius: '6px',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              height: '220px',
              minHeight: '120px',
              maxHeight: '520px',
              resize: 'vertical',
            }}
          >
            <div
              style={{
                padding: '8px 10px',
                fontSize: '12px',
                fontWeight: '600',
                color: '#374151',
                backgroundColor: '#f9fafb',
                borderBottom: '1px solid #e5e7eb',
              }}
            >
              {t('labelAgentPanel.resultsTitle')}
            </div>
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '8px',
                fontSize: '12px',
                lineHeight: '1.5',
                color: '#111827',
              }}
            >
              {batchAnnotationResults.map((r, i) => {
                const label = r.path
                  ? r.path.replace(/^.*[/\\]/, '').replace(/\.(hdf5|h5)$/i, '')
                  : r.episode_id;
                const text = r.error
                  ? `${t('labelAgentPanel.failedPrefix')}: ${normalizeBusyError(r.error)}`
                  : (r.instruction || '');
                return (
                  <div
                    key={r.episode_id + String(i)}
                    style={{
                      marginBottom: i < batchAnnotationResults.length - 1 ? '10px' : 0,
                      paddingBottom: i < batchAnnotationResults.length - 1 ? '10px' : 0,
                      borderBottom: i < batchAnnotationResults.length - 1 ? '1px solid #f3f4f6' : 'none',
                    }}
                  >
                    <span style={{ fontWeight: '600', color: '#059669' }}>{label}:</span>{' '}
                    <span style={{ color: r.error ? '#dc2626' : '#374151', wordBreak: 'break-word' }}>
                      {text}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      {/* 进度条：运行中时显示，避免从 10% 直接跳到 100% 的突兀感 */}
      {isAgentRunning && (
        <div style={{ marginTop: '12px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '6px',
              fontSize: '12px',
              color: '#6b7280',
            }}
          >
            <span>{t('labelAgentPanel.generating')}</span>
            <span>{Math.round(displayProgress)}%</span>
          </div>
          <div
            style={{
              height: '8px',
              borderRadius: '4px',
              backgroundColor: '#e5e7eb',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${displayProgress}%`,
                backgroundColor: '#2563eb',
                borderRadius: '4px',
                transition: 'width 0.3s ease-out',
              }}
            />
          </div>
        </div>
      )}
      </div>

      {/* 生成的标注结果 */}
      {generatedDescription && (
        <div
          style={{
            padding: '16px',
            borderBottom: '1px solid #e5e7eb',
            backgroundColor: '#f9fafb',
          }}
        >
          <div
            style={{
              fontSize: '13px',
              fontWeight: '600',
              color: '#111827',
              marginBottom: '8px',
            }}
          >
            {t('labelAgentPanel.generatedTitle')}
          </div>
          <div
            style={{
              fontSize: '13px',
              color: '#374151',
              lineHeight: '1.6',
              marginBottom: '12px',
              padding: '12px',
              backgroundColor: '#ffffff',
              borderRadius: '6px',
              border: '1px solid #e5e7eb',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {generatedDescription}
          </div>
        </div>
      )}

      {/* 输出框 */}
      <div
        ref={logContainerRef}
        style={{
          flex: 1,
          padding: '12px',
          overflowY: 'auto',
          backgroundColor: '#ffffff',
          fontSize: '12px',
          fontFamily: 'monospace',
          color: '#374151',
          lineHeight: '1.6',
        }}
      >
        {agentLogs.length === 0 ? (
          <div
            style={{
              color: '#9ca3af',
              textAlign: 'center',
              padding: '40px 0',
            }}
          >
            {batchAnnotationResults.length > 0
              ? t('labelAgentPanel.batchDoneHint')
              : t('labelAgentPanel.waitingOutput')}
          </div>
        ) : (
          agentLogs.map((log, index) => (
            <div
              key={index}
              style={{
                marginBottom: '4px',
                wordBreak: 'break-word',
              }}
            >
              <span style={{ color: '#6b7280' }}>[{log.timestamp}]</span>{' '}
              <span style={{ color: '#111827' }}>{log.message}</span>
            </div>
          ))
        )}
      </div>

      {/* 右下角：模型选择下拉 + 设置 */}
      <div
        style={{
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '10px 16px',
          borderTop: '1px solid #e5e7eb',
          backgroundColor: '#f9fafb',
        }}
      >
        <span style={{ fontSize: '12px', fontWeight: '500', color: '#374151', whiteSpace: 'nowrap' }}>
          {t('labelAgentPanel.modelLabel')}
        </span>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={isAgentRunning || batchAnnotationLoading}
          style={{
            flex: 1,
            minWidth: 0,
            height: '32px',
            padding: '0 10px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            backgroundColor: isAgentRunning || batchAnnotationLoading ? '#f3f4f6' : '#ffffff',
            color: '#111827',
            fontSize: '12px',
            outline: 'none',
          }}
        >
          {modelOptions.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          title={t('labelAgentPanel.manageModelsTooltip')}
          disabled={!canManageModelConfig}
          style={{
            width: '32px',
            height: '32px',
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            background: '#ffffff',
            cursor: !canManageModelConfig ? 'not-allowed' : 'pointer',
            color: !canManageModelConfig ? '#cbd5e1' : '#6b7280',
            fontSize: '14px',
          }}
        >
          ⚙
        </button>
      </div>

      <ModelManageModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        projectId={projectId}
        canEdit={canManageModelConfig}
        onSave={(config) => {
          setApiConfig({ apiKey: config.apiKey, baseUrl: config.baseUrl });
          setStoredApiConfig({ apiKey: config.apiKey, baseUrl: config.baseUrl }, apiConfigScope);
          if (config.modelNames && config.modelNames.length > 0) {
            setProviderModelOptions(config.modelNames.map((name) => ({ id: name, label: name })));
          } else {
            setProviderModelOptions(null);
          }
          if (config.selectedModelName) {
            setSelectedModel(config.selectedModelName);
            setStoredSelectedModel(config.selectedModelName);
          }
          setSettingsOpen(false);
        }}
      />

      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}

