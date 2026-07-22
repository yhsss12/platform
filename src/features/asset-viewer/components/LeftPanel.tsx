'use client';

import { useState, useEffect, useCallback, useRef, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  getEpisodes,
  getInstruction,
  saveInstruction,
  getTaskInstructions,
  getAnnotationDownloadOne,
  getAnnotationDownloadBatch,
  getAssetAnnotationDownloadOne,
  getAssetAnnotationDownloadBatch,
  type Episode,
} from '../api/labelApi';
import { useI18n } from '@/components/common/I18nProvider';

interface LeftPanelProps {
  selectedEpisode: string | null;
  onSelectEpisode: (episodeId: string) => void;
  onDoubleClickEpisode?: (episodeId: string) => void;
  onNewAnnotation: () => void;
  onSave: () => void;
  taskId: string;
  /** 数据查看页传入，用于从数据仓库下载标注 */
  assetId?: string;
  episodes?: Episode[];
  episodesLoading?: boolean;
  episodesError?: string;
  onDescriptionUpdateRef?: (ref: (desc: string) => void) => void;
  /** 父组件可调用 .refresh() 刷新左侧每条标注状态（如批量标注完成后） */
  instructionsRefreshRef?: React.MutableRefObject<{ refresh: () => void } | null>;
  /** 保存成功后由父组件刷新 episodes 列表（以更新 instruction_text） */
  onEpisodesRefresh?: () => void | Promise<void>;
  /** 是否显示「新建标注」「保存」按钮（数据查看页隐藏） */
  showLabelActions?: boolean;
}

export default function LeftPanel({
  selectedEpisode,
  onSelectEpisode,
  onDoubleClickEpisode,
  onNewAnnotation,
  onSave,
  taskId,
  assetId,
  episodes: externalEpisodes,
  episodesLoading: externalEpisodesLoading,
  episodesError: externalEpisodesError,
  onDescriptionUpdateRef,
  instructionsRefreshRef,
  onEpisodesRefresh,
  showLabelActions = true,
}: LeftPanelProps) {
  const { t } = useI18n();
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [description, setDescription] = useState('');
  const [showSaveToast, setShowSaveToast] = useState(false);
  const [showDownloadToast, setShowDownloadToast] = useState(false);
  const [toastMsg, setToastMsg] = useState<{ text: string; isError?: boolean } | null>(null);
  const showToast = useCallback((text: string, isError?: boolean) => {
    setToastMsg({ text, isError });
    setTimeout(() => setToastMsg(null), 2200);
  }, []);
  const [loading, setLoading] = useState(false);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [episodesError, setEpisodesError] = useState<string>('');
  const [taskInstructions, setTaskInstructions] = useState<string[]>([]);
  const [downloadMenuOpen, setDownloadMenuOpen] = useState(false);
  const [downloadMenuPosition, setDownloadMenuPosition] = useState<{ top: number; left: number; width: number } | null>(null);
  const downloadTriggerRef = useRef<HTMLDivElement>(null);

  // 如果外部传入了 episodes，使用外部的；否则从 API 加载
  const useExternalEpisodes = externalEpisodes !== undefined;
  const displayEpisodes = useExternalEpisodes ? externalEpisodes : episodes;
  const displayEpisodesLoading = useExternalEpisodes ? (externalEpisodesLoading ?? false) : episodesLoading;
  const displayEpisodesError = useExternalEpisodes ? (externalEpisodesError ?? '') : episodesError;

  // 加载 episode 列表（仅在未传入外部 episodes 时执行）
  useEffect(() => {
    if (!useExternalEpisodes) {
      loadEpisodes();
    }
  }, [useExternalEpisodes]);

  // 加载任务下所有标注结果（用于左侧列表展示与整数据集下载）
  const loadTaskInstructions = useCallback(async () => {
    if (!taskId) return;
    try {
      const res = await getTaskInstructions(taskId);
      if (res.ok && res.data?.instructions) {
        setTaskInstructions(res.data.instructions);
      } else {
        setTaskInstructions([]);
      }
    } catch {
      setTaskInstructions([]);
    }
  }, [taskId]);

  // 下载下拉打开时测量按钮位置，用于 Portal 定位，避免被父级 overflow 裁剪
  useLayoutEffect(() => {
    if (!downloadMenuOpen || !downloadTriggerRef.current) {
      setDownloadMenuPosition(null);
      return;
    }
    const rect = downloadTriggerRef.current.getBoundingClientRect();
    const width = Math.max(140, Math.round(rect.width));
    setDownloadMenuPosition({ top: rect.bottom + 4, left: rect.left, width });
  }, [downloadMenuOpen]);

  useEffect(() => {
    if (taskId && displayEpisodes.length > 0) {
      loadTaskInstructions();
    } else {
      setTaskInstructions([]);
    }
  }, [taskId, displayEpisodes.length, loadTaskInstructions]);

  // 加载当前 episode 的标注：优先使用数据资产 instruction_text，无则回退到 instruction.json
  useEffect(() => {
    if (selectedEpisode) {
      const ep = displayEpisodes.find((e) => e.id === selectedEpisode);
      if (ep?.instruction_text !== undefined && ep.instruction_text !== null) {
        setDescription(String(ep.instruction_text).trim());
      } else {
        loadInstruction(selectedEpisode);
      }
    } else {
      setDescription('');
    }
  }, [selectedEpisode, displayEpisodes]);

  // 将更新函数传递给父组件
  useEffect(() => {
    if (onDescriptionUpdateRef) {
      onDescriptionUpdateRef((desc: string) => {
        setDescription(desc);
      });
    }
  }, [onDescriptionUpdateRef]);

  useEffect(() => {
    if (instructionsRefreshRef) {
      instructionsRefreshRef.current = { refresh: loadTaskInstructions };
      return () => {
        instructionsRefreshRef.current = null;
      };
    }
  }, [instructionsRefreshRef, loadTaskInstructions]);

  const loadEpisodes = async () => {
    setEpisodesLoading(true);
    setEpisodesError('');
    try {
      // 按 taskId 过滤 episodes
      const response = await getEpisodes(taskId);
      if (response.ok && response.data) {
        setEpisodes(response.data);
      } else {
        setEpisodesError(response.error || t('labelExecutePage.alertNoDatasets'));
      }
    } catch (error) {
      console.error('Failed to load episodes:', error);
      setEpisodesError(t('feedback.requestFailed'));
    } finally {
      setEpisodesLoading(false);
    }
  };

  const loadInstruction = async (episodeId: string) => {
    try {
      // 传递 taskId 以便后端正确查找 episode
      const response = await getInstruction(episodeId, taskId);
      if (response.ok && response.data) {
        setDescription(response.data.instruction || '');
      }
    } catch (error) {
      console.error('Failed to load instruction:', error);
    }
  };

  const extractEpisodeIndex = (episodeId: string): number => {
    // 优先使用在列表中的真实下标（episode ID 常为 hash，不能当索引用）
    const idx = displayEpisodes.findIndex((ep) => ep.id === episodeId);
    if (idx >= 0) return idx;
    const match = episodeId.match(/^\d+$/);
    return match ? parseInt(match[0], 10) : 0;
  };

  const handleSave = async () => {
    if (!selectedEpisode) return;
    
    setLoading(true);
    try {
      const episodeIndex = extractEpisodeIndex(selectedEpisode);
      // 传递 taskId 以便后端正确查找 episode
      const response = await saveInstruction(selectedEpisode, description, episodeIndex, taskId);
        if (response.ok) {
        onSave();
        setShowSaveToast(true);
        setTimeout(() => setShowSaveToast(false), 2000);
        loadTaskInstructions(); // 刷新任务内 instructions
        onEpisodesRefresh?.(); // 刷新 episodes 以更新 instruction_text 与「已标注」状态
      } else {
        showToast(`${t('labelExecutePage.alertSaveFailed')}: ${response.error}`, true);
      }
    } catch (error) {
      console.error('Failed to save instruction:', error);
      showToast(t('labelExecutePage.alertSaveFailedRetry'), true);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!selectedEpisode) {
      showToast(t('labelExecutePage.alertSelectDatasetFirst'), true);
      return;
    }
    const hasTask = !!taskId;
    const hasAsset = assetId != null && assetId !== '';
    if (!hasTask && !hasAsset) {
      showToast(t('labelExecutePage.alertNoDownloadContentHint') || '缺少任务或资产上下文', true);
      return;
    }
    setDownloadMenuOpen(false);
    try {
      const res = hasTask
        ? await getAnnotationDownloadOne(taskId, selectedEpisode)
        : await getAssetAnnotationDownloadOne(String(assetId), selectedEpisode);
      if (!res.ok || res.data == null) {
        throw new Error(res.error || '获取标注失败');
      }
      const jsonData = { instructions: [res.data.instruction ?? ''] };
      const jsonString = JSON.stringify(jsonData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'instruction.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setShowDownloadToast(true);
      setTimeout(() => setShowDownloadToast(false), 2000);
    } catch (error) {
      console.error('Failed to download annotation:', error);
      showToast(t('labelExecutePage.alertDownloadFailedRetry'), true);
    }
  };

  const handleDownloadAll = async () => {
    setDownloadMenuOpen(false);
    if (displayEpisodes.length === 0) {
      showToast(t('labelExecutePage.alertNoDatasets'), true);
      return;
    }
    const hasTask = !!taskId;
    const hasAsset = assetId != null && assetId !== '';
    if (!hasTask && !hasAsset) {
      showToast(t('labelExecutePage.alertNoDownloadContentHint') || '缺少任务或资产上下文', true);
      return;
    }
    try {
      const res = hasTask
        ? await getAnnotationDownloadBatch(taskId)
        : await getAssetAnnotationDownloadBatch(String(assetId));
      if (!res.ok || res.data == null) {
        throw new Error(res.error || '获取标注失败');
      }
      const instructions = (res.data.items ?? []).map((item) => item.instruction ?? '');
      const jsonData = { instructions };
      const jsonString = JSON.stringify(jsonData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'instruction.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setShowDownloadToast(true);
      setTimeout(() => setShowDownloadToast(false), 2000);
    } catch (error) {
      console.error('Failed to download annotations:', error);
      showToast(t('labelExecutePage.alertDownloadFailedRetry'), true);
    }
  };

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#ffffff',
        borderRight: '1px solid #e5e7eb',
        overflow: 'hidden',
      }}
    >
      {/* 数据集列表 */}
      <div style={{ flex: '0 0 auto', borderBottom: '1px solid #e5e7eb' }}>
        <div
          style={{
            padding: '12px 16px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#111827',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          {t('labelDatasetPanel.datasetLabel')}
        </div>
        <div
          style={{
            maxHeight: '300px',
            overflowY: 'auto',
            padding: '8px 0',
          }}
        >
          {displayEpisodesLoading ? (
            <div
              style={{
                padding: '20px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '13px',
              }}
            >
              {t('common.loading')}
            </div>
          ) : displayEpisodesError ? (
            <div
              style={{
                padding: '20px',
                textAlign: 'center',
                color: '#ef4444',
                fontSize: '13px',
              }}
            >
              {displayEpisodesError}
            </div>
          ) : displayEpisodes.length === 0 ? (
            <div
              style={{
                padding: '20px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '13px',
              }}
            >
              {useExternalEpisodes ? t('labelDatasetPanel.emptyInDir') : t('labelExecutePage.alertNoDatasets')}
            </div>
          ) : (
            displayEpisodes.map((episode, idx) => {
              const hasAnnotation = (episode.instruction_text ?? '').trim() !== '';
              const isSelected = selectedEpisode === episode.id;
              return (
                <div
                  key={episode.id}
                  onClick={() => onSelectEpisode(episode.id)}
                  onDoubleClick={() => onDoubleClickEpisode?.(episode.id)}
                  style={{
                    padding: '10px 16px 10px 14px',
                    borderLeft: isSelected ? '2px solid #2563eb' : '2px solid transparent',
                    fontSize: '13px',
                    color: isSelected ? '#2563eb' : '#374151',
                    fontWeight: isSelected ? 500 : 400,
                    cursor: 'pointer',
                    backgroundColor: isSelected ? '#eff6ff' : 'transparent',
                    transition: 'background-color 0.15s, color 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '8px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = isSelected ? '#dbeafe' : '#f9fafb';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = isSelected ? '#eff6ff' : 'transparent';
                  }}
                >
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {episode.name}
                  </span>
                  <span
                    style={{
                      flexShrink: 0,
                      fontSize: '11px',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: hasAnnotation ? '#d1fae5' : '#f3f4f6',
                      color: hasAnnotation ? '#065f46' : '#6b7280',
                    }}
                  >
                    {hasAnnotation ? t('labelExecutePage.statusLabeled') : t('labelExecutePage.statusUnlabeled')}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* 按钮行 */}
      <div
        style={{
          padding: '12px 16px',
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid #e5e7eb',
          flexWrap: 'nowrap',
        }}
      >
        {showLabelActions && (
          <>
            <button
              onClick={handleSave}
              disabled={loading || !selectedEpisode}
              style={{
                flex: 1,
                minWidth: '80px',
                height: '32px',
                padding: '0 12px',
                backgroundColor: loading || !selectedEpisode ? '#f3f4f6' : '#f9fafb',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                color: loading || !selectedEpisode ? '#9ca3af' : '#374151',
                fontSize: '13px',
                cursor: loading || !selectedEpisode ? 'not-allowed' : 'pointer',
                fontWeight: '500',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                if (!loading && selectedEpisode) {
                  e.currentTarget.style.backgroundColor = '#f3f4f6';
                  e.currentTarget.style.borderColor = '#9ca3af';
                }
              }}
              onMouseLeave={(e) => {
                if (!loading && selectedEpisode) {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                  e.currentTarget.style.borderColor = '#d1d5db';
                }
              }}
            >
              {loading ? t('labelExecutePage.saving') : t('labelExecutePage.actionSave')}
            </button>
          </>
        )}
        <div ref={downloadTriggerRef} style={{ position: 'relative', flex: 1, minWidth: '80px' }}>
          <button
            onClick={() => setDownloadMenuOpen((v) => !v)}
            disabled={displayEpisodes.length === 0}
            style={{
              width: '100%',
              height: '32px',
              padding: '0 12px',
              backgroundColor: displayEpisodes.length === 0 ? '#f3f4f6' : '#2563eb',
              border: 'none',
              borderRadius: '6px',
              color: displayEpisodes.length === 0 ? '#9ca3af' : '#ffffff',
              fontSize: '13px',
              cursor: displayEpisodes.length === 0 ? 'not-allowed' : 'pointer',
              fontWeight: '500',
              transition: 'all 0.2s',
            }}
          >
            {t('labelExecutePage.downloadMenu')}
          </button>
          {downloadMenuOpen && downloadMenuPosition && typeof document !== 'undefined' && createPortal(
            <>
              <div
                style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
                onClick={() => setDownloadMenuOpen(false)}
                aria-hidden="true"
              />
              <div
                style={{
                  position: 'fixed',
                  top: downloadMenuPosition.top,
                  left: downloadMenuPosition.left,
                  width: downloadMenuPosition.width,
                  zIndex: 9999,
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  overflow: 'hidden',
                }}
              >
                <button
                  onClick={handleDownload}
                  disabled={!selectedEpisode || (!taskId && (assetId == null || assetId === ''))}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '10px 14px',
                    textAlign: 'left',
                    border: 'none',
                    backgroundColor: 'transparent',
                    fontSize: '13px',
                    color: !selectedEpisode || (!taskId && (assetId == null || assetId === '')) ? '#9ca3af' : '#374151',
                    cursor: !selectedEpisode || (!taskId && (assetId == null || assetId === '')) ? 'not-allowed' : 'pointer',
                  }}
                >
                  {t('labelExecutePage.downloadCurrent')}
                </button>
                <button
                  onClick={handleDownloadAll}
                  style={{
                    display: 'block',
                    width: '100%',
                    padding: '10px 14px',
                    textAlign: 'left',
                    border: 'none',
                    borderTop: '1px solid #e5e7eb',
                    backgroundColor: 'transparent',
                    fontSize: '13px',
                    color: '#374151',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {t('labelExecutePage.actionDownloadAll')}
                </button>
              </div>
            </>,
            document.body
          )}
        </div>
      </div>

      {/* 任务描述输入框 */}
      <div style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column' }}>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="在这里输入任务描述 (例如：pick the block from the pliers)"
          style={{
            flex: 1,
            width: '100%',
            padding: '12px',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            color: '#111827',
            fontSize: '13px',
            fontFamily: 'inherit',
            resize: 'none',
            outline: 'none',
            boxSizing: 'border-box',
            transition: 'all 0.2s',
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = '#2563eb';
            e.currentTarget.style.boxShadow = '0 0 0 3px rgba(37, 99, 235, 0.1)';
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = '#d1d5db';
            e.currentTarget.style.boxShadow = 'none';
          }}
        />
      </div>

      {/* 保存成功 Toast */}
      {showSaveToast && (
        <div
          style={{
            position: 'absolute',
            bottom: '60px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '12px 20px',
            backgroundColor: '#111827',
            color: '#ffffff',
            borderRadius: '6px',
            fontSize: '13px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            zIndex: 1000,
          }}
        >
          {t('labelExecutePage.toastSaved')}
        </div>
      )}

      {/* 下载成功 Toast */}
      {showDownloadToast && (
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '12px 20px',
            backgroundColor: '#10b981',
            color: '#ffffff',
            borderRadius: '6px',
            fontSize: '13px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            zIndex: 1000,
          }}
        >
          {t('labelExecutePage.toastDownloaded')}
        </div>
      )}

      {/* 错误/轻提示 Toast */}
      {toastMsg && (
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '12px 20px',
            backgroundColor: toastMsg.isError ? '#fef2f2' : 'rgba(17,24,39,0.92)',
            color: toastMsg.isError ? '#b91c1c' : '#ffffff',
            borderRadius: '6px',
            fontSize: '13px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            zIndex: 1000,
          }}
        >
          {toastMsg.text}
        </div>
      )}
    </div>
  );
}

