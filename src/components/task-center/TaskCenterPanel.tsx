'use client';

import { useMemo, useState, useCallback, useEffect, useRef } from 'react';
import { useTaskCenter } from './TaskCenterContext';
import type { BackgroundTask } from './types';
import TaskCenterItem from './TaskCenterItem';
import { useI18n } from '@/components/common/I18nProvider';

type TabKey = 'active' | 'done' | 'failed';

/** 与 maxHeight 一致，保证列向 flex 主尺寸确定，中间列表才能稳定出现滚动区 */
const PANEL_HEIGHT_CSS = 'min(560px, calc(100vh - 120px))';

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

export default function TaskCenterPanel() {
  const { t } = useI18n();
  const {
    tasks,
    removeTask,
    clearTaskGroup,
    deleteTaskResult,
    cancelBackgroundTask,
    panelOpen,
    setPanelOpen,
    assistantVisible,
    assistantPosition,
  } = useTaskCenter();
  const [tab, setTab] = useState<TabKey>('active');
  const [deleteConfirmTask, setDeleteConfirmTask] = useState<BackgroundTask | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [cancelConfirmTask, setCancelConfirmTask] = useState<BackgroundTask | null>(null);
  const [cancelLoading, setCancelLoading] = useState(false);
  const [toast, setToast] = useState<{ message: string; isError?: boolean } | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [panelSize, setPanelSize] = useState<{ w: number; h: number }>({ w: 400, h: 360 });

  const showToast = useCallback((message: string, isError?: boolean) => {
    setToast({ message, isError });
    setTimeout(() => setToast(null), 2500);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!deleteConfirmTask) return;
    setDeleteLoading(true);
    const res = await deleteTaskResult(deleteConfirmTask);
    setDeleteLoading(false);
    if (res.ok) {
      removeTask(deleteConfirmTask.id);
      setDeleteConfirmTask(null);
      showToast(t('backgroundTasks.deleteSuccess'));
    } else {
      const msg =
        res.error == null
          ? t('backgroundTasks.deleteFailedRetry')
          : res.error.includes('.')
            ? t(res.error)
            : res.error;
      showToast(msg, true);
    }
  }, [deleteConfirmTask, deleteTaskResult, removeTask, showToast, t]);

  const handleConfirmCancel = useCallback(async () => {
    if (!cancelConfirmTask) return;
    setCancelLoading(true);
    const res = await cancelBackgroundTask(cancelConfirmTask);
    setCancelLoading(false);
    if (res.ok) {
      setCancelConfirmTask(null);
      if (res.displayOnly) {
        showToast(t('backgroundTasks.syncRemovedFromListOnly'));
      } else {
        showToast(t('backgroundTasks.cancelSuccess'));
      }
    } else {
      const msg =
        res.error == null
          ? t('backgroundTasks.cancelFailed')
          : res.error.includes('.')
            ? t(res.error)
            : res.error;
      showToast(msg, true);
    }
  }, [cancelConfirmTask, cancelBackgroundTask, showToast, t]);

  const { active, done, failed, doneTotal, failedTotal } = useMemo(() => {
    const active: BackgroundTask[] = [];
    const done: BackgroundTask[] = [];
    const failed: BackgroundTask[] = [];
    tasks.forEach((t) => {
      const activeInPanel = t.status === 'queued' || t.status === 'running' || t.status === 'paused';
      if (t.status === 'failed' || t.status === 'cancelled') failed.push(t);
      else if (activeInPanel) active.push(t);
      else done.push(t);
    });
    const sortByUpdatedDesc = (a: BackgroundTask, b: BackgroundTask) =>
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
    done.sort(sortByUpdatedDesc);
    failed.sort(sortByUpdatedDesc);
    const maxList = 50;
    return {
      active,
      done: done.slice(0, maxList),
      doneTotal: done.length,
      failed: failed.slice(0, maxList),
      failedTotal: failed.length,
    };
  }, [tasks]);

  const list = tab === 'active' ? active : tab === 'done' ? done : failed;
  const hasMore = (tab === 'done' && doneTotal > 50) || (tab === 'failed' && failedTotal > 50);
  const canClearCurrent = list.length > 0;

  // 打开时测量面板尺寸，用于智能避让机器人位置（必须在 early return 前调用，遵守 Hooks 规则）
  useEffect(() => {
    if (!panelOpen) return;
    const el = panelRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      setPanelSize({ w: rect.width, h: rect.height });
    }
  }, [panelOpen, tab, tasks.length]);

  const defaultPanelStyle: React.CSSProperties = useMemo(() => {
    // 保持原有风格：fixed 浮层 + 不重做 UI；只做定位与边界控制
    const pad = 12;
    const vw = typeof window !== 'undefined' ? window.innerWidth : 1200;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 800;

    // 如果机器人隐藏/不可用，则回退到原来的右下角固定位置
    if (!assistantVisible) {
      return {
        position: 'fixed',
        right: 24,
        bottom: 96,
      };
    }

    const robotSize = 64;
    const gap = 12;
    const rx = assistantPosition.x + robotSize; // robot right
    const ry = assistantPosition.y + robotSize; // robot bottom

    // 优先放在机器人“附近”，根据空间自动选择展开方向
    const preferLeft = rx + gap + panelSize.w > vw - pad; // 右侧空间不足则向左
    const preferUp = ry + gap + panelSize.h > vh - pad; // 下方空间不足则向上

    const left = preferLeft ? assistantPosition.x - gap - panelSize.w : rx + gap;
    const top = preferUp ? assistantPosition.y - gap - panelSize.h : ry + gap;

    const clampedLeft = clamp(left, pad, Math.max(pad, vw - panelSize.w - pad));
    const clampedTop = clamp(top, pad, Math.max(pad, vh - panelSize.h - pad));

    return {
      position: 'fixed',
      left: clampedLeft,
      top: clampedTop,
    };
  }, [assistantVisible, assistantPosition.x, assistantPosition.y, panelSize.h, panelSize.w]);

  /** 用户拖动后的位置；关闭面板后清空，下次打开恢复默认布局 */
  const [manualPanelPosition, setManualPanelPosition] = useState<{ left: number; top: number } | null>(null);
  const panelDragRef = useRef<{
    startClientX: number;
    startClientY: number;
    originLeft: number;
    originTop: number;
  } | null>(null);

  useEffect(() => {
    if (!panelOpen) setManualPanelPosition(null);
  }, [panelOpen]);

  const panelPositionStyle: React.CSSProperties = manualPanelPosition
    ? { position: 'fixed', left: manualPanelPosition.left, top: manualPanelPosition.top }
    : defaultPanelStyle;

  const onPanelDragHandleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const el = panelRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    panelDragRef.current = {
      startClientX: e.clientX,
      startClientY: e.clientY,
      originLeft: r.left,
      originTop: r.top,
    };
    setManualPanelPosition({ left: r.left, top: r.top });

    const onMove = (ev: MouseEvent) => {
      const d = panelDragRef.current;
      const panelEl = panelRef.current;
      if (!d || !panelEl) return;
      const dx = ev.clientX - d.startClientX;
      const dy = ev.clientY - d.startClientY;
      const pr = panelEl.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const pad = 8;
      setManualPanelPosition({
        left: clamp(d.originLeft + dx, pad - pr.width + 48, vw - 48),
        top: clamp(d.originTop + dy, pad, vh - 48),
      });
    };
    const onUp = () => {
      panelDragRef.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  if (!panelOpen) return null;

  return (
    <div
      ref={panelRef}
      style={{
        ...panelPositionStyle,
        width: 400,
        maxWidth: 'calc(100vw - 48px)',
        height: PANEL_HEIGHT_CSS,
        maxHeight: PANEL_HEIGHT_CSS,
        boxSizing: 'border-box',
        backgroundColor: '#fff',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        boxShadow: '0 12px 40px rgba(15,23,42,0.12)',
        zIndex: 1305,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        minHeight: 0,
        pointerEvents: 'auto',
      }}
    >
      <style>{`
        @keyframes taskCenterExportZipIndeterminate{0%,100%{opacity:.35}50%{opacity:.95}}
        .task-center-panel__list::-webkit-scrollbar{width:6px}
        .task-center-panel__list::-webkit-scrollbar-track{background:transparent}
        .task-center-panel__list::-webkit-scrollbar-thumb{background:rgba(15,23,42,.18);border-radius:999px}
        .task-center-panel__list::-webkit-scrollbar-thumb:hover{background:rgba(15,23,42,.28)}
      `}</style>
      <div
        style={{
          flexShrink: 0,
          padding: '14px 16px',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
        }}
      >
        <div
          role="presentation"
          onMouseDown={onPanelDragHandleMouseDown}
          style={{
            flex: 1,
            minWidth: 0,
            cursor: 'grab',
            userSelect: 'none',
            WebkitUserSelect: 'none',
          }}
        >
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: '#111827' }}>{t('backgroundTasks.title')}</h3>
        </div>
        <button
          type="button"
          onClick={() => setPanelOpen(false)}
          onMouseDown={(e) => e.stopPropagation()}
          aria-label={t('backgroundTasks.close')}
          style={{
            flexShrink: 0,
            width: 28,
            height: 28,
            borderRadius: 8,
            border: 'none',
            backgroundColor: 'transparent',
            color: '#6b7280',
            cursor: 'pointer',
            fontSize: 18,
            lineHeight: 1,
          }}
        >
          ×
        </button>
      </div>
      <div
        style={{
          flexShrink: 0,
          display: 'flex',
          borderBottom: '1px solid #e5e7eb',
          padding: '0 12px',
        }}
      >
        {(['active', 'done', 'failed'] as const).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            style={{
              padding: '10px 14px',
              fontSize: 13,
              fontWeight: 600,
              border: 'none',
              backgroundColor: 'transparent',
              color: tab === key ? '#2563eb' : '#6b7280',
              borderBottom: tab === key ? '2px solid #2563eb' : '2px solid transparent',
              marginBottom: -1,
              cursor: 'pointer',
            }}
          >
            {key === 'active'
              ? t('backgroundTasks.tabRunning')
              : key === 'done'
                ? t('backgroundTasks.tabCompleted')
                : t('backgroundTasks.tabFailed')}
          </button>
        ))}
      </div>
      <div
        className="task-center-panel__list"
        style={{
          flex: '1 1 0%',
          minHeight: 0,
          minWidth: 0,
          overflowX: 'hidden',
          overflowY: 'auto',
          WebkitOverflowScrolling: 'touch',
          overscrollBehavior: 'contain',
          padding: 12,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'stretch',
          gap: 10,
          scrollbarGutter: 'stable',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(15, 23, 42, 0.22) transparent',
        }}
      >
        {list.length === 0 ? (
          <div style={{ color: '#9ca3af', fontSize: 13, textAlign: 'center', padding: 24 }}>
            {tab === 'active'
              ? t('backgroundTasks.emptyRunning')
              : tab === 'done'
                ? t('backgroundTasks.emptyCompleted')
                : t('backgroundTasks.emptyFailed')}
          </div>
        ) : (
          <>
            {list.map((task) => (
              <TaskCenterItem
                key={task.id}
                task={task}
                onRemove={removeTask}
                onRequestDelete={(tk) => setDeleteConfirmTask(tk)}
                onRequestCancel={(tk) => setCancelConfirmTask(tk)}
                cancelLoadingTaskId={cancelLoading ? cancelConfirmTask?.id ?? null : null}
                onNotify={showToast}
              />
            ))}
            {hasMore && (
              <div
                style={{
                  flexShrink: 0,
                  fontSize: 11,
                  color: '#9ca3af',
                  textAlign: 'center',
                  padding: '8px 0',
                }}
              >
                {t('backgroundTasks.onlyShowRecent', { n: 50 })}
              </div>
            )}
          </>
        )}
      </div>
      {canClearCurrent && (
        <div
          style={{
            flexShrink: 0,
            padding: 10,
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <button
            type="button"
            onClick={() => clearTaskGroup(tab)}
            style={{
              padding: '6px 12px',
              fontSize: 12,
              fontWeight: 600,
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              backgroundColor: '#fff',
              color: '#6b7280',
              cursor: 'pointer',
            }}
          >
            {t('backgroundTasks.clearList')}
          </button>
        </div>
      )}

      {cancelConfirmTask && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15,23,42,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1300,
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget && !cancelLoading) setCancelConfirmTask(null);
          }}
        >
          <div
            style={{
              backgroundColor: '#fff',
              borderRadius: 12,
              padding: '18px 20px 16px',
              width: '100%',
              maxWidth: 360,
              boxShadow: '0 12px 40px rgba(15,23,42,0.15)',
              border: '1px solid #e5e7eb',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h4 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700, color: '#111827', lineHeight: 1.35 }}>
              {t('backgroundTasks.confirmCancelTitle')}
            </h4>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                type="button"
                onClick={() => !cancelLoading && setCancelConfirmTask(null)}
                disabled={cancelLoading}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  borderRadius: 8,
                  border: '1px solid #d1d5db',
                  backgroundColor: '#fff',
                  color: '#374151',
                  cursor: cancelLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {t('backgroundTasks.cancel')}
              </button>
              <button
                type="button"
                onClick={handleConfirmCancel}
                disabled={cancelLoading}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  borderRadius: 8,
                  border: 'none',
                  backgroundColor: '#ea580c',
                  color: '#fff',
                  cursor: cancelLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {cancelLoading ? `${t('backgroundTasks.actionCancel')}…` : t('dialog.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteConfirmTask && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15,23,42,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1300,
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget && !deleteLoading) setDeleteConfirmTask(null);
          }}
        >
          <div
            style={{
              backgroundColor: '#fff',
              borderRadius: 12,
              padding: 20,
              width: '100%',
              maxWidth: 360,
              boxShadow: '0 12px 40px rgba(15,23,42,0.15)',
              border: '1px solid #e5e7eb',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h4 style={{ margin: '0 0 8px', fontSize: 15, fontWeight: 700, color: '#111827' }}>
              {deleteConfirmTask.type === 'export'
                ? t('backgroundTasks.confirmDeleteExportTitle')
                : deleteConfirmTask.type === 'convert'
                  ? t('backgroundTasks.confirmDeleteConvertTitle')
                  : deleteConfirmTask.type === 'import'
                    ? t('backgroundTasks.confirmDeleteImportTitle')
                    : t('backgroundTasks.confirmDeleteGenericTitle')}
            </h4>
            <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.5 }}>
              {deleteConfirmTask.type === 'export'
                ? t('backgroundTasks.confirmDeleteExportDescription')
                : deleteConfirmTask.type === 'convert'
                  ? t('backgroundTasks.confirmDeleteConvertDescription')
                  : deleteConfirmTask.type === 'import'
                    ? t('backgroundTasks.confirmDeleteImportDescription')
                    : t('backgroundTasks.confirmDeleteGenericDescription')}
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                type="button"
                onClick={() => !deleteLoading && setDeleteConfirmTask(null)}
                disabled={deleteLoading}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  borderRadius: 8,
                  border: '1px solid #d1d5db',
                  backgroundColor: '#fff',
                  color: '#374151',
                  cursor: deleteLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {t('backgroundTasks.cancel')}
              </button>
              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={deleteLoading}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  borderRadius: 8,
                  border: 'none',
                  backgroundColor: '#dc2626',
                  color: '#fff',
                  cursor: deleteLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {deleteLoading ? `${t('backgroundTasks.confirmDelete')}…` : t('backgroundTasks.confirmDelete')}
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 13,
            zIndex: 1301,
            backgroundColor: toast.isError ? '#fef2f2' : 'rgba(17,24,39,0.92)',
            color: toast.isError ? '#b91c1c' : '#fff',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
