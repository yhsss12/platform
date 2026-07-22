'use client';

import { useState, useEffect } from 'react';
import { fsList, listDirs, agentFsList, type FsListItem } from '@/features/data-platform/api/fsApi';
import { useI18n } from '@/components/common/I18nProvider';

export type PathPickerMode = 'files' | 'dir';
export type PathPickerSource = 'platform' | 'agent';

interface PathPickerModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (items: { path: string; name: string; type: 'file' | 'dir' }[]) => void;
  /** 选择文件/多选（默认） 或 仅选择当前目录（用于输出路径等） */
  mode?: PathPickerMode;
  /** mode=files 时是否允许选择任意文件（默认仅显示数据资产文件 mcap/hdf5/h5） */
  allowAllFiles?: boolean;
  /** 选择路径来源：platform=平台服务器文件系统；agent=采集端文件系统（通过隧道） */
  source?: PathPickerSource;
  /** source=agent 时用于解析采集端 Agent */
  agentId?: string;
  deviceId?: string;
  /** 弹窗标题，不传则用 i18n 的 pathPicker.titleFile / titleFolder */
  title?: string;
  /** 打开时初始路径，仅 mode=dir 时使用（如转换模块的 outputPath） */
  initialPath?: string;
}

interface EntryView {
  name: string;
  type: 'file' | 'dir';
}

function isSupportedFile(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.endsWith('.mcap') || lower.endsWith('.hdf5') || lower.endsWith('.h5');
}

export default function PathPickerModal({
  open,
  onClose,
  onConfirm,
  mode = 'files',
  allowAllFiles = false,
  source = 'platform',
  agentId,
  deviceId,
  title: titleProp,
  initialPath = '',
}: PathPickerModalProps) {
  const { t } = useI18n();
  const [currentPath, setCurrentPath] = useState<string>('');
  const [entries, setEntries] = useState<EntryView[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [selectedNames, setSelectedNames] = useState<Set<string>>(new Set());

  const isDirMode = mode === 'dir';

  useEffect(() => {
    if (open) {
      (async () => {
        const startPath = initialPath ? initialPath : '';
        if (startPath) {
          await loadEntries(startPath);
          return;
        }
        if (source === 'agent') {
          await loadEntries('');
          return;
        }
        const dirRes = await listDirs();
        const base = dirRes.ok && dirRes.data?.base ? dirRes.data.base : '/';
        await loadEntries(base);
      })();
    } else {
      setCurrentPath('');
      setEntries([]);
      setError('');
      setSelectedNames(new Set());
    }
  }, [open, isDirMode, initialPath]);

  const loadEntries = async (path: string) => {
    setLoading(true);
    setError('');
    try {
      const res =
        source === 'agent'
          ? await agentFsList({ path, agentId, deviceId })
          : await fsList(path);
      if (res.ok && res.data) {
        setCurrentPath(res.data.path);
        const list: EntryView[] = [];
        res.data.items.forEach((item: FsListItem) => {
          if (item.type === 'dir') {
            list.push({ name: item.name, type: 'dir' });
          } else if (allowAllFiles || isSupportedFile(item.name)) {
            list.push({ name: item.name, type: 'file' });
          }
        });
        setEntries(list);
        setSelectedNames(new Set());
      } else {
        setError(res.error || t('pathPicker.loadError'));
        setEntries([]);
      }
    } catch (err: any) {
      setError(err.message || t('pathPicker.loadError'));
      setEntries([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDirDoubleClick = (name: string) => {
    const base = currentPath || '';
    const next = base.endsWith('/') ? base + name : `${base}/${name}`;
    loadEntries(next);
  };

  const handleGoUp = () => {
    if (!currentPath || currentPath === '/') return;
    const normalized = currentPath.replace(/\/+$/, '');
    const parts = normalized.split('/').filter(Boolean);
    parts.pop();
    const parent = parts.length > 0 ? '/' + parts.join('/') : '/';
    loadEntries(parent);
  };

  const toggleSelect = (name: string) => {
    setSelectedNames((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const handleConfirm = () => {
    if (isDirMode) {
      if (!currentPath) {
        onClose();
        return;
      }
      const name = currentPath.replace(/\/+$/, '').split('/').filter(Boolean).pop() || '';
      onConfirm([{ path: currentPath, name, type: 'dir' }]);
      onClose();
      return;
    }
    if (!currentPath || selectedNames.size === 0) {
      onClose();
      return;
    }
    const items: { path: string; name: string; type: 'file' | 'dir' }[] = [];
    entries.forEach((entry) => {
      if (selectedNames.has(entry.name)) {
        const full = currentPath.endsWith('/')
          ? currentPath + entry.name
          : `${currentPath}/${entry.name}`;
        items.push({ path: full, name: entry.name, type: entry.type });
      }
    });
    onConfirm(items);
    onClose();
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          width: '760px',
          maxHeight: '80vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 头部 */}
        <div
          style={{
            padding: '20px 24px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <h3
            style={{
              fontSize: '18px',
              fontWeight: '600',
              color: '#111827',
              margin: 0,
            }}
          >
            {titleProp ?? (isDirMode ? t('pathPicker.titleFolder') : t('pathPicker.titleFile'))}
          </h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#6b7280',
              fontSize: '20px',
              cursor: 'pointer',
              padding: '4px',
              lineHeight: 1,
              width: '24px',
              height: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '4px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
              e.currentTarget.style.color = '#111827';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = '#6b7280';
            }}
          >
            ✕
          </button>
        </div>

        {/* 当前路径显示 */}
        <div
          style={{
            padding: '12px 24px',
            backgroundColor: '#f9fafb',
            borderBottom: '1px solid #e5e7eb',
            fontSize: '13px',
            color: '#6b7280',
            wordBreak: 'break-all',
          }}
        >
          {t('pathPicker.currentPath')}: {currentPath || (loading ? t('pathPicker.loading') : t('pathPicker.notSelected'))}
        </div>

        {/* 列表 */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '8px 0',
            minHeight: '320px',
            maxHeight: '420px',
          }}
          onContextMenu={(e) => {
            // 阻止浏览器默认右键菜单
            e.preventDefault();
          }}
        >
          {loading ? (
            <div
              style={{
                padding: '40px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}
            >
              {t('pathPicker.loading')}
            </div>
          ) : error ? (
            <div
              style={{
                padding: '20px',
                textAlign: 'center',
                color: '#ef4444',
                fontSize: '13px',
              }}
            >
              {error}
            </div>
          ) : entries.length === 0 ? (
            <div
              style={{
                padding: '40px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}
            >
              {t('pathPicker.emptyDirOrFiles')}
            </div>
          ) : (
            <>
              {currentPath && (
                <div
                  onClick={handleGoUp}
                  style={{
                    padding: '10px 24px',
                    fontSize: '13px',
                    color: '#2563eb',
                    cursor: 'pointer',
                    backgroundColor: 'transparent',
                    transition: 'background-color 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f3f4f6';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <span>←</span>
                  <span>{t('pathPicker.goUp')}</span>
                </div>
              )}
              {entries.map((entry) => {
                const isDir = entry.type === 'dir';
                const isSelected = !isDirMode && selectedNames.has(entry.name);
                return (
                  <div
                    key={`${entry.type}-${entry.name}`}
                    onDoubleClick={(e) => {
                      if (isDir) {
                        e.preventDefault();
                        handleDirDoubleClick(entry.name);
                      }
                    }}
                    onMouseDown={(e) => {
                      if (e.button === 0 && !isDirMode) {
                        document.body.dataset.pathPickerDragging = 'true';
                        toggleSelect(entry.name);
                      }
                    }}
                    onMouseEnter={(e) => {
                      const dragging = document.body.dataset.pathPickerDragging === 'true';
                      if (dragging && !isDirMode) {
                        toggleSelect(entry.name);
                      } else if (!isSelected) {
                        e.currentTarget.style.backgroundColor = '#f9fafb';
                      }
                    }}
                    onMouseLeave={(e) => {
                      const dragging = document.body.dataset.pathPickerDragging === 'true';
                      if (!dragging && !isSelected) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                    onMouseUp={(e) => {
                      if (e.button === 0) {
                        document.body.dataset.pathPickerDragging = 'false';
                      }
                    }}
                    style={{
                      padding: '10px 24px',
                      fontSize: '13px',
                      color: isDir ? '#374151' : '#111827',
                      cursor: 'pointer',
                      backgroundColor: isSelected ? '#eff6ff' : 'transparent',
                      transition: 'background-color 0.15s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <span style={{ fontSize: '16px' }}>{isDir ? '📁' : '📄'}</span>
                    <span style={{ flex: 1 }}>{entry.name}</span>
                    {!isDirMode && (
                      <span
                        style={{
                          width: 16,
                          height: 16,
                          borderRadius: 4,
                          border: isSelected ? 'none' : '1px solid #d1d5db',
                          backgroundColor: isSelected ? '#2563eb' : '#ffffff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 12,
                          color: '#ffffff',
                        }}
                      >
                        {isSelected ? '✓' : ''}
                      </span>
                    )}
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* 底部按钮 */}
        <div
          style={{
            padding: '20px 24px',
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'center',
            gap: '12px',
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: '10px 24px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              fontWeight: '500',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#ffffff';
            }}
          >
            {t('common.cancel')}
          </button>
          <button
            onClick={handleConfirm}
            disabled={isDirMode ? !currentPath : !currentPath || selectedNames.size === 0}
            style={{
              padding: '10px 24px',
              backgroundColor: isDirMode
                ? (currentPath ? '#2563eb' : '#d1d5db')
                : (currentPath && selectedNames.size > 0 ? '#2563eb' : '#d1d5db'),
              border: 'none',
              borderRadius: '6px',
              color: '#ffffff',
              fontSize: '14px',
              cursor: isDirMode
                ? (currentPath ? 'pointer' : 'not-allowed')
                : (currentPath && selectedNames.size > 0 ? 'pointer' : 'not-allowed'),
              fontWeight: '500',
              boxShadow: (isDirMode ? currentPath : currentPath && selectedNames.size > 0) ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none',
              transition: 'all 0.2s',
            }}
          >
            {isDirMode ? t('pathPicker.confirmFolder') : t('pathPicker.confirmSelect')}
          </button>
        </div>
      </div>
    </div>
  );
}
