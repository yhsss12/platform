'use client';

import { useState, useEffect } from 'react';
import { listDirs, listAgentDirs } from '../../api/fsApi';
import { useI18n } from '@/components/common/I18nProvider';

interface FolderPickerModalProps {
  open: boolean;
  onClose: () => void;
  onSelect: (folderPath: string) => void;
  /** 初始基础目录，用于覆盖后端默认 ROOT_DIR 或 Agent 默认目录 */
  defaultBase?: string;
  /** 是否通过采集端 Agent 浏览文件系统 */
  useAgentFs?: boolean;
  /** 设备 ID，用于解析 Agent（useAgentFs=true 时生效） */
  deviceId?: string;
}

export default function FolderPickerModal({
  open,
  onClose,
  onSelect,
  defaultBase,
  useAgentFs,
  deviceId,
}: FolderPickerModalProps) {
  const { t } = useI18n();
  const [currentPath, setCurrentPath] = useState<string>('');
  const [dirs, setDirs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // 当弹窗打开时，加载根目录或指定基础目录
  useEffect(() => {
    if (open) {
      // 如果传入了 defaultBase，则以它为起点，否则让后端/Agent 使用默认根目录
      loadDirs(defaultBase);
    } else {
      // 关闭时重置状态
      setCurrentPath('');
      setDirs([]);
      setError('');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, defaultBase]);

  const loadDirs = async (base?: string) => {
    setLoading(true);
    setError('');
    try {
      const response = useAgentFs
        ? await listAgentDirs(deviceId, base)
        : await listDirs(base);

      if (response.ok && response.data) {
        setCurrentPath(response.data.base);
        setDirs(response.data.dirs);
      } else {
        setError(response.error || t('pathPicker.loadError'));
      }
    } catch (err: any) {
      setError(err.message || t('pathPicker.loadError'));
    } finally {
      setLoading(false);
    }
  };

  const handleDirClick = (dirName: string) => {
    // 构建下一级路径
    const nextPath = currentPath 
      ? (currentPath.endsWith('/') ? currentPath + dirName : `${currentPath}/${dirName}`)
      : dirName;
    loadDirs(nextPath);
  };

  const handleGoUp = () => {
    if (!currentPath || currentPath === '/') return;
    
    // 获取父目录
    // 移除末尾的斜杠（如果有）
    const normalizedPath = currentPath.replace(/\/+$/, '');
    const parts = normalizedPath.split('/').filter(Boolean);
    
    // 移除最后一部分，构建父路径
    parts.pop();
    const parentPath = parts.length > 0 ? '/' + parts.join('/') : '/';
    loadDirs(parentPath);
  };

  const handleSelect = () => {
    onSelect(currentPath);
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
          width: '600px',
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
            {t('pathPicker.titleFolder')}
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

        {/* 目录列表 */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '8px 0',
            minHeight: '300px',
            maxHeight: '400px',
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
          ) : dirs.length === 0 ? (
            <div
              style={{
                padding: '40px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}
            >
              {t('pathPicker.emptyDir')}
            </div>
          ) : (
            <>
              {/* 返回上一级按钮 */}
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
              {/* 目录列表 */}
              {dirs.map((dir) => (
                <div
                  key={dir}
                  onClick={() => handleDirClick(dir)}
                  style={{
                    padding: '10px 24px',
                    fontSize: '13px',
                    color: '#374151',
                    cursor: 'pointer',
                    backgroundColor: 'transparent',
                    transition: 'background-color 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f9fafb';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <span style={{ fontSize: '16px' }}>📁</span>
                  <span>{dir}</span>
                </div>
              ))}
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
            onClick={handleSelect}
            disabled={!currentPath}
            style={{
              padding: '10px 24px',
              backgroundColor: currentPath ? '#2563eb' : '#d1d5db',
              border: 'none',
              borderRadius: '6px',
              color: '#ffffff',
              fontSize: '14px',
              cursor: currentPath ? 'pointer' : 'not-allowed',
              fontWeight: '500',
              boxShadow: currentPath ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              if (currentPath) {
                e.currentTarget.style.backgroundColor = '#1d4ed8';
              }
            }}
            onMouseLeave={(e) => {
              if (currentPath) {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }
            }}
          >
            {t('pathPicker.confirmFolder')}
          </button>
        </div>
      </div>
    </div>
  );
}

