'use client';

import { useState, useEffect } from 'react';
import { useI18n } from '@/components/common/I18nProvider';

const STORAGE_KEY_API_KEY = 'label_agent_api_key';
const STORAGE_KEY_BASE_URL = 'label_agent_base_url';

export interface ApiConfig {
  apiKey: string;
  baseUrl: string;
}

export interface ApiConfigScope {
  userId?: string | null;
  projectId?: string | null;
}

interface ApiConfigDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (config: ApiConfig) => void;
  initialConfig?: ApiConfig | null;
}

function scopedStorageKey(base: string, scope?: ApiConfigScope): string {
  const uid = (scope?.userId || '').trim();
  const pid = (scope?.projectId || '').trim();
  if (!uid && !pid) return base;
  return `${base}::u=${uid || '-'}::p=${pid || '-'}`;
}

export function getStoredApiConfig(scope?: ApiConfigScope): ApiConfig {
  if (typeof window === 'undefined') {
    return { apiKey: '', baseUrl: '' };
  }
  const kApi = scopedStorageKey(STORAGE_KEY_API_KEY, scope);
  const kBase = scopedStorageKey(STORAGE_KEY_BASE_URL, scope);
  return {
    apiKey: localStorage.getItem(kApi) || '',
    baseUrl: localStorage.getItem(kBase) || '',
  };
}

export function setStoredApiConfig(config: ApiConfig, scope?: ApiConfigScope): void {
  if (typeof window === 'undefined') return;
  const kApi = scopedStorageKey(STORAGE_KEY_API_KEY, scope);
  const kBase = scopedStorageKey(STORAGE_KEY_BASE_URL, scope);
  if (config.apiKey) localStorage.setItem(kApi, config.apiKey);
  else localStorage.removeItem(kApi);
  if (config.baseUrl) localStorage.setItem(kBase, config.baseUrl);
  else localStorage.removeItem(kBase);
}

export default function ApiConfigDialog({
  open,
  onClose,
  onSave,
  initialConfig,
}: ApiConfigDialogProps) {
  const { t } = useI18n();
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testMessage, setTestMessage] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (open) {
      const cfg = initialConfig ?? getStoredApiConfig();
      setApiKey(cfg.apiKey);
      setBaseUrl(cfg.baseUrl);
      setTestMessage(null);
    }
  }, [open, initialConfig]);

  const handleTest = async () => {
    const url = (baseUrl || '').trim().replace(/\/$/, '');
    const key = apiKey.trim();
    if (!url || !key) {
      setTestMessage({ type: 'error', text: '请填写 API 地址和 API 密钥' });
      return;
    }
    setTesting(true);
    setTestMessage(null);
    try {
      const res = await fetch(`${url}/v1/models`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${key}` },
      });
      if (res.ok) {
        setTestMessage({ type: 'ok', text: '连接成功' });
      } else {
        const t = await res.text();
        setTestMessage({ type: 'error', text: `请求失败: ${res.status} ${t.slice(0, 80)}` });
      }
    } catch (e) {
      setTestMessage({
        type: 'error',
        text: `请求异常: ${e instanceof Error ? e.message : String(e)}`,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    const config: ApiConfig = {
      apiKey: apiKey.trim(),
      baseUrl: baseUrl.trim().replace(/\/$/, ''),
    };
    setStoredApiConfig(config);
    onSave(config);
    onClose();
  };

  if (!open) return null;

  const previewUrl = baseUrl.trim() ? `${baseUrl.replace(/\/$/, '')}/v1beta/models` : '';

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0,0,0,0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
          width: '90%',
          maxWidth: '440px',
          padding: '20px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ fontSize: '16px', fontWeight: '600', color: '#111827', marginBottom: '16px' }}>
          API 配置
        </div>

        <div style={{ marginBottom: '14px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '6px',
            }}
          >
            <span style={{ fontSize: '13px', fontWeight: '500', color: '#374151' }}>API 密钥</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type={showApiKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="请输入 API Key"
              style={{
                flex: 1,
                height: '36px',
                padding: '0 10px',
                borderRadius: '6px',
                border: '1px solid #e5e7eb',
                fontSize: '13px',
                outline: 'none',
              }}
            />
            <button
              type="button"
              onClick={() => setShowApiKey((v) => !v)}
              style={{
                padding: '0 10px',
                height: '36px',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                background: '#f9fafb',
                cursor: 'pointer',
                fontSize: '12px',
                color: '#6b7280',
              }}
            >
              {showApiKey ? '隐藏' : '显示'}
            </button>
            <button
              type="button"
              onClick={handleTest}
              disabled={testing}
              style={{
                padding: '0 12px',
                height: '36px',
                border: 'none',
                borderRadius: '6px',
                background: testing ? '#e5e7eb' : '#2563eb',
                color: testing ? '#9ca3af' : '#fff',
                cursor: testing ? 'not-allowed' : 'pointer',
                fontSize: '13px',
              }}
            >
              {testing ? '检测中...' : '检测'}
            </button>
          </div>
          <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '4px' }}>
            多个密钥使用逗号分隔
          </div>
          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontSize: '12px', color: '#2563eb', marginTop: '4px', display: 'inline-block' }}
          >
            点击这里获取密钥
          </a>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '13px', fontWeight: '500', color: '#374151', marginBottom: '6px' }}>
            API 地址
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://x666.me"
              style={{
                flex: 1,
                height: '36px',
                padding: '0 10px',
                borderRadius: '6px',
                border: '1px solid #e5e7eb',
                fontSize: '13px',
                outline: 'none',
              }}
            />
            <button
              type="button"
              onClick={() => setBaseUrl('')}
              style={{
                padding: '0 10px',
                height: '36px',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                background: '#fff',
                color: '#dc2626',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              {t('common.reset')}
            </button>
          </div>
          {previewUrl && (
            <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '4px' }}>
              预览: {previewUrl}
            </div>
          )}
        </div>

        {testMessage && (
          <div
            style={{
              marginBottom: '12px',
              padding: '8px 10px',
              borderRadius: '6px',
              fontSize: '12px',
              backgroundColor: testMessage.type === 'ok' ? '#ecfdf5' : '#fef2f2',
              color: testMessage.type === 'ok' ? '#059669' : '#dc2626',
            }}
          >
            {testMessage.text}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '0 16px',
              height: '36px',
              border: '1px solid #e5e7eb',
              borderRadius: '6px',
              background: '#fff',
              color: '#374151',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleSave}
            style={{
              padding: '0 16px',
              height: '36px',
              border: 'none',
              borderRadius: '6px',
              background: '#2563eb',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
}
