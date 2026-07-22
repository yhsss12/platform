'use client';

import { useEffect, useState, useCallback } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import {
  listProviders,
  getProviderDetail,
  updateUserProvider,
  verifyProvider,
  updateUserModels,
  createModel,
  deleteModel,
  updateModel,
  type ProviderItem,
  type ProviderDetail,
  type ModelItem,
} from '../api/llmApi';

const GUIDE_URLS: Record<string, string> = {
  openai: 'https://platform.openai.com/api-keys',
  deepseek: 'https://platform.deepseek.com/api_keys',
  qwen: 'https://dashscope.console.aliyun.com/apiKey',
  gemini: 'https://aistudio.google.com/apikey',
  claude: 'https://console.anthropic.com/settings/keys',
  zhipu: 'https://open.bigmodel.cn/usercenter/apikeys',
};

export interface ModelManageModalProps {
  open: boolean;
  onClose: () => void;
  projectId?: string;
  canEdit?: boolean;
  onSave?: (config: {
    apiKey: string;
    baseUrl: string;
    selectedModelName: string;
    providerId: number;
    modelNames: string[];
  }) => void;
}

export default function ModelManageModal({
  open,
  onClose,
  projectId,
  canEdit = true,
  onSave,
}: ModelManageModalProps) {
  const { t } = useI18n();
  const [search, setSearch] = useState('');
  const [providers, setProviders] = useState<ProviderItem[]>([]);
  const [currentProvider, setCurrentProvider] = useState<ProviderDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');
  const [verifyMessage, setVerifyMessage] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiBase, setApiBase] = useState('');
  const [isEnabled, setIsEnabled] = useState(false);
  const [modelSearch, setModelSearch] = useState('');
  const [modelFormOpen, setModelFormOpen] = useState(false);
  const [modelFormMode, setModelFormMode] = useState<'create' | 'edit'>('create');
  const [editingModel, setEditingModel] = useState<ModelItem | null>(null);
  const [modelFormName, setModelFormName] = useState('');
  const [modelFormDisplayName, setModelFormDisplayName] = useState('');
  const [modelFormError, setModelFormError] = useState('');
  const [deleteModelTarget, setDeleteModelTarget] = useState<ModelItem | null>(null);

  const loadProviders = useCallback(async () => {
    if (!open) return;
    setLoading(true);
    try {
      const res = await listProviders(search || undefined, projectId);
      if (res.ok && res.data) setProviders(res.data);
      else setProviders([]);
    } catch {
      setProviders([]);
    } finally {
      setLoading(false);
    }
  }, [open, search, projectId]);

  const loadProviderDetail = useCallback(
    async (providerId: number) => {
      setLoading(true);
      try {
        const res = await getProviderDetail(providerId, projectId);
        if (res.ok && res.data) {
          setCurrentProvider(res.data);
          setApiBase(res.data.api_base || res.data.base_url || '');
          setIsEnabled(res.data.is_enabled);
          setApiKey(''); // 不回显明文，仅显示 api_key_masked
          setVerifyStatus('idle');
          setVerifyMessage('');
        } else {
          setCurrentProvider(null);
        }
      } catch {
        setCurrentProvider(null);
      } finally {
        setLoading(false);
      }
    },
    [projectId]
  );

  useEffect(() => {
    if (!open) return;
    loadProviders();
  }, [open, loadProviders]);

  useEffect(() => {
    if (open && providers.length > 0 && !currentProvider) {
      const first = providers[0];
      loadProviderDetail(first.id);
    }
  }, [open, providers, currentProvider, loadProviderDetail]);

  const handleSelectProvider = (p: ProviderItem) => {
    loadProviderDetail(p.id);
  };

  const openCreateModelForm = () => {
    if (!currentProvider) return;
    setModelFormMode('create');
    setEditingModel(null);
    setModelFormName('');
    setModelFormDisplayName('');
    setModelFormError('');
    setModelFormOpen(true);
  };

  const openEditModelForm = (model: ModelItem) => {
    if (!currentProvider) return;
    setModelFormMode('edit');
    setEditingModel(model);
    setModelFormName(model.name);
    setModelFormDisplayName(model.display_name || model.name);
    setModelFormError('');
    setModelFormOpen(true);
  };

  const handleSubmitModelForm = async () => {
    if (!currentProvider) return;
    if (!projectId) {
      setModelFormError('缺少项目 ID，无法保存模型配置');
      return;
    }
    const name = modelFormName.trim();
    const displayName = modelFormDisplayName.trim();
    if (!name) {
      setModelFormError(t('labelModelManage.modelNameRequired') || '模型名称不能为空');
      return;
    }
    try {
      if (modelFormMode === 'create') {
        await createModel({
          project_id: projectId,
          provider_id: currentProvider.id,
          name,
          display_name: displayName || undefined,
        });
      } else if (editingModel) {
        await updateModel(editingModel.id, {
          name,
          display_name: displayName || name,
        }, projectId);
      }
      await loadProviderDetail(currentProvider.id);
      setModelFormOpen(false);
    } catch (e) {
      console.error(e);
      setModelFormError(t('labelModelManage.verifyFailed'));
    }
  };

  const handleDeleteModel = (model: ModelItem) => {
    if (!currentProvider) return;
    setDeleteModelTarget(model);
  };

  const handleConfirmDeleteModel = async () => {
    if (!currentProvider || !deleteModelTarget || !projectId) return;
    try {
      await deleteModel(deleteModelTarget.id, projectId);
      await loadProviderDetail(currentProvider.id);
    } catch (e) {
      console.error(e);
    } finally {
      setDeleteModelTarget(null);
    }
  };

  const handleVerify = async () => {
    if (!currentProvider) return;
    if (!canEdit) {
      setVerifyMessage('当前角色无权限配置模型');
      setVerifyStatus('error');
      return;
    }
    const key = apiKey.trim();
    if (!key && !currentProvider.has_api_key) {
      setVerifyMessage(t('labelModelManage.verifyNeedApiKey'));
      setVerifyStatus('error');
      return;
    }
    setVerifyStatus('loading');
    setVerifyMessage('');
    const res = await verifyProvider({
      provider_id: currentProvider.id,
      project_id: projectId || '',
      api_key: key || undefined,
      api_base: apiBase.trim() || undefined,
    });
    if (res.ok && res.data?.success) {
      setVerifyStatus('ok');
      setVerifyMessage(t('labelModelManage.verifySuccess'));
    } else {
      setVerifyStatus('error');
      setVerifyMessage(res.error || t('labelModelManage.verifyFailed'));
    }
  };

  const handleSave = async () => {
    if (!currentProvider) return;
    if (!canEdit) return;
    setSaving(true);
    try {
      await updateUserProvider({
        provider_id: currentProvider.id,
        project_id: projectId || '',
        api_key: apiKey.trim() || undefined,
        api_base: apiBase.trim() || undefined,
        is_enabled: isEnabled,
      });
      // 目前将该厂商下的全部模型都标记为“已启用”
      await updateUserModels({
        provider_id: currentProvider.id,
        project_id: projectId || '',
        model_ids: currentProvider.models.map((m) => m.id),
      });
      const firstSelected = currentProvider.models[0];
      const modelName = firstSelected?.name || '';
      const baseUrl = (apiBase.trim() || currentProvider.base_url || '').replace(/\/$/, '');
      onSave?.({
        apiKey: apiKey.trim(),
        baseUrl,
        selectedModelName: modelName,
        providerId: currentProvider.id,
        modelNames: currentProvider.models.map((m) => m.name),
      });
      onClose();
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  const filteredModels =
    currentProvider?.models.filter((m) => {
      if (!modelSearch.trim()) return true;
      const kw = modelSearch.trim().toLowerCase();
      return (
        (m.name || '').toLowerCase().includes(kw) ||
        (m.display_name || '').toLowerCase().includes(kw)
      );
    }) || [];

  if (!open) return null;

  const guideUrl = currentProvider ? GUIDE_URLS[currentProvider.code] || '#' : '#';

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0,0,0,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#fff',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.15)',
          width: '95%',
          maxWidth: '560px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ padding: '16px', borderBottom: '1px solid #e5e7eb', flexShrink: 0 }}>
          <div style={{ fontSize: '18px', fontWeight: '600', color: '#111827', marginBottom: '12px' }}>
            {t('labelAgentPanel.manageModelsTitle')}
          </div>
          <input
            type="search"
            name="llm-provider-filter"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            data-lpignore="true"
            placeholder={t('labelAgentPanel.manageModelsSearchPlaceholder')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: '100%',
              height: '36px',
              padding: '0 12px',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              fontSize: '14px',
              outline: 'none',
            }}
          />
        </div>

        {/* 厂商快速切换 */}
        <div
          style={{
            padding: '10px 16px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            gap: '8px',
            overflowX: 'auto',
            flexShrink: 0,
          }}
        >
          {providers.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => handleSelectProvider(p)}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: currentProvider?.id === p.id ? '1px solid #2563eb' : '1px solid #e5e7eb',
                backgroundColor: currentProvider?.id === p.id ? '#eff6ff' : '#fff',
                color: currentProvider?.id === p.id ? '#2563eb' : '#374151',
                fontSize: '13px',
                whiteSpace: 'nowrap',
                cursor: 'pointer',
              }}
            >
              {p.name}
            </button>
          ))}
        </div>

        {loading && !currentProvider ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#6b7280' }}>{t('labelModelManage.loading')}</div>
        ) : currentProvider ? (
          <>
            {/* 当前厂商配置卡片 */}
            <div
              style={{
                padding: '16px',
                borderBottom: '1px solid #e5e7eb',
                flexShrink: 0,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span style={{ fontSize: '15px', fontWeight: '600', color: '#111827' }}>
                  {currentProvider.name}
                </span>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={isEnabled}
                    onChange={(e) => setIsEnabled(e.target.checked)}
                    disabled={!canEdit}
                  />
                  <span style={{ fontSize: '13px', color: '#374151' }}>ON</span>
                </label>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>{t('labelModelManage.apiKeyLabel')}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    name="llm-api-key"
                    autoComplete={showApiKey ? 'off' : 'new-password'}
                    autoCorrect="off"
                    spellCheck={false}
                    data-lpignore="true"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    disabled={!canEdit}
                    placeholder={
                      currentProvider.has_api_key && !apiKey
                        ? `${t('labelModelManage.apiKeySet')} (${currentProvider.api_key_masked || '***'})`
                        : t('labelModelManage.apiKeyPlaceholder')
                    }
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
                    disabled={!canEdit}
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
                    {showApiKey ? t('labelModelManage.hide') : t('labelModelManage.show')}
                  </button>
                  <button
                    type="button"
                    onClick={handleVerify}
                    disabled={verifyStatus === 'loading' || !canEdit}
                    style={{
                      padding: '0 12px',
                      height: '36px',
                      border: 'none',
                      borderRadius: '6px',
                      background: verifyStatus === 'loading' ? '#e5e7eb' : '#2563eb',
                      color: '#fff',
                      cursor: verifyStatus === 'loading' ? 'not-allowed' : 'pointer',
                      fontSize: '13px',
                    }}
                  >
                    {verifyStatus === 'loading' ? t('labelModelManage.verifying') : t('labelModelManage.verify')}
                  </button>
                </div>
                {verifyMessage && (
                  <div
                    style={{
                      marginTop: '6px',
                      fontSize: '12px',
                      color: verifyStatus === 'ok' ? '#059669' : '#dc2626',
                    }}
                  >
                    {verifyMessage}
                  </div>
                )}
                <div style={{ marginTop: '6px', fontSize: '12px', color: '#9ca3af' }}>
                  {t('labelModelManage.multiKeyHint')}
                </div>
                <a
                  href={guideUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ fontSize: '12px', color: '#2563eb', marginTop: '4px', display: 'inline-block' }}
                >
                  {t('labelModelManage.getKeyHint')}
                </a>
              </div>
              <div>
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>{t('labelModelManage.apiBaseUrlLabel')}</div>
                <input
                  type="url"
                  name="llm-api-base"
                  autoComplete="off"
                  autoCorrect="off"
                  spellCheck={false}
                  data-lpignore="true"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  disabled={!canEdit}
                  placeholder={currentProvider.base_url || t('labelModelManage.apiBaseUrlPlaceholder')}
                  style={{
                    width: '100%',
                    height: '34px',
                    padding: '0 10px',
                    borderRadius: '6px',
                    border: '1px solid #e5e7eb',
                    fontSize: '13px',
                    outline: 'none',
                  }}
                />
              </div>
            </div>

            {/* 模型列表 */}
            <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <div style={{ padding: '8px 16px', borderBottom: '1px solid #e5e7eb', flexShrink: 0 }}>
                <input
                  type="search"
                  name="llm-model-filter"
                  autoComplete="off"
                  autoCorrect="off"
                  spellCheck={false}
                  data-lpignore="true"
                  placeholder={t('labelModelManage.searchModelPlaceholder')}
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  style={{
                    width: '100%',
                    height: '32px',
                    padding: '0 10px',
                    borderRadius: '6px',
                    border: '1px solid #e5e7eb',
                    fontSize: '13px',
                    outline: 'none',
                  }}
                />
              </div>
              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  padding: '12px 16px',
                }}
              >
                {filteredModels.length === 0 ? (
                  <div style={{ color: '#9ca3af', fontSize: '13px' }}>{t('labelModelManage.emptyModels')}</div>
                ) : (
                  filteredModels.map((m) => (
                    <div
                      key={m.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '10px',
                        padding: '8px 0',
                        borderBottom: '1px solid #f3f4f6',
                      }}
                    >
                      <span style={{ fontSize: '13px', color: '#111827', flex: 1 }}>
                        {m.display_name || m.name}
                      </span>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <button
                          type="button"
                          onClick={() => openEditModelForm(m)}
                          disabled={!canEdit}
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            border: '1px solid #d1d5db',
                            background: '#f9fafb',
                            color: '#374151',
                            fontSize: '12px',
                            cursor: 'pointer',
                          }}
                        >
                          {t('labelModelManage.edit')}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteModel(m)}
                          disabled={!canEdit}
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            border: '1px solid #fecaca',
                            background: '#fef2f2',
                            color: '#b91c1c',
                            fontSize: '12px',
                            cursor: 'pointer',
                          }}
                        >
                          {t('labelModelManage.delete')}
                        </button>
                      </div>
                    </div>
                  ))
                )}
                {currentProvider && (
                  <a
                    href={guideUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'inline-block',
                      marginTop: '12px',
                      fontSize: '12px',
                      color: '#2563eb',
                    }}
                  >
                    {t('labelModelManage.getKeyHint')}
                  </a>
                )}
              </div>
            </div>

            {/* 底部操作栏 */}
            <div
              style={{
                padding: '12px 16px',
                borderTop: '1px solid #e5e7eb',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexShrink: 0,
              }}
            >
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={openCreateModelForm}
                  disabled={!canEdit}
                  style={{
                    padding: '6px 12px',
                    border: '1px solid #e5e7eb',
                    borderRadius: '6px',
                    background: '#fff',
                    fontSize: '13px',
                    color: '#374151',
                    cursor: 'pointer',
                  }}
                >
                  + {t('labelModelManage.addModel')}
                </button>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={onClose}
                  style={{
                    padding: '8px 16px',
                    border: '1px solid #e5e7eb',
                    borderRadius: '6px',
                    background: '#fff',
                    fontSize: '13px',
                    color: '#374151',
                    cursor: 'pointer',
                  }}
                >
                  {t('labelModelManage.cancel')}
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saving || !canEdit}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderRadius: '6px',
                    background: saving ? '#e5e7eb' : '#2563eb',
                    color: '#fff',
                    fontSize: '13px',
                    cursor: saving ? 'not-allowed' : 'pointer',
                  }}
                >
                  {saving ? t('labelModelManage.saving') : t('labelModelManage.save')}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: '#6b7280' }}>
            {t('labelModelManage.noProviders')}
          </div>
        )}
      </div>
      {/* 模型名称输入弹窗（新增/编辑模型） */}
      {modelFormOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 11000,
          }}
          onClick={() => setModelFormOpen(false)}
        >
          <div
            style={{
              width: 420,
              maxWidth: '96vw',
              backgroundColor: '#ffffff',
              borderRadius: 12,
              border: '1px solid #e5e7eb',
              boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
              padding: 20,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
              {modelFormMode === 'create'
                ? t('labelModelManage.promptNewModelName')
                : t('labelModelManage.promptEditModelName')}
            </div>
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 13, color: '#4b5563', marginBottom: 4, display: 'block' }}>
                {t('labelModelManage.modelNameLabel') || '模型名称'}
              </label>
              <input
                type="text"
                name="llm-model-internal-name"
                autoComplete="off"
                data-lpignore="true"
                value={modelFormName}
                onChange={(e) => {
                  setModelFormName(e.target.value);
                  if (modelFormError) setModelFormError('');
                }}
                style={{
                  width: '100%',
                  height: 36,
                  padding: '0 10px',
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  fontSize: 13,
                  outline: 'none',
                }}
              />
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 13, color: '#4b5563', marginBottom: 4, display: 'block' }}>
                {t('labelModelManage.modelDisplayNameLabel') || '展示名称'}
              </label>
              <input
                type="text"
                name="llm-model-display-name"
                autoComplete="off"
                data-lpignore="true"
                value={modelFormDisplayName}
                onChange={(e) => setModelFormDisplayName(e.target.value)}
                style={{
                  width: '100%',
                  height: 36,
                  padding: '0 10px',
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  fontSize: 13,
                  outline: 'none',
                }}
              />
            </div>
            {modelFormError && (
              <div style={{ color: '#dc2626', fontSize: 12, marginBottom: 12 }}>{modelFormError}</div>
            )}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                type="button"
                onClick={() => setModelFormOpen(false)}
                style={{
                  padding: '8px 14px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#ffffff',
                  fontSize: 13,
                  color: '#374151',
                  cursor: 'pointer',
                }}
              >
                {t('labelModelManage.cancel')}
              </button>
              <button
                type="button"
                onClick={handleSubmitModelForm}
                disabled={!canEdit}
                style={{
                  padding: '8px 14px',
                  borderRadius: 8,
                  border: 'none',
                  backgroundColor: '#2563eb',
                  color: '#ffffff',
                  fontSize: 13,
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                {t('labelModelManage.confirm') || '确认'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 删除模型确认弹窗：使用统一 ConfirmDialog，确认按钮为危险态 */}
      <ConfirmDialog
        open={!!deleteModelTarget}
        title={t('labelModelManage.deleteModelTitle') || '删除模型'}
        description={
          deleteModelTarget
            ? t('labelModelManage.confirmDeleteModel', {
                name: deleteModelTarget.display_name || deleteModelTarget.name,
              })
            : t('labelModelManage.confirmDeleteModel', { name: '' })
        }
        confirmText={t('dialog.confirm')}
        cancelText={t('dialog.cancel')}
        onCancel={() => setDeleteModelTarget(null)}
        onConfirm={handleConfirmDeleteModel}
      />
    </div>
  );
}
