'use client';

import { useState, useRef, useEffect } from 'react';
import type { LabelTask } from '../../models/labelTask';
import FolderPickerModal from './FolderPickerModal';
import DatasetMultiSelectModal from './DatasetMultiSelectModal';
import { getDataAsset, getDataAssetWarehouseDisplayPath } from '@/features/data-platform/api/dataAssetsApi';
import * as projectService from '@/lib/projects/projectService';
import type { Project } from '@/lib/projects/types';
import { useI18n } from '@/components/common/I18nProvider';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import ProjectMemberSelect from './ProjectMemberSelect';

/** 编辑时提交的补丁（不含 id/createdAt，updatedAt 由调用方设置） */
export type LabelTaskEditPatch = {
  name: string;
  dataCount: number;
  projectId: string;
  labeler?: string;
  reviewer?: string;
  collector: string;
  datasetIds?: number[];
  datasetDir?: string;
};

interface CreateLabelTaskModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (task: Omit<LabelTask, 'id' | 'createdAt' | 'updatedAt'>) => void;
  /** 编辑模式：传入当前任务，弹窗标题为“编辑任务”，提交调用 onSave */
  initialTask?: LabelTask | null;
  onSave?: (taskId: string, patch: LabelTaskEditPatch) => void;
  /** 从数据资产页批量标注跳转时预选的数据集 ID 列表 */
  initialDatasetIds?: number[];
  /** 从数据资产页跳转时预填的所属项目 ID */
  initialProjectId?: string;
  /** 从数据资产页跳转时已知的文件路径（单条时由 URL 传入） */
  initialFilePaths?: string[];
  /** 是否来自数据资产页（用于按 data asset id 拉取路径） */
  initialFromDataAssets?: boolean;
  /** 是否正在提交创建（禁用确认按钮并显示“创建中…”） */
  isSubmitting?: boolean;
}

export default function CreateLabelTaskModal({ open, onClose, onSubmit, initialTask, onSave, initialDatasetIds, initialProjectId, initialFilePaths = [], initialFromDataAssets, isSubmitting = false }: CreateLabelTaskModalProps) {
  const { t } = useI18n();
  const isEdit = !!initialTask;
  const [formData, setFormData] = useState({
    name: '',
    datasetDir: '',
    dataCount: '',
    projectId: '',
    labeler: '',
    reviewer: '',
    collector: '默认采集员',
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  // 本地选择的 HDF5 文件列表（仅在当前弹窗 state 中保存，后续步骤再考虑持久化）
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [hdf5Count, setHdf5Count] = useState(0);
  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [showDatasetSelector, setShowDatasetSelector] = useState(false);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);
  const [dataAssetPaths, setDataAssetPaths] = useState<string[]>([]);
  const [dataAssetPathsLoading, setDataAssetPathsLoading] = useState(false);
  const [copyToast, setCopyToast] = useState(false);
  const [projectList, setProjectList] = useState<Project[]>([]);

  // 隐藏的目录选择 input，用于触发 webkitdirectory 文件夹选择
  const directoryInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let active = true;
    projectService.listAsync(false)
      .then((result) => {
        if (!active) return;
        const projects = Array.isArray(result) ? result : result.projects;
        setProjectList(projects.filter((p) => p.status !== '已归档'));
      })
      .catch(() => {
        if (!active) return;
        setProjectList([]);
      });
    return () => {
      active = false;
    };
  }, []);

  // 编辑模式：回填表单
  useEffect(() => {
    if (open && initialTask) {
      setFormData({
        name: initialTask.name || '',
        datasetDir: initialTask.datasetDir || '',
        dataCount: initialTask.dataCount != null ? String(initialTask.dataCount) : '',
        projectId: initialTask.projectId || '',
        labeler: initialTask.labeler || '',
        reviewer: initialTask.reviewer || '',
        collector: initialTask.collector || '默认采集员',
      });
      setSelectedDatasetIds(Array.isArray(initialTask.datasetIds) ? [...initialTask.datasetIds] : []);
    }
  }, [open, initialTask]);

  // 从数据资产页批量标注跳转：预填 projectId 和 datasetIds
  useEffect(() => {
    if (open && !initialTask && (initialDatasetIds?.length || initialProjectId)) {
      if (initialDatasetIds?.length) {
        setSelectedDatasetIds([...initialDatasetIds]);
        setFormData((prev) => ({ ...prev, dataCount: String(initialDatasetIds.length) }));
      }
      if (initialProjectId) {
        setFormData((prev) => ({ ...prev, projectId: initialProjectId }));
      }
    }
  }, [open, initialTask, initialDatasetIds, initialProjectId]);

  // 选择完数据后自动拉取并显示数据路径（两种入口统一：从数据资产页带参数 / 弹窗内点「+」选择）
  useEffect(() => {
    if (!open || selectedDatasetIds.length === 0) {
      setDataAssetPaths([]);
      setDataAssetPathsLoading(false);
      return;
    }
    let cancelled = false;
    setDataAssetPathsLoading(true);
    const fetchPaths = async () => {
      const fetched: string[] = [];
      for (let i = 0; i < selectedDatasetIds.length; i++) {
        if (cancelled) return;
        try {
          const res = await getDataAsset(selectedDatasetIds[i]);
          if (res.ok && res.data) {
            const display = getDataAssetWarehouseDisplayPath(res.data);
            if (display) fetched.push(display);
          }
        } catch {
          /* ignore */
        }
      }
      if (!cancelled) {
        setDataAssetPaths(fetched);
        setDataAssetPathsLoading(false);
      }
    };
    fetchPaths();
    return () => { cancelled = true; };
  }, [open, selectedDatasetIds]);

  // 弹窗关闭时清空表单与选择（避免下次打开带旧数据；创建失败时不在此清空，由下面 handleSubmit 不重置保证表单保留）
  useEffect(() => {
    if (!open) {
      setFormData({
        name: '',
        datasetDir: '',
        dataCount: '',
        projectId: '',
        labeler: '',
        reviewer: '',
        collector: '默认采集员',
      });
      setErrors({});
      setSelectedDatasetIds([]);
      setDataAssetPaths([]);
      setDataAssetPathsLoading(false);
    }
  }, [open]);

  if (!open) return null;

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = t('labelTasksPage.nameRequired');
    }
    const hasDataAssets = selectedDatasetIds.length > 0;
    const hasLocalPath = !!(formData.datasetDir && formData.datasetDir.trim());
    const hasExistingDataset = isEdit && initialTask && (initialTask.datasetDir || (initialTask.datasetIds && initialTask.datasetIds.length > 0));
    if (!hasDataAssets && !hasLocalPath && !hasExistingDataset) {
      newErrors.datasetDir = t('labelTasksPage.datasetOrPathRequired');
    }
    const dataCountNum = parseInt(formData.dataCount, 10);
    if (!formData.dataCount || isNaN(dataCountNum) || dataCountNum <= 0) {
      newErrors.dataCount = t('labelTasksPage.dataCountInvalid');
    }
    if (!formData.projectId.trim()) {
      newErrors.projectId = t('labelTasksPage.projectRequired');
    }
    // 标注员、审核员为可选

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validate()) {
      return;
    }

    const patch = {
      name: formData.name.trim(),
      dataCount: parseInt(formData.dataCount, 10),
      projectId: formData.projectId.trim(),
      labeler: formData.labeler.trim() || undefined,
      reviewer: formData.reviewer.trim() || undefined,
      collector: formData.collector.trim() || '默认采集员',
      datasetIds: selectedDatasetIds.length > 0 ? selectedDatasetIds : undefined,
      datasetDir: formData.datasetDir.trim() || undefined,
    };

    if (isEdit && initialTask && onSave) {
      onSave(initialTask.id, patch);
      handleClose();
      return;
    }

    // 新建：仅提交，不在此清空表单；成功时由父组件关闭弹窗并触发 open→false 从而清空
    onSubmit({
      name: patch.name,
      datasetDir: patch.datasetDir ?? '',
      dataCount: patch.dataCount,
      projectId: patch.projectId,
      labeler: patch.labeler,
      reviewer: patch.reviewer,
      collector: patch.collector,
      datasetIds: selectedDatasetIds.length > 0 ? [...selectedDatasetIds] : [],
      fromDataAssets: initialFromDataAssets ?? selectedDatasetIds.length > 0,
    } as Parameters<typeof onSubmit>[0]);
    // 不再在此处 setFormData/setErrors 清空，避免请求失败后表单被清空用户无法重试
  };

  const handleClose = () => {
    setFormData({
      name: '',
      datasetDir: '',
      dataCount: '',
      projectId: '',
      labeler: '',
      reviewer: '',
      collector: '默认采集员',
    });
    setErrors({});
    setSelectedFiles([]);
    setHdf5Count(0);
    setSelectedDatasetIds([]);
    setDataAssetPaths([]);
    setDataAssetPathsLoading(false);
    onClose();
  };

  const handleFolderFilesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);

    // 过滤出 .hdf5 / .h5 文件
    const hdf5Files = files.filter((file) => {
      const name = file.name.toLowerCase();
      return name.endsWith('.hdf5') || name.endsWith('.h5');
    });

    setSelectedFiles(hdf5Files);
    setHdf5Count(hdf5Files.length);

    // 尝试从第一个文件的 webkitRelativePath 中提取“文件夹名称”（仅用于显示）
    let folderLabel = '';
    if (hdf5Files.length > 0) {
      const first: any = hdf5Files[0];
      const relPath: string | undefined = first.webkitRelativePath;
      if (relPath) {
        const parts = relPath.split('/');
        if (parts.length > 1) {
          folderLabel = parts[0];
        }
      }
    }

    setFormData((prev) => ({
      ...prev,
      datasetDir: folderLabel || prev.datasetDir || '本地数据集目录',
      // 如果尚未填写数据数量，则默认用识别到的 HDF5 数量填充
      dataCount:
        !prev.dataCount && hdf5Files.length > 0
          ? String(hdf5Files.length)
          : prev.dataCount,
    }));

    if (errors.datasetDir) {
      setErrors({ ...errors, datasetDir: '' });
    }
  };

  const handleFolderSelect = (folderPath: string) => {
    setFormData((prev) => ({
      ...prev,
      datasetDir: folderPath,
    }));
    if (errors.datasetDir) {
      setErrors({ ...errors, datasetDir: '' });
    }
  };

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
        zIndex: 1000,
      }}
      onClick={handleClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          width: '600px',
          maxHeight: '90vh',
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
            {isEdit ? t('labelTasksPage.editModalTitle') : t('labelTasksPage.createModalTitle')}
          </h3>
          <ModalCloseButton onClick={handleClose} />
        </div>

        {/* 表单内容 */}
        <div
          style={{
            padding: '24px',
            overflowY: 'auto',
            flex: 1,
          }}
        >
          {/* 任务名称 */}
          <div style={{ marginBottom: '20px' }}>
            <label
              style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: '#374151',
                marginBottom: '8px',
              }}
            >
              {t('labelTasksPage.taskNameLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="text"
              placeholder={t('labelTasksPage.taskNamePlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.name ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                color: '#111827',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'all 0.2s',
              }}
              value={formData.name}
              onChange={(e) => {
                setFormData({ ...formData, name: e.target.value });
                if (errors.name) setErrors({ ...errors, name: '' });
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = '#2563eb';
                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(37, 99, 235, 0.1)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = errors.name ? '#ef4444' : '#d1d5db';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            {errors.name && (
              <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>
                {errors.name}
              </div>
            )}
          </div>

          {/* 数据集 */}
          <div style={{ marginBottom: '20px' }}>
            <label
              style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: '#374151',
                marginBottom: '8px',
              }}
            >
              {t('labelTasksPage.datasetLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <div
                style={{
                  flex: 1,
                  minHeight: '40px',
                  padding: selectedDatasetIds.length > 0 ? '8px 12px' : '0 12px',
                  backgroundColor: '#ffffff',
                  border: errors.datasetDir ? '1px solid #ef4444' : '1px solid #d1d5db',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onClick={() => setShowDatasetSelector(true)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#2563eb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = errors.datasetDir ? '#ef4444' : '#d1d5db';
                }}
              >
                {selectedDatasetIds.length > 0 ? (
                  <div style={{ fontSize: '14px', color: '#111827' }}>
                    已选 {selectedDatasetIds.length} 条
                  </div>
                ) : (
                  <span style={{ fontSize: '14px', color: '#9ca3af' }}>
                    {t('labelTasksPage.datasetPlaceholder')}
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => setShowDatasetSelector(true)}
                style={{
                  width: '40px',
                  height: '40px',
                  padding: 0,
                  backgroundColor: '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: '20px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: '500',
                  transition: 'all 0.2s',
                  flexShrink: 0,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#1d4ed8';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#2563eb';
                }}
              >
                +
              </button>
            </div>
            {errors.datasetDir && (
              <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>
                {errors.datasetDir}
              </div>
            )}
          </div>

          {/* 数据数量 */}
          <div style={{ marginBottom: '20px' }}>
            <label
              style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: '#374151',
                marginBottom: '8px',
              }}
            >
              {t('labelTasksPage.dataCountLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="number"
              placeholder={t('labelTasksPage.dataCountPlaceholder')}
              min="1"
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.dataCount ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                color: '#111827',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'all 0.2s',
              }}
              value={formData.dataCount}
              onChange={(e) => {
                const value = e.target.value;
                if (value === '' || /^\d+$/.test(value)) {
                  setFormData({ ...formData, dataCount: value });
                  if (errors.dataCount) setErrors({ ...errors, dataCount: '' });
                }
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = '#2563eb';
                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(37, 99, 235, 0.1)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = errors.dataCount ? '#ef4444' : '#d1d5db';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            {errors.dataCount && (
              <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>
                {errors.dataCount}
              </div>
            )}
          </div>

          {/* 所属项目 */}
          <div style={{ marginBottom: '20px' }}>
            <label
              style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: '#374151',
                marginBottom: '8px',
              }}
            >
              {t('labelTasksPage.projectLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <select
              value={formData.projectId}
              onChange={(e) => {
                setFormData({
                  ...formData,
                  projectId: e.target.value,
                  labeler: '',
                  reviewer: '',
                });
                if (errors.projectId) setErrors({ ...errors, projectId: '' });
                if (errors.labeler) setErrors({ ...errors, labeler: '' });
                if (errors.reviewer) setErrors({ ...errors, reviewer: '' });
              }}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.projectId ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                color: '#111827',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'all 0.2s',
                cursor: 'pointer',
              }}
            >
              <option value="">{t('labelTasksPage.projectPlaceholder')}</option>
              {projectList.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            {errors.projectId && (
              <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>
                {errors.projectId}
              </div>
            )}
          </div>

          {/* 标注员：仅从项目成员点选 */}
          <div style={{ marginBottom: '20px' }}>
            <ProjectMemberSelect
              projectId={formData.projectId}
              value={formData.labeler}
              onChange={(username) => {
                setFormData({ ...formData, labeler: username });
                if (errors.labeler) setErrors({ ...errors, labeler: '' });
              }}
              label={t('labelTasksPage.labelerLabel')}
              placeholder={t('labelTasksPage.labelerPlaceholder')}
              disabled={isSubmitting}
              error={errors.labeler}
            />
          </div>

          {/* 审核员：仅从项目成员点选 */}
          <div style={{ marginBottom: '20px' }}>
            <ProjectMemberSelect
              projectId={formData.projectId}
              value={formData.reviewer}
              onChange={(username) => {
                setFormData({ ...formData, reviewer: username });
                if (errors.reviewer) setErrors({ ...errors, reviewer: '' });
              }}
              label={t('labelTasksPage.reviewerLabel')}
              placeholder={t('labelTasksPage.reviewerPlaceholder')}
              disabled={isSubmitting}
              error={errors.reviewer}
            />
          </div>
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
            onClick={handleClose}
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
            onClick={handleSubmit}
            disabled={isSubmitting}
            style={{
              padding: '10px 24px',
              backgroundColor: isSubmitting ? '#9ca3af' : '#2563eb',
              border: 'none',
              borderRadius: '6px',
              color: '#ffffff',
              fontSize: '14px',
              cursor: isSubmitting ? 'not-allowed' : 'pointer',
              fontWeight: '500',
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              if (!isSubmitting) {
                e.currentTarget.style.backgroundColor = '#1d4ed8';
              }
            }}
            onMouseLeave={(e) => {
              if (!isSubmitting) {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }
            }}
          >
            {isSubmitting ? t('labelTasksPage.creating') : isEdit ? t('common.save') : t('common.confirm')}
          </button>
        </div>
      </div>

      {/* 文件夹选择器弹窗 */}
      <FolderPickerModal
        open={showFolderPicker}
        onClose={() => setShowFolderPicker(false)}
        onSelect={handleFolderSelect}
      />

      {/* 数据资产选择器弹窗 */}
      <DatasetMultiSelectModal
        open={showDatasetSelector}
        onClose={() => setShowDatasetSelector(false)}
        onConfirm={(ids) => {
          setSelectedDatasetIds(ids);
          // 自动更新数据数量为所选数据集的数量
          setFormData((prev) => ({
            ...prev,
            dataCount: ids.length > 0 ? String(ids.length) : prev.dataCount,
          }));
          if (errors.datasetDir) {
            setErrors({ ...errors, datasetDir: '' });
          }
          if (errors.dataCount) {
            setErrors({ ...errors, dataCount: '' });
          }
        }}
        initialSelectedIds={selectedDatasetIds}
        syncedOnly
      />
      {copyToast && (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '8px 16px',
            borderRadius: 8,
            backgroundColor: '#111827',
            color: '#ffffff',
            fontSize: 13,
            boxShadow: '0 16px 40px rgba(15,23,42,0.35)',
            zIndex: 1600,
          }}
        >
          路径已复制到剪贴板
        </div>
      )}
    </div>
  );
}

