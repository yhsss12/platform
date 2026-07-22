'use client';

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import * as projectService from '@/lib/projects/projectService';
import type { Project } from '@/lib/projects/types';
import {
  planFolderTreeImport,
  buildAssetPerspectiveSummary,
  jobsToImportAssetRows,
  planFlatFilesImportRows,
  fileKey,
  type ImportAssetRow,
} from '@/components/assets/folderImportPlanner';
import { useTaskCenter } from '@/components/task-center/TaskCenterContext';
import {
  executeDataAssetImportBackground,
  buildImportTaskTitle,
  resolveImportModeSummary,
  formatImportModeSummaryLabel,
} from '@/components/assets/dataAssetImportRunner';

interface ImportDataDialogProps {
  open: boolean;
  onClose: () => void;
  /** 导入成功（含部分成功）后调用；message 为展示用摘要 */
  onSuccess: (message?: string) => void;
}

/** 与 webkitdirectory 一致：相对路径中含 / 视为目录型批次；普通多选文件通常无斜杠 */
function isTreeUploadFile(f: File): boolean {
  const rel = ((f as File & { webkitRelativePath?: string }).webkitRelativePath || '').replace(/\\/g, '/');
  return rel.includes('/');
}

const ENABLE_DIRECT_MULTI = process.env.NEXT_PUBLIC_DIRECT_UPLOAD_MULTI !== '0';

/** 多文件或文件夹：可走 Phase2 直传（受环境变量控制） */
function shouldUseDirectMultiBatch(files: File[]): boolean {
  if (files.length === 0) return false;
  if (files.some(isTreeUploadFile)) return true;
  return files.length >= 2;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function ImportDataDialog({ open, onClose, onSuccess }: ImportDataDialogProps) {
  const { runDataImportJob } = useTaskCenter();
  const [projectId, setProjectId] = useState('');
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [projectList, setProjectList] = useState<Project[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    projectService
      .listAsync(false)
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

  useEffect(() => {
    if (!open) {
      setPendingFiles([]);
      setProjectId('');
      setError(null);
    }
  }, [open]);

  useEffect(() => {
    setError(null);
  }, [pendingFiles, projectId]);

  const mergeFiles = useCallback((incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    if (arr.length === 0) return;

    const incomingTree = arr.some(isTreeUploadFile);
    const incomingUniform = arr.every((f) => isTreeUploadFile(f) === incomingTree);
    if (!incomingUniform) {
      queueMicrotask(() =>
        setError('同一批所选文件结构不一致：请只通过「选择文件夹」选目录，或只通过「选择文件」选多个文件。'),
      );
      return;
    }

    setPendingFiles((prev) => {
      if (prev.length > 0) {
        const prevTree = prev.some(isTreeUploadFile);
        if (prevTree !== incomingTree) {
          queueMicrotask(() =>
            setError(
              '不能混选「多个单文件」与「整个文件夹」。请先点「清空列表」，再只使用一种添加方式。',
            ),
          );
          return prev;
        }
      }
      const seen = new Set(prev.map(fileKey));
      const next = [...prev];
      for (const f of arr) {
        const k = fileKey(f);
        if (!seen.has(k)) {
          seen.add(k);
          next.push(f);
        }
      }
      return next;
    });
  }, []);

  const importAssetsView = useMemo(() => {
    if (!pendingFiles.length) {
      return {
        rows: [] as ImportAssetRow[],
        summaryGreen: null as string | null,
        summaryYellow: null as string | null,
      };
    }
    if (pendingFiles.some(isTreeUploadFile)) {
      const plan = planFolderTreeImport(pendingFiles);
      if (!plan.ok) {
        return { rows: [], summaryGreen: null, summaryYellow: plan.message };
      }
      return {
        rows: jobsToImportAssetRows(plan.jobs),
        summaryGreen: buildAssetPerspectiveSummary(plan),
        summaryYellow: null,
      };
    }
    const flat = planFlatFilesImportRows(pendingFiles);
    if (flat.blocked) {
      return { rows: [], summaryGreen: null, summaryYellow: flat.blockMessage };
    }
    return { rows: flat.rows, summaryGreen: flat.summary, summaryYellow: null };
  }, [pendingFiles]);

  const recognizedBytes = useMemo(
    () => importAssetsView.rows.reduce((s, r) => s + r.sizeBytes, 0),
    [importAssetsView.rows],
  );

  const canConfirm =
    !!projectId.trim() &&
    importAssetsView.rows.length > 0 &&
    !importAssetsView.summaryYellow;

  const handleConfirm = () => {
    if (!canConfirm) return;
    setError(null);
    const filesSnapshot = [...pendingFiles];
    const pid = projectId.trim();
    const pname = projectList.find((p) => p.id === pid)?.name ?? pid;
    const rowsSnapshot = [...importAssetsView.rows];
    const totalUnits = rowsSnapshot.length;
    const treeBatch = filesSnapshot.some(isTreeUploadFile);

    if (ENABLE_DIRECT_MULTI && treeBatch && shouldUseDirectMultiBatch(filesSnapshot)) {
      const plan = planFolderTreeImport(filesSnapshot);
      if (!plan.ok) {
        setError(plan.message);
        return;
      }
    }

    const modeSummary = formatImportModeSummaryLabel(resolveImportModeSummary(filesSnapshot, treeBatch));
    const title = buildImportTaskTitle(rowsSnapshot);
    const namesPreview = rowsSnapshot
      .slice(0, 5)
      .map((r) => r.displayName)
      .join('、');

    runDataImportJob({
      title,
      totalUnits,
      projectId: pid,
      projectName: pname,
      modeSummary,
      assetNamesPreview: namesPreview.length > 120 ? `${namesPreview.slice(0, 120)}…` : namesPreview,
      onImportFinished: onSuccess,
      run: (onProgress, opts) =>
        executeDataAssetImportBackground({
          files: filesSnapshot,
          projectId: pid,
          projectName: pname,
          treeBatch,
          totalUnits,
          onProgress,
          signal: opts?.signal,
          onUploadSessionReady: opts?.onUploadSessionReady,
          onDirectInitSnapshot: opts?.onDirectInitSnapshot,
        }),
    });

    onClose();
  };

  const handleClose = () => {
    setPendingFiles([]);
    setProjectId('');
    setError(null);
    onClose();
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => e.target === e.currentTarget && handleClose()}
    >
      <div
        style={{
          width: '90%',
          maxWidth: 900,
          maxHeight: '90vh',
          background: '#fff',
          borderRadius: 12,
          boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>导入数据</h3>
          <button
            type="button"
            onClick={handleClose}
            style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#6b7280' }}
          >
            ×
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".hdf5,.h5,.mcap,.zip,application/x-hdf5,application/octet-stream"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files?.length) mergeFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files?.length) mergeFiles(e.target.files);
            e.target.value = '';
          }}
        />

        <div style={{ padding: '20px 24px', borderBottom: '1px solid #e5e7eb', backgroundColor: '#f9fafb' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              style={{
                padding: '10px 18px',
                borderRadius: 8,
                border: '1px solid #2563eb',
                background: '#eff6ff',
                color: '#1d4ed8',
                cursor: 'pointer',
                fontSize: 14,
              }}
            >
              选择文件
            </button>
            <button
              type="button"
              onClick={() => folderInputRef.current?.click()}
              style={{
                padding: '10px 18px',
                borderRadius: 8,
                border: '1px solid #059669',
                background: '#ecfdf5',
                color: '#047857',
                cursor: 'pointer',
                fontSize: 14,
              }}
            >
              选择文件夹
            </button>
            {pendingFiles.length > 0 && (
              <button
                type="button"
                onClick={() => setPendingFiles([])}
                style={{
                  padding: '10px 14px',
                  borderRadius: 8,
                  border: '1px solid #d1d5db',
                  background: '#fff',
                  color: '#374151',
                  cursor: 'pointer',
                  fontSize: 13,
                }}
              >
                清空列表
              </button>
            )}
          </div>
        </div>

        <div style={{ padding: '20px 24px 18px', backgroundColor: '#ffffff', flex: 1, overflow: 'auto' }}>
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8, color: '#374151' }}>
              待导入资产（{importAssetsView.rows.length} 条，约 {formatBytes(recognizedBytes)}）
            </div>
            {pendingFiles.length === 0 ? (
              <div
                style={{
                  padding: 12,
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  fontSize: 13,
                  color: '#9ca3af',
                }}
              >
                尚未选择文件
              </div>
            ) : importAssetsView.rows.length === 0 ? (
              <div
                style={{
                  padding: 12,
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  fontSize: 13,
                  color: '#9ca3af',
                }}
              >
                当前选择无法形成可导入资产，请见下方说明或调整文件。
              </div>
            ) : (
              <div
                style={{
                  maxHeight: 220,
                  overflow: 'auto',
                  border: '1px solid #e5e7eb',
                  borderRadius: 8,
                }}
              >
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f9fafb' }}>
                      <th style={{ textAlign: 'left', padding: '6px 10px' }}>资产名称</th>
                      <th style={{ textAlign: 'left', padding: '6px 10px', width: 88 }}>资产类型</th>
                      <th style={{ textAlign: 'right', padding: '6px 10px', width: 90 }}>大小</th>
                      <th style={{ width: 56 }} />
                    </tr>
                  </thead>
                  <tbody>
                    {importAssetsView.rows.map((row) => (
                      <tr key={row.id} style={{ borderTop: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '6px 10px', wordBreak: 'break-all' }}>{row.displayName}</td>
                        <td style={{ padding: '6px 10px', color: '#4b5563' }}>{row.assetType}</td>
                        <td style={{ padding: '6px 10px', textAlign: 'right', color: '#6b7280' }}>
                          {formatBytes(row.sizeBytes)}
                        </td>
                        <td style={{ padding: '4px' }}>
                          <button
                            type="button"
                            onClick={() => {
                              const drop = new Set(row.fileKeysToRemove);
                              setPendingFiles((prev) => prev.filter((x) => !drop.has(fileKey(x))));
                            }}
                            style={{
                              border: 'none',
                              background: 'transparent',
                              color: '#dc2626',
                              cursor: 'pointer',
                              fontSize: 12,
                            }}
                          >
                            移除
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {importAssetsView.summaryGreen ? (
            <div
              style={{
                marginBottom: 12,
                padding: '10px 12px',
                borderRadius: 8,
                background: '#ecfdf5',
                border: '1px solid #a7f3d0',
                fontSize: 13,
                color: '#065f46',
                lineHeight: 1.5,
              }}
            >
              {importAssetsView.summaryGreen}
            </div>
          ) : null}
          {importAssetsView.summaryYellow ? (
            <div
              style={{
                marginBottom: 12,
                padding: '10px 12px',
                borderRadius: 8,
                background: '#fffbeb',
                border: '1px solid #fde68a',
                fontSize: 13,
                color: '#92400e',
                lineHeight: 1.5,
              }}
            >
              {importAssetsView.summaryYellow}
            </div>
          ) : null}

          <div style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 6 }}>
              所属项目 <span style={{ color: '#dc2626' }}>*</span>
            </label>
            <select
              value={projectId}
              onChange={(e) => setProjectId(e.target.value)}
              style={{
                width: '100%',
                maxWidth: 320,
                padding: '8px 12px',
                border: '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
              }}
            >
              <option value="">请选择所属项目</option>
              {projectList.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {error ? (
            <div style={{ marginBottom: 12, fontSize: 13, color: '#dc2626', lineHeight: 1.45 }}>{error}</div>
          ) : null}
        </div>

        <div
          style={{
            padding: 16,
            borderTop: '1px solid #e5e7eb',
            backgroundColor: '#f9fafb',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 12,
          }}
        >
          <button
            type="button"
            onClick={handleClose}
            style={{
              padding: '10px 20px',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              background: '#fff',
              cursor: 'pointer',
              fontSize: 14,
            }}
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={!canConfirm}
            style={{
              padding: '10px 20px',
              border: 'none',
              borderRadius: 6,
              background: canConfirm ? '#2563eb' : '#9ca3af',
              color: '#fff',
              cursor: canConfirm ? 'pointer' : 'not-allowed',
              fontSize: 14,
            }}
          >
            确认导入
          </button>
        </div>
      </div>
    </div>
  );
}
