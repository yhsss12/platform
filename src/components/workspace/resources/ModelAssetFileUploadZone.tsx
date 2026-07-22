'use client';

import { useId, useRef, type ChangeEvent, type MouseEvent } from 'react';
import { Upload } from 'lucide-react';

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ModelAssetFileUploadZone({
  accept,
  emptyTitle,
  emptySubtitle,
  file,
  onFileChange,
  onInvalidFile,
}: {
  accept: string;
  emptyTitle: string;
  emptySubtitle: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  onInvalidFile?: (message: string) => void;
}) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptList = accept
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);

  const validateExtension = (name: string): boolean => {
    const ext = name.includes('.') ? `.${name.split('.').pop()?.toLowerCase()}` : '';
    return acceptList.some((item) => item === ext || item.endsWith(ext));
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = event.target.files?.[0] ?? null;
    if (!next) {
      onFileChange(null);
      return;
    }
    if (!validateExtension(next.name)) {
      onInvalidFile?.(`文件格式不支持，请选择 ${acceptList.join(' / ')}`);
      event.target.value = '';
      return;
    }
    onFileChange(next);
  };

  const clearFile = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    onFileChange(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={handleChange}
      />
      <label
        htmlFor={inputId}
        className={`ws-file-upload${file ? ' ws-file-upload-has-file' : ''}`}
      >
        <span className="ws-file-upload-icon" aria-hidden>
          <Upload size={18} strokeWidth={2} />
        </span>
        <span className="ws-file-upload-text">
          <div className="ws-file-upload-title">{file ? file.name : emptyTitle}</div>
          <div className="ws-file-upload-subtitle">
            {file ? formatFileSize(file.size) : emptySubtitle}
          </div>
        </span>
        {file ? (
          <button type="button" className="ws-file-upload-clear" onClick={clearFile}>
            清除
          </button>
        ) : null}
      </label>
    </div>
  );
}
