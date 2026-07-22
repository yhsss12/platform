'use client';

import React, { useEffect, useState } from 'react';
import type { CreateProjectInput } from '@/lib/projects/createProject';
import { useI18n } from '@/components/common/I18nProvider';

const MAX_TAGS = 4;

export default function CreateProjectModal({
  open,
  onClose,
  onSubmit,
  initialValues,
  title,
  confirmText,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (input: CreateProjectInput) => void;
  initialValues?: {
    name?: string;
    description?: string;
    tags?: string[];
  };
  title?: string;
  confirmText?: string;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [nameError, setNameError] = useState('');
  const { t } = useI18n();

  useEffect(() => {
    if (open) {
      setName(initialValues?.name ?? '');
      setDescription(initialValues?.description ?? '');
      setTags(Array.isArray(initialValues?.tags) ? initialValues!.tags!.slice(0, MAX_TAGS) : []);
      setTagInput('');
      setNameError('');
    }
  }, [open, initialValues]);

  const addTag = () => {
    const t = tagInput.trim();
    if (!t) return;
    if (tags.length >= MAX_TAGS) return;
    if (tags.includes(t)) return;
    setTags((prev) => [...prev, t]);
    setTagInput('');
  };

  const removeTag = (tag: string) => {
    setTags((prev) => prev.filter((x) => x !== tag));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const n = name.trim();
    if (!n) {
      setNameError(t('adminProjectsPage.nameRequired'));
      return;
    }
    setNameError('');
    onSubmit({
      name: n,
      description: description.trim() || undefined,
      tags: tags.length ? tags : undefined,
      ownerId: '', // 由调用方注入
      ownerName: '',
    });
    onClose();
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15,23,42,0.45)',
        zIndex: 1600,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '520px',
          maxWidth: '96vw',
          backgroundColor: '#ffffff',
          borderRadius: '12px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '16px 18px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ fontSize: '16px', fontWeight: 800, color: '#111827' }}>
            {title || t('adminProjectsPage.createTitle')}
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              width: '34px',
              height: '34px',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              backgroundColor: '#ffffff',
              cursor: 'pointer',
              color: '#374151',
              fontSize: '18px',
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ padding: '18px' }}>
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: '6px' }}>
              {t('adminProjectsPage.projectNameLabel')} <span style={{ color: '#dc2626' }}>*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (nameError) setNameError('');
              }}
              placeholder={t('adminProjectsPage.projectNamePlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                border: nameError ? '1px solid #dc2626' : '1px solid #d1d5db',
                borderRadius: '10px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {nameError && <div style={{ fontSize: '12px', color: '#dc2626', marginTop: '4px' }}>{nameError}</div>}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: '6px' }}>
              {t('adminProjectsPage.descriptionLabel')}
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder={t('adminProjectsPage.descriptionPlaceholder')}
              rows={3}
              style={{
                width: '100%',
                padding: '10px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '10px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                resize: 'vertical',
              }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, color: '#374151', marginBottom: '6px' }}>
              {t('adminProjectsPage.tagsLabelOptional', { n: MAX_TAGS })}
            </label>
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addTag();
                }
              }}
              placeholder={t('adminProjectsPage.tagPlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                border: '1px solid #d1d5db',
                borderRadius: '10px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {tags.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
                {tags.map((t) => (
                  <span
                    key={t}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '4px 10px',
                      backgroundColor: '#f3f4f6',
                      borderRadius: '999px',
                      fontSize: '12px',
                      color: '#374151',
                      border: '1px solid #e5e7eb',
                    }}
                  >
                    {t}
                    <button
                      type="button"
                      onClick={() => removeTag(t)}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: '#6b7280',
                        padding: 0,
                        lineHeight: 1,
                        fontSize: '14px',
                      }}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                height: '38px',
                padding: '0 14px',
                borderRadius: '10px',
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                color: '#374151',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              {t('adminProjectsPage.cancel')}
            </button>
            <button
              type="submit"
              style={{
                height: '38px',
                padding: '0 14px',
                borderRadius: '10px',
                border: 'none',
                backgroundColor: '#2563eb',
                color: '#ffffff',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 600,
              }}
            >
              {confirmText || t('adminProjectsPage.confirmCreate')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
