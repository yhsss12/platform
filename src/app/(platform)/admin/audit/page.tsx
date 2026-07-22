'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiGet } from '@/lib/api/authClient';
import { useI18n } from '@/components/common/I18nProvider';
import { useAuthStore } from '@/store/authStore';
import type { Role } from '@/lib/api/types';
import { isSuperAdmin, isTeamAdminAccount, normalizeRole } from '@/lib/api/roleLabels';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageFilterCard,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import { AuditDateTimeFilterInput } from '@/components/admin/AuditDateTimeFilterInput';
import * as projectService from '@/lib/projects/projectService';
import type { Project } from '@/lib/projects/types';

export interface AuditLogRow {
  /** 旧库为 UUID 字符串，新库为数字字符串 */
  id: string;
  created_at: string;
  user_id: string | null;
  username: string | null;
  role: string | null;
  project_id: string | null;
  project_name: string | null;
  team_id: string | null;
  team_name: string | null;
  action_type: string;
  action_label: string;
  resource_type: string | null;
  resource_id: string | null;
  resource_name: string | null;
  result: string;
  ip: string | null;
  user_agent: string | null;
  detail_json: Record<string, unknown> | null;
  error_message: string | null;
}

interface AuditListResponse {
  items: AuditLogRow[];
  total: number;
}

interface ActionItem {
  code: string;
  label: string;
}

/** 与后端 audit_resources 及历史小写值兼容 */
const RESOURCE_TYPE_ZH: Record<string, string> = {
  USER: '用户',
  PROJECT: '项目',
  TEAM: '团队',
  PROJECT_MEMBER: '项目成员',
  DATA_ASSET: '数据资产',
  TASK: '任务',
  COLLECTION_JOB: '采集作业',
  LABEL_JOB: '标注任务',
  CONVERT_JOB: '转换任务',
  SESSION: '会话',
  user: '用户',
  project: '项目',
  team: '团队',
  project_member: '项目成员',
  data_asset: '数据资产',
  collection_task: '采集任务',
  collection_job: '采集作业',
  label_task: '标注任务',
  conversion_job: '转换任务',
  session: '会话',
};

function formatActionDisplay(log: AuditLogRow): string {
  const label = (log.action_label || '').trim();
  if (label) return label;
  return log.action_type || '—';
}

function formatResourceTypeDisplay(rt: string | null, locale: string): string {
  if (!rt) return '—';
  if (locale === 'zh-CN') return RESOURCE_TYPE_ZH[rt] || rt;
  return rt;
}

function formatJson(v: unknown): string {
  if (v == null) return '';
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function displayStr(v: string | null | undefined): string {
  const s = (v ?? '').trim();
  return s.length ? s : '—';
}

/** 列表「所属项目」列：有名优先名，否则 id，否则 — */
function formatAuditProjectCell(log: AuditLogRow): string {
  const name = (log.project_name ?? '').trim();
  if (name) return name;
  const id = (log.project_id ?? '').trim();
  if (id) return id;
  return '—';
}

/** 列表「所属团队」：无 team_id 的历史日志显示 —；有 id 时优先名称 */
function formatAuditTeamCell(log: AuditLogRow): string {
  const tid = (log.team_id ?? '').trim();
  if (!tid) return '—';
  const name = (log.team_name ?? '').trim();
  return name.length ? name : tid;
}

/** 列表/详情：24 小时制 `YYYY-MM-DD HH:mm:ss`（本地时区，与筛选条风格一致） */
function formatAuditLogDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${y}-${mo}-${day} ${h}:${mi}:${s}`;
}

/** 仅放在「高级信息」展示，不在业务详情复述 */
const DETAIL_ADVANCED_ONLY_KEYS = new Set(['jti', 'legacy_id']);

const DETAIL_FIELD_LABELS_ZH: Record<string, string> = {
  new_username: '新建用户名',
  new_role: '新角色',
  old_role: '原角色',
  target_user: '目标用户',
  target_username: '目标用户名',
  deleted_username: '被删除用户名',
  member_username: '成员用户名',
  member_user_id: '成员用户 ID',
  operation: '操作类型',
  is_active: '账号启用',
  imported_count: '成功导入数量',
  failed_count: '失败数量',
  imported: '导入条目（摘要）',
  failed: '失败条目',
  requested_count: '请求删除数量',
  deleted_count: '已删除数量',
  deleted_asset_ids: '已删除资产 ID',
  deleted_filenames: '已删除文件名',
  errors: '错误信息',
  asset_ids: '资产 ID 列表',
  count: '数量',
  job_id: '任务 / 导出 ID',
  output_path: '输出目录',
  task_id: '任务 ID',
  job_number: '作业编号',
  control: '控制操作',
  domain: '领域',
  dataset_source: '数据集来源',
  delete_file: '同时删除本地文件',
  path: '路径',
  type: '类型',
  minio_registered: '已登记 MinIO',
  row_count: '行数',
  format: '格式',
  outputFormat: '输出格式',
  assetId: '资产 ID',
  inputDatasetId: '输入数据集 ID',
  legacy_detail: '历史明细',
};

const DETAIL_FIELD_LABELS_EN: Record<string, string> = {
  new_username: 'New username',
  new_role: 'New role',
  old_role: 'Previous role',
  target_user: 'Target user',
  target_username: 'Target username',
  deleted_username: 'Deleted username',
  member_username: 'Member username',
  member_user_id: 'Member user ID',
  operation: 'Operation',
  is_active: 'Account active',
  imported_count: 'Imported count',
  failed_count: 'Failed count',
  imported: 'Imported items (summary)',
  failed: 'Failed items',
  requested_count: 'Requested delete count',
  deleted_count: 'Deleted count',
  deleted_asset_ids: 'Deleted asset IDs',
  deleted_filenames: 'Deleted filenames',
  errors: 'Errors',
  asset_ids: 'Asset IDs',
  count: 'Count',
  job_id: 'Job / export ID',
  output_path: 'Output path',
  task_id: 'Task ID',
  job_number: 'Job number',
  control: 'Control',
  domain: 'Domain',
  dataset_source: 'Dataset source',
  delete_file: 'Delete local file too',
  path: 'Path',
  type: 'Type',
  minio_registered: 'MinIO registered',
  row_count: 'Row count',
  format: 'Format',
  outputFormat: 'Output format',
  assetId: 'Asset ID',
  inputDatasetId: 'Input dataset ID',
  legacy_detail: 'Legacy detail text',
};

const DETAIL_KEY_PRIORITY: string[] = [
  'new_username',
  'new_role',
  'old_role',
  'deleted_username',
  'target_username',
  'operation',
  'is_active',
  'member_username',
  'member_user_id',
  'imported_count',
  'failed_count',
  'imported',
  'failed',
  'requested_count',
  'deleted_count',
  'deleted_asset_ids',
  'deleted_filenames',
  'errors',
  'asset_ids',
  'count',
  'job_id',
  'output_path',
  'task_id',
  'job_number',
  'control',
  'domain',
  'dataset_source',
  'delete_file',
  'path',
  'type',
  'minio_registered',
  'row_count',
  'format',
  'outputFormat',
  'assetId',
  'inputDatasetId',
  'legacy_detail',
];

function detailFieldLabel(key: string, locale: string): string {
  const m = locale === 'zh-CN' ? DETAIL_FIELD_LABELS_ZH : DETAIL_FIELD_LABELS_EN;
  return m[key] || key;
}

/** 旧库：动作码不像 UPPER_SNAKE 常量，视为历史长句类型 */
function isLegacyAuditLog(log: AuditLogRow): boolean {
  const at = (log.action_type || '').trim();
  if (!at) return true;
  return !/^[A-Z][A-Z0-9_]*$/.test(at);
}

type AuditT = (path: string, vars?: Record<string, string | number>) => string;

const ROLE_DETAIL_KEYS = new Set(['new_role', 'old_role', 'role']);

function auditRoleI18nKey(
  role: Role,
):
  | 'adminAuditPage.auditRoleSuperAdmin'
  | 'adminAuditPage.auditRoleAdmin'
  | 'adminAuditPage.auditRoleOwner'
  | 'adminAuditPage.auditRoleUser' {
  switch (role) {
    case 'SUPER_ADMIN':
      return 'adminAuditPage.auditRoleSuperAdmin';
    case 'ADMIN':
      return 'adminAuditPage.auditRoleAdmin';
    case 'OWNER':
      return 'adminAuditPage.auditRoleOwner';
    default:
      return 'adminAuditPage.auditRoleUser';
  }
}

/** 审计列表/详情：四层角色统一展示（含 ADMINISTRATOR/MEMBER 等历史码） */
function formatAuditRoleDisplay(raw: string | null | undefined, t: AuditT): string {
  const s = raw == null ? '' : String(raw).trim();
  if (!s) return '—';
  return t(auditRoleI18nKey(normalizeRole(s)));
}

function formatDetailValue(key: string, value: unknown, locale: string, t: AuditT): string {
  if (value === null || value === undefined) return '—';
  if (ROLE_DETAIL_KEYS.has(key) && typeof value === 'string') {
    return formatAuditRoleDisplay(value, t);
  }
  if (key === 'operation' && typeof value === 'string') {
    const v = value.toLowerCase();
    if (locale === 'zh-CN') {
      if (v === 'disable') return '禁用';
      if (v === 'enable') return '启用';
    }
    return value;
  }
  if (key === 'is_active') {
    if (typeof value === 'boolean') return locale === 'zh-CN' ? (value ? '是' : '否') : value ? 'Yes' : 'No';
  }
  if (key === 'control' && typeof value === 'string') {
    if (locale === 'zh-CN') {
      if (value === 'pause') return '暂停';
      if (value === 'resume') return '恢复';
    }
    return value;
  }
  if (key === 'delete_file' || key === 'minio_registered') {
    if (typeof value === 'boolean') return locale === 'zh-CN' ? (value ? '是' : '否') : value ? 'Yes' : 'No';
  }
  if (Array.isArray(value)) {
    const max = key === 'imported' || key === 'failed' ? 8 : 24;
    const parts = value.slice(0, max).map((x) => {
      if (x !== null && typeof x === 'object') {
        try {
          return JSON.stringify(x);
        } catch {
          return String(x);
        }
      }
      return String(x);
    });
    let s = parts.join(locale === 'zh-CN' ? '，' : ', ');
    if (value.length > max) {
      s +=
        locale === 'zh-CN'
          ? ` …（共 ${value.length} 项，完整列表见下方 JSON）`
          : ` … (${value.length} items total; see JSON below)`;
    }
    return s || '—';
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function buildBusinessRows(
  log: AuditLogRow,
  locale: string,
  t: AuditT
): { rows: { label: string; value: string }[]; mode: 'normal' | 'legacy' | 'empty' } {
  if (isLegacyAuditLog(log)) {
    return { rows: [], mode: 'legacy' };
  }
  const dj =
    log.detail_json && typeof log.detail_json === 'object' && !Array.isArray(log.detail_json)
      ? (log.detail_json as Record<string, unknown>)
      : null;
  const rows: { label: string; value: string }[] = [];
  const pn = displayStr(log.project_name);
  const pid = displayStr(log.project_id);
  if (pn !== '—' || pid !== '—') {
    rows.push({
      label: t('adminAuditPage.detailSummaryProject'),
      value: pn !== '—' ? (pid !== '—' && pn !== pid ? `${pn}（${pid}）` : pn) : pid,
    });
  }
  const tname = (log.team_name ?? '').trim();
  const tid = (log.team_id ?? '').trim();
  rows.push({
    label: t('adminAuditPage.detailSummaryTeam'),
    value: tid ? (tname || tid) : '—',
  });
  const fail = (log.result || '').toUpperCase() === 'FAIL';
  const err = (log.error_message || '').trim();
  if (fail && err) {
    rows.push({ label: t('adminAuditPage.detailErrorLabel'), value: err });
  }
  if (!dj) {
    return { rows, mode: rows.length ? 'normal' : 'empty' };
  }
  const keys = Object.keys(dj).filter((k) => !DETAIL_ADVANCED_ONLY_KEYS.has(k));
  const ordered: string[] = [];
  for (const k of DETAIL_KEY_PRIORITY) {
    if (keys.includes(k)) ordered.push(k);
  }
  for (const k of keys.sort()) {
    if (!ordered.includes(k)) ordered.push(k);
  }
  for (const k of ordered) {
    const val = dj[k];
    if (val === undefined) continue;
    rows.push({
      label: detailFieldLabel(k, locale),
      value: formatDetailValue(k, val, locale, t),
    });
  }
  return { rows, mode: rows.length ? 'normal' : 'empty' };
}

/** 与数据资产页 `FiltersBar` 输入/下拉一致（单行筛选视觉对齐） */
const dataPageFilterInput: React.CSSProperties = {
  padding: '8px 12px',
  backgroundColor: '#ffffff',
  border: '1px solid #d1d5db',
  borderRadius: '6px',
  color: '#111827',
  fontSize: '14px',
  outline: 'none',
  boxSizing: 'border-box',
};

/** 审计筛选栏：时间控件略收窄，仍够展示 `YYYY-MM-DD HH:mm` 与日历按钮 */
const AUDIT_FILTER_DATETIME_WIDTH = 156;

/** 团队 / 项目 / 动作 / 结果 下拉统一固定宽，避免长短不一 */
const AUDIT_FILTER_SELECT_WIDTH = 124;

const auditFilterSelectStyle: React.CSSProperties = {
  ...dataPageFilterInput,
  width: AUDIT_FILTER_SELECT_WIDTH,
  minWidth: AUDIT_FILTER_SELECT_WIDTH,
  maxWidth: AUDIT_FILTER_SELECT_WIDTH,
  flex: `0 0 ${AUDIT_FILTER_SELECT_WIDTH}px`,
  cursor: 'pointer',
};

/**
 * 关键词搜索：禁止 flex-grow 占满左侧（原 `flex: 1 1 240px` 会挤占右侧按钮区域）。
 * 使用 max-width 封顶，窄屏下可随父级收缩。
 */
const auditFilterKeywordWrap: React.CSSProperties = {
  position: 'relative',
  flex: '0 1 200px',
  width: 200,
  minWidth: 152,
  maxWidth: 220,
};

function sectionTitleStyle(): React.CSSProperties {
  return {
    fontSize: 12,
    fontWeight: 600,
    color: '#6b7280',
    marginBottom: 8,
  };
}

function AuditDetailModal({
  log,
  onClose,
  locale,
  t,
}: {
  log: AuditLogRow;
  onClose: () => void;
  locale: string;
  t: AuditT;
}) {
  const [advancedOpen, setAdvancedOpen] = React.useState(false);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const dj =
    log.detail_json && typeof log.detail_json === 'object' && !Array.isArray(log.detail_json)
      ? (log.detail_json as Record<string, unknown>)
      : null;
  const jsonText = formatJson(log.detail_json);
  const hasJson = (() => {
    const d = log.detail_json;
    if (d == null) return false;
    if (Array.isArray(d)) return d.length > 0 && jsonText.trim().length > 0;
    if (typeof d === 'object') return Object.keys(d as Record<string, unknown>).length > 0 && jsonText.trim().length > 0;
    return jsonText.trim().length > 0;
  })();
  const jti = dj && typeof dj.jti === 'string' ? dj.jti : null;
  const hasUserAgent = Boolean((log.user_agent || '').trim());
  const { rows: businessRows, mode: bizMode } = buildBusinessRows(log, locale, t);
  const resourceLine = (() => {
    const rt = formatResourceTypeDisplay(log.resource_type, locale);
    const rn = displayStr(log.resource_name);
    if (rt === '—' && rn === '—') return '—';
    if (rt === '—') return rn;
    if (rn === '—') return rt;
    return `${rt} · ${rn}`;
  })();
  const operatorLine = displayStr(log.username);
  const roleSummaryLine = formatAuditRoleDisplay(log.role, t);

  return (
    <div
      role="presentation"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.45)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
      }}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal
        aria-labelledby="audit-detail-title"
        style={{
          width: 'min(680px, 100%)',
          maxHeight: 'min(88vh, 900px)',
          overflow: 'auto',
          background: '#fff',
          borderRadius: 12,
          boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)',
          border: '1px solid #e5e7eb',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            position: 'sticky',
            top: 0,
            background: '#fff',
            zIndex: 1,
          }}
        >
          <h2 id="audit-detail-title" style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827' }}>
            {t('adminAuditPage.detailTitle')}
          </h2>
          <button
            type="button"
            onClick={onClose}
            style={{
              border: 'none',
              background: '#f3f4f6',
              width: 36,
              height: 36,
              borderRadius: 8,
              fontSize: 20,
              lineHeight: 1,
              cursor: 'pointer',
              color: '#4b5563',
            }}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div style={{ padding: '16px 20px 20px' }}>
          <div style={sectionTitleStyle()}>{t('adminAuditPage.detailSectionSummary')}</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
              gap: '10px 20px',
              padding: 12,
              background: '#f9fafb',
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              marginBottom: 18,
            }}
          >
            <div style={{ gridColumn: '1 / -1' }}>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryAction')}</div>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#111827' }}>{formatActionDisplay(log)}</div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryOperator')}</div>
              <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{operatorLine}</div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryRole')}</div>
              <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{roleSummaryLine}</div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryTime')}</div>
              <div style={{ fontSize: 14, color: '#111827' }}>{formatAuditLogDateTime(log.created_at)}</div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryResource')}</div>
              <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{resourceLine}</div>
            </div>
            <div style={{ gridColumn: '1 / -1' }}>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{t('adminAuditPage.detailSummaryResult')}</div>
              <div>
                <ResultCell result={log.result} t={t} />
              </div>
            </div>
          </div>

          <div style={sectionTitleStyle()}>{t('adminAuditPage.detailSectionBusiness')}</div>
          <div
            style={{
              padding: '12px 14px',
              background: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: 8,
              marginBottom: 16,
              minHeight: 48,
            }}
          >
            {bizMode === 'legacy' ? (
              <div style={{ fontSize: 14, color: '#374151', lineHeight: 1.6 }}>
                <div>
                  <span style={{ color: '#6b7280' }}>{t('adminAuditPage.detailLegacyActionLabel')}：</span>
                  {formatActionDisplay(log)}
                  {log.action_type && log.action_type !== log.action_label ? (
                    <span style={{ color: '#9ca3af', fontSize: 13 }}>（{log.action_type}）</span>
                  ) : null}
                </div>
                {(log.result || '').toUpperCase() === 'FAIL' && (log.error_message || '').trim() ? (
                  <div style={{ marginTop: 10, color: '#b91c1c', fontSize: 14 }}>
                    <span style={{ color: '#6b7280' }}>{t('adminAuditPage.detailErrorLabel')}：</span>
                    {(log.error_message || '').trim()}
                  </div>
                ) : null}
                <div style={{ marginTop: 10, fontSize: 13, color: '#6b7280' }}>{t('adminAuditPage.detailLegacyHint')}</div>
              </div>
            ) : bizMode === 'empty' ? (
              <div style={{ fontSize: 14, color: '#6b7280' }}>{t('adminAuditPage.detailNoBusinessExtra')}</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(100px, 34%) 1fr', gap: '8px 12px', fontSize: 14 }}>
                {businessRows.map((row, i) => (
                  <React.Fragment key={`${row.label}-${i}`}>
                    <div style={{ color: '#6b7280', fontWeight: 500 }}>{row.label}</div>
                    <div style={{ color: '#111827', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{row.value}</div>
                  </React.Fragment>
                ))}
              </div>
            )}
          </div>

          {hasJson ? (
            <>
              <div style={sectionTitleStyle()}>{t('adminAuditPage.detailJsonLabel')}</div>
              <pre
                style={{
                  margin: '0 0 16px',
                  padding: 12,
                  fontSize: 12,
                  lineHeight: 1.45,
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                  color: '#374151',
                  background: '#f3f4f6',
                  border: '1px solid #e5e7eb',
                  borderRadius: 8,
                  overflow: 'auto',
                  maxHeight: 220,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {jsonText}
              </pre>
            </>
          ) : null}

          <button
            type="button"
            onClick={() => setAdvancedOpen((o) => !o)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              width: '100%',
              padding: '10px 12px',
              marginBottom: 8,
              background: '#f9fafb',
              border: '1px solid #e5e7eb',
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 500,
              color: '#374151',
            }}
          >
            <span style={{ transform: advancedOpen ? 'rotate(90deg)' : 'rotate(0deg)', display: 'inline-block' }}>▸</span>
            {t('adminAuditPage.detailSectionAdvanced')}
            <span style={{ marginLeft: 'auto', fontWeight: 400, color: '#6b7280' }}>
              {advancedOpen ? t('adminAuditPage.detailAdvancedToggleCollapse') : t('adminAuditPage.detailAdvancedToggleExpand')}
            </span>
          </button>
          {advancedOpen ? (
            <div
              style={{
                padding: '12px 14px',
                background: '#fafafa',
                border: '1px solid #e5e7eb',
                borderRadius: 8,
                fontSize: 13,
                color: '#374151',
                display: 'grid',
                gridTemplateColumns: 'minmax(88px, 32%) 1fr',
                gap: '8px 10px',
              }}
            >
              <div style={{ color: '#6b7280' }}>{t('adminAuditPage.tableIp')}</div>
              <div style={{ wordBreak: 'break-all' }}>{displayStr(log.ip)}</div>
              <div style={{ color: '#6b7280' }}>{t('adminAuditPage.detailActionCode')}</div>
              <div style={{ wordBreak: 'break-all' }}>{displayStr(log.action_type)}</div>
              <div style={{ color: '#6b7280' }}>{t('adminAuditPage.detailAdvancedResourceTypeRaw')}</div>
              <div style={{ wordBreak: 'break-all' }}>{displayStr(log.resource_type)}</div>
              <div style={{ color: '#6b7280' }}>{t('adminAuditPage.detailAdvancedResourceIdLabel')}</div>
              <div style={{ wordBreak: 'break-all' }}>{displayStr(log.resource_id)}</div>
              {jti ? (
                <>
                  <div style={{ color: '#6b7280' }}>{t('adminAuditPage.detailAdvancedJti')}</div>
                  <div style={{ wordBreak: 'break-all' }}>{jti}</div>
                </>
              ) : null}
              {hasUserAgent ? (
                <>
                  <div
                    style={{
                      gridColumn: '1 / -1',
                      height: 1,
                      background: '#e5e7eb',
                      margin: '4px 0',
                    }}
                  />
                  <div style={{ color: '#9ca3af', fontSize: 12 }}>{t('adminAuditPage.detailUserAgentLabel')}</div>
                  <div
                    style={{
                      wordBreak: 'break-all',
                      fontSize: 12,
                      color: '#6b7280',
                      lineHeight: 1.45,
                    }}
                  >
                    {(log.user_agent || '').trim()}
                  </div>
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ResultCell({
  result,
  t,
}: {
  result: string;
  t: (path: string, vars?: Record<string, string | number>) => string;
}) {
  const ok = (result || '').toUpperCase() === 'SUCCESS';
  const label = ok ? t('adminAuditPage.resultSuccess') : t('adminAuditPage.resultFail');
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 12,
        fontWeight: 500,
        backgroundColor: ok ? '#dcfce7' : '#fee2e2',
        color: ok ? '#166534' : '#991b1b',
      }}
    >
      {label}
    </span>
  );
}

export default function AuditPage() {
  const router = useRouter();
  const { t, locale } = useI18n();
  const user = useAuthStore((s) => s.user);
  const isHydrated = useAuthStore((s) => s.isHydrated);

  const [items, setItems] = useState<AuditLogRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [detailRow, setDetailRow] = useState<AuditLogRow | null>(null);
  const [actionItems, setActionItems] = useState<ActionItem[]>([]);

  const [createdFrom, setCreatedFrom] = useState('');
  const [createdTo, setCreatedTo] = useState('');
  const [filterTeamId, setFilterTeamId] = useState('');
  const [filterProjectId, setFilterProjectId] = useState('');
  const [filterAction, setFilterAction] = useState('');
  const [filterResult, setFilterResult] = useState('');
  const [filterQ, setFilterQ] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 50;
  const [projectOptions, setProjectOptions] = useState<Project[]>([]);
  const [teamOptions, setTeamOptions] = useState<{ id: string; name: string }[]>([]);

  const canAccess = Boolean(
    isHydrated && user && (isSuperAdmin(user.role) || isTeamAdminAccount(user.role)),
  );

  /** 团队管理员账号：与超管共用本页 UI，数据范围由后端 scope 限制；前端隐藏「所属团队」筛选且不传 team_id */
  const isTeamAdminViewer = Boolean(
    user && isTeamAdminAccount(user.role) && !isSuperAdmin(user.role),
  );

  useEffect(() => {
    if (!isHydrated) return;
    if (!user) {
      router.replace('/login');
      return;
    }
    if (!isSuperAdmin(user.role) && !isTeamAdminAccount(user.role)) {
      router.replace('/forbidden');
    }
  }, [isHydrated, user, router]);

  const loadMeta = useCallback(async () => {
    if (!canAccess) return;
    try {
      const m = await apiGet<{ action_items?: ActionItem[] }>('/audit/meta');
      if (m?.action_items?.length) setActionItems(m.action_items);
    } catch {
      setActionItems([]);
    }
  }, [canAccess]);

  const loadAuditLogs = useCallback(async () => {
    if (!canAccess) return;
    try {
      setLoading(true);
      setError('');
      const q = new URLSearchParams();
      if (createdFrom) q.set('created_from', new Date(createdFrom).toISOString());
      if (createdTo) q.set('created_to', new Date(createdTo).toISOString());
      if (!isTeamAdminViewer && filterTeamId.trim()) q.set('team_id', filterTeamId.trim());
      if (filterProjectId.trim()) q.set('project_id', filterProjectId.trim());
      if (filterAction.trim()) q.set('action_type', filterAction.trim());
      if (filterResult.trim()) q.set('result', filterResult.trim().toUpperCase());
      if (filterQ.trim()) q.set('q', filterQ.trim());
      q.set('limit', String(pageSize));
      q.set('offset', String(page * pageSize));
      const qs = q.toString();
      const data = await apiGet<AuditListResponse>(`/audit?${qs}`);
      setItems(data?.items ?? []);
      setTotal(data?.total ?? 0);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('403') || msg.includes('Forbidden')) {
        router.replace('/forbidden');
        return;
      }
      setError(msg || t('adminAuditPage.loadFailed'));
    } finally {
      setLoading(false);
    }
  }, [
    canAccess,
    createdFrom,
    createdTo,
    filterTeamId,
    filterProjectId,
    filterAction,
    filterResult,
    filterQ,
    page,
    router,
    t,
    isTeamAdminViewer,
  ]);

  useEffect(() => {
    if (!canAccess) return;
    loadMeta();
  }, [canAccess, loadMeta]);

  useEffect(() => {
    if (!canAccess) return;
    let cancelled = false;
    projectService
      .listAsync(false)
      .then((res) => {
        if (cancelled) return;
        const projects = Array.isArray(res) ? res : [];
        setProjectOptions(projects.filter((p) => p.status !== '已归档'));
      })
      .catch(() => {
        if (!cancelled) setProjectOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [canAccess]);

  useEffect(() => {
    if (!canAccess || isTeamAdminViewer) {
      setTeamOptions([]);
      return;
    }
    let cancelled = false;
    apiGet<{ items?: { id: string; name: string }[] }>('/audit/team-filter-options')
      .then((res) => {
        if (cancelled) return;
        setTeamOptions(Array.isArray(res?.items) ? res.items : []);
      })
      .catch(() => {
        if (!cancelled) setTeamOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [canAccess, isTeamAdminViewer]);

  useEffect(() => {
    if (!canAccess) return;
    loadAuditLogs();
  }, [canAccess, loadAuditLogs]);

  const resetFilters = () => {
    setCreatedFrom('');
    setCreatedTo('');
    setFilterTeamId('');
    setFilterProjectId('');
    setFilterAction('');
    setFilterResult('');
    setFilterQ('');
    setPage(0);
  };

  if (!isHydrated || !user) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ color: '#6b7280' }}>{t('adminAuditPage.accessRedirecting')}</div>
      </div>
    );
  }

  if (!isSuperAdmin(user.role) && !isTeamAdminAccount(user.role)) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ color: '#6b7280' }}>{t('adminAuditPage.accessRedirecting')}</div>
      </div>
    );
  }

  if (loading && items.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ color: '#6b7280' }}>{t('common.loading')}</div>
      </div>
    );
  }

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <ModulePageContainer>
      <ModulePageHeader title={t('adminAuditPage.title')} />

      <ModulePageFilterCard>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            flexWrap: 'wrap',
            rowGap: 12,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              flex: '1 1 0',
              minWidth: 0,
              flexWrap: 'wrap',
            }}
          >
            <AuditDateTimeFilterInput
              value={createdFrom}
              placeholder={t('adminAuditPage.filterTimePlaceholderStart')}
              ariaLabel={t('adminAuditPage.filterTimeFrom')}
              pickAriaLabel={t('adminAuditPage.filterPickDateTime')}
              confirmLabel={t('adminAuditPage.filterTimeModalConfirm')}
              clearLabel={t('adminAuditPage.filterTimeModalClear')}
              dateFieldLabel={t('adminAuditPage.filterTimeModalDateLabel')}
              timeFieldLabel={t('adminAuditPage.filterTimeModalTimeLabel')}
              pickDatePlaceholder={t('adminAuditPage.filterTimeModalPickDatePh')}
              timeInputPlaceholder={t('adminAuditPage.filterTimeModalTimeInputPh')}
              timeInvalidHint={t('adminAuditPage.filterTimeModalTimeInvalid')}
              timeRangeEnd="start"
              width={AUDIT_FILTER_DATETIME_WIDTH}
              onChange={(v) => {
                setPage(0);
                setCreatedFrom(v);
              }}
            />
            <AuditDateTimeFilterInput
              value={createdTo}
              placeholder={t('adminAuditPage.filterTimePlaceholderEnd')}
              ariaLabel={t('adminAuditPage.filterTimeTo')}
              pickAriaLabel={t('adminAuditPage.filterPickDateTime')}
              confirmLabel={t('adminAuditPage.filterTimeModalConfirm')}
              clearLabel={t('adminAuditPage.filterTimeModalClear')}
              dateFieldLabel={t('adminAuditPage.filterTimeModalDateLabel')}
              timeFieldLabel={t('adminAuditPage.filterTimeModalTimeLabel')}
              pickDatePlaceholder={t('adminAuditPage.filterTimeModalPickDatePh')}
              timeInputPlaceholder={t('adminAuditPage.filterTimeModalTimeInputPh')}
              timeInvalidHint={t('adminAuditPage.filterTimeModalTimeInvalid')}
              timeRangeEnd="end"
              width={AUDIT_FILTER_DATETIME_WIDTH}
              onChange={(v) => {
                setPage(0);
                setCreatedTo(v);
              }}
            />
            {!isTeamAdminViewer ? (
              <select
                value={filterTeamId}
                onChange={(e) => {
                  setPage(0);
                  setFilterTeamId(e.target.value);
                }}
                aria-label={t('adminAuditPage.tableTeam')}
                style={auditFilterSelectStyle}
              >
                <option value="">{t('adminAuditPage.filterTeamAll')}</option>
                {filterTeamId.trim() &&
                !teamOptions.some((tm) => tm.id === filterTeamId.trim()) ? (
                  <option value={filterTeamId.trim()}>{filterTeamId.trim()}</option>
                ) : null}
                {teamOptions.map((tm) => (
                  <option key={tm.id} value={tm.id}>
                    {tm.name}
                  </option>
                ))}
              </select>
            ) : null}
            <select
              value={filterProjectId}
              onChange={(e) => {
                setPage(0);
                setFilterProjectId(e.target.value);
              }}
              aria-label={t('adminAuditPage.tableProject')}
              style={auditFilterSelectStyle}
            >
              <option value="">{t('adminAuditPage.filterProjectPh')}</option>
              {filterProjectId.trim() &&
              !projectOptions.some((p) => p.id === filterProjectId.trim()) ? (
                <option value={filterProjectId.trim()}>{filterProjectId.trim()}</option>
              ) : null}
              {projectOptions.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <select
              value={filterAction}
              onChange={(e) => {
                setPage(0);
                setFilterAction(e.target.value);
              }}
              aria-label={t('adminAuditPage.tableAction')}
              style={auditFilterSelectStyle}
            >
              <option value="">{t('adminAuditPage.filterActionAll')}</option>
              {actionItems.map((a) => (
                <option key={a.code} value={a.code}>
                  {a.label}
                </option>
              ))}
            </select>
            <select
              value={filterResult}
              onChange={(e) => {
                setPage(0);
                setFilterResult(e.target.value);
              }}
              aria-label={t('adminAuditPage.tableResult')}
              style={auditFilterSelectStyle}
            >
              <option value="">{t('adminAuditPage.filterResultAll')}</option>
              <option value="SUCCESS">{t('adminAuditPage.filterResultSuccess')}</option>
              <option value="FAIL">{t('adminAuditPage.filterResultFail')}</option>
            </select>
            <div style={auditFilterKeywordWrap}>
              <input
                type="text"
                value={filterQ}
                onChange={(e) => {
                  setPage(0);
                  setFilterQ(e.target.value);
                }}
                placeholder={t('adminAuditPage.filterKeywordPh')}
                aria-label={t('adminAuditPage.filterKeyword')}
                style={{
                  ...dataPageFilterInput,
                  width: '100%',
                  paddingLeft: 36,
                }}
              />
              <svg
                style={{
                  position: 'absolute',
                  left: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: 16,
                  height: 16,
                  fill: '#6b7280',
                  pointerEvents: 'none',
                }}
                viewBox="0 0 24 24"
                aria-hidden
              >
                <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
              </svg>
            </div>
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              flexShrink: 0,
              flexGrow: 0,
            }}
          >
            <button
              type="button"
              onClick={() => loadAuditLogs()}
              style={{
                padding: '8px 16px',
                backgroundColor: '#2563eb',
                border: 'none',
                borderRadius: 6,
                color: '#ffffff',
                fontSize: 14,
                cursor: 'pointer',
                fontWeight: 500,
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                flexShrink: 0,
                whiteSpace: 'nowrap',
                minWidth: 'fit-content',
              }}
            >
              {t('adminAuditPage.search')}
            </button>
            <button
              type="button"
              onClick={resetFilters}
              style={{
                padding: '8px 16px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                color: '#374151',
                fontSize: '14px',
                cursor: 'pointer',
                outline: 'none',
                flexShrink: 0,
                whiteSpace: 'nowrap',
                minWidth: 'fit-content',
              }}
            >
              {t('adminAuditPage.reset')}
            </button>
          </div>
        </div>
      </ModulePageFilterCard>

      {error && (
        <div
          style={{
            padding: '12px 14px',
            marginBottom: 16,
            backgroundColor: '#fef2f2',
            border: '1px solid #fecdd3',
            borderRadius: 12,
            color: '#b91c1c',
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      <ModulePageTableCard>
        <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed', minWidth: 1080 }}>
          <colgroup>
            <col style={{ width: '14%' }} />
            <col style={{ width: '8%' }} />
            <col style={{ width: '8%' }} />
            <col style={{ width: '10%' }} />
            <col style={{ width: '10%' }} />
            <col style={{ width: '14%' }} />
            <col style={{ width: '9%' }} />
            <col />
            <col style={{ width: '72px' }} />
            <col style={{ width: '96px' }} />
            <col style={{ width: '76px' }} />
          </colgroup>
          <thead>
            <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e8eaee' }}>
              <th style={th}>{t('adminAuditPage.tableTime')}</th>
              <th style={th}>{t('adminAuditPage.tableUser')}</th>
              <th style={th}>{t('adminAuditPage.tableRole')}</th>
              <th style={th}>{t('adminAuditPage.tableTeam')}</th>
              <th style={th}>{t('adminAuditPage.tableProject')}</th>
              <th style={th}>{t('adminAuditPage.tableAction')}</th>
              <th style={th}>{t('adminAuditPage.tableResourceType')}</th>
              <th style={th}>{t('adminAuditPage.tableResourceName')}</th>
              <th style={th}>{t('adminAuditPage.tableResult')}</th>
              <th style={th}>{t('adminAuditPage.tableIp')}</th>
              <th style={{ ...th, textAlign: 'center' }}>{t('adminAuditPage.tableDetail')}</th>
            </tr>
          </thead>
          <tbody>
            {items.map((log) => {
              const resName = log.resource_name || log.resource_id || '—';
              const resTitleFull =
                resName !== '—' ? String((log.resource_name || log.resource_id || '').trim()) : '';
              return (
                <tr key={log.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{formatAuditLogDateTime(log.created_at)}</td>
                  <td style={td}>{log.username || '—'}</td>
                  <td style={td}>{formatAuditRoleDisplay(log.role, t)}</td>
                  <td style={td}>{formatAuditTeamCell(log)}</td>
                  <td style={td}>{formatAuditProjectCell(log)}</td>
                  <td style={{ ...td, wordBreak: 'break-word' }}>{log.action_label || log.action_type}</td>
                  <td style={td}>{formatResourceTypeDisplay(log.resource_type, locale)}</td>
                  <td style={tdResourceName} title={resTitleFull.length > 18 ? resTitleFull : undefined}>
                    {resName}
                  </td>
                  <td style={td}>
                    <ResultCell result={log.result} t={t} />
                  </td>
                  <td style={{ ...tdIp }}>{log.ip || '—'}</td>
                  <td style={{ ...td, textAlign: 'center' }}>
                    <button
                      type="button"
                      onClick={() => setDetailRow(log)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: '#2563eb',
                        cursor: 'pointer',
                        fontSize: 13,
                        fontWeight: 500,
                        padding: '4px 8px',
                        borderRadius: 6,
                      }}
                    >
                      {t('adminAuditPage.detailOpen')}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {items.length === 0 && !loading && (
          <div style={{ padding: '48px', textAlign: 'center', color: '#94a3b8', fontSize: 14 }}>{t('adminAuditPage.empty')}</div>
        )}
        </div>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '14px 24px',
            borderTop: '1px solid #e5e7eb',
            backgroundColor: '#fafafa',
          }}
        >
          <span style={{ fontSize: 13, color: '#64748b' }}>{t('adminAuditPage.total', { count: total })}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button
              type="button"
              disabled={page <= 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              style={{
                ...pageBtn,
                opacity: page <= 0 ? 0.45 : 1,
                cursor: page <= 0 ? 'not-allowed' : 'pointer',
              }}
            >
              {t('adminAuditPage.prev')}
            </button>
            <span
              style={{
                fontSize: 13,
                color: '#475569',
                fontVariantNumeric: 'tabular-nums',
                minWidth: 72,
                textAlign: 'center',
              }}
            >
              {page + 1} / {totalPages}
            </span>
            <button
              type="button"
              disabled={page + 1 >= totalPages}
              onClick={() => setPage((p) => p + 1)}
              style={{
                ...pageBtn,
                opacity: page + 1 >= totalPages ? 0.45 : 1,
                cursor: page + 1 >= totalPages ? 'not-allowed' : 'pointer',
              }}
            >
              {t('adminAuditPage.next')}
            </button>
          </div>
        </div>
      </ModulePageTableCard>

      {detailRow ? (
        <AuditDetailModal
          key={detailRow.id}
          log={detailRow}
          onClose={() => setDetailRow(null)}
          locale={locale}
          t={t}
        />
      ) : null}
    </ModulePageContainer>
  );
}

const th: React.CSSProperties = {
  padding: '11px 12px',
  textAlign: 'left',
  fontSize: 12,
  fontWeight: 600,
  color: '#64748b',
  whiteSpace: 'nowrap',
  letterSpacing: '0.01em',
};

const td: React.CSSProperties = {
  padding: '10px 12px',
  fontSize: 13,
  color: '#1e293b',
  verticalAlign: 'middle',
};

const tdResourceName: React.CSSProperties = {
  ...td,
  maxWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const tdIp: React.CSSProperties = {
  ...td,
  fontSize: 12,
  fontVariantNumeric: 'tabular-nums',
  color: '#475569',
};

const pageBtn: React.CSSProperties = {
  height: 32,
  padding: '0 14px',
  borderRadius: 8,
  border: '1px solid #e2e8f0',
  background: '#fff',
  fontSize: 13,
  fontWeight: 500,
  color: '#475569',
};
