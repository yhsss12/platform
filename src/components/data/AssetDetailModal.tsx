'use client';

import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import type { Project } from '@/lib/projects/types';
import type { DataAssetItem } from '@/features/data-platform/api/dataAssetsApi';
import {
  getDataAsset,
  getDataAssetWarehouseDisplayPath,
  getSourceLabel,
  isDataAssetSynced,
} from '@/features/data-platform/api/dataAssetsApi';
import { listDevices } from '@/features/data-platform/api/deviceApi';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { useI18n } from '@/components/common/I18nProvider';

type DerivedFrom = {
  asset_id?: string | null;
  input_path?: string | null;
  project_id?: string | null;
  project_name?: string | null;
};

function pathBasename(p: string): string {
  const s = (p || '').replace(/\\/g, '/').trim();
  if (!s) return '';
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

function inferFormatLabel(filenameOrPath: string): string {
  const lower = filenameOrPath.toLowerCase();
  if (lower.endsWith('.mcap')) return 'MCAP';
  if (lower.endsWith('.hdf5') || lower.endsWith('.h5')) return 'HDF5';
  if (lower.endsWith('.zip')) return 'LeRobot';
  return '—';
}

function parseDerivedFrom(meta: string | null | undefined): DerivedFrom | null {
  if (!meta) return null;
  try {
    const o = JSON.parse(meta) as { derived_from?: DerivedFrom };
    const d = o?.derived_from;
    if (!d || typeof d !== 'object') return null;
    return d;
  } catch {
    return null;
  }
}

function syncStatusLabel(status: string | undefined): string {
  const s = (status || 'synced').toLowerCase();
  if (s === 'unsynced') return '未同步';
  if (s === 'syncing') return '同步中';
  if (s === 'failed') return '同步失败';
  return '已同步';
}

const rowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '120px 1fr',
  gap: '8px 16px',
  fontSize: 13,
  lineHeight: 1.5,
  padding: '8px 0',
  borderBottom: '1px solid #f3f4f6',
};

const labelStyle: CSSProperties = { color: '#6b7280', textAlign: 'right' };
const valueStyle: CSSProperties = { color: '#111827', wordBreak: 'break-word' };

export interface AssetDetailModalProps {
  open: boolean;
  asset: DataAssetItem | null;
  projectList: Project[];
  onClose: () => void;
}

export default function AssetDetailModal({ open, asset, projectList, onClose }: AssetDetailModalProps) {
  const { t } = useI18n();
  const [resolved, setResolved] = useState<DataAssetItem | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [deviceNameById, setDeviceNameById] = useState<Map<string, string>>(new Map());

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listDevices()
      .then((res) => {
        if (cancelled || !res.ok || !res.data) return;
        const m = new Map<string, string>();
        for (const d of res.data) {
          m.set(String(d.id), d.name);
        }
        setDeviceNameById(m);
      })
      .catch(() => {
        if (!cancelled) setDeviceNameById(new Map());
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open || !asset) {
      setResolved(null);
      return;
    }
    setResolved(asset);
    if (!isDataAssetSynced(asset)) return;

    let cancelled = false;
    setLoadingDetail(true);
    getDataAsset(asset.id)
      .then((res) => {
        if (cancelled) return;
        if (res.ok && res.data) setResolved(res.data);
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, asset?.id]);

  const display = resolved || asset;
  const derived = useMemo(() => parseDerivedFrom(display?.meta ?? null), [display?.meta]);
  const isConverted = useMemo(() => {
    const s = (display?.source || '').toString().trim().toLowerCase();
    if (s === 'convert' || s === 'conversion') return true;
    return Boolean((display?.conversion_task_name || '').trim());
  }, [display?.source, display?.conversion_task_name]);

  const projectDisplay = useMemo(() => {
    if (!display) return '—';
    if (display.project_name) return display.project_name;
    const pv = display.project_id ?? '';
    if (!pv) return '—';
    const byId = projectList.find((p) => p.id === pv);
    if (byId) return byId.name;
    const byName = projectList.find((p) => p.name === pv);
    if (byName) return byName.name;
    return pv;
  }, [display, projectList]);

  const deviceLine = useMemo(() => {
    if (!display?.device_id || !String(display.device_id).trim()) return '—';
    const did = String(display.device_id).trim();
    const name = deviceNameById.get(did);
    return name ? `${name}（ID ${did}）` : did;
  }, [display?.device_id, deviceNameById]);

  const formatBytes = (bytes: number) => {
    if (!bytes) return '—';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
    return `${mb.toFixed(2)} MB`;
  };

  if (!open || !display) return null;

  const warehouse = getDataAssetWarehouseDisplayPath(display);
  const sourceFile = derived?.input_path ? pathBasename(derived.input_path) : '';
  const sourceFormat =
    derived?.input_path && sourceFile ? inferFormatLabel(sourceFile) : (derived?.input_path ? inferFormatLabel(derived.input_path) : '—');

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="asset-detail-title"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 2000,
        backgroundColor: 'rgba(15, 23, 42, 0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 'min(560px, 100%)',
          maxHeight: 'min(88vh, 720px)',
          overflow: 'auto',
          backgroundColor: '#ffffff',
          borderRadius: 12,
          boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)',
          border: '1px solid #e5e7eb',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '18px 20px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <h2 id="asset-detail-title" style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827' }}>
            {t('dataPage.assetDetailTitle')}
          </h2>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              border: '1px solid #d1d5db',
              background: '#fff',
              fontSize: 13,
              cursor: 'pointer',
              color: '#374151',
            }}
          >
            {t('common.close')}
          </button>
        </div>

        <div style={{ padding: '16px 20px 20px' }}>
          {loadingDetail ? (
            <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 8 }}>{t('common.loading')}</div>
          ) : null}

          <section style={{ marginBottom: 4 }}>
            <div style={{ ...rowStyle, borderBottom: 'none', paddingTop: 0 }}>
              <div style={labelStyle}>{t('dataPage.assetDetailDatasetId')}</div>
              <div style={valueStyle}>{(display.dataset_id || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailCode')}</div>
              <div style={valueStyle}>{display.code || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.fileName')}</div>
              <div style={valueStyle}>{display.filename || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.format')}</div>
              <div style={valueStyle}>{(display.format || '—').toUpperCase()}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.source')}</div>
              <div style={valueStyle}>{getSourceLabel(display.source, t)}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.project')}</div>
              <div style={valueStyle}>{projectDisplay}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.operator')}</div>
              <div style={valueStyle}>{(display.operator_name || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailDevice')}</div>
              <div style={valueStyle}>{deviceLine}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.collectTaskName')}</div>
              <div style={valueStyle}>{(display.collect_task_name || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.labelTaskName')}</div>
              <div style={valueStyle}>{(display.label_task_name || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.conversionTaskName')}</div>
              <div style={valueStyle}>{(display.conversion_task_name || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailAnnotation')}</div>
              <div style={valueStyle}>{(display.instruction_text || '').trim() || '—'}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.fileSize')}</div>
              <div style={valueStyle}>{formatBytes(display.file_size_bytes)}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailCreatedAt')}</div>
              <div style={valueStyle}>{formatDateTimeMinuteYmdSlash(display.created_at)}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.updatedAt')}</div>
              <div style={valueStyle}>{formatDateTimeMinuteYmdSlash(display.updated_at ?? display.created_at)}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailSyncStatus')}</div>
              <div style={valueStyle}>{syncStatusLabel(display.sync_status)}</div>
            </div>
            <div style={rowStyle}>
              <div style={labelStyle}>{t('dataPage.assetDetailParseStatus')}</div>
              <div style={valueStyle}>{display.parse_status || '—'}</div>
            </div>
            <div style={{ ...rowStyle, borderBottom: '1px solid #f3f4f6' }}>
              <div style={labelStyle}>{t('dataPage.assetDetailStoragePath')}</div>
              <div style={{ ...valueStyle, fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>{warehouse || '—'}</div>
            </div>
          </section>

          {isConverted && derived ? (
            <section style={{ marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 8 }}>
                {t('dataPage.assetDetailLineageSection')}
              </div>
              <div style={rowStyle}>
                <div style={labelStyle}>{t('dataPage.assetDetailOriginalFilename')}</div>
                <div style={valueStyle}>{sourceFile || pathBasename(derived.input_path || '') || '—'}</div>
              </div>
              <div style={rowStyle}>
                <div style={labelStyle}>{t('dataPage.assetDetailOriginalFormat')}</div>
                <div style={valueStyle}>{sourceFormat}</div>
              </div>
              <div style={rowStyle}>
                <div style={labelStyle}>{t('dataPage.assetDetailSourceAssetId')}</div>
                <div style={valueStyle}>{(derived.asset_id || '').trim() || '—'}</div>
              </div>
              {derived.input_path && pathBasename(derived.input_path) !== derived.input_path ? (
                <div style={{ ...rowStyle, borderBottom: 'none' }}>
                  <div style={labelStyle}>{t('dataPage.assetDetailSourcePath')}</div>
                  <div style={{ ...valueStyle, fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>{derived.input_path}</div>
                </div>
              ) : null}
            </section>
          ) : null}

          {(() => {
            const syncStatus = (display.sync_status || '').toString().trim().toLowerCase();
            const parseStatus = (display.parse_status || '').toString().trim().toLowerCase();
            const shouldShowSyncError = syncStatus === 'failed' && (display.sync_error || '').trim();
            const shouldShowParseError =
              (parseStatus === '失败' || parseStatus === 'failed') && (display.error_msg || '').trim();
            if (!shouldShowSyncError && !shouldShowParseError) return null;
            return (
            <section style={{ marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#b91c1c', marginBottom: 8 }}>
                {t('dataPage.assetDetailErrors')}
              </div>
              {shouldShowSyncError ? (
                <div style={{ fontSize: 12, color: '#991b1b', marginBottom: 6 }}>同步：{display.sync_error}</div>
              ) : null}
              {shouldShowParseError ? (
                <div style={{ fontSize: 12, color: '#991b1b' }}>解析：{display.error_msg}</div>
              ) : null}
            </section>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
