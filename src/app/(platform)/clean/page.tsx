'use client';

import { useI18n } from '@/components/common/I18nProvider';

/**
 * 清洗功能暂未开放，兜底页：用户手动输入 /clean 时显示空状态
 */
export default function CleanPage() {
  const { t } = useI18n();
  return (
    <div
      style={{
        padding: 48,
        minHeight: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f6f7f9',
        color: '#6b7280',
      }}
    >
      <div
        style={{
          fontSize: '15px',
          fontWeight: 500,
          color: '#374151',
          marginBottom: '8px',
        }}
      >
        {t('cleanPage.notAvailable')}
      </div>
      <div style={{ fontSize: '14px' }}>{t('cleanPage.comingSoon')}</div>
    </div>
  );
}
