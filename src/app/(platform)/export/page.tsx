'use client';

import { useI18n } from '@/components/common/I18nProvider';

export default function ExportPage() {
  const { t } = useI18n();
  return (
    <div style={{ padding: '24px' }}>
      <h2 style={{ fontSize: '18px', fontWeight: 600, color: '#fff', margin: '0 0 24px 0' }}>
        {t('shellPage.exportTitle')}
      </h2>
      <div style={{
        textAlign: 'center',
        padding: '60px 24px',
        color: '#666',
      }}>
        <p style={{ fontSize: '16px', margin: 0 }}>{t('shellPage.comingSoon')}</p>
      </div>
    </div>
  );
}


