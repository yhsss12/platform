'use client';

import { useCallback, useState } from 'react';
import { PhysicsProxyLibraryTable } from '@/components/workspace/PhysicsProxyLibraryTable';
import { useI18n } from '@/components/common/I18nProvider';

export default function PhysicsProxiesPage() {
  const { t } = useI18n();
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const showToast = useCallback((text: string) => {
    setToastMsg(text);
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  return (
    <>
      <PhysicsProxyLibraryTable
        title={t('physicsProxy.libraryTitle')}
        subtitle={t('physicsProxy.librarySubtitle')}
        onToast={showToast}
      />
      {toastMsg ? (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 14,
            fontWeight: 500,
            zIndex: 1700,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#fff',
          }}
        >
          {toastMsg}
        </div>
      ) : null}
    </>
  );
}
