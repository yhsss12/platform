'use client';

import type { ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageHeader,
  ModulePageFilterCard,
} from '@/components/layout/ModulePageLayout';
import { useAuthStore } from '@/store/authStore';
import { useI18n } from '@/components/common/I18nProvider';
import { changeOwnPassword, getMe, patchProfile } from '@/lib/api/authClient';
import { getRoleLabelKey, normalizeRole } from '@/lib/api/roleLabels';
import type { MeResponse } from '@/lib/api/types';

export default function AccountSettingsPage() {
  const router = useRouter();
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  const isHydrated = useAuthStore((s) => s.isHydrated);
  const [me, setMe] = useState<MeResponse | null>(null);
  const [usernameEdit, setUsernameEdit] = useState('');
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState<string | null>(null);
  const [profileMsgIsError, setProfileMsgIsError] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [pwdSaving, setPwdSaving] = useState(false);
  const [pwdMsg, setPwdMsg] = useState<string | null>(null);
  const [pwdMsgIsError, setPwdMsgIsError] = useState(false);
  const [loadError, setLoadError] = useState('');

  const refreshMe = useCallback(async () => {
    const u = await getMe();
    setMe(u);
    setUsernameEdit(u.username);
    setUser(u);
  }, [setUser]);

  useEffect(() => {
    if (!isHydrated) return;
    if (!authUser) {
      router.replace('/login');
      return;
    }
    void (async () => {
      try {
        setLoadError('');
        await refreshMe();
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [isHydrated, authUser, router, refreshMe]);

  const roleLabel = useMemo(() => {
    if (!me) return '';
    const key = getRoleLabelKey(normalizeRole(me.role));
    return t(key);
  }, [me, t]);

  const handleSaveProfile = async () => {
    if (!me) return;
    const next = usernameEdit.trim();
    if (!next) return;
    setProfileMsg(null);
    setProfileMsgIsError(false);
    setProfileSaving(true);
    try {
      const updated = await patchProfile(next);
      setMe(updated);
      setUser(updated);
      setProfileMsgIsError(false);
      setProfileMsg(t('accountSettingsPage.saveSuccess'));
      window.setTimeout(() => setProfileMsg(null), 2200);
    } catch (e) {
      setProfileMsgIsError(true);
      setProfileMsg(e instanceof Error ? e.message : t('accountSettingsPage.saveFailed'));
    } finally {
      setProfileSaving(false);
    }
  };

  const handleChangePassword = async () => {
    setPwdMsg(null);
    setPwdMsgIsError(false);
    if (newPassword.length < 6) {
      setPwdMsgIsError(true);
      setPwdMsg(t('adminUsersPage.resetPasswordTooShort'));
      return;
    }
    if (newPassword !== confirmPassword) {
      setPwdMsgIsError(true);
      setPwdMsg(t('accountSettingsPage.passwordMismatch'));
      return;
    }
    setPwdSaving(true);
    try {
      await changeOwnPassword(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setPwdMsgIsError(false);
      setPwdMsg(t('accountSettingsPage.passwordChangeSuccess'));
      window.setTimeout(() => setPwdMsg(null), 2200);
    } catch (e) {
      setPwdMsgIsError(true);
      setPwdMsg(e instanceof Error ? e.message : t('accountSettingsPage.passwordChangeFailed'));
    } finally {
      setPwdSaving(false);
    }
  };

  if (!isHydrated || !authUser) {
    return (
      <ModulePageContainer>
        <div style={{ fontSize: 14, color: '#6b7280' }}>{t('common.loading')}</div>
      </ModulePageContainer>
    );
  }

  if (loadError || !me) {
    return (
      <ModulePageContainer>
        <ModulePageHeader title={t('accountSettingsPage.title')} />
        <p style={{ color: '#b91c1c', fontSize: 14 }}>{loadError || t('common.loading')}</p>
      </ModulePageContainer>
    );
  }

  const fieldRow = (label: string, value: ReactNode, editable?: ReactNode) => (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '160px 1fr',
        gap: 12,
        alignItems: 'center',
        padding: '10px 0',
        borderBottom: '1px solid #f3f4f6',
        fontSize: 14,
      }}
    >
      <div style={{ color: '#6b7280', fontWeight: 500 }}>{label}</div>
      <div style={{ color: '#111827' }}>{editable ?? value}</div>
    </div>
  );

  const lastLogin =
    me.last_login_at && me.last_login_at.trim()
      ? me.last_login_at
      : t('accountSettingsPage.lastLoginNever');

  return (
    <ModulePageContainer>
      <ModulePageHeader title={t('accountSettingsPage.title')} subtitle={t('accountSettingsPage.subtitle')} />

      <ModulePageFilterCard>
        <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 600, color: '#111827' }}>
          {t('accountSettingsPage.cardProfile')}
        </h3>
        {fieldRow(t('accountSettingsPage.fieldAccountId'), me.account_id)}
        {fieldRow(
          t('accountSettingsPage.fieldUsername'),
          null,
          <input
            value={usernameEdit}
            onChange={(e) => setUsernameEdit(e.target.value)}
            style={{
              maxWidth: 320,
              padding: '8px 12px',
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              fontSize: 14,
            }}
          />
        )}
        {fieldRow(t('accountSettingsPage.fieldRole'), roleLabel)}
        {fieldRow(
          t('accountSettingsPage.fieldStatus'),
          me.is_active ? t('accountSettingsPage.statusActive') : t('accountSettingsPage.statusDisabled')
        )}
        {fieldRow(t('accountSettingsPage.fieldCreatedAt'), me.created_at)}
        {fieldRow(t('accountSettingsPage.fieldLastLogin'), lastLogin)}
        {profileMsg ? (
          <p style={{ margin: '12px 0 0', fontSize: 13, color: profileMsgIsError ? '#b91c1c' : '#059669' }}>
            {profileMsg}
          </p>
        ) : null}
        <div style={{ marginTop: 16 }}>
          <button
            type="button"
            onClick={() => void handleSaveProfile()}
            disabled={profileSaving}
            style={{
              padding: '8px 20px',
              borderRadius: 8,
              border: 'none',
              backgroundColor: '#111827',
              color: '#fff',
              fontSize: 14,
              cursor: profileSaving ? 'wait' : 'pointer',
            }}
          >
            {profileSaving ? t('login.loggingIn') : t('accountSettingsPage.saveUsername')}
          </button>
        </div>
      </ModulePageFilterCard>

      <ModulePageFilterCard>
        <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 600, color: '#111827' }}>
          {t('accountSettingsPage.cardSecurity')}
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 360 }}>
          <label style={{ fontSize: 13, color: '#374151' }}>
            {t('accountSettingsPage.currentPassword')}
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              style={{
                display: 'block',
                width: '100%',
                marginTop: 6,
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                fontSize: 14,
              }}
            />
          </label>
          <label style={{ fontSize: 13, color: '#374151' }}>
            {t('accountSettingsPage.newPassword')}
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              style={{
                display: 'block',
                width: '100%',
                marginTop: 6,
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                fontSize: 14,
              }}
            />
          </label>
          <label style={{ fontSize: 13, color: '#374151' }}>
            {t('accountSettingsPage.confirmPassword')}
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              style={{
                display: 'block',
                width: '100%',
                marginTop: 6,
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                fontSize: 14,
              }}
            />
          </label>
        </div>
        {pwdMsg ? (
          <p style={{ margin: '12px 0 0', fontSize: 13, color: pwdMsgIsError ? '#b91c1c' : '#059669' }}>
            {pwdMsg}
          </p>
        ) : null}
        <div style={{ marginTop: 16 }}>
          <button
            type="button"
            onClick={() => void handleChangePassword()}
            disabled={pwdSaving}
            style={{
              padding: '8px 20px',
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              backgroundColor: '#ffffff',
              color: '#111827',
              fontSize: 14,
              cursor: pwdSaving ? 'wait' : 'pointer',
            }}
          >
            {pwdSaving ? t('login.loggingIn') : t('accountSettingsPage.changePassword')}
          </button>
        </div>
      </ModulePageFilterCard>
    </ModulePageContainer>
  );
}
