'use client';

import Link from 'next/link';
import { useAuthStore } from '@/store/authStore';

export default function ForbiddenPage() {
  const { user } = useAuthStore();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      backgroundColor: '#f6f7f9',
      padding: '20px',
    }}>
      <div style={{
        textAlign: 'center',
        maxWidth: '500px',
      }}>
        <h1 style={{
          fontSize: '48px',
          fontWeight: '700',
          color: '#dc2626',
          marginBottom: '16px',
        }}>
          403
        </h1>
        <h2 style={{
          fontSize: '24px',
          fontWeight: '600',
          color: '#111827',
          marginBottom: '12px',
        }}>
          无权限访问
        </h2>
        <p style={{
          fontSize: '16px',
          color: '#6b7280',
          marginBottom: '32px',
        }}>
          您没有权限访问此页面
        </p>

        <div style={{
          display: 'flex',
          gap: '12px',
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}>
          {user ? (
            <>
              <Link
                href="/workspace/simulation"
                style={{
                  padding: '12px 24px',
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#ffffff',
                  backgroundColor: '#2563eb',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  transition: 'background-color 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#1d4ed8';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#2563eb';
                }}
              >
                返回工作台
              </Link>
              <Link
                href="/"
                style={{
                  padding: '12px 24px',
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#374151',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f9fafb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#ffffff';
                }}
              >
                返回首页
              </Link>
            </>
          ) : (
            <Link
              href="/login"
              style={{
                padding: '12px 24px',
                fontSize: '14px',
                fontWeight: '500',
                color: '#ffffff',
                backgroundColor: '#2563eb',
                borderRadius: '6px',
                textDecoration: 'none',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#1d4ed8';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }}
            >
              去登录
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

















