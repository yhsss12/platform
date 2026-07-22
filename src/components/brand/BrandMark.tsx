'use client';

import React from 'react';

/**
 * 品牌标识组件
 * 圆角方形徽章 + 字母/图形
 */
export default function BrandMark({ size = 24, variant = 'default' }: { size?: number; variant?: 'default' | 'topbar' }) {
  if (variant === 'topbar') {
    // Topbar 版本：类似 Z 或编织图案的 logo
    return (
      <div
        style={{
          width: `${size}px`,
          height: `${size}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <svg
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M12 2L2 7L12 12L22 7L12 2Z"
            stroke="#2563eb"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M2 17L12 22L22 17"
            stroke="#2563eb"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M2 12L12 17L22 12"
            stroke="#2563eb"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }
  
  // Sidebar 版本：圆角方形徽章
  return (
    <div
      style={{
        width: `${size}px`,
        height: `${size}px`,
        backgroundColor: '#2563eb',
        borderRadius: '6px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#ffffff',
        fontSize: `${size * 0.5}px`,
        fontWeight: '700',
        flexShrink: 0,
      }}
    >
      一
    </div>
  );
}

