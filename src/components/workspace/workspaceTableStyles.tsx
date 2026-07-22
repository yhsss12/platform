'use client';

import Link from 'next/link';
import type { CSSProperties, ReactNode } from 'react';

/** 数据中心表格样式规范 — 三中心统一引用 */

export const workspaceThStyle: CSSProperties = {
  padding: '12px 16px',
  textAlign: 'left',
  borderBottom: '1px solid #e5e7eb',
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  backgroundColor: '#f9fafb',
  whiteSpace: 'nowrap',
};

export const workspaceTdStyle: CSSProperties = {
  padding: '12px 16px',
  fontSize: 13,
  fontWeight: 400,
  color: '#111827',
  borderBottom: '1px solid #f3f4f6',
  verticalAlign: 'top',
};

/** 表格长文本单行省略（需配合 table-layout: fixed 与 col maxWidth: 0） */
export const workspaceTdEllipsisStyle: CSSProperties = {
  ...workspaceTdStyle,
  maxWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  verticalAlign: 'middle',
};

export const workspaceTdMiddleStyle: CSSProperties = {
  ...workspaceTdStyle,
  verticalAlign: 'middle',
};

export const workspaceTdTimeStyle: CSSProperties = {
  ...workspaceTdStyle,
  fontSize: 12,
  color: '#374151',
  whiteSpace: 'nowrap',
};

export const workspaceCheckboxStyle: CSSProperties = {
  cursor: 'pointer',
  width: 16,
  height: 16,
};

export const workspaceBtnPrimary: CSSProperties = {
  padding: '4px 10px',
  backgroundColor: '#2563eb',
  border: 'none',
  borderRadius: 6,
  color: '#fff',
  fontSize: 12,
  fontWeight: 500,
  cursor: 'pointer',
  textDecoration: 'none',
  display: 'inline-flex',
  alignItems: 'center',
  flexShrink: 0,
};

export const workspaceBtnLink: CSSProperties = {
  padding: '4px 6px',
  fontSize: 12,
  color: '#2563eb',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  textDecoration: 'none',
  flexShrink: 0,
};

export const workspaceBtnDanger: CSSProperties = {
  ...workspaceBtnLink,
  color: '#dc2626',
};

export const workspaceTableStyle: CSSProperties = {
  width: '100%',
  minWidth: 960,
  borderCollapse: 'collapse',
  backgroundColor: '#fff',
};

export function WorkspaceTableWrap({ children }: { children: ReactNode }) {
  return <div style={{ overflowX: 'auto' }}>{children}</div>;
}

export function WorkspaceRowCheckbox({
  checked,
  onChange,
  ariaLabel,
  disabled,
}: {
  checked: boolean;
  onChange: () => void;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  return (
    <input
      type="checkbox"
      checked={checked}
      onChange={onChange}
      disabled={disabled}
      style={workspaceCheckboxStyle}
      aria-label={ariaLabel}
    />
  );
}

export function WorkspaceTableHeaderCheckbox({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <input
      type="checkbox"
      checked={checked}
      disabled={disabled}
      onChange={onChange}
      style={workspaceCheckboxStyle}
      aria-label="全选当前页"
    />
  );
}

export function workspaceTableRowHoverHandlers() {
  return {
    onMouseEnter: (e: React.MouseEvent<HTMLTableRowElement>) => {
      e.currentTarget.style.backgroundColor = '#f9fafb';
    },
    onMouseLeave: (e: React.MouseEvent<HTMLTableRowElement>) => {
      e.currentTarget.style.backgroundColor = 'transparent';
    },
  };
}

export function WorkspaceActionBar({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '4px 6px' }}>
      {children}
    </div>
  );
}

export function WorkspaceActionLink({
  label,
  href,
  onClick,
  variant = 'link',
  disabled,
  title,
}: {
  label: string;
  href?: string;
  onClick?: () => void;
  variant?: 'link' | 'primary' | 'danger';
  disabled?: boolean;
  title?: string;
}) {
  const style =
    variant === 'primary'
      ? workspaceBtnPrimary
      : variant === 'danger'
        ? workspaceBtnDanger
        : workspaceBtnLink;

  const mergedStyle: CSSProperties = {
    ...style,
    opacity: disabled ? 0.4 : 1,
    cursor: disabled ? 'not-allowed' : style.cursor,
    color: variant === 'primary' ? '#fff' : style.color,
  };

  if (href && !disabled) {
    return (
      <Link href={href} style={mergedStyle} title={title}>
        {label}
      </Link>
    );
  }

  return (
    <button
      type="button"
      style={mergedStyle}
      disabled={disabled}
      title={title}
      onClick={() => {
        if (!disabled) onClick?.();
      }}
    >
      {label}
    </button>
  );
}

export function WorkspaceTableEmptyCell({
  colSpan,
  message,
}: {
  colSpan: number;
  message: string;
}) {
  return (
    <tr>
      <td
        colSpan={colSpan}
        style={{ ...workspaceTdStyle, textAlign: 'center', color: '#6b7280', padding: 40 }}
      >
        {message}
      </td>
    </tr>
  );
}
