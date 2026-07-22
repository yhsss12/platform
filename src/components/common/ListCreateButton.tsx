'use client';

import React, { useState, type CSSProperties, type PropsWithChildren } from 'react';

export interface ListCreateButtonProps extends PropsWithChildren {
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  style?: CSSProperties;
}

export function ListCreateButton(props: ListCreateButtonProps) {
  const { disabled, title, onClick, style, children } = props;
  const [hover, setHover] = useState(false);
  const [active, setActive] = useState(false);
  const [focus, setFocus] = useState(false);

  const baseStyle: CSSProperties = {
    padding: '8px 16px',
    borderRadius: 6,
    border: 'none',
    backgroundColor: disabled ? '#e5e7eb' : hover || active ? '#1d4ed8' : '#2563eb',
    color: disabled ? '#9ca3af' : '#ffffff',
    fontSize: 14,
    fontWeight: 500,
    cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    outline: 'none',
    boxShadow: focus
      ? '0 0 0 2px rgba(191, 219, 254, 0.9), 0 10px 24px rgba(15, 23, 42, 0.18)'
      : '0 10px 24px rgba(15, 23, 42, 0.12)',
    transform: active ? 'translateY(1px)' : 'translateY(0)',
    transition:
      'background-color 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease, color 0.15s ease, border-color 0.15s ease',
    whiteSpace: 'nowrap',
    ...style,
  };

  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={() => {
        if (disabled) return;
        onClick?.();
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => {
        setHover(false);
        setActive(false);
      }}
      onMouseDown={() => {
        if (disabled) return;
        setActive(true);
      }}
      onMouseUp={() => setActive(false)}
      onFocus={() => {
        if (disabled) return;
        setFocus(true);
      }}
      onBlur={() => {
        setFocus(false);
        setActive(false);
      }}
      style={baseStyle}
    >
      {children}
    </button>
  );
}

export default ListCreateButton;

