'use client';

import { useState, useId } from 'react';

export type FloatingFieldProps = {
  id?: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password';
  autoComplete?: string;
  rightSlot?: React.ReactNode;
  onFocus?: () => void;
  onBlur?: () => void;
};

export function FloatingField({
  id: idProp,
  label,
  value,
  onChange,
  type = 'text',
  autoComplete,
  rightSlot,
  onFocus,
  onBlur,
}: FloatingFieldProps) {
  const [focused, setFocused] = useState(false);
  const [hovered, setHovered] = useState(false);
  const floated = focused || value.length > 0;
  const reactId = useId();
  const id = idProp ?? reactId;

  const borderColor = focused
    ? '#94a3b8' // slate-400
    : hovered
      ? '#cbd5e1' // slate-300
      : '#e2e8f0'; // slate-200

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: 64, // h-16
        borderRadius: 12, // rounded-xl
        border: `1px solid ${borderColor}`,
        backgroundColor: '#ffffff',
        boxShadow: focused ? '0 0 0 4px rgba(15,23,42,0.05)' : 'none', // ring-4 ring-slate-900/5
        transition: 'border-color 0.15s ease-out, box-shadow 0.15s ease-out',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <style>
        {`
          .floating-field-label {
            transition: all 150ms ease-out;
          }
        `}
      </style>

      <label
        htmlFor={id}
        className="floating-field-label"
        style={{
          position: 'absolute',
          left: 16, // left-4
          top: floated ? 8 : 18, // top-2 when floated, top-[18px] default (略偏上)
          transform: floated ? 'translateY(0)' : 'none',
          margin: 0,
          fontSize: floated ? 12 : 18,
          fontWeight: 500,
          color: floated ? '#64748b' : '#334155', // text-slate-500 / slate-700
          pointerEvents: 'none',
          ...(floated
            ? { backgroundColor: '#ffffff', paddingLeft: 4, paddingRight: 4 }
            : {}),
        }}
      >
        {label}
      </label>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          height: '100%',
          paddingLeft: 16, // px-4
          paddingRight: rightSlot ? 8 : 16,
          paddingTop: 28, // pt-7 so text sits below label
          paddingBottom: 12, // pb-3
          boxSizing: 'border-box',
        }}
      >
        <input
          id={id}
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => {
            setFocused(true);
            onFocus?.();
          }}
          onBlur={() => {
            setFocused(false);
            onBlur?.();
          }}
          autoComplete={autoComplete}
          style={{
            flex: 1,
            width: '100%',
            minWidth: 0,
            border: 'none',
            outline: 'none',
            background: 'transparent',
            fontSize: 16,
            color: '#0f172a',
            padding: 0,
            margin: 0,
            height: '100%',
          }}
        />
        {rightSlot && (
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              marginLeft: 4,
            }}
          >
            {rightSlot}
          </div>
        )}
      </div>
    </div>
  );
}
