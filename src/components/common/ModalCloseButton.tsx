'use client';

/**
 * 与「新建标注任务」CreateLabelTaskModal 右上角关闭按钮一致，供各业务弹窗复用。
 */
export function ModalCloseButton({ onClick, ariaLabel = '关闭' }: { onClick: () => void; ariaLabel?: string }) {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      onClick={onClick}
      style={{
        background: 'none',
        border: 'none',
        color: '#6b7280',
        fontSize: '20px',
        cursor: 'pointer',
        padding: '4px',
        lineHeight: 1,
        width: '24px',
        height: '24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '4px',
        transition: 'all 0.2s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = '#f3f4f6';
        e.currentTarget.style.color = '#111827';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent';
        e.currentTarget.style.color = '#6b7280';
      }}
    >
      ✕
    </button>
  );
}
