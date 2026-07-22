'use client';

import { useParams } from 'next/navigation';

export default function LabelTaskExecutePage() {
  const params = useParams();
  const taskId = params.id as string;

  return (
    <div style={{ padding: '24px', backgroundColor: '#f6f7f9', minHeight: '100vh' }}>
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '8px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
          padding: '40px',
          textAlign: 'center',
        }}
      >
        <h2
          style={{
            fontSize: '20px',
            fontWeight: '600',
            color: '#111827',
            marginBottom: '12px',
          }}
        >
          执行标注任务
        </h2>
        <p style={{ color: '#6b7280', fontSize: '14px', margin: 0 }}>
          任务 ID: {taskId}
        </p>
        <p style={{ color: '#6b7280', fontSize: '14px', marginTop: '8px' }}>
          执行页面开发中...
        </p>
      </div>
    </div>
  );
}


