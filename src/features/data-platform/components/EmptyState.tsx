export default function EmptyState({ message, hint }: { message: string; hint?: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '60px 24px',
      color: '#666',
      textAlign: 'center',
    }}>
      <p style={{ fontSize: '18px', margin: '0 0 8px 0' }}>{message}</p>
      {hint && <p style={{ fontSize: '14px', margin: 0, color: '#555' }}>{hint}</p>}
    </div>
  );
}


