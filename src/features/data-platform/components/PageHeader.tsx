export default function PageHeader({ title }: { title: string }) {
  return (
    <h2 style={{
      fontSize: '24px',
      fontWeight: 600,
      margin: 0,
      color: '#fff',
    }}>
      {title}
    </h2>
  );
}


