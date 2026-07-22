import type { Metadata } from 'next';
import { PLATFORM_DESCRIPTION, PLATFORM_NAME } from '@/config/brand';

export const metadata: Metadata = {
  title: PLATFORM_NAME,
  description: PLATFORM_DESCRIPTION,
};

import { AuthBootstrap } from '@/components/AuthBootstrap';
import { I18nProvider } from '@/components/common/I18nProvider';
import { QueryProvider } from '@/lib/query/QueryProvider';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // suppressHydrationWarning：浏览器扩展（如沉浸式翻译）会在 <html> 上注入 data-* 属性，导致与服务端 HTML 不一致
    <html lang="zh-CN" suppressHydrationWarning>
      <body
        style={{ margin: 0, padding: 0, fontFamily: 'system-ui, -apple-system, sans-serif' }}
        suppressHydrationWarning
      >
        <AuthBootstrap />
        <QueryProvider>
          <I18nProvider>{children}</I18nProvider>
        </QueryProvider>
      </body>
    </html>
  );
}

