import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Middleware：登录态不在 Cookie（使用 sessionStorage token + X-Session-Id）。
 *
 * 注意：
 * - Next middleware 无法读取 sessionStorage
 * - 浏览器页面导航也不会自动携带 Authorization header
 *
 * 因此这里不做基于 cookie 的登录拦截，避免误判导致登录后跳转循环（一直加载）。
 */
export function middleware(request: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * 匹配所有路径，除了：
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - login, forbidden (公开页面)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|login|forbidden).*)',
  ],
};


