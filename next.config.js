/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  productionBrowserSourceMaps: false,
  // Next.js rewrite 代理默认仅 30s（proxy-request.js），长耗时接口会先被代理掐断并返回 500，
  // 而后端仍会继续执行（如数据同步到 MinIO），导致「前端失败、后端日志成功、MinIO 已有数据」。
  // 与后端 agent_data_sync_proxy 单次 httpx 超时（1800s）对齐；若需数小时级批量同步，应改为异步任务+轮询（见 docs/ARCHITECTURE_AGENT_AND_STREAMING.md §9），勿无限加大此处。
  experimental: {
    proxyTimeout: 1_800_000,
  },
  // LAN 访问 dev 时浏览器 Origin 为网卡 IP（非 localhost），须放行否则 /_next/* 会被 block
  allowedDevOrigins: [
    ...(process.env.FRONTEND_LAN_HOST ? [process.env.FRONTEND_LAN_HOST] : ['172.18.0.101']),
    'localhost',
    '127.0.0.1',
  ],
  webpack: (config, { isServer, dev }) => {
    // 修复 lucide-react 在 Next.js 15 中的兼容性问题
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    // dev 模式禁用 webpack 持久化 cache，避免 HMR 后 .next/server chunk 引用丢失（500/502）
    if (dev) {
      config.cache = false;
    }
    return config;
  },
  // API 代理：将所有 /api/* 请求转发到 FastAPI 后端
  async rewrites() {
    // 仅服务端 rewrite 使用：Docker 同容器（START_MODE=all）内应指向本机 8000，勿用宿主机局域网 IP（易 ECONNRESET）
    // 浏览器端应走相对路径 /api/*，由本 rewrite 代理到 INTERNAL_API_URL
    const apiUrl =
      process.env.INTERNAL_API_URL ||
      'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      // Agent 一键安装脚本从 ${PUBLIC_HOST}:3001/static/bin/... 拉取包时，需转发到 FastAPI（与 /api 同源后端）
      {
        source: '/static/bin/:path*',
        destination: `${apiUrl}/static/bin/:path*`,
      },
    ];
  },
}

module.exports = nextConfig

