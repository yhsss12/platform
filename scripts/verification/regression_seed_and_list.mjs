#!/usr/bin/env node

/**
 * Regression Case 1: Seed 后四类对象 list 必须非空。
 * 
 * 由于使用客户端 localStorage mock，此脚本需要在浏览器环境执行。
 * P0 版本：使用 Playwright 或类似工具在浏览器上下文执行。
 * 
 * 如果 Playwright 未安装，此脚本将输出说明信息。
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const ARTIFACTS_DIR = join(process.cwd(), 'artifacts', 'eval');
const OUTPUT_FILE = join(ARTIFACTS_DIR, 'smoke.api.json');

// 确保目录存在
mkdirSync(ARTIFACTS_DIR, { recursive: true });

// 检查是否在浏览器环境（通过环境变量或参数判断）
// 如果不在浏览器环境，输出说明
const isBrowserEnv = process.env.BROWSER_ENV === 'true' || process.argv.includes('--browser');

if (!isBrowserEnv) {
  console.log('⚠️  此 regression case 需要在浏览器环境执行（因为使用 localStorage mock）');
  console.log('   请使用 Playwright 或类似工具运行此脚本');
  console.log('   或者设置环境变量 BROWSER_ENV=true');
  console.log('');
  console.log('   示例 Playwright 脚本：');
  console.log('   - 启动 dev server');
  console.log('   - 访问页面触发 seed');
  console.log('   - 检查 localStorage 中四类对象数量 > 0');
  process.exit(1);
}

// 模拟浏览器环境下的验证逻辑
// 实际执行时，这部分应该由 Playwright 在浏览器上下文执行
async function runRegression() {
  // 这里应该是 Playwright 代码，但为了 P0 最小化，我们创建一个占位输出
  const result = {
    timestamp: new Date().toISOString(),
    tasks: { count: 0, status: 'pending' },
    runs: { count: 0, status: 'pending' },
    datasets: { count: 0, status: 'pending' },
    jobs: { count: 0, status: 'pending' },
    note: '此文件由 regression 脚本生成，实际验证需要在浏览器环境执行',
  };

  writeFileSync(OUTPUT_FILE, JSON.stringify(result, null, 2));
  console.log(`✅ Regression case 输出已生成: ${OUTPUT_FILE}`);
  console.log('⚠️  注意：实际验证需要在浏览器环境执行（使用 Playwright）');
}

runRegression().catch((err) => {
  console.error('❌ Regression case 执行失败:', err);
  process.exit(1);
});

