/**
 * 五大中心页面切换性能采样（production / dev）
 *
 * 用法:
 *   BASE_URL=http://127.0.0.1:3000 node scripts/verification/browser/perf-page-switch.mjs
 *   BASE_URL=http://127.0.0.1:3001 node scripts/verification/browser/perf-page-switch.mjs --mode=dev
 *   node scripts/verification/browser/perf-page-switch.mjs --storage-state=.auth/perf-storage-state.json
 *
 * 环境变量:
 *   BASE_URL, EAI_USERNAME, EAI_PASSWORD, STORAGE_STATE, STALE_WAIT_MS, PERF_MODE
 */
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { chromium } from '@playwright/test';

const args = process.argv.slice(2);
const modeArg =
  args.find((a) => a.startsWith('--mode='))?.split('=')[1] || process.env.PERF_MODE || 'prod';
const storageStateArg =
  args.find((a) => a.startsWith('--storage-state='))?.split('=')[1] ||
  process.env.STORAGE_STATE ||
  '.auth/perf-storage-state.json';

const BASE_URL =
  process.env.BASE_URL ||
  (modeArg === 'dev' ? 'http://127.0.0.1:3001' : 'http://127.0.0.1:3000');
const USERNAME = process.env.EAI_USERNAME || 'admin';
const PASSWORD = process.env.EAI_PASSWORD || 'admin123';
const STALE_WAIT_MS = Number(process.env.STALE_WAIT_MS || 35000);
const DEFAULT_STORAGE = path.resolve(process.cwd(), storageStateArg);

const PAGES = [
  { name: 'DataCenter', path: '/workspace/data', apiPrefix: '/api/workspace/datasets' },
  {
    name: 'TrainingCenter',
    path: '/workspace/training',
    apiPrefix: '/api/workspace/training/jobs',
  },
  {
    name: 'EvaluationCenter',
    path: '/workspace/evaluation',
    apiPrefix: '/api/workspace/evaluation/jobs',
  },
  {
    name: 'ModelAssets',
    path: '/workspace/resources/model-assets',
    apiPrefix: '/api/workspace/model-assets',
  },
  {
    name: 'TaskTemplates',
    path: '/workspace/resources/task-templates',
    apiPrefix: '/api/workspace/task-templates',
  },
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isClientApiRequest(url, method) {
  if (!url.includes('/api/')) return false;
  const m = (method || 'GET').toUpperCase();
  return m === 'GET' || m === 'POST' || m === 'PUT' || m === 'PATCH' || m === 'DELETE';
}

function stripQuery(url) {
  try {
    const u = new URL(url);
    return `${u.pathname}`;
  } catch {
    return url.split('?')[0];
  }
}

function formatMs(ms) {
  return `${Math.round(ms)}ms`;
}

function pad(str, len) {
  const s = String(str ?? '');
  return s.length >= len ? s : s + ' '.repeat(len - s.length);
}

async function loginViaApi(page) {
  const sessionId = randomUUID();
  const resp = await page.request.post(`${BASE_URL}/api/auth/login`, {
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': sessionId,
    },
    data: { username: USERNAME, password: PASSWORD },
  });
  const raw = await resp.json().catch(() => ({}));
  if (!resp.ok() || raw?.ok === false) {
    throw new Error(raw?.error || `Login failed (${resp.status()})`);
  }
  const tokenData = raw?.data ?? raw;
  const accessToken = tokenData?.access_token;
  if (!accessToken) {
    throw new Error('Login response missing access_token');
  }

  await page.goto(`${BASE_URL}/login`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(
    ({ accessToken, sessionId, refreshToken }) => {
      window.sessionStorage.setItem('auth.access_token', accessToken);
      window.sessionStorage.setItem('auth.sessionId', sessionId);
      if (refreshToken) {
        window.sessionStorage.setItem('auth.refresh_token', refreshToken);
      }
    },
    {
      accessToken,
      sessionId,
      refreshToken: tokenData?.refresh_token ?? null,
    }
  );
}

async function loginViaUi(page) {
  await page.goto(`${BASE_URL}/login`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#login-username input, #login-username', { timeout: 15000 });
  const usernameInput = page.locator('#login-username input').first();
  if (await usernameInput.count()) {
    await usernameInput.fill(USERNAME);
  } else {
    await page.locator('#login-username').fill(USERNAME);
  }
  const passwordInput = page.locator('#login-password input').first();
  if (await passwordInput.count()) {
    await passwordInput.fill(PASSWORD);
  } else {
    await page.locator('#login-password').fill(PASSWORD);
  }
  await page.locator('button[type="submit"]').first().click();
  await page.waitForFunction(
    () => window.sessionStorage.getItem('auth.access_token'),
    { timeout: 30000 }
  );
}

async function ensureAuthenticated(browser, storagePath) {
  let context;
  if (fs.existsSync(storagePath)) {
    console.log(`[auth] reuse storageState: ${storagePath}`);
    context = await browser.newContext({ storageState: storagePath });
  } else {
    context = await browser.newContext();
  }
  const page = await context.newPage();
  await page.goto(`${BASE_URL}/workspace/data`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  let token = await page.evaluate(() => window.sessionStorage.getItem('auth.access_token'));
  if (!token) {
    try {
      await loginViaApi(page);
    } catch (err) {
      console.warn(`[auth] API login failed (${err.message}), fallback UI login`);
      await loginViaUi(page);
    }
    await page.goto(`${BASE_URL}/workspace/data`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    token = await page.evaluate(() => window.sessionStorage.getItem('auth.access_token'));
    if (!token) {
      throw new Error('Login failed: no access token in sessionStorage');
    }
    fs.mkdirSync(path.dirname(storagePath), { recursive: true });
    await context.storageState({ path: storagePath });
    console.log(`[auth] saved storageState: ${storagePath}`);
  }
  return { context, page };
}

function createNetworkCollector(page, apiPrefix) {
  const pending = new Map();
  const apiCalls = [];
  const perfLogs = [];

  const onRequest = (req) => {
    const url = req.url();
    const type = req.resourceType();
    if (type !== 'fetch' && type !== 'xhr') return;
    if (!isClientApiRequest(url, req.method())) return;
    pending.set(req, { url, method: req.method(), startedAt: Date.now() });
  };

  const onResponse = (res) => {
    const req = res.request();
    const meta = pending.get(req);
    if (!meta) return;
    pending.delete(req);
    const durationMs = Date.now() - meta.startedAt;
    apiCalls.push({
      url: meta.url,
      path: stripQuery(meta.url),
      method: meta.method,
      status: res.status(),
      durationMs,
      isPrimary: meta.url.includes(apiPrefix),
    });
  };

  const onRequestFailed = (req) => {
    const meta = pending.get(req);
    if (!meta) return;
    pending.delete(req);
    apiCalls.push({
      url: meta.url,
      path: stripQuery(meta.url),
      method: meta.method,
      status: 0,
      durationMs: Date.now() - meta.startedAt,
      isPrimary: meta.url.includes(apiPrefix),
      failed: true,
    });
  };

  const onConsole = (msg) => {
    const text = msg.text();
    if (text.includes('[Perf]')) perfLogs.push(text);
  };

  page.on('request', onRequest);
  page.on('response', onResponse);
  page.on('requestfailed', onRequestFailed);
  page.on('console', onConsole);

  return {
    detach() {
      page.off('request', onRequest);
      page.off('response', onResponse);
      page.off('requestfailed', onRequestFailed);
      page.off('console', onConsole);
    },
    getSnapshot() {
      const primaryCalls = apiCalls.filter((c) => c.isPrimary);
      const slowest =
        apiCalls.length > 0
          ? apiCalls.reduce((a, b) => (a.durationMs >= b.durationMs ? a : b))
          : null;
      const readyLog =
        perfLogs.find((l) => l.includes(' ready in ')) ||
        perfLogs.filter((l) => l.includes('ready')).pop() ||
        null;
      return {
        apiCalls: [...apiCalls],
        primaryApiCount: primaryCalls.length,
        slowestApi: slowest
          ? `${slowest.path} ${slowest.status} ${formatMs(slowest.durationMs)}`
          : '—',
        perfReady: readyLog,
        perfLogs: [...perfLogs],
      };
    },
  };
}

async function waitForPageReady(page, spec, collector, timeoutMs = 60000) {
  const started = Date.now();
  await page.goto(`${BASE_URL}${spec.path}`, { waitUntil: 'domcontentloaded', timeout: timeoutMs });

  const deadline = started + timeoutMs;
  while (Date.now() < deadline) {
    const snap = collector.getSnapshot();
    if (snap.perfReady) break;
    const elapsed = Date.now() - started;
    const snap2 = collector.getSnapshot();
    if (snap2.primaryApiCount > 0 && elapsed > 1200) break;
    await sleep(100);
  }

  await sleep(200);
  return Date.now() - started;
}

async function sampleVisit(page, spec) {
  const collector = createNetworkCollector(page, spec.apiPrefix);
  const elapsedMs = await waitForPageReady(page, spec, collector);
  const snap = collector.getSnapshot();
  collector.detach();

  const readyMatch = snap.perfReady?.match(/ready in (\d+)ms/);
  const perfReadyMs = readyMatch ? Number(readyMatch[1]) : null;

  return {
    elapsedMs,
    perfReadyMs,
    perfReady: snap.perfReady,
    apiCount: snap.apiCalls.length,
    primaryApiCount: snap.primaryApiCount,
    slowestApi: snap.slowestApi,
    apiCalls: snap.apiCalls,
  };
}

function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function summarizeApiCalls(apiCalls) {
  const byPath = new Map();
  for (const call of apiCalls) {
    const key = `${call.method} ${call.path}`;
    const bucket = byPath.get(key) ?? [];
    bucket.push(call.durationMs);
    byPath.set(key, bucket);
  }
  return Array.from(byPath.entries())
    .map(([key, durations]) => ({
      key,
      count: durations.length,
      p50: percentile(durations, 50),
      p95: percentile(durations, 95),
    }))
    .sort((a, b) => b.p95 - a.p95);
}

function analyzeBlocking(visit, apiPrefix) {
  const primaryCalls = visit.apiCalls.filter((c) => c.path.includes(apiPrefix.replace('/api', '')) || c.isPrimary);
  if (!primaryCalls.length || visit.perfReadyMs == null) {
    return { blockedBy: '—', note: visit.perfReadyMs == null ? 'no perf log' : 'no primary api' };
  }
  const slowest = primaryCalls.reduce((a, b) => (a.durationMs >= b.durationMs ? a : b));
  const blocked = visit.perfReadyMs >= slowest.durationMs * 0.8;
  return {
    blockedBy: blocked ? `${slowest.method} ${slowest.path}` : '—',
    note: blocked
      ? `ready ${visit.perfReadyMs}ms ~ api ${slowest.durationMs}ms`
      : `ready ${visit.perfReadyMs}ms faster than api ${slowest.durationMs}ms`,
  };
}

function inferQueryCacheHit(first, second) {
  if (second.primaryApiCount === 0) return 'hit';
  if (second.primaryApiCount < first.primaryApiCount) return 'partial';
  return 'miss';
}

function printTable(mode, baseUrl, rows) {
  console.log(`\n=== 五大中心页面切换性能 (${mode}) @ ${baseUrl} ===\n`);
  const header = [
    pad('页面', 18),
    pad('首次', 8),
    pad('二次', 8),
    pad('ready2', 8),
    pad('主API2', 7),
    pad('RQ缓存', 8),
    pad('阻塞API', 28),
  ].join(' | ');
  console.log(header);
  console.log('-'.repeat(header.length));

  for (const row of rows) {
    const primarySecond = row.second.apiCalls.find((c) => c.isPrimary);
    const line = [
      pad(row.page, 18),
      pad(formatMs(row.first.elapsedMs), 8),
      pad(formatMs(row.second.elapsedMs), 8),
      pad(row.second.perfReadyMs != null ? formatMs(row.second.perfReadyMs) : 'n/a', 8),
      pad(primarySecond ? formatMs(primarySecond.durationMs) : '0', 7),
      pad(row.queryCache, 8),
      pad(row.blocking.blockedBy.slice(0, 28), 28),
    ].join(' | ');
    console.log(line);
  }

  console.log('\n--- API p50/p95（二次进入）---');
  for (const row of rows) {
    const stats = summarizeApiCalls(row.second.apiCalls);
    console.log(`\n[${row.page}] blocking: ${row.blocking.note}`);
    if (stats.length === 0) {
      console.log('  (no api calls)');
      continue;
    }
    for (const stat of stats.slice(0, 6)) {
      console.log(`  ${stat.key} n=${stat.count} p50=${formatMs(stat.p50)} p95=${formatMs(stat.p95)}`);
    }
  }

  console.log('\n--- 详情（首次 / 二次）---');
  for (const row of rows) {
    console.log(`\n[${row.page}] queryCache=${row.queryCache}`);
    for (const label of ['first', 'second']) {
      const v = row[label];
      const blocking = label === 'second' ? row.blocking : analyzeBlocking(v, PAGES.find((p) => p.name === row.page)?.apiPrefix || '');
      console.log(
        `  ${label}: elapsed=${formatMs(v.elapsedMs)} primaryApi=${v.primaryApiCount} ready=${v.perfReadyMs ?? 'n/a'} blockedBy=${blocking.blockedBy}`
      );
      if (v.apiCalls.length > 0) {
        const top = [...v.apiCalls]
          .sort((a, b) => b.durationMs - a.durationMs)
          .slice(0, 5)
          .map((c) => `    ${c.method} ${c.path} ${c.status} ${formatMs(c.durationMs)}`)
          .join('\n');
        console.log(top);
      }
    }
  }
  console.log('');
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const { context, page } = await ensureAuthenticated(browser, DEFAULT_STORAGE);

  const report = [];
  for (const spec of PAGES) {
    const first = await sampleVisit(page, spec);
    await page.goto(`${BASE_URL}/workspace/overview`, { waitUntil: 'domcontentloaded' }).catch(() => {});
    await sleep(400);
    const second = await sampleVisit(page, spec);
    report.push({
      page: spec.name,
      first,
      second,
      queryCache: inferQueryCacheHit(first, second),
      blocking: analyzeBlocking(second, spec.apiPrefix),
    });
    await page.goto(`${BASE_URL}/workspace/overview`, { waitUntil: 'domcontentloaded' }).catch(() => {});
    await sleep(300);
  }

  printTable(modeArg, BASE_URL, report);

  if (process.env.RUN_STALE === '1') {
    console.log(`\n--- staleTime 采样 (${STALE_WAIT_MS}ms) ---`);
    await sleep(STALE_WAIT_MS);
    for (const spec of PAGES) {
      const afterStale = await sampleVisit(page, spec);
      console.log(
        `${spec.name} after stale: ${formatMs(afterStale.elapsedMs)} api=${afterStale.apiCount} perf=${afterStale.perfReady ?? 'n/a'}`
      );
    }
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
