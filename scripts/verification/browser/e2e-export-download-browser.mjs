/**
 * 浏览器内验证：流式下载 + Content-Length 进度 + zip 完整性 + 401/空包（网络层模拟）
 *
 * 不依赖真实登录与导出任务：对 Next 站点注入 route 拦截 /api/data-assets/export/download。
 * 运行：node scripts/verification/browser/e2e-export-download-browser.mjs
 * 环境：BASE_URL 默认 http://127.0.0.1:3001；需 dev 服务可访问（任意页面即可发起同源 fetch）
 */
import { readFileSync, existsSync } from 'node:fs';
import { chromium } from 'playwright';

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:3001';
const SMALL_ZIP = '/tmp/e2e_small.zip';
const LARGE_ZIP = '/tmp/e2e_large.zip';

async function streamDownloadInPage(page, label) {
  return page.evaluate(async ({ label: _label }) => {
    const log = [];
    const res = await fetch('/api/data-assets/export/download?jobId=e2e-mock', {
      credentials: 'include',
    });
    if (!res.ok) {
      const text = await res.text();
      let err = `HTTP ${res.status}`;
      try {
        const j = JSON.parse(text);
        if (j.error) err = j.error;
        else if (j.detail) err = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      } catch {
        if (text) err = text.slice(0, 200);
      }
      return { ok: false, label: _label, error: err, status: res.status };
    }
    const cl = res.headers.get('Content-Length');
    const total = cl ? parseInt(cl, 10) : null;
    log.push({ ev: 'headers', loaded: 0, total, disposition: res.headers.get('Content-Disposition') });
    const body = res.body;
    if (!body) {
      const blob = await res.blob();
      log.push({ ev: 'blob_fallback', size: blob.size });
      return {
        ok: true,
        label: _label,
        mode: 'blob',
        loaded: blob.size,
        total,
        log,
        bytes: Array.from(new Uint8Array(await blob.arrayBuffer())),
      };
    }
    const reader = body.getReader();
    const chunks = [];
    let loaded = 0;
    let maxPct = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loaded += value.byteLength;
        const pct = total && total > 0 ? Math.round((loaded / total) * 100) : null;
        if (pct != null) maxPct = Math.max(maxPct, pct);
        if (log.length < 12 || done) log.push({ ev: 'chunk', loaded, total, pct });
      }
    }
    const blob = new Blob(chunks, { type: 'application/zip' });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return {
      ok: true,
      label: _label,
      mode: 'stream',
      loaded,
      total,
      maxPct,
      blobSize: blob.size,
      logLen: log.length,
      log,
      bytes: Array.from(bytes),
    };
  }, { label });
}

function verifyZipMagic(bytes) {
  if (bytes.length < 4) return false;
  return bytes[0] === 0x50 && bytes[1] === 0x4b;
}

async function runBrowser(channel, name) {
  const launchOpts = { headless: true };
  if (channel) launchOpts.channel = channel;
  const browser = await chromium.launch(launchOpts);
  const context = await browser.newContext({ baseURL: BASE_URL });
  const page = await context.newPage();

  /** 去掉 FSA，强制走 Blob 回退（与 Safari/禁用权限等场景一致） */
  await page.addInitScript(() => {
    try {
      delete window.showSaveFilePicker;
    } catch {
      /* ignore */
    }
  });

  const smallBuf = readFileSync(SMALL_ZIP);
  const largeBuf = existsSync(LARGE_ZIP) ? readFileSync(LARGE_ZIP) : smallBuf;

  const results = {
    browser: name,
    small: null,
    large: null,
    unauthorized: null,
    empty: null,
    paintDefer: null,
  };

  await page.route('**/api/data-assets/export/download**', async (route) => {
    const u = route.request().url();
    if (u.includes('jobId=e2e-401')) {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ ok: false, error: 'Missing Authorization header or refresh cookie' }),
      });
      return;
    }
    if (u.includes('jobId=e2e-empty')) {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/zip',
          'Content-Disposition': 'attachment; filename="empty.zip"',
          'Content-Length': '0',
        },
        body: '',
      });
      return;
    }
    const useLarge = u.includes('jobId=e2e-large');
    const buf = useLarge ? largeBuf : smallBuf;
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename="mock.zip"',
        'Content-Length': String(buf.length),
      },
      body: buf,
    });
  });

  await page.goto(`${BASE_URL}/login`, { waitUntil: 'domcontentloaded', timeout: 20000 }).catch(() => {});

  results.small = await streamDownloadInPage(page, 'small');
  results.small.zipMagic = results.small.ok && verifyZipMagic(results.small.bytes);
  delete results.small.bytes;

  results.large = await page.evaluate(async () => {
    const res = await fetch('/api/data-assets/export/download?jobId=e2e-large', { credentials: 'include' });
    const total = parseInt(res.headers.get('Content-Length') || '0', 10);
    const reader = res.body.getReader();
    let loaded = 0;
    let chunks = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        loaded += value.byteLength;
        chunks += 1;
      }
    }
    return { ok: res.ok, status: res.status, loaded, total, chunks };
  });

  results.unauthorized = await page.evaluate(async () => {
    const res = await fetch('/api/data-assets/export/download?jobId=e2e-401', { credentials: 'include' });
    const text = await res.text();
    let msg = `HTTP ${res.status}`;
    try {
      const j = JSON.parse(text);
      if (j.error) msg = j.error;
    } catch {
      /* ignore */
    }
    return { status: res.status, msg, isJson: text.trim().startsWith('{') };
  });

  results.empty = await page.evaluate(async () => {
    const res = await fetch('/api/data-assets/export/download?jobId=e2e-empty', { credentials: 'include' });
    const reader = res.body?.getReader();
    let loaded = 0;
    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) loaded += value.byteLength;
      }
    } else {
      const b = await res.blob();
      loaded = b.size;
    }
    return { ok: res.ok, status: res.status, loaded };
  });

  /** 双 rAF 后是否在同帧前让出绘制（仅测 API 存在与计数） */
  results.paintDefer = await page.evaluate(() => {
    return new Promise((resolve) => {
      let raf1 = false;
      let raf2 = false;
      requestAnimationFrame(() => {
        raf1 = true;
        requestAnimationFrame(() => {
          raf2 = true;
          resolve({ raf1, raf2 });
        });
      });
    });
  });

  await browser.close();
  return results;
}

async function main() {
  if (!existsSync(SMALL_ZIP)) {
    console.error('缺少', SMALL_ZIP, '请先: echo test > /tmp/e2e_one.txt && zip -qj /tmp/e2e_small.zip /tmp/e2e_one.txt');
    process.exit(1);
  }

  const out = {
    baseUrl: BASE_URL,
    chromium: await runBrowser(undefined, 'Chromium (bundled)'),
  };

  try {
    out.edge = await runBrowser('msedge', 'Microsoft Edge');
  } catch (e) {
    out.edge = { skipped: true, reason: String(e.message || e) };
  }

  console.log(JSON.stringify(out, null, 2));

  const sm = out.chromium.small;
  const okSmall =
    sm &&
    sm.ok &&
    sm.zipMagic &&
    sm.loaded === sm.total &&
    sm.total > 50 &&
    (sm.maxPct === 100 || sm.mode === 'blob');

  const lg = out.chromium.large;
  const okLarge = lg && lg.ok && lg.loaded === lg.total && lg.total > 1_000_000 && lg.chunks >= 1;

  const ok401 = out.chromium.unauthorized?.status === 401 && typeof out.chromium.unauthorized.msg === 'string';

  const okEmpty = out.chromium.empty?.ok && out.chromium.empty.loaded === 0;

  const pass = okSmall && okLarge && ok401 && okEmpty;
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
