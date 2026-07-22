/**
 * Third-batch i18n verification: login page language init + logged-in page checks.
 * 1) Login page: epi_locale zh-CN / en / sv, reload, assert first-screen text.
 * 2) Login with LOGIN_USER / LOGIN_PASS (default admin / admin123), then verify
 *    /overview, /datasets, /jobs, /runs, /tasks with locale and assertions.
 * Run: LOGIN_USER=admin LOGIN_PASS=admin123 node scripts/verification/browser/verify-i18n-batch3.mjs
 * Requires: npm run dev (or next start) on port 3000, backend with valid user.
 */
import { chromium } from 'playwright';

const BASE = process.env.BASE_URL || 'http://localhost:3000';
const LOGIN_USER = process.env.LOGIN_USER || 'admin';
const LOGIN_PASS = process.env.LOGIN_PASS || 'admin123';

const CORE_ROUTES = ['/overview', '/datasets', '/jobs', '/runs', '/tasks'];

const LOGIN_EXPECT = {
  'zh-CN': { re: /登录|账号|密码/, name: 'ZH' },
  en: { re: /Login|Account|Password/i, name: 'EN' },
  sv: { re: /Logga in|Användarnamn|Lösenord/i, name: 'SV' },
};

const PAGE_TITLES = {
  '/overview': { 'zh-CN': /概览/, en: /Overview/i, sv: /Översikt/i },
  '/datasets': { 'zh-CN': /数据集/, en: /Datasets/i, sv: /Dataset/i },
  '/jobs': { 'zh-CN': /Jobs/, en: /Jobs/i, sv: /Jobs/i },
  '/runs': { 'zh-CN': /Runs/, en: /Runs/i, sv: /Runs/i },
  '/tasks': { 'zh-CN': /任务管理/, en: /Task management/i, sv: /Uppgiftshantering/i },
};

function reportSection(title, lines) {
  console.log('\n## ' + title + '\n');
  (Array.isArray(lines) ? lines : [lines]).forEach((l) => console.log(l));
}

async function main() {
  const results = {
    login: { zh: null, en: null, sv: null, persisted: null, consoleErrors: [] },
    loggedInPages: {},
    consoleErrors: [],
  };
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ locale: 'en' });
  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      results.consoleErrors.push({ url: page.url(), text: msg.text() });
    }
  });

  try {
    // ---------- 1) Login page: set epi_locale, reload, wait for form, assert first-screen text ----------
    for (const [localeKey, expect] of Object.entries(LOGIN_EXPECT)) {
      await page.goto(BASE + '/login', { waitUntil: 'networkidle', timeout: 15000 });
      await page.evaluate((key) => localStorage.setItem('epi_locale', key), localeKey);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      const body = await page.locator('body').innerText();
      const ok = expect.re.test(body);
      results.login[localeKey === 'zh-CN' ? 'zh' : localeKey === 'en' ? 'en' : 'sv'] = ok;
    }
    results.login.persisted = await page.evaluate(() => localStorage.getItem('epi_locale'));

    // ---------- 2) Login: wait for form to appear (after mounted) ----------
    await page.goto(BASE + '/login', { waitUntil: 'networkidle', timeout: 20000 });
    await page.waitForFunction(
      () => {
        const body = document.body?.innerText || '';
        return /Login|Logga in|登录/.test(body) && (document.querySelector('#login-username') || document.querySelector('input[type="text"]'));
      },
      { timeout: 20000 }
    );
    await page.waitForTimeout(400);
    const userSel = '#login-username';
    const passSel = '#login-password';
    if (await page.locator(userSel).count()) {
      await page.fill(userSel, LOGIN_USER);
      await page.fill(passSel, LOGIN_PASS);
    } else {
      await page.fill('input[type="text"]', LOGIN_USER);
      await page.fill('input[type="password"]', LOGIN_PASS);
    }
    await page.locator('button[type="submit"]').click();
    await page.waitForURL((u) => !u.pathname.includes('/login'), { timeout: 15000 }).catch(() => {});
    const afterLoginUrl = page.url();
    const isLoggedIn = !afterLoginUrl.includes('/login');

    if (!isLoggedIn) {
      reportSection('Login result', 'Login failed or still on /login. Skip logged-in page checks.');
    } else {
      // ---------- 3) For each core route: set zh-CN / en / sv, visit, assert ----------
      for (const route of CORE_ROUTES) {
        const pageResult = { threeLang: { zh: null, en: null, sv: null }, keyStrings: false, rawEnum: false, residualZh: false, consoleOnRoute: 0 };
        for (const localeKey of ['zh-CN', 'en', 'sv']) {
          await page.evaluate((key) => localStorage.setItem('epi_locale', key), localeKey);
          await page.goto(BASE + route, { waitUntil: 'networkidle', timeout: 12000 });
          await page.waitForTimeout(400);
          const path = page.url().replace(BASE, '').split('?')[0];
          if (path.includes('/login')) {
            pageResult.threeLang[localeKey === 'zh-CN' ? 'zh' : localeKey === 'en' ? 'en' : 'sv'] = 'redirected';
            continue;
          }
          const body = await page.locator('body').innerText();
          const titles = PAGE_TITLES[route];
          const re = titles && titles[localeKey];
          const titleOk = re ? re.test(body) : true;
          pageResult.threeLang[localeKey === 'zh-CN' ? 'zh' : localeKey === 'en' ? 'en' : 'sv'] = titleOk ? 'ok' : 'fail';
          if (localeKey === 'zh-CN') {
            pageResult.keyStrings = /overviewPage\.|datasetsPage\.|dashboard\.|shellPage\.|common\./.test(body);
            pageResult.rawEnum = /\b(PENDING|RUNNING|COMPLETED|PAUSED|FAILED)\b/.test(body);
            pageResult.residualZh = /暂无|加载中|新建|搜索|任务管理|概览|导出|图表|字典|技能|项目：/.test(body);
          }
        }
        const errs = results.consoleErrors.filter((e) => e.url.includes(route));
        pageResult.consoleOnRoute = errs.length;
        results.loggedInPages[route] = pageResult;
      }
    }

    // ---------- 4) Report ----------
    reportSection('1) Login page first-screen language', [
      'zh-CN: ' + (results.login.zh ? 'OK' : 'FAIL'),
      'en: ' + (results.login.en ? 'OK' : 'FAIL'),
      'sv: ' + (results.login.sv ? 'OK' : 'FAIL'),
      'epi_locale persisted: ' + results.login.persisted,
    ]);
    reportSection('2) Login result', [
      'After submit URL: ' + afterLoginUrl,
      'Logged in: ' + isLoggedIn,
    ]);
    if (isLoggedIn) {
      reportSection('3) Logged-in core pages', []);
      for (const route of CORE_ROUTES) {
        const r = results.loggedInPages[route] || {};
        console.log(route + ':');
        console.log('  threeLang zh/en/sv: ' + [r.threeLang?.zh, r.threeLang?.en, r.threeLang?.sv].join(', '));
        console.log('  keyStrings: ' + (r.keyStrings ? 'YES' : 'no') + ', rawEnum: ' + (r.rawEnum ? 'YES' : 'no') + ', residualZh: ' + (r.residualZh ? 'YES' : 'no'));
        console.log('  consoleErrors on route: ' + (r.consoleOnRoute || 0));
        console.log('');
      }
    }
    const non404 = results.consoleErrors.filter((e) => !e.text.includes('404'));
    if (non404.length) reportSection('Console errors (excluding 404)', non404.slice(0, 8).map((e) => e.url + ' => ' + e.text));
  } catch (err) {
    console.error('Script error:', err.message);
  } finally {
    await browser.close();
  }
}

main();
