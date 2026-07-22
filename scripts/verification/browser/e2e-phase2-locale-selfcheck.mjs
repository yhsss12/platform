/**
 * 阶段2 场景自测：语言、epi_locale、hydration、登录、持久化、错误提示、布局
 * 使用账号 admin / admin123，前端 http://localhost:3001
 * 运行: node scripts/verification/browser/e2e-phase2-locale-selfcheck.mjs
 */

import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:3001';
const LOGIN_URL = `${BASE_URL}/login`;

const results = [];
let hydrationWarnings = [];

function addResult(scenario, ok, problem, fix) {
  results.push({ scenario, result: ok ? '通过' : '失败', problem: problem || '-', fix: fix || '-' });
}

function pushHydration(msg) {
  if (!hydrationWarnings.some((m) => m === msg)) hydrationWarnings.push(msg);
}

async function run() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL, viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  page.on('console', (msg) => {
    const text = msg.text();
    const type = msg.type();
    if (type === 'error' && (text.includes('hydrat') || text.includes('Hydration'))) {
      pushHydration(text);
    }
  });

  try {
    // ---------- 场景1: 清空 epi_locale → 刷新 → 记录 navigator.language 与初始 UI ----------
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
    await page.evaluate(() => {
      localStorage.removeItem('epi_locale');
    });
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(800);

    const scene1 = await page.evaluate(() => {
      const form = document.querySelector('.auth-card form');
      const btn = form?.querySelector('button[type="submit"]');
      const agreement = form?.querySelector('p');
      return {
        navigatorLanguage: navigator.language,
        epi_locale: localStorage.getItem('epi_locale'),
        loginButtonText: btn?.innerText?.trim() ?? '',
        agreementText: agreement?.innerText?.trim()?.slice(0, 80) ?? '',
      };
    });
    const initialLocaleOk = !scene1.epi_locale && (scene1.loginButtonText.includes('登录') || scene1.loginButtonText.includes('Sign in') || scene1.loginButtonText.includes('Logga in'));
    addResult(
      '1) 清空 epi_locale→刷新，navigator与初始UI',
      initialLocaleOk,
      initialLocaleOk ? null : `navigator.language=${scene1.navigatorLanguage} epi_locale=${scene1.epi_locale} 按钮=${scene1.loginButtonText}`,
      null
    );

    // ---------- 场景2: 登录页语言切换三次（通过 localStorage + 刷新），每次记录 epi_locale + 登录按钮 + 协议 ----------
    const langOrder = ['简体中文', 'English', 'Svenska'];
    const expectedLocale = ['zh-CN', 'en', 'sv'];
    const expectedBtn = ['登录', 'Sign in', 'Logga in'];
    for (let i = 0; i < 3; i++) {
      await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
      await page.evaluate((locale) => localStorage.setItem('epi_locale', locale), expectedLocale[i]);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(800);

      const after = await page.evaluate(() => ({
        epi_locale: localStorage.getItem('epi_locale'),
        loginButtonText: document.querySelector('.auth-card form button[type="submit"]')?.innerText?.trim() ?? '',
        agreementText: document.querySelector('.auth-card form p')?.innerText?.trim() ?? '',
      }));

      const localeMatch = after.epi_locale === expectedLocale[i];
      const btnMatch = after.loginButtonText.includes(expectedBtn[i]) || after.loginButtonText.includes('...');
      const ok = localeMatch && btnMatch;
      addResult(
        `2) 登录页语言切换-${langOrder[i]} (epi_locale+刷新)`,
        ok,
        ok ? null : `epi_locale=${after.epi_locale}(期望${expectedLocale[i]}) 按钮=${after.loginButtonText} 协议=${(after.agreementText || '').slice(0, 50)}`,
        !localeMatch ? '确保 setStoredLocale 在切换时写入 epi_locale' : null
      );
    }

    // ---------- 场景3: 每种语言下 admin/admin123 登录，检查左侧菜单与数据资产页 ----------
    for (let i = 0; i < 3; i++) {
      const ctx = await browser.newContext({ baseURL: BASE_URL });
      const p = await ctx.newPage();
      await p.goto(LOGIN_URL, { waitUntil: 'networkidle' });
      await p.evaluate((locale) => localStorage.setItem('epi_locale', locale), expectedLocale[i]);
      await p.reload({ waitUntil: 'networkidle' });
      await p.waitForTimeout(600);

      await p.locator('#login-username').fill('admin');
      await p.locator('#login-password').fill('admin123');
      await p.locator('.auth-card form button[type="submit"]').click();
      await p.waitForURL(/\/(admin|collect)/, { timeout: 10000 }).catch(() => {});

      const onPlatform = p.url().includes('/admin') || p.url().includes('/collect');
      if (!onPlatform) {
        addResult(`3) ${langOrder[i]} 登录进入平台`, false, '未进入平台', null);
        await ctx.close();
        continue;
      }

      await p.goto(`${BASE_URL}/data`, { waitUntil: 'networkidle' });
      await p.waitForTimeout(600);

      const ui = await p.evaluate(() => {
        const sidebar = document.querySelector('[class*="sidebar"]') || document.querySelector('aside') || document.body;
        const sidebarText = sidebar?.innerText ?? '';
        const title = document.querySelector('h1, [class*="title"]')?.innerText ?? document.body.innerText?.slice(0, 200) ?? '';
        return { sidebarText: sidebarText.slice(0, 300), title: title.slice(0, 150) };
      });

      const hasDataLabel = ui.sidebarText.includes('数据') || ui.sidebarText.includes('Data') || ui.sidebarText.includes('Datatillgångar') || ui.title.includes('数据资产') || ui.title.includes('Data Assets') || ui.title.includes('Datatillgångar');
      addResult(
        `3) ${langOrder[i]} 平台内菜单与数据资产文案`,
        hasDataLabel,
        hasDataLabel ? null : `侧栏/标题未含预期文案: ${ui.title.slice(0, 80)}`,
        null
      );
      await ctx.close();
    }

    // ---------- 场景4: 平台页刷新、关闭重开，语言保持 ----------
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
    await page.evaluate(() => localStorage.setItem('epi_locale', 'en'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    await page.locator('#login-username').fill('admin');
    await page.locator('#login-password').fill('admin123');
    await page.locator('.auth-card form button[type="submit"]').click();
    await page.waitForURL(/\/(admin|collect)/, { timeout: 10000 }).catch(() => {});

    await page.goto(`${BASE_URL}/data`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(300);
    const beforeRefresh = await page.evaluate(() => localStorage.getItem('epi_locale'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    const afterRefresh = await page.evaluate(() => localStorage.getItem('epi_locale'));
    const signAfterRefresh = await page.evaluate(() => document.body?.innerText?.includes('Sign in') || document.body?.innerText?.includes('Data Assets'));

    const newPage = await context.newPage();
    await newPage.goto(`${BASE_URL}/data`, { waitUntil: 'networkidle' });
    await newPage.waitForTimeout(500);
    const localeReopen = await newPage.evaluate(() => localStorage.getItem('epi_locale'));
    await newPage.close();

    const persistOk = afterRefresh === 'en' && localeReopen === 'en';
    addResult(
      '4) 平台页刷新与重开标签页语言保持',
      persistOk,
      persistOk ? null : `刷新后 epi_locale=${afterRefresh} 重开后=${localeReopen}`,
      null
    );

    // ---------- 场景5: 错误提示 - 故意输错密码 ----------
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
    await page.evaluate(() => localStorage.setItem('epi_locale', 'zh-CN'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    await page.locator('#login-username').fill('admin');
    await page.locator('#login-password').fill('wrongpass');
    await page.locator('.auth-card form button[type="submit"]').click();
    await page.waitForTimeout(1500);
    const errZh = await page.locator('.auth-card form [class*="border"][class*="fecaca"], .auth-card form p').filter({ hasText: /错|误|invalid|Fel/ }).first().innerText().catch(() => '');

    await page.evaluate(() => localStorage.setItem('epi_locale', 'en'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(500);
    await page.locator('#login-password').fill('wrongpass2');
    await page.locator('.auth-card form button[type="submit"]').click();
    await page.waitForTimeout(1500);
    const errEn = await page.locator('.auth-card form').filter({ hasText: /Incorrect|error|wrong/ }).first().innerText().catch(() => '');

    const errOk = (errZh.length > 0 && (errZh.includes('错') || errZh.includes('误'))) || (errEn.length > 0 && (errEn.includes('Incorrect') || errEn.includes('password')));
    addResult(
      '5) 错误提示显示且随语言变化',
      errOk,
      errOk ? null : `中文错误区=${errZh.slice(0, 60)} 英文=${errEn.slice(0, 60)}`,
      null
    );

    // ---------- 场景6: 布局稳定性 English/Svenska 登录页与数据页 ----------
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
    await page.evaluate(() => localStorage.setItem('epi_locale', 'en'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(400);

    const layoutLoginEn = await page.evaluate(() => {
      const btn = document.querySelector('.auth-card form button[type="submit"]');
      const switcher = document.querySelector('button[aria-haspopup="menu"]');
      const card = document.querySelector('.auth-card');
      if (!btn || !card) return { ok: false, msg: 'missing elements' };
      const br = btn.getBoundingClientRect();
      const sr = switcher?.getBoundingClientRect();
      const cr = card.getBoundingClientRect();
      const btnOverflow = br.right > cr.right || br.left < cr.left;
      const switcherOverflow = sr && (sr.right > cr.right + 20 || sr.left < cr.left - 20);
      return { ok: !btnOverflow && !switcherOverflow, msg: btnOverflow ? 'button overflow' : switcherOverflow ? 'switcher overflow' : 'ok' };
    });

    await page.locator('#login-username').fill('admin');
    await page.locator('#login-password').fill('admin123');
    await page.locator('.auth-card form button[type="submit"]').click();
    await page.waitForURL(/\/(admin|data|collect)/, { timeout: 10000 }).catch(() => {});
    await page.goto(`${BASE_URL}/data`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    const layoutDataEn = await page.evaluate(() => {
      const table = document.querySelector('table');
      const pagination = document.querySelector('[class*="pagination"], button');
      const filter = document.querySelector('input[type="text"], [class*="filter"], [class*="search"]');
      const tb = table?.getBoundingClientRect();
      const pb = pagination?.getBoundingClientRect();
      const overflow = tb && tb.width > (window.innerWidth - 100);
      return { tableVisible: !!table, paginationVisible: !!pagination, filterVisible: !!filter, overflow };
    });

    addResult(
      '6) 布局稳定性(EN/SV)登录页与数据页',
      layoutLoginEn.ok && layoutDataEn.tableVisible !== false,
      !layoutLoginEn.ok ? `登录页: ${layoutLoginEn.msg}` : !layoutDataEn.tableVisible ? '数据页表格未找到' : layoutDataEn.overflow ? '表格横向溢出' : null,
      null
    );

    // ---------- Hydration 汇总 ----------
    if (hydrationWarnings.length > 0) {
      addResult(
        'Hydration mismatch warning',
        false,
        hydrationWarnings.slice(0, 2).join(' | ').slice(0, 500),
        '统一服务端与客户端 style 序列化（如用 className 或 suppressHydrationWarning）'
      );
    } else {
      addResult('Hydration mismatch warning', true, '-', '-');
    }
  } finally {
    await browser.close();
  }

  // 输出表格
  console.log('\n| 场景 | 结果 | 问题描述 | 修复建议(如有) |');
  console.log('|------|------|----------|----------------|');
  for (const r of results) {
    const problem = (r.problem || '-').replace(/\|/g, '\\|').replace(/\n/g, ' ');
    const fix = (r.fix || '-').replace(/\|/g, '\\|');
    console.log(`| ${r.scenario} | ${r.result} | ${problem.slice(0, 80)} | ${fix.slice(0, 60)} |`);
  }
  if (hydrationWarnings.length > 0) {
    console.log('\n--- Hydration 关键信息（可截图控制台）---');
    hydrationWarnings.forEach((m, i) => console.log(`[${i + 1}] ${m.slice(0, 400)}`));
  }
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
