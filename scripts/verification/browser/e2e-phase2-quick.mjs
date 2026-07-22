/**
 * 阶段2 快速自测：仅场景1、2、Hydration（不依赖登录/后端）
 * 运行: node scripts/verification/browser/e2e-phase2-quick.mjs
 */
import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:3001';
const LOGIN_URL = `${BASE_URL}/login`;

const results = [];
let hydrationWarnings = [];

function addResult(scenario, ok, problem, fix) {
  results.push({ scenario, result: ok ? '通过' : '失败', problem: problem || '-', fix: fix || '-' });
}

async function run() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL, viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();

  page.on('console', (msg) => {
    const text = msg.text();
    if (msg.type() === 'error' && (text.includes('hydrat') || text.includes('Hydration'))) {
      if (!hydrationWarnings.includes(text)) hydrationWarnings.push(text);
    }
  });

  try {
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
    await page.evaluate(() => localStorage.removeItem('epi_locale'));
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
    const initialOk = !scene1.epi_locale && (scene1.loginButtonText.includes('登录') || scene1.loginButtonText.includes('Sign in') || scene1.loginButtonText.includes('Logga in'));
    addResult('1) 清空 epi_locale→刷新，navigator与初始UI', initialOk, initialOk ? null : `navigator.language=${scene1.navigatorLanguage} epi_locale=${scene1.epi_locale} 按钮=${scene1.loginButtonText}`, null);

    const langOrder = ['简体中文', 'English', 'Svenska'];
    const expectedLocale = ['zh-CN', 'en', 'sv'];
    const expectedBtn = ['登录', 'Sign in', 'Logga in'];
    for (let i = 0; i < 3; i++) {
      await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });
      await page.evaluate((locale) => localStorage.setItem('epi_locale', locale), expectedLocale[i]);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      const after = await page.evaluate(() => ({
        epi_locale: localStorage.getItem('epi_locale'),
        loginButtonText: document.querySelector('.auth-card form button[type="submit"]')?.innerText?.trim() ?? '',
        agreementText: document.querySelector('.auth-card form p')?.innerText?.trim() ?? '',
      }));
      const localeOk = after.epi_locale === expectedLocale[i];
      const btnOk = after.loginButtonText.includes(expectedBtn[i]) || after.loginButtonText.includes('...');
      const ok = localeOk && btnOk;
      addResult(`2) 登录页语言切换-${langOrder[i]}`, ok, ok ? null : `epi_locale=${after.epi_locale}(期望${expectedLocale[i]}) 按钮=${after.loginButtonText}(期望含${expectedBtn[i]})`, !localeOk ? '确保 setStoredLocale 写入 epi_locale' : !btnOk ? 'hydrate 后 UI 需随 locale 更新，或延长等待' : null);
    }

    if (hydrationWarnings.length > 0) {
      addResult('Hydration mismatch warning', false, hydrationWarnings[0].slice(0, 400), '统一服务端与客户端 style 序列化');
    } else {
      addResult('Hydration mismatch warning', true, '-', '-');
    }
  } finally {
    await browser.close();
  }

  console.log('\n| 场景 | 结果 | 问题描述 | 修复建议(如有) |');
  console.log('|------|------|----------|----------------|');
  for (const r of results) {
    const problem = (r.problem || '-').replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 80);
    const fix = (r.fix || '-').replace(/\|/g, '\\|').slice(0, 60);
    console.log(`| ${r.scenario} | ${r.result} | ${problem} | ${fix} |`);
  }
  if (hydrationWarnings.length > 0) {
    console.log('\n--- Hydration 关键信息（可截图控制台）---');
    console.log(hydrationWarnings[0].slice(0, 600));
  }
}

run().catch((e) => { console.error(e); process.exit(1); });
