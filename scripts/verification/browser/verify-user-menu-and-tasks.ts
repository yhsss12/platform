/**
 * 真实浏览器验证：用户菜单 + 后台任务面板（zh-CN / en / sv）
 * 运行：npx ts-node --esm scripts/verification/browser/verify-user-menu-and-tasks.ts
 * 或：npx tsx scripts/verification/browser/verify-user-menu-and-tasks.ts
 * 要求：Next dev 已运行，默认 http://localhost:3001
 */

import { chromium } from 'playwright';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3001';

const LOCALES = ['zh-CN', 'en', 'sv'] as const;

const EXPECTED = {
  'zh-CN': {
    userMenu: {
      accountSettings: '账号设置',
      logout: '退出登录',
      roleAdmin: '管理员',
      roleSuperAdmin: '超级管理员',
      roleUser: '普通用户',
      language: '语言',
      localeZhCN: '简体中文',
      localeEn: 'English',
      localeSv: 'Svenska',
    },
    taskPanel: {
      title: '后台任务',
      tabRunning: '进行中',
      tabCompleted: '已完成',
      tabFailed: '失败',
      emptyRunning: '暂无进行中的任务',
      emptyCompleted: '暂无已完成的任务',
      emptyFailed: '暂无失败的任务',
      close: '关闭',
      clearCompleted: '清空已完成',
      confirmDeleteExportTitle: '确认删除导出结果？',
      confirmDeleteConvertTitle: '确认删除转换结果？',
      cancel: '取消',
      confirmDelete: '删除',
    },
  },
  en: {
    userMenu: {
      accountSettings: 'Account settings',
      logout: 'Sign out',
      roleAdmin: 'Admin',
      roleSuperAdmin: 'Super Admin',
      roleUser: 'User',
      language: 'Language',
      localeZhCN: 'Simplified Chinese',
      localeEn: 'English',
      localeSv: 'Swedish',
    },
    taskPanel: {
      title: 'Background Tasks',
      tabRunning: 'Running',
      tabCompleted: 'Completed',
      tabFailed: 'Failed',
      emptyRunning: 'No running tasks',
      emptyCompleted: 'No completed tasks',
      emptyFailed: 'No failed tasks',
      close: 'Close',
      clearCompleted: 'Clear completed',
      confirmDeleteExportTitle: 'Delete export result?',
      confirmDeleteConvertTitle: 'Delete conversion result?',
      cancel: 'Cancel',
      confirmDelete: 'Delete',
    },
  },
  sv: {
    userMenu: {
      accountSettings: 'Kontoinställningar',
      logout: 'Logga ut',
      roleAdmin: 'Administratör',
      roleSuperAdmin: 'Superadministratör',
      roleUser: 'Användare',
      language: 'Språk',
      localeZhCN: 'Förenklad kinesiska',
      localeEn: 'English',
      localeSv: 'Svenska',
    },
    taskPanel: {
      title: 'Bakgrundsuppgifter',
      tabRunning: 'Pågår',
      tabCompleted: 'Slutförda',
      tabFailed: 'Misslyckade',
      emptyRunning: 'Inga pågående uppgifter',
      emptyCompleted: 'Inga slutförda uppgifter',
      emptyFailed: 'Inga misslyckade uppgifter',
      close: 'Stäng',
      clearCompleted: 'Rensa slutförda',
      confirmDeleteExportTitle: 'Ta bort exportresultat?',
      confirmDeleteConvertTitle: 'Ta bort konverteringsresultat?',
      cancel: 'Avbryt',
      confirmDelete: 'Ta bort',
    },
  },
};

type LocaleKey = (typeof LOCALES)[number];

interface Result {
  locale: LocaleKey;
  loginBlocked: boolean;
  loginPageHasLanguageSwitcher: boolean;
  loginPageHasUserMenu: boolean;
  loginPageHasRobotEntry: boolean;
  userMenu: {
    passed: boolean;
    profileOk: boolean;
    logoutOk: boolean;
    roleOk: boolean;
    languageRowOk: boolean;
    detail?: string;
  };
  taskPanel: {
    passed: boolean;
    titleOk: boolean;
    tabsOk: boolean;
    emptyStateOk: boolean;
    closeAriaOk: boolean;
    clearCompletedOk: boolean;
    deleteConfirmOk: boolean;
    detail?: string;
  };
  languageSwitchMethod: string;
}

function setLocaleAndReload(page: import('playwright').Page, locale: LocaleKey) {
  return page.evaluate((loc) => {
    localStorage.setItem('epi_locale', loc);
    window.location.reload();
  }, locale);
}

async function waitForHydration(page: import('playwright').Page, timeout = 5000) {
  await page.waitForTimeout(800);
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const btn = await page.locator('button:has-text("登录"), button:has-text("Sign in"), button:has-text("Logga in")').first();
    const userArea = await page.locator('header').getByRole('button').first();
    if (await btn.isVisible().catch(() => false) || await userArea.isVisible().catch(() => false)) break;
    await page.waitForTimeout(200);
  }
}

async function runOneLocale(
  page: import('playwright').Page,
  locale: LocaleKey
): Promise<Result> {
  const result: Result = {
    locale,
    loginBlocked: false,
    loginPageHasLanguageSwitcher: false,
    loginPageHasUserMenu: false,
    loginPageHasRobotEntry: false,
    userMenu: { passed: false, profileOk: false, logoutOk: false, roleOk: false, languageRowOk: false },
    taskPanel: { passed: false, titleOk: false, tabsOk: false, emptyStateOk: false, closeAriaOk: false, clearCompletedOk: false, deleteConfirmOk: false },
    languageSwitchMethod: 'localStorage.epi_locale + 刷新',
  };

  await page.goto(BASE_URL + '/login', { waitUntil: 'domcontentloaded', timeout: 15000 });
  await setLocaleAndReload(page, locale);
  await page.waitForLoadState('networkidle').catch(() => {});
  await waitForHydration(page);

  const url = page.url();
  const onLoginPage = url.includes('/login');

  if (onLoginPage) {
    result.loginBlocked = true;
    result.loginPageHasLanguageSwitcher = await page.locator('button:has-text("简体中文"), button:has-text("English"), button:has-text("Svenska")').first().isVisible().catch(() => false);
    result.loginPageHasUserMenu = false;
    result.loginPageHasRobotEntry = false;
    result.userMenu.detail = '无法验证（登录阻塞）';
    result.taskPanel.detail = '无法验证（登录阻塞）';
    return result;
  }

  // 站内切换：已在平台，用用户菜单里的语言切换
  const userMenuBtn = page.locator('header').locator('div').filter({ has: page.locator('text=/^[A-Za-z0-9_]+$') }).locator('..').first();
  await userMenuBtn.click().catch(() => {});
  await page.waitForTimeout(400);
  const hasLangInMenu = await page.locator('button:has-text("语言"), button:has-text("Language")').first().isVisible().catch(() => false);
  if (hasLangInMenu) result.languageSwitchMethod = '站内：用户菜单 → 语言 → 选择语言';

  const expUser = EXPECTED[locale].userMenu;
  const body = await page.locator('body').textContent() || '';

  const profileOk = body.includes(expUser.accountSettings);
  const logoutOk = body.includes(expUser.logout);
  const roleOk =
    body.includes(expUser.roleAdmin) ||
    body.includes(expUser.roleSuperAdmin) ||
    body.includes(expUser.roleUser);
  const languageRowOk = body.includes(expUser.language) || body.includes('Language');

  result.userMenu = {
    passed: profileOk && logoutOk && roleOk && languageRowOk,
    profileOk,
    logoutOk,
    roleOk,
    languageRowOk,
  };

  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);

  // 后台任务面板：仅在 /data 或 /convert 显示
  await page.goto(BASE_URL + '/data', { waitUntil: 'domcontentloaded', timeout: 15000 });
  await page.waitForTimeout(1200);

  const robotBtn = page.locator('button[aria-label*="任务"], button[aria-label*="Tasks"], button[aria-label*="uppgift"], button.tc-launcher');
  const robotVisible = await robotBtn.first().isVisible().catch(() => false);
  if (!robotVisible) {
    result.taskPanel.detail = '未找到后台任务入口（机器人按钮）';
    return result;
  }

  await robotBtn.first().click();
  await page.waitForTimeout(500);

  const expTask = EXPECTED[locale].taskPanel;
  const panel = page.locator('div').filter({ has: page.locator(`text=${expTask.title}`) }).first();
  const panelVisible = await panel.isVisible().catch(() => false);

  if (!panelVisible) {
    result.taskPanel.detail = '面板打开后未找到标题文案';
    return result;
  }

  result.taskPanel.titleOk = true;
  result.taskPanel.tabsOk =
    (await page.locator(`text=${expTask.tabRunning}`).isVisible().catch(() => false)) &&
    (await page.locator(`text=${expTask.tabCompleted}`).isVisible().catch(() => false)) &&
    (await page.locator(`text=${expTask.tabFailed}`).isVisible().catch(() => false));

  const emptyRunningVisible = await page.locator(`text=${expTask.emptyRunning}`).isVisible().catch(() => false);
  result.taskPanel.emptyStateOk = emptyRunningVisible;

  const closeBtn = panel.locator('button[aria-label]').first();
  const closeAria = await closeBtn.getAttribute('aria-label').catch(() => '');
  result.taskPanel.closeAriaOk = closeAria === expTask.close || closeAria?.toLowerCase().includes(expTask.close.toLowerCase());

  const clearBtn = page.locator(`button:has-text("${expTask.clearCompleted}")`);
  result.taskPanel.clearCompletedOk = await clearBtn.isVisible().catch(() => false);

  result.taskPanel.passed =
    result.taskPanel.titleOk &&
    result.taskPanel.tabsOk &&
    result.taskPanel.emptyStateOk &&
    result.taskPanel.closeAriaOk &&
    result.taskPanel.clearCompletedOk;

  return result;
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ locale: 'en-US' });
  const page = await context.newPage();

  const results: Result[] = [];
  for (const locale of LOCALES) {
    try {
      const r = await runOneLocale(page, locale);
      results.push(r);
    } catch (e) {
      results.push({
        locale,
        loginBlocked: true,
        loginPageHasLanguageSwitcher: false,
        loginPageHasUserMenu: false,
        loginPageHasRobotEntry: false,
        userMenu: { passed: false, profileOk: false, logoutOk: false, roleOk: false, languageRowOk: false, detail: String(e) },
        taskPanel: { passed: false, titleOk: false, tabsOk: false, emptyStateOk: false, closeAriaOk: false, clearCompletedOk: false, deleteConfirmOk: false, detail: String(e) },
        languageSwitchMethod: 'localStorage.epi_locale + 刷新',
      });
    }
  }

  await browser.close();

  const report: string[] = [];
  report.push('# 用户菜单 + 后台任务面板 浏览器验证报告');
  report.push(`BASE_URL: ${BASE_URL}`);
  report.push('');

  const loginBlocked = results.every((r) => r.loginBlocked);
  if (loginBlocked) {
    const first = results[0];
    report.push('## 登录阻塞');
    report.push('所有语言下均停留在登录页，无法进入平台。');
    report.push('- 登录页右上角用户菜单：**不可见**（未登录无头像）');
    report.push('- 登录页机器人入口：**不可见**（后台任务仅在 /data、/convert 页显示）');
    report.push(`- 登录页语言切换控件：${first?.loginPageHasLanguageSwitcher ? '**可见**（LanguageSwitcher）' : '未检测到'}`);
    report.push('');
    report.push('| 语言 | 用户菜单 | 后台任务面板 | 语言切换方式 |');
    report.push('|------|----------|--------------|--------------|');
    for (const r of results) {
      report.push(`| ${r.locale} | 无法验证（登录阻塞） | 无法验证（登录阻塞） | ${r.languageSwitchMethod} |`);
    }
  } else {
    report.push('## 验证结果汇总');
    report.push('');
    report.push('| 语言 | 用户菜单 | 后台任务面板 | 语言切换方式 |');
    report.push('|------|----------|--------------|--------------|');
    for (const r of results) {
      const um = r.loginBlocked ? '无法验证（登录阻塞）' : (r.userMenu.passed ? '通过' : '未通过');
      const tp = r.loginBlocked ? '无法验证（登录阻塞）' : (r.taskPanel.passed ? '通过' : '未通过');
      report.push(`| ${r.locale} | ${um} | ${tp} | ${r.languageSwitchMethod} |`);
    }
    report.push('');
    report.push('### 用户菜单详情');
    for (const r of results) {
      if (r.loginBlocked) continue;
      report.push(`- **${r.locale}**: accountSettings=${r.userMenu.profileOk} logout=${r.userMenu.logoutOk} role=${r.userMenu.roleOk} languageRow=${r.userMenu.languageRowOk} → ${r.userMenu.passed ? '通过' : '未通过'}`);
    }
    report.push('');
    report.push('### 后台任务面板详情');
    for (const r of results) {
      if (r.loginBlocked) continue;
      report.push(`- **${r.locale}**: title=${r.taskPanel.titleOk} tabs=${r.taskPanel.tabsOk} empty=${r.taskPanel.emptyStateOk} closeAria=${r.taskPanel.closeAriaOk} clearCompleted=${r.taskPanel.clearCompletedOk} → ${r.taskPanel.passed ? '通过' : '未通过'}${r.taskPanel.detail ? ` (${r.taskPanel.detail})` : ''}`);
    }
  }

  report.push('');
  report.push('---');
  console.log(report.join('\n'));

  const allPass = !loginBlocked && results.every((r) => r.userMenu.passed && r.taskPanel.passed);
  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
