/**
 * 5 步语言真实交互验证：localhost:3001/login。
 * 1) 清空 epi_locale → 刷新 → 记录初始语言 + navigator.language
 * 2) 切换到 English → 确认登录页文案 + 记录 epi_locale
 * 3) admin/admin123 登录 → 用户下拉 Language 区块与三项语言、左侧菜单与数据资产页英文
 * 4) 平台内切到 Svenska → 左侧/数据页瑞典语，刷新后保持
 * 5) 退出登录回登录页 → 仍保持 Svenska
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const baseURL = process.env.BASE_URL || "http://localhost:3001";
const outDir =
  process.env.OUT_DIR ||
  path.join(process.cwd(), "artifacts", `locale-verify-5steps-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`);

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function safeFileName(s) {
  return String(s).replaceAll("/", "_").replaceAll(" ", "_").replaceAll(":", "-");
}

async function snap(page, name) {
  const file = path.join(outDir, `${safeFileName(name)}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

const results = [];

function addResult(step, passed, detail, screenshot) {
  results.push({ step, passed, detail: detail || "", screenshot: screenshot || "" });
}

async function run() {
  ensureDir(outDir);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  await context.clearCookies();
  const page = await context.newPage();

  const getNavLang = () => page.evaluate(() => navigator.language);
  const getEpiLocale = () => page.evaluate(() => localStorage.getItem("epi_locale"));
  const getLoginButtonText = () =>
    page.locator("form button[type='submit']").first().innerText().catch(() => "");
  const getLoginFormSnippet = () =>
    page.locator("form").first().innerText().then((t) => t.replace(/\s+/g, " ").slice(0, 400)).catch(() => "");

  try {
    // ---------- Step 1: 清空 epi_locale，刷新，记录初始语言 + navigator.language ----------
    await page.goto(`${baseURL}/login`, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForTimeout(500);
    await page.evaluate(() => localStorage.removeItem("epi_locale"));
    await page.goto(`${baseURL}/login`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(2000);

    const navLang = await getNavLang();
    const epiLocale1 = await getEpiLocale();
    const initialButtonText = await getLoginButtonText();
    const hasInitialUi = initialButtonText.includes("登录") || initialButtonText.includes("Sign in") || initialButtonText.includes("Logga in");
    const formVisible = await page.locator("form").first().isVisible().catch(() => false);
    const step1Pass = Boolean(navLang) && (hasInitialUi || formVisible);
    const shot1 = await snap(page, "step1-after-clear-refresh");
    addResult(
      "1) 清空 epi_locale→刷新，记录初始语言",
      step1Pass,
      `navigator.language=${navLang}, epi_locale=${epiLocale1 ?? "null"}（hydrate 后可能被写回）, 登录按钮=${initialButtonText}`,
      shot1
    );

    // ---------- Step 2: 切换到 English，确认登录页文案，记录 epi_locale ----------
    const langTrigger = page.locator("button").filter({ hasText: /简体中文|English|Svenska/ }).first();
    await langTrigger.waitFor({ state: "visible", timeout: 8000 });
    await langTrigger.click({ timeout: 3000 });
    await page.waitForTimeout(200);
    const englishItem = page.getByRole("menuitem", { name: "English" }).first();
    await englishItem.click({ timeout: 2000 });
    await page.waitForTimeout(800);

    const epiLocale2 = await getEpiLocale();
    const formText2 = await getLoginFormSnippet();
    const btn2 = await getLoginButtonText();
    const agreementEn = formText2.includes("By signing in, you agree") || formText2.includes("usage rules");
    const btnEn = btn2.includes("Sign in") || btn2.includes("Signing in");
    const step2Pass = epiLocale2 === "en" && btnEn && agreementEn;
    const shot2 = await snap(page, "step2-english-login-page");
    addResult(
      "2) 切换到 English，登录页文案为英文",
      step2Pass,
      `epi_locale=${epiLocale2}, 按钮=${btn2}, 协议含英文=${agreementEn}; 表单片段=${formText2.slice(0, 120)}`,
      shot2
    );

    // ---------- Step 3: admin/admin123 登录，用户下拉 Language 区块 + 左侧菜单与数据资产页英文 ----------
    await page.locator("form input").first().fill("admin");
    await page.locator("form input[type='password']").fill("admin123");
    await page.locator("form button[type='submit']").first().click();
    await page.waitForURL((u) => !u.pathname.includes("/login"), { timeout: 12000 }).catch(() => {});

    const onPlatform = !page.url().includes("/login");
    if (!onPlatform) {
      addResult("3) 登录并进入平台", false, `登录后未跳转，当前 URL=${page.url()}`, await snap(page, "step3-login-failed"));
    } else {
      await page.waitForTimeout(1500);
      await page.waitForFunction(
        () => document.body.innerText.includes("admin") && (document.querySelector("aside") || document.querySelector("nav")),
        { timeout: 15000 }
      ).catch(() => {});
      await page.waitForTimeout(500);
      const userDropdownTrigger = page.locator("header").getByText("admin").first();
      await userDropdownTrigger.click({ timeout: 5000 });
      await page.waitForTimeout(400);

      const hasLanguageLabel = await page.getByText("Language", { exact: true }).isVisible().catch(() => false);
      const hasZh = await page.getByText("简体中文").first().isVisible().catch(() => false);
      const hasEn = await page.getByText("English").first().isVisible().catch(() => false);
      const hasSv = await page.getByText("Svenska").first().isVisible().catch(() => false);
      const hasThreeLangs = hasZh && hasEn && hasSv;
      const hasCheck = await page.locator("button").filter({ has: page.locator("svg") }).first().isVisible().catch(() => false);

      await page.goto(`${baseURL}/data`, { waitUntil: "domcontentloaded" }).catch(() => {});
      await page.waitForTimeout(600);
      const sidebarText = await page.locator("aside").nth(1).innerText().catch(() => "") || await page.locator("aside").first().innerText().catch(() => "");
      const mainText = await page.locator("main").first().innerText().catch(() => "");
      const pageTitle = await page.locator("h1, h2").first().innerText().catch(() => "");

      const menuEn = sidebarText.includes("Data") || sidebarText.includes("Collection");
      const dataPageEn = pageTitle.includes("Data Assets") || mainText.includes("Data Assets") || mainText.includes("Import Data");
      const step3Pass = onPlatform && (hasLanguageLabel || hasThreeLangs) && (menuEn && dataPageEn);
      const shot3 = await snap(page, "step3-platform-en");
      addResult(
        "3) 平台内：用户下拉 Language+三项语言+勾；左侧菜单与数据资产页英文",
        step3Pass,
        `Language区块=${hasLanguageLabel}, 三项语言=${hasThreeLangs}, 有勾=${hasCheck}, 侧栏含Data=${menuEn}, 数据页英文=${dataPageEn}, title=${pageTitle.slice(0, 60)}`,
        shot3
      );
    }

    if (onPlatform) {
      // ---------- Step 4: 用户下拉切到 Svenska，左侧与数据页瑞典语，刷新后保持 ----------
      const userTrigger2 = page.locator("header").getByText("admin").first();
      await userTrigger2.click({ timeout: 3000 });
      await page.waitForTimeout(300);
      const svenskaBtn = page.getByRole("button", { name: "Svenska" }).or(page.getByRole("menuitem", { name: "Svenska" })).first();
      await svenskaBtn.click({ timeout: 2000 });
      await page.waitForTimeout(600);

      const sidebarAfter = await page.locator("aside").nth(1).innerText().catch(() => "") || await page.locator("aside").first().innerText().catch(() => "");
      const mainAfter = await page.locator("main").first().innerText().catch(() => "");
      const titleAfter = await page.locator("h1, h2").first().innerText().catch(() => "");
      const menuSv =
        sidebarAfter.includes("Data") ||
        sidebarAfter.includes("Insamling") ||
        sidebarAfter.includes("Datatillgångar") ||
        sidebarAfter.includes("Konvertering");
      const dataSv = titleAfter.includes("Datatillgångar") || mainAfter.includes("Datatillgångar") || mainAfter.includes("Importera data");
      const step4aPass = menuSv && dataSv;
      const shot4a = await snap(page, "step4-sv-before-refresh");
      addResult(
        "4a) 平台内切到 Svenska，左侧与数据资产页为瑞典语",
        step4aPass,
        `侧栏=${sidebarAfter.slice(0, 120)}, title=${titleAfter}, 数据页瑞典语=${dataSv}`,
        shot4a
      );

      await page.reload({ waitUntil: "domcontentloaded" });
      await page.waitForTimeout(600);
      const epiLocaleAfter = await getEpiLocale();
      const sidebarRefreshed = await page.locator("aside").nth(1).innerText().catch(() => "") || await page.locator("aside").first().innerText().catch(() => "");
      const titleRefreshed = await page.locator("h1, h2").first().innerText().catch(() => "");
      const stillSv =
        epiLocaleAfter === "sv" &&
        (titleRefreshed.includes("Datatillgångar") || sidebarRefreshed.includes("Insamling") || sidebarRefreshed.includes("Data"));
      const step4bPass = stillSv;
      const shot4b = await snap(page, "step4-sv-after-refresh");
      addResult(
        "4b) 刷新后仍保持瑞典语",
        step4bPass,
        `epi_locale=${epiLocaleAfter}, title=${titleRefreshed}, 侧栏片段=${sidebarRefreshed.slice(0, 80)}`,
        shot4b
      );

      // ---------- Step 5: 退出登录回登录页，仍保持 Svenska ----------
      const userTrigger3 = page.locator("header").getByText("admin").first();
      await userTrigger3.click({ timeout: 3000 });
      await page.waitForTimeout(300);
      const logoutBtn = page.getByRole("button", { name: /退出登录/ }).first();
      await logoutBtn.click({ timeout: 2000 });
      await page.waitForTimeout(400);
      const confirmBtn = page.getByRole("button", { name: /退出登录|确认/ }).first();
      if (await confirmBtn.isVisible().catch(() => false)) await confirmBtn.click();
      await page.waitForURL((u) => u.pathname.includes("/login"), { timeout: 5000 }).catch(() => {});

      await page.waitForTimeout(500);
      const onLoginAgain = page.url().includes("/login");
      const epiLocale5 = await getEpiLocale();
      const loginBtn5 = await getLoginButtonText();
      const stillSvLogin = epiLocale5 === "sv" && (loginBtn5.includes("Logga in") || loginBtn5.includes("Loggar in"));
      const step5Pass = onLoginAgain && stillSvLogin;
      const shot5 = await snap(page, "step5-login-page-still-sv");
      addResult(
        "5) 退出登录回登录页，仍保持 Svenska",
        step5Pass,
        `epi_locale=${epiLocale5}, 登录按钮=${loginBtn5}, 当前URL=${page.url()}`,
        shot5
      );
    } else {
      addResult("4) 平台内切到 Svenska（略）", false, "因步骤3未进入平台，跳过", "");
      addResult("5) 退出登录回登录页（略）", false, "因步骤3未进入平台，跳过", "");
    }
  } catch (err) {
    addResult("执行异常", false, String(err.message || err), await snap(page, "error").catch(() => ""));
  }

  await context.close();
  await browser.close();

  // 输出表格
  const tableRows = results.map((r) => ({
    step: r.step,
    passed: r.passed ? "通过" : "未通过",
    detail: r.detail,
    screenshot: r.screenshot,
  }));

  const md = [
    "| 步骤 | 是否通过 | 说明 / 失败时页面·元素 |",
    "|------|----------|------------------------|",
    ...tableRows.map(
      (r) =>
        `| ${r.step} | **${r.passed}** | ${(r.detail || "-").replace(/\|/g, "\\|").replace(/\n/g, " ").slice(0, 200)}${r.screenshot ? ` 截图: \`${path.basename(r.screenshot)}\`` : ""} |`
    ),
  ].join("\n");

  const fullReport = md + "\n\n截图目录: " + outDir + "\n";
  fs.writeFileSync(path.join(outDir, "report.md"), fullReport);
  fs.writeFileSync(path.join(outDir, "report.json"), JSON.stringify({ baseURL, results: tableRows, outDir }, null, 2));

  process.stdout.write(fullReport);
  return results;
}

run()
  .then((r) => process.exit(r.every((x) => x.passed) ? 0 : 1))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
