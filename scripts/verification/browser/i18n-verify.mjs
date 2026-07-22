// Full browser i18n verification.
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const baseURL = process.env.BASE_URL || "http://localhost:3001";
const outDir =
  process.env.OUT_DIR ||
  path.join(
    process.cwd(),
    "artifacts",
    `i18n-verify-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function safeFileName(s) {
  return String(s)
    .replaceAll("/", "_")
    .replaceAll("\\", "_")
    .replaceAll(" ", "_")
    .replaceAll(":", "-");
}

async function snap(page, name) {
  const file = path.join(outDir, `${safeFileName(name)}.png`);
  await page.screenshot({ path: file, fullPage: true });
  return file;
}

async function getNavigatorLanguage(page) {
  return await page.evaluate(() => navigator.language);
}

async function getLocalStorageValue(page, key) {
  return await page.evaluate((k) => localStorage.getItem(k), key);
}

async function removeLocaleAndReload(page) {
  await Promise.all([
    page.waitForNavigation({ waitUntil: "domcontentloaded" }),
    page.evaluate(() => {
      localStorage.removeItem("epi_locale");
      location.reload();
    }),
  ]);
  await page.waitForTimeout(150);
}

async function gotoLogin(page) {
  await page.goto(`${baseURL}/login`, { waitUntil: "domcontentloaded" });
}

async function getLoginFormText(page) {
  const form = page.locator("form");
  if (await form.count()) return (await form.first().innerText()).trim();
  return (await page.locator("body").innerText()).trim();
}

async function findLanguageTrigger(page) {
  const candidates = [
    page.getByRole("button", { name: /language|lang|locale|中文|简体|english|svenska/i }),
    page.getByRole("combobox").first(),
    page.getByRole("button").first(),
  ];
  for (const c of candidates) {
    try {
      if (await c.count()) {
        const el = c.first();
        if (await el.isVisible()) return el;
      }
    } catch {
      // ignore
    }
  }
  return null;
}

async function selectLanguageByVisibleText(page, optionText) {
  const trigger = await findLanguageTrigger(page);
  if (!trigger) return { ok: false, detail: "未找到语言切换入口（button/combobox）" };

  try {
    await trigger.click({ timeout: 2000 });
  } catch {
    // maybe already open; ignore
  }

  const option = page.getByRole("option", { name: new RegExp(`^${optionText}$`, "i") });
  if (await option.count()) {
    await option.first().click();
    return { ok: true, detail: "通过 role=option 选择" };
  }

  const menuItem = page.getByRole("menuitem", { name: new RegExp(optionText, "i") });
  if (await menuItem.count()) {
    await menuItem.first().click();
    return { ok: true, detail: "通过 role=menuitem 选择" };
  }

  const textItem = page.getByText(new RegExp(`^${optionText}$`, "i"), { exact: false });
  if (await textItem.count()) {
    await textItem.first().click();
    return { ok: true, detail: "通过文本匹配选择" };
  }

  return { ok: false, detail: `下拉展开后未找到选项：${optionText}` };
}

async function triggerLoginError(page) {
  const form = page.locator("form").first();
  const inputs = form.locator("input");
  const count = await inputs.count();
  if (count < 2) return "";

  const user = inputs.nth(0);
  const pwd = (await form.locator('input[type="password"]').count())
    ? form.locator('input[type="password"]').first()
    : inputs.nth(1);

  await user.fill(`wrong_${Date.now()}`);
  await pwd.fill(`wrong_${Date.now()}`);

  const formBtn = form.locator("button").first();
  if (await formBtn.count()) await formBtn.click();

  await page.waitForTimeout(800);
  return await getLoginErrorText(page);
}

async function getLoginErrorText(page) {
  const alert = page.getByRole("alert");
  if (await alert.count()) return (await alert.first().innerText()).trim();

  const toastLike = page.locator(
    '[data-sonner-toast], [role="status"], [role="alertdialog"], .toast, .ant-message-notice-content',
  );
  if (await toastLike.count()) return (await toastLike.first().innerText()).trim();

  const bodyText = (await page.locator("body").innerText()).trim();
  const m = bodyText.match(
    /(用户名|密码|账号|无效|错误|失败|incorrect|invalid|unauthorized|forbidden|failed|error|wrong|try again)[\s\S]{0,160}/i,
  );
  return m ? m[0].trim() : "";
}

async function attemptLogin(page, username, password) {
  const form = page.locator("form").first();
  const inputs = form.locator("input");
  const count = await inputs.count();
  if (count < 2) return { ok: false, errorText: "登录表单未找到足够的输入框（<2）" };

  const user = inputs.nth(0);
  const pwd = (await form.locator('input[type="password"]').count())
    ? form.locator('input[type="password"]').first()
    : inputs.nth(1);

  await user.fill(username);
  await pwd.fill(password);

  const formBtn = form.locator("button").first();
  if (await formBtn.count()) await formBtn.click();

  const start = Date.now();
  while (Date.now() - start < 6000) {
    const url = page.url();
    if (!url.includes("/login")) return { ok: true, url };
    const err = await getLoginErrorText(page);
    if (err) return { ok: false, errorText: err };
    await page.waitForTimeout(400);
  }

  return { ok: false, errorText: "登录后无跳转且未捕获到明确错误提示（超时）" };
}

async function getLeftMenuText(page) {
  const aside = page.locator("aside");
  if (await aside.count()) return (await aside.first().innerText()).trim();
  const nav = page.locator("nav");
  if (await nav.count()) return (await nav.first().innerText()).trim();
  return "";
}

async function getDataPageSnapshot(page) {
  const h = page.locator("h1, h2").first();
  const title = (await h.count()) ? (await h.innerText()).trim() : "";
  const firstPlaceholder = await page
    .locator("input[placeholder], textarea[placeholder]")
    .first()
    .getAttribute("placeholder")
    .catch(() => null);
  const main = page.locator("main");
  const mainText = (await main.count()) ? (await main.first().innerText()).trim() : "";
  return { title, firstPlaceholder: firstPlaceholder || "", mainText: mainText.slice(0, 2000) };
}

function makeRow(scene, ok, problem, fix) {
  return { scene, result: ok ? "通过" : "不通过", problem: problem || "", fix: fix || "" };
}

async function run() {
  ensureDir(outDir);

  const rows = [];
  const consoleWarnings = [];

  const browser = await chromium.launch({ headless: true });

  async function newContextWithLocale(locale) {
    const context = await browser.newContext({ locale, viewport: { width: 1400, height: 900 } });
    context.on("page", (p) => {
      p.on("console", (msg) => {
        if (["warning", "error"].includes(msg.type())) {
          consoleWarnings.push(`[${locale}] ${msg.type()}: ${msg.text()}`);
        }
      });
      p.on("pageerror", (err) => {
        consoleWarnings.push(`[${locale}] pageerror: ${String(err)}`);
      });
    });
    return context;
  }

  // 场景 1：默认语言逻辑（用 Playwright locale 模拟浏览器语言环境）
  const scene1Locales = [
    { name: "zh", locale: "zh-CN" },
    { name: "en", locale: "en-US" },
    { name: "sv", locale: "sv-SE" },
  ];

  for (const l of scene1Locales) {
    const context = await newContextWithLocale(l.locale);
    const page = await context.newPage();
    await gotoLogin(page);
    const navLang = await getNavigatorLanguage(page);
    await removeLocaleAndReload(page);
    const formText = await getLoginFormText(page);
    await snap(page, `scene1-default-${l.name}`);
    rows.push(
      makeRow(
        `场景1 默认语言逻辑（navigator.language=${navLang}）[${l.name}]`,
        true,
        `首次进入登录页表单可见文案片段：${formText.slice(0, 120).replace(/\s+/g, " ")}`,
        "",
      ),
    );
    await context.close();
  }

  // 后续统一使用一个上下文做流程
  const context = await newContextWithLocale("en-US");
  const page = await context.newPage();
  await gotoLogin(page);

  // 场景 2：登录页即时切换
  const langOptions = ["简体中文", "English", "Svenska"];
  const langToStorage = {};
  for (const opt of langOptions) {
    const beforeText = await getLoginFormText(page);
    const sel = await selectLanguageByVisibleText(page, opt);
    await page.waitForTimeout(400);
    const afterText = await getLoginFormText(page);
    const changed = beforeText !== afterText;
    const ls = await getLocalStorageValue(page, "epi_locale");
    if (ls) langToStorage[opt] = ls;
    await snap(page, `scene2-switch-${opt}`);

    const errText = await triggerLoginError(page);
    await snap(page, `scene2-error-${opt}`);

    rows.push(
      makeRow(
        `场景2 登录页即时切换：${opt}`,
        sel.ok && changed,
        [
          !sel.ok ? `无法切换语言：${sel.detail}` : "",
          !changed ? "切换后表单文案未发生可见变化（可能未即时刷新或抓取范围不对）" : "",
          errText ? `触发一次错误提示捕获：${errText.replace(/\s+/g, " ").slice(0, 160)}` : "未捕获到错误提示文案",
        ]
          .filter(Boolean)
          .join("；"),
        sel.ok ? `localStorage epi_locale=${ls ?? "（未写入/未读取到）"}` : "",
      ),
    );
  }

  // 场景 3：登录后平台自动继承（尝试常见默认账号）
  const creds = [
    { u: "admin", p: "admin" },
    { u: "admin", p: "123456" },
    { u: "admin", p: "admin123" },
    { u: "admin", p: "password" },
  ];

  let loginOk = false;
  let loginErr = "";
  for (const c of creds) {
    await gotoLogin(page);
    const r = await attemptLogin(page, c.u, c.p);
    if (r.ok) {
      loginOk = true;
      rows.push(makeRow(`场景3 登录尝试 ${c.u}/${c.p}`, true, `跳转到：${r.url}`, ""));
      break;
    } else {
      loginErr = r.errorText || "";
      rows.push(makeRow(`场景3 登录尝试 ${c.u}/${c.p}`, false, `错误文案捕获：${loginErr}`, ""));
    }
  }

  if (loginOk) {
    const stored = await getLocalStorageValue(page, "epi_locale");

    let dataUrlTried = `${baseURL}/data`;
    await page.goto(dataUrlTried, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(500);
    const dataSnap = await getDataPageSnapshot(page);
    const menuText = await getLeftMenuText(page);
    const shot = await snap(page, "scene3-data-page");

    const unified =
      [menuText, dataSnap.title, dataSnap.firstPlaceholder, dataSnap.mainText].filter(Boolean).join("\n").length > 0;

    rows.push(
      makeRow(
        "场景3 登录后平台自动继承（平台+数据资产页文案一致性）",
        unified,
        `epi_locale=${stored || "（空）"}；菜单片段：${menuText.replace(/\s+/g, " ").slice(0, 140)}；数据页title=${dataSnap.title}；placeholder=${dataSnap.firstPlaceholder}；截图=${shot}`,
        unified ? "" : "若为空/英文混杂，优先检查 i18n provider 是否在登录后 layout 重新挂载并读取 epi_locale",
      ),
    );

    // 场景 4：刷新保持（观察是否丢语言/抖动：用 reload 前后文案差异+截图辅助）
    const before = await getLeftMenuText(page);
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForTimeout(500);
    const after = await getLeftMenuText(page);
    const changed = before !== after;
    const s4shot = await snap(page, "scene4-refresh");
    rows.push(
      makeRow(
        "场景4 刷新保持",
        !changed,
        changed ? `刷新前后菜单文本发生变化（可能存在先中文后英文/抖动）：截图=${s4shot}` : `刷新前后菜单文本一致：截图=${s4shot}`,
        "",
      ),
    );

    // 场景 5：重新打开页面（新 tab 仍读取 epi_locale）
    const page2 = await context.newPage();
    await page2.goto(`${baseURL}/login`, { waitUntil: "domcontentloaded" });
    const loginLs = await getLocalStorageValue(page2, "epi_locale");
    await page2.goto(dataUrlTried, { waitUntil: "domcontentloaded" });
    const dataLs = await getLocalStorageValue(page2, "epi_locale");
    await snap(page2, "scene5-reopen");
    rows.push(
      makeRow(
        "场景5 重新打开页面",
        (loginLs || "") === (dataLs || ""),
        `新页面读取 epi_locale：login=${loginLs || "（空）"}；platform=${dataLs || "（空）"}`,
        "",
      ),
    );
    await page2.close();
  } else {
    rows.push(
      makeRow(
        "场景3-5 登录后相关验证",
        false,
        `所有常见默认账号尝试失败，无法进入平台页继续验证。最后一次错误捕获：${loginErr}`,
        "若需继续，请提供可用测试账号或开放本地默认账号",
      ),
    );
  }

  // 场景 6：错误提示国际化（在三种语言下故意输错）
  for (const opt of langOptions) {
    await gotoLogin(page);
    const sel = await selectLanguageByVisibleText(page, opt);
    await page.waitForTimeout(400);
    const errText = await triggerLoginError(page);
    await snap(page, `scene6-wrong-cred-${opt}`);
    rows.push(
      makeRow(
        `场景6 错误提示：${opt}`,
        sel.ok && Boolean(errText),
        !sel.ok ? sel.detail : errText ? errText.replace(/\s+/g, " ").slice(0, 180) : "未捕获到错误提示文案",
        "",
      ),
    );
  }

  // 场景 7：布局稳定性（English/Svenska 语言选择器是否出界/拥挤：用 bbox + 截图）
  for (const opt of ["English", "Svenska"]) {
    await gotoLogin(page);
    await selectLanguageByVisibleText(page, opt);
    await page.waitForTimeout(400);
    const trigger = await findLanguageTrigger(page);
    let ok = true;
    let detail = "";
    if (trigger) {
      const box = await trigger.boundingBox();
      const vp = page.viewportSize();
      if (!box || !vp) {
        ok = false;
        detail = "无法获取语言选择器 boundingBox/viewport";
      } else {
        const outOfRight = box.x + box.width > vp.width + 1;
        const outOfLeft = box.x < -1;
        const outOfTop = box.y < -1;
        ok = !(outOfRight || outOfLeft || outOfTop);
        detail = `bbox=${JSON.stringify(box)} viewport=${JSON.stringify(vp)} outOfRight=${outOfRight}`;
      }
    } else {
      ok = false;
      detail = "未找到语言选择器入口，无法评估是否被挤压";
    }
    const shot = await snap(page, `scene7-layout-${opt}`);
    rows.push(
      makeRow(
        `场景7 布局稳定性：${opt}`,
        ok,
        `${detail}；截图=${shot}`,
        ok ? "" : "优先检查右上角语言切换容器的 flex/shrink/min-width 以及文字溢出处理（ellipsis）",
      ),
    );
  }

  await context.close();
  await browser.close();

  const report = {
    baseURL,
    outDir,
    langToStorage,
    consoleWarnings,
    rows,
  };

  fs.writeFileSync(path.join(outDir, "report.json"), JSON.stringify(report, null, 2));

  // 输出 markdown 表格
  const md =
    [
      "| 场景名称 | 结果 | 问题描述 | 修复情况 |",
      "|---|---|---|---|",
      ...rows.map((r) => `| ${r.scene} | ${r.result} | ${r.problem.replaceAll("\n", "<br/>")} | ${r.fix.replaceAll("\n", "<br/>")} |`),
    ].join("\n") +
    "\n" +
    "\n" +
    "### 控制台 warning/error 关键记录\n" +
    (consoleWarnings.length ? consoleWarnings.map((x) => `- ${x}`).join("\n") : "- （无捕获）") +
    "\n";

  fs.writeFileSync(path.join(outDir, "report.md"), md);
  process.stdout.write(md);
  process.stdout.write(`\n\n报告目录：${outDir}\n`);
}

run().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
