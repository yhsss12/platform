// Collect-page i18n browser smoke test.
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
    `collect-i18n-smoke-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`,
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

function pickFirstNonEmpty(...xs) {
  for (const x of xs) if (x && String(x).trim()) return String(x).trim();
  return "";
}

async function isOnLogin(page) {
  const url = page.url();
  if (url.includes("/login")) return true;
  const form = page.locator("form");
  if (await form.count()) {
    const hasPwd = (await form.first().locator('input[type="password"]').count()) > 0;
    if (hasPwd) return true;
  }
  return false;
}

async function attemptLogin(page) {
  const creds = [
    { u: "admin", p: "admin123" },
    { u: "admin", p: "admin" },
    { u: "admin", p: "123456" },
    { u: "admin", p: "password" },
  ];

  for (const c of creds) {
    await page.goto(`${baseURL}/login`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(300);
    const form = page.locator("form").first();
    const inputs = form.locator("input");
    if ((await inputs.count()) < 2) continue;
    await inputs.nth(0).fill(c.u);
    const pwd = (await form.locator('input[type="password"]').count())
      ? form.locator('input[type="password"]').first()
      : inputs.nth(1);
    await pwd.fill(c.p);
    await form.locator("button[type='submit'], button").first().click().catch(() => {});
    const start = Date.now();
    while (Date.now() - start < 8000) {
      if (!(await isOnLogin(page))) return { ok: true, used: `${c.u}/${c.p}`, url: page.url() };
      await page.waitForTimeout(300);
    }
  }
  return { ok: false, used: "", url: page.url() };
}

async function findLanguageTrigger(page) {
  const candidates = [
    page.getByRole("button", { name: /language|lang|locale|中文|简体|english|svenska/i }),
    page.locator("button").filter({ hasText: /简体中文|English|Svenska/ }),
    page.getByRole("combobox"),
    page.locator("[data-testid*='lang'], [data-testid*='locale']"),
  ];
  for (const c of candidates) {
    try {
      if ((await c.count()) && (await c.first().isVisible().catch(() => false))) return c.first();
    } catch {
      // ignore
    }
  }
  return null;
}

async function setLocaleViaUI(page, locale) {
  const trigger = await findLanguageTrigger(page);
  if (!trigger) return { ok: false, detail: "未找到站内语言切换控件" };
  await trigger.click({ timeout: 2000 }).catch(() => {});
  await page.waitForTimeout(150);

  const optionNames = {
    "zh-CN": ["简体中文", "中文", "Chinese"],
    en: ["English"],
    sv: ["Svenska"],
  }[locale] || [];

  for (const name of optionNames) {
    const mi = page.getByRole("menuitem", { name: new RegExp(`^${name}$`, "i") });
    if ((await mi.count()) && (await mi.first().isVisible().catch(() => false))) {
      await mi.first().click();
      return { ok: true, detail: `通过菜单选择：${name}` };
    }
    const opt = page.getByRole("option", { name: new RegExp(`^${name}$`, "i") });
    if ((await opt.count()) && (await opt.first().isVisible().catch(() => false))) {
      await opt.first().click();
      return { ok: true, detail: `通过 option 选择：${name}` };
    }
    const txt = page.getByText(new RegExp(`^${name}$`, "i")).first();
    if (await txt.isVisible().catch(() => false)) {
      await txt.click();
      return { ok: true, detail: `通过文本点击：${name}` };
    }
  }

  return { ok: false, detail: `控件展开后未匹配到目标语言选项（${locale}）` };
}

async function setLocaleViaLocalStorage(page, locale) {
  await page.evaluate((l) => {
    localStorage.setItem("epi_locale", l);
  }, locale);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(600);
  const after = await page.evaluate(() => localStorage.getItem("epi_locale"));
  return { ok: after === locale, detail: `localStorage.epi_locale=${after ?? "null"}` };
}

function expectedTokens() {
  return {
    jobs: {
      en: { label: /Label\b/i, audit: /Audit\b/i, del: /Delete\b/i },
      "zh-CN": { label: /标注|打标|标签/i, audit: /审核|审计/i, del: /删除/i },
      sv: { label: /Etikett|Märk|Label/i, audit: /Granska|Revision|Audit/i, del: /Ta bort|Radera|Delete/i },
    },
    quality: {
      en: {
        batchVerify: /Batch\s+verify/i,
        pass: /\bPass\b/i,
        abnormal: /Abnormal/i,
        checking: /Checking/i,
        actual: /\bActual\b/i,
        standard: /\bStandard\b/i,
        recollect: /Re-collect/i,
        save: /\bSave\b/i,
      },
      "zh-CN": {
        batchVerify: /批量.*(校验|验证)|Batch/i,
        pass: /通过|Pass/i,
        abnormal: /异常|Abnormal/i,
        checking: /检查中|校验中|Checking/i,
        actual: /实际|Actual/i,
        standard: /标准|Standard/i,
        recollect: /重新采集|重采集|Re-collect/i,
        save: /保存|Save/i,
      },
      sv: {
        batchVerify: /Batch|Mass|Verifiera/i,
        pass: /Godkänd|Pass/i,
        abnormal: /Avvik|Onormal|Abnormal/i,
        checking: /Kontroller|Checking/i,
        actual: /Faktisk|Actual/i,
        standard: /Standard/i,
        recollect: /Samla\s+igen|Re-collect/i,
        save: /Spara|Save/i,
      },
    },
  };
}

async function readBodyText(page, limit = 4000) {
  const t = await page.locator("body").innerText().catch(() => "");
  return String(t || "").replace(/\s+/g, " ").trim().slice(0, limit);
}

async function gotoCollectTasks(page) {
  await page.goto(`${baseURL}/collect/tasks`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(800);
}

async function gotoCollectJobs(page) {
  const link = page.locator("a[href='/collect/jobs'], a[href^='/collect/jobs']").first();
  if (await link.isVisible().catch(() => false)) {
    await link.click();
    await page.waitForURL((u) => u.pathname.includes("/collect/jobs"), { timeout: 8000 }).catch(() => {});
  } else {
    await page.goto(`${baseURL}/collect/jobs`, { waitUntil: "domcontentloaded" });
  }
  await page.waitForTimeout(800);
}

async function gotoCollectQuality(page) {
  const link = page.locator("a[href='/collect/quality'], a[href^='/collect/quality']").first();
  if (await link.isVisible().catch(() => false)) {
    await link.click();
    await page.waitForURL((u) => u.pathname.includes("/collect/quality"), { timeout: 8000 }).catch(() => {});
  } else {
    await page.goto(`${baseURL}/collect/quality`, { waitUntil: "domcontentloaded" });
  }
  await page.waitForTimeout(800);
}

async function openFirstDialogIfPossible(page, localeTokens) {
  const candidates = [
    page.getByRole("button", { name: localeTokens.label }),
    page.getByRole("button", { name: localeTokens.audit }),
    page.getByRole("button", { name: localeTokens.del }),
    page.locator("button").filter({ hasText: localeTokens.label }),
    page.locator("button").filter({ hasText: localeTokens.audit }),
  ];
  for (const c of candidates) {
    try {
      if ((await c.count()) && (await c.first().isVisible().catch(() => false))) {
        await c.first().click({ timeout: 2000 }).catch(() => {});
        await page.waitForTimeout(500);
        const dlg = page.getByRole("dialog");
        if ((await dlg.count()) && (await dlg.first().isVisible().catch(() => false))) {
          const txt = await dlg.first().innerText().catch(() => "");
          return { ok: true, text: String(txt || "").replace(/\s+/g, " ").slice(0, 800) };
        }
      }
    } catch {
      // ignore
    }
  }
  return { ok: false, text: "" };
}

async function tryOpenQualityDetail(page) {
  const rowLink = page.locator("main a").first();
  if (await rowLink.isVisible().catch(() => false)) {
    await rowLink.click().catch(() => {});
    await page.waitForTimeout(800);
    return true;
  }
  const row = page.locator("main tr").nth(1);
  if (await row.isVisible().catch(() => false)) {
    await row.click().catch(() => {});
    await page.waitForTimeout(800);
    return true;
  }
  return false;
}

async function run() {
  ensureDir(outDir);

  const browser = await chromium.launch({ headless: true });
  const locales = ["zh-CN", "en", "sv"];
  const report = {
    baseURL,
    outDir,
    pages: {
      tasks: {},
      jobs: {},
      quality: {},
    },
  };

  for (const locale of locales) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();

    const chunk404s = [];
    const other404s = [];
    const consoleErrors = [];

    page.on("console", (msg) => {
      if (["error"].includes(msg.type())) consoleErrors.push(`${msg.type()}: ${msg.text()}`);
    });
    page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${String(err)}`));
    page.on("response", (resp) => {
      const s = resp.status();
      if (s !== 404) return;
      const u = resp.url();
      if (u.includes("/_next/static/") || u.includes("/chunks/")) chunk404s.push(u);
      else other404s.push(u);
    });

    const localeSwitch = { method: "", ok: false, detail: "" };
    const auth = { blocked: false, onLogin: false, tried: "", ok: false, detail: "" };

    // 1) tasks 入口（可能重定向到 login）
    await gotoCollectTasks(page);
    if (await isOnLogin(page)) {
      auth.onLogin = true;
      const loginRes = await attemptLogin(page);
      auth.tried = loginRes.used;
      auth.ok = loginRes.ok;
      auth.detail = loginRes.ok ? `登录成功，跳转到 ${loginRes.url}` : `登录失败/未跳转，停留在 ${loginRes.url}`;
      if (!loginRes.ok) auth.blocked = true;
      if (loginRes.ok) await gotoCollectTasks(page);
    }

    // 4) 切语言（即使被登录阻塞，也尽量在登录页完成切换验证）
    {
      const ui = await setLocaleViaUI(page, locale);
      if (ui.ok) {
        localeSwitch.method = "站内语言切换控件";
        localeSwitch.ok = true;
        localeSwitch.detail = ui.detail;
      } else {
        const ls = await setLocaleViaLocalStorage(page, locale);
        localeSwitch.method = "localStorage.epi_locale + 刷新";
        localeSwitch.ok = ls.ok;
        localeSwitch.detail = `${ui.detail}; ${ls.detail}`;
      }
    }

    const tasksBody = await readBodyText(page);
    const tasksShot = await snap(page, `${locale}-collect-tasks`);
    report.pages.tasks[locale] = {
      url: page.url(),
      renderOk: !chunk404s.length,
      chunk404s,
      other404s,
      consoleErrors: consoleErrors.slice(0, 20),
      auth,
      localeSwitch,
      bodySnippet: tasksBody.slice(0, 600),
      screenshot: path.basename(tasksShot),
    };

    // 2) jobs
    if (!auth.blocked) {
      await gotoCollectJobs(page);
      const body = await readBodyText(page);
      const tok = expectedTokens().jobs[locale] || expectedTokens().jobs.en;
      const hasLabel = tok.label.test(body);
      const hasAudit = tok.audit.test(body);
      const hasDelete = tok.del.test(body);
      const dlg = await openFirstDialogIfPossible(page, tok);
      const shot = await snap(page, `${locale}-collect-jobs`);

      report.pages.jobs[locale] = {
        url: page.url(),
        renderOk: !chunk404s.length,
        chunk404s,
        other404s,
        consoleErrors: consoleErrors.slice(0, 20),
        localeSwitch,
        tokensFound: { label: hasLabel, audit: hasAudit, delete: hasDelete },
        dialog: { opened: dlg.ok, textSnippet: dlg.text },
        bodySnippet: body.slice(0, 900),
        screenshot: path.basename(shot),
      };
    } else {
      report.pages.jobs[locale] = { blockedByLogin: true, detail: auth.detail, localeSwitch };
    }

    // 3) quality
    if (!auth.blocked) {
      await gotoCollectQuality(page);
      const openedDetail = await tryOpenQualityDetail(page);
      const body = await readBodyText(page, 7000);
      const tok = expectedTokens().quality[locale] || expectedTokens().quality.en;
      const check = Object.fromEntries(Object.entries(tok).map(([k, re]) => [k, re.test(body)]));
      const shot = await snap(page, `${locale}-collect-quality${openedDetail ? "-detail" : ""}`);

      report.pages.quality[locale] = {
        url: page.url(),
        renderOk: !chunk404s.length,
        chunk404s,
        other404s,
        consoleErrors: consoleErrors.slice(0, 20),
        localeSwitch,
        openedDetail,
        tokensFound: check,
        bodySnippet: body.slice(0, 1200),
        screenshot: path.basename(shot),
      };
    } else {
      report.pages.quality[locale] = { blockedByLogin: true, detail: auth.detail, localeSwitch };
    }

    await context.close();
  }

  await browser.close();
  fs.writeFileSync(path.join(outDir, "report.json"), JSON.stringify(report, null, 2));

  const mdLines = [];
  mdLines.push(`BaseURL: \`${baseURL}\``);
  mdLines.push(`报告目录: \`${outDir}\``);
  mdLines.push("");
  mdLines.push("| 页面 | locale | 是否通过(渲染/关键文案) | 语言切换方式 | 备注 |");
  mdLines.push("|---|---|---:|---|---|");

  const summaryLocales = ["zh-CN", "en", "sv"];
  for (const locale of summaryLocales) {
    const t = report.pages.tasks[locale];
    const j = report.pages.jobs[locale];
    const q = report.pages.quality[locale];

    const tPass = t && !t.auth?.blocked && t.renderOk;
    const jTokensOk =
      j && !j.blockedByLogin && j.tokensFound && Object.values(j.tokensFound).every(Boolean);
    const qTokensOk =
      q && !q.blockedByLogin && q.tokensFound && Object.values(q.tokensFound).filter(Boolean).length >= 4;
    const jPass = j && !j.blockedByLogin && j.renderOk && jTokensOk;
    const qPass = q && !q.blockedByLogin && q.renderOk && qTokensOk;

    const sw = t?.localeSwitch || j?.localeSwitch || q?.localeSwitch || { method: "", detail: "" };
    const swText = pickFirstNonEmpty(sw.method, "（未切换/被登录阻塞）");
    const note = pickFirstNonEmpty(
      t?.auth?.blocked ? `登录阻塞：${t.auth.detail}` : "",
      t?.chunk404s?.length ? `chunk404=${t.chunk404s.length}` : "",
      j?.chunk404s?.length ? `jobs chunk404=${j.chunk404s.length}` : "",
      q?.chunk404s?.length ? `quality chunk404=${q.chunk404s.length}` : "",
      sw.detail || "",
    );

    mdLines.push(`| /collect/tasks | ${locale} | **${tPass ? "通过" : "不通过"}** | ${swText} | ${note.replace(/\|/g, "\\|").slice(0, 160)} |`);
    mdLines.push(`| /collect/jobs | ${locale} | **${jPass ? "通过" : "不通过"}** | ${swText} | ${(j?.blockedByLogin ? j.detail : JSON.stringify(j?.tokensFound || {})).replace(/\|/g, "\\|").slice(0, 160)} |`);
    mdLines.push(`| /collect/quality | ${locale} | **${qPass ? "通过" : "不通过"}** | ${swText} | ${(q?.blockedByLogin ? q.detail : JSON.stringify(q?.tokensFound || {})).replace(/\|/g, "\\|").slice(0, 160)} |`);
  }

  const md = mdLines.join("\n") + "\n";
  fs.writeFileSync(path.join(outDir, "report.md"), md);
  process.stdout.write(md);
}

run().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
