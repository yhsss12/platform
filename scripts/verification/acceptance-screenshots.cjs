const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '../../logs/acceptance-screenshots');
const BASE = process.env.BASE_URL || 'http://127.0.0.1:3001';
const USER = process.env.EAI_USERNAME || 'Pibot0001';
const PASS = process.env.EAI_PASSWORD || 'jinlian1234';

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  await page.goto(`${BASE}/login`, { waitUntil: 'domcontentloaded' });
  await page.fill('#login-username', USER);
  await page.fill('#login-password', PASS);
  await page.getByRole('button', { name: /登录|Sign in/i }).click();
  await page.waitForURL(/\/(overview|workspace|admin)/, { timeout: 30000 });

  await page.goto(`${BASE}/workspace/resources/task-templates`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const experimentalToggle = page.getByRole('button', { name: /实验|展开|更多/i });
  if (await experimentalToggle.count()) {
    await experimentalToggle.first().click().catch(() => {});
    await page.waitForTimeout(800);
  }
  await page.screenshot({
    path: path.join(OUT, '01-task-templates-gate.png'),
    fullPage: true,
  });

  const disabledGenerate = await page.locator('button[disabled]:has-text("生成数据")').count();
  const enabledGenerate = await page.locator('button:not([disabled]):has-text("生成数据")').count();

  await page.goto(
    `${BASE}/workspace/replay?jobId=ct_gen_20260617_202234_4507&taskType=cable_threading`,
    { waitUntil: 'networkidle' }
  );
  await page.waitForTimeout(3000);
  await page.screenshot({
    path: path.join(OUT, '02-ct-gen-replay.png'),
    fullPage: true,
  });
  const ctGenText = await page.locator('body').innerText();

  await page.goto(
    `${BASE}/workspace/replay?replayType=evaluation&evalId=ct_eval_20260627_222156_3c30&taskType=cable_threading`,
    { waitUntil: 'networkidle' }
  );
  await page.waitForTimeout(3000);
  await page.screenshot({
    path: path.join(OUT, '03-ct-eval-replay.png'),
    fullPage: true,
  });
  const ctEvalText = await page.locator('body').innerText();

  await page.goto(
    `${BASE}/workspace/replay?replayType=evaluation&evalJobId=eval_20260626_103509_1b68&taskType=dual_arm_cable_manipulation`,
    { waitUntil: 'networkidle' }
  );
  await page.waitForTimeout(3000);
  await page.screenshot({
    path: path.join(OUT, '04-dual-arm-eval-replay.png'),
    fullPage: true,
  });
  const dualText = await page.locator('body').innerText();

  await browser.close();

  const report = {
    at: new Date().toISOString(),
    baseUrl: BASE,
    taskTemplates: {
      disabledGenerateButtons: disabledGenerate,
      enabledGenerateButtons: enabledGenerate,
      screenshot: 'logs/acceptance-screenshots/01-task-templates-gate.png',
    },
    ctGenReplay: {
      hasExpertRollout: ctGenText.includes('生成过程回放 / Expert Rollout') || ctGenText.includes('Expert Rollout'),
      hasGeneratePreview: ctGenText.includes('生成预览'),
      screenshot: 'logs/acceptance-screenshots/02-ct-gen-replay.png',
    },
    ctEvalReplay: {
      hasModelOrExpertEval:
        ctEvalText.includes('模型评测回放 / Model Rollout') ||
        ctEvalText.includes('专家策略评测 / Expert Policy Evaluation'),
      noGeneratePreview: !ctEvalText.includes('生成预览') && !ctEvalText.includes('Expert Rollout'),
      screenshot: 'logs/acceptance-screenshots/03-ct-eval-replay.png',
    },
    dualArmEvalReplay: {
      hasStabilityLabel: dualText.includes('专家管线稳定性评测 / Expert Pipeline Stability Evaluation'),
      screenshot: 'logs/acceptance-screenshots/04-dual-arm-eval-replay.png',
    },
  };

  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
