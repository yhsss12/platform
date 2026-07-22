#!/usr/bin/env node
// Generate the repository route report artifact.

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const ARTIFACTS_DIR = join(process.cwd(), 'artifacts', 'report');
const OUTPUT_FILE = join(ARTIFACTS_DIR, 'routes.md');

mkdirSync(ARTIFACTS_DIR, { recursive: true });

const routes = `# Platform Routes

## P0 路由清单

### 平台导航路由组 (platform)

#### 默认入口
- \`/\` - 自动重定向到 \`/collect/tasks\`

#### 艾欧风格 Sidebar 菜单路由
- \`/data\` - 数据页面
- \`/upload\` - 上传页面
- \`/collect/tasks\` - 采集 / 任务列表（默认入口）
- \`/collect/jobs\` - 采集 / 作业中心
- \`/collect/realtime\` - 采集 / 实时采集
- \`/collect/quality\` - 采集 / 质量校验
- \`/label\` - 标注页面
- \`/dict\` - 字典页面
- \`/charts\` - 图表页面
- \`/skills\` - 技能页面
- \`/teleop\` - 遥操作页面
- \`/export\` - 导出页面
- \`/lerobot\` - LeRobot 页面
- \`/settings\` - 设置页面

#### 旧路由（保留，但不在 Sidebar 显示）
- \`/datasets\` - Datasets 列表页
- \`/tasks\` - Tasks 列表页
- \`/runs\` - Runs 列表页
- \`/jobs\` - Jobs 列表页

## 路由结构

\`\`\`
src/app/(platform)/
├── layout.tsx                    # 平台布局（艾欧风格 Sidebar + Topbar）
├── page.tsx                      # 首页（重定向到 /collect/tasks）
├── data/page.tsx                 # 数据页面
├── upload/page.tsx               # 上传页面
├── collect/
│   ├── tasks/page.tsx            # 采集 / 任务列表（复用 TaskList 组件）
│   ├── jobs/page.tsx             # 采集 / 作业中心
│   ├── realtime/page.tsx         # 采集 / 实时采集
│   └── quality/page.tsx          # 采集 / 质量校验
├── label/page.tsx                # 标注页面
├── dict/page.tsx                 # 字典页面
├── charts/page.tsx               # 图表页面
├── skills/page.tsx               # 技能页面
├── teleop/page.tsx               # 遥操作页面
├── export/page.tsx               # 导出页面
├── lerobot/page.tsx              # LeRobot 页面
├── settings/page.tsx             # 设置页面
├── datasets/page.tsx              # Datasets 页面（旧路由，保留）
├── tasks/page.tsx                # Tasks 页面（旧路由，保留）
├── runs/page.tsx                 # Runs 页面（旧路由，保留）
└── jobs/page.tsx                 # Jobs 页面（旧路由，保留）
\`\`\`

## Sidebar 菜单项

1. 数据 📊 - /data
2. 上传 📤 - /upload
3. 采集 📝 - /collect/tasks（默认入口，支持子路径高亮）
4. 标注 🏷️ - /label
5. 字典 📖 - /dict
6. 图表 📈 - /charts
7. 技能 ⚡ - /skills
8. 遥操作 🎮 - /teleop
9. 导出 💾 - /export
10. LeRobot 🤖 - /lerobot
11. 设置 ⚙️ - /settings

## 生成时间

${new Date().toISOString()}
`;

writeFileSync(OUTPUT_FILE, routes);
console.log(`✅ Routes report generated: ${OUTPUT_FILE}`);
