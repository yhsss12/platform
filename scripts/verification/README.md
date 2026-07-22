# 验收与回归脚本

这里存放平台级验收、浏览器回归和集成冒烟工具。它们不会被前后端服务或任务 worker
作为生产入口调用。

脚本移动到本目录后仍以仓库根目录作为工作目录，运行命令统一为：

```bash
python scripts/verification/<script-name>.py
node scripts/verification/<script-name>.mjs
bash scripts/verification/<script-name>.sh
```

`browser/` 子目录存放 Playwright、i18n、页面交互及性能类浏览器检查。
