# 历史验证脚本

- `verify_full_flow.py`：早期任务创建与文件上传 API 冒烟流程。
- `verify_job_update.py`：早期任务更新 API 冒烟流程。
- `cookies.txt`：早期 curl 验证留下的空 Cookie 文件，仅作为历史文件归档。

当前验证脚本和维护工具应优先放在 `scripts/` 的对应分类中；正式后端测试位于 `backend/tests/`，浏览器端到端测试位于 `e2e/`。
