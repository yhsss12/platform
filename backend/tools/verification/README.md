# 后端冒烟验证工具

这里保存需要真实后端环境、模型运行时或仿真环境的独立 smoke 脚本。它们不属于自动单元测试，
也不会由 API 或 worker 自动调度。

一般从 `backend/` 目录运行，例如：

```bash
python tools/verification/run_pi0_platform_clean_smoke.py
```
