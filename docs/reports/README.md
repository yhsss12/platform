# 测试与审计报告

本目录保存平台测试、问题修复验证和历史数据审计产生的报告。

- `evaluation_flow_test_report.md`：评测流程全链路测试记录。
- `evaluation_replay_episodes_fix_report.md`：评测回放 episode 与视频数量问题的修复记录。
- `evaluation_type_audit_report.md`：历史评测类型与展示标签审计结果；由 `backend/tools/verification/evaluation_type_audit.py` 生成。

运行时日志、截图和机器生成的临时产物不应放入本目录，应写入仓库外的数据根目录或被忽略的测试产物目录。
