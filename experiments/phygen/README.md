# PhyGen 研究实验

本目录保存 PhyGen 相关的研究代码、实验脚本和小型复现记录，不参与平台前后端生产运行。

通用的 PhyGen 包代码位于仓库根目录 `phygen/`；其中
`phygen/adapters/mimicgen/coffee_repair.py` 保存 CoffeePreparation 的数据修复与反馈生成实现，
`scripts/` 下只保留可执行入口和诊断脚本。

- `runs/coffee_preparation_compare/`：CoffeePreparation 对比实验。
- `runs/proxyq_stack_three_20260718/`：StackThree Proxy-Q 实验。
- `runs/stack_eval/`：StackThree 评测与 selector 结果。
- `runs/stack_three_bc/`：StackThree BC / BC-RNN 对比实验。
- `scripts/`：CoffeePreparation 数据转换、扫参、诊断和 PhyGen 研究评测脚本。

这些实验目录中的历史 JSON 可能记录绝对产物路径。目录迁移时已更新到当前仓库位置；未来新增大型输出应写入仓库外的数据根。
