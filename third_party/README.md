# 第三方源码

本目录集中保存独立检出或 vendored 的第三方项目。平台自研适配代码应放在 `integrations/`，不要直接写入第三方目录。

- `robosuite-task-zoo/`：ARISE Initiative 的 robosuite 扩展任务库，以 Git 子模块固定版本，
  当前不被平台运行链路直接引用。克隆仓库时可使用 `git clone --recurse-submodules`。

`mimicgen/` 保存 MimicGen 嵌套仓库及官方源数据；`IsaacLab/` 保存物块堆叠使用的 Isaac Lab
运行时源码。平台配置和任务集成统一引用这里，不再使用仓库根目录的旧路径。

`backend/integrations/` 不是第三方源码副本：其中是由后端直接调度的双臂数据导出、Isaac Lab
运行脚本和 pi0 runner，因此继续保留在后端应用边界内。

Isaac Sim Franka 专家任务包已归入 `integrations/IsaacSimFrankaPickPlace/`；其模板仍保持
`integration_pending`，不会混入四个稳定模板的可用任务列表。
