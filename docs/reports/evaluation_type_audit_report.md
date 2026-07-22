# 评测类型历史任务审计报告

- 总任务数：82
- 修复前标签匹配：32
- 修复前标签不匹配：50

主要错误类型：
- `episode_stability` 被旧逻辑映射为「稳定性评测」，应统一为「专家策略评测」
- 缺少 evaluationMode / evaluationObject 的历史任务被旧逻辑映射为「评测任务」

| job_id | task_name | 当前显示 | 应显示 | 判断依据 | 置信度 | 是否匹配 |
|---|---|---|---|---|---|---|
| eval_20260624_142321_92ac | 离线数据集评测 · 线缆穿杆数据_20260623_2312 | 数据集评测 | 数据集评测 | evaluationType=dataset | high | 是 |
| eval_20260624_123244_e61c | 线缆整理稳定性评测_20260624_1232 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| eval_20260624_112336_b217 | 线缆整理稳定性评测_20260624_1123 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260624_bc_accept_01 | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260624_110006_e65c | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260624_100914_d038 | 线缆穿杆专家策略评测_20260624_1009 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260624_093922_46f0 | 线缆穿杆专家策略评测_20260624_0939 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260624_092114_efae | 线缆穿杆专家策略评测_20260624_0921 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260624_091925_15a9 | 线缆穿杆专家策略评测_20260624_0919 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260624_091529_e0b0 | 线缆穿杆专家策略评测_20260624_0915 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260624_085422_61f0 | 线缆穿杆专家策略评测_20260624_0854 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_smoke200_ep68_20260623 | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_fix_bn_20260623 | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260623_185004_ed3c | 线缆穿杆模型评测_20260623_1850 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_180043_89e0 | 线缆穿杆模型评测_20260623_1800 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_174038_7979 | 线缆穿杆专家策略评测_20260623_1740 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260623_173313_ecba | 线缆穿杆专家策略评测_20260623_1733 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260623_172535_daad | 线缆穿杆模型评测_20260623_1725 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_171728_980c | 线缆穿杆模型评测_20260623_1717 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_171025_ef22 | 线缆穿杆模型评测_20260623_1710 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_170856_13c3 | 线缆穿杆模型评测_20260623_1708 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_165942_35e5 | 线缆穿杆模型评测_20260623_1659 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_165053_9cdc | 线缆穿杆模型评测_20260623_1650 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260623_160159_e4eb | 线缆穿杆专家策略评测_20260623_1601 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260622_110652_75e8 | 线缆穿杆专家策略评测_20260622_1106 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260622_110402_c067 | 线缆穿杆专家策略评测_20260622_1104 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_smoke_20260617_212304_obsfix | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260618_100917_63da | 线缆穿杆专家策略评测_20260618_1009 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260618_091110_aad8 | 线缆穿杆专家策略评测_20260618_0911 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260618_085541_bf70 | 线缆穿杆模型评测_20260618_0855 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260618_084935_2a32 | 线缆穿杆专家策略评测_20260618_0849 | 专家策略评测 | 专家策略评测 | evaluationTypeLabel=专家策略评测 | high | 是 |
| ct_eval_20260618_084910_04c1 | 线缆穿杆模型评测_20260618_0849 | 模型评测 | 模型评测 | evaluationTypeLabel=模型评测 | high | 是 |
| ct_eval_20260617_235649_c094 | 线缆穿杆评测_20260617_124 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_234733_1abf | 线缆穿杆评测_20260617_769 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_234256_648f | 线缆穿杆评测_20260617_926 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_234256_ceb4 | 线缆穿杆评测_20260617_926 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_234256_aaaa | 线缆穿杆评测_20260617_926 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_233655_7be5 | regression model__163406_b1a9_ff10ee0ee9 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_233257_e0e6 | 线缆穿杆评测_20260617_126 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_231820_b726 | 线缆穿杆评测_20260617_620 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_221925_2ff0 | regression model__163406_b1a9_ff10ee0ee9 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_221725_cc7f | regression model__163406_b1a9_ff10ee0ee9 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_213335_8530 | 线缆穿杆回归验收 model__163406_b1a9_ff10ee0ee9 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_210257_80d3 | 线缆穿杆数据_20260617_1627 · Final_评测_20260617_2102 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_203454_36fb | 线缆穿杆评测demo8 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_193429_1821 | 线缆穿杆评测_demo5 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| isaac_eval_20260617_193339_c9cf | 线缆穿杆评测_20260617_268 | 模型评测 | 模型评测 | model, trained_model_evaluation, model__132914_c503_33be3945a2 | high | 是 |
| ct_eval_20260617_190925_48ce | 线缆穿杆评测_20260617_495 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_185613_255f | 线缆穿杆评测demo3 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_183203_684c | 线缆穿杆评测demo2 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_182125_8ab8 | 线缆穿杆评测_20260617_513 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_180559_be6c | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_164838_a891 | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_164536_32d1 | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_163802_697f | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_163420_4a3a | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260617_162501_6cc1 | 离线数据集评测 · 线缆穿杆数据_20260617_1613 | 数据集评测 | 数据集评测 | evaluationType=dataset | high | 是 |
| ct_eval_20260617_132553_c0bb | 线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260617_132553_4a9a | 线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| eval_20260617_132530_b452 | 线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| eval_20260617_132145_df53 | 线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260617_121301_0bf6 | 单臂线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260617_113411_8355 | 单臂线缆穿杆 | 模型评测 | 模型评测 | model, trained_model_evaluation, model_20260616_191328_c8fc | high | 是 |
| isaac_eval_20260617_095040_b265 | block_stacking | 模型评测 | 模型评测 | model, trained_model_evaluation, model_20260617_093433_10a4 | high | 是 |
| ct_eval_20260617_093330_92f5 | 单臂线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260617_084817_0195 | 离线数据集评测 · 物块堆叠数据_20260616_1327 | 数据集评测 | 数据集评测 | evaluationType=dataset | high | 是 |
| eval_20260616_154635_f6b4 | 离线数据集评测 · task_cable_threading_v1 · ct_gen_20260615_102019_8f58 | 数据集评测 | 数据集评测 | evaluationType=dataset | high | 是 |
| eval_20260614_223802_1336 | 双臂线缆操控 | 稳定性评测 | 数据集评测 | dataset, episode_stability, ds_dac_gen_20260612_154646_3c5b | high | 否 |
| eval_20260614_222154_ff13 | 双臂线缆操控 | 模型评测 | 模型评测 | model, trained_model_evaluation, model_20260614_221500_82da | high | 是 |
| eval_20260614_222120_118d | 双臂线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260614_215237_f45a | 单臂线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260614_215237_2987 | 双臂线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260614_215215_f075 | 单臂线缆穿杆 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260614_215215_4f35 | 双臂线缆操控 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260614_124403_d99e | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260614_124403_4663 | 双臂线缆操控任务 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260614_123352_52c1 | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| eval_20260614_123352_32fd | 双臂线缆操控任务 | 稳定性评测 | 专家策略评测 | expert_policy, episode_stability | high | 否 |
| ct_eval_20260613_204905_626d | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260612_162118_883a | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260612_162043_bcac | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
| ct_eval_20260612_150754_b877 | 线缆穿杆任务 | 评测任务 | 专家策略评测 | fallback default expert_policy | low | 否 |
