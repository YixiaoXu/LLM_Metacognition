# 当前连续元信息实验流程

活动代码支持同一套“外部语义参考 + 连续 module”机制在三类任务上运行：

- BeaverTails：安全行为指标。
- MathQA：数学正确率、可解析率、推理长度与解题结构指标。
- UltraChat：通用对话的回答长度、组织结构、解释方式和互动语气指标。

三个 profile 的指标和 FDR family 完全隔离：MathQA 不报告安全指标，UltraChat 不报告数学或安全指标。

## 入口

- `run_experiment.sh`：BeaverTails 面向用户的运行入口，负责数据、激活缓存和双 GPU 模型队列。
- `run_mathqa_experiment.sh`：MathQA 多步激活、跨模型语义对齐和双 GPU 模型组合队列。
- `run_ultrachat_experiment.sh`：UltraChat 多步激活和连续风格分析入口。
- `run_generated_step_experiment.sh`：MathQA 与 UltraChat 共用的任务无关编排器。
- `smoke_mathqa_pipeline.sh`：真实小规模端到端冒烟测试。
- `run_model_pair.sh`：单个“目标模型 + 外部语义参考模型”组合的阶段编排。
- `train_external_semantic_modules.sh`：训练 support=64/128 的外部语义参考模块。
- `run_continuous_module.sh`：构建连续轴后，用 direct trust-region 执行连续
  baseline、正/反向 dose、临界 held-out 和持续生成。主线不再训练或加载 runtime bridge。

MathQA 的 target cache 额外保存精确 prompt token IDs，下游不再对多步前缀重新
tokenize。外部语义模型用自己的 chat 模板渲染原问题，再追加目标模型
已生成的 assistant 前缀。

科学参数存放在：

`configs/beavertails330k_external_semantic_continuous_v1.env`

`configs/mathqa_external_semantic_continuous_v1.env`

`configs/ultrachat_external_semantic_continuous_v1.env`

Bash 中仅保留 GPU、输出目录、是否强制重跑等运行时设置。每次运行保存
`RUN_ROOT/input_config.env`和 SHA-256。若配置继承了 MathQA 基础配置（如 smoke），
同时保存 `RUN_ROOT/base_config.env`。

## 日志

- 每个模型组合任务只写一个日志：
  `RUN_ROOT/<target>__semantic_<reference>/task.log`
- 该日志按时间顺序包含 support 训练、模块选择、axis、baseline、
  连续关联、screen A/B、临界样本和持续生成分析，阶段间用时间戳标题分隔。
- 同一模型组合内的确定性 association baseline 在严格 ID 完整性校验后复用；
  prototype、screen 和临界生成仍按 module 独立运行。
- 激活提取是可被多个模型组合复用的独立任务，日志位于
  `RUN_ROOT/activation_prep/<profile>.log`。复用已有缓存时，日志会明确记录 `reuse`。
- `RUN_ROOT/task_logs.tsv` 是本次实验所有任务与日志路径的唯一索引。顶层
  `nohup` 日志只保留队列启动和最终状态，不再承载阶段级详情。

## 统计口径

- 模块选择：held-out residual fraction、residual gain、连续语义可预测性 R2。
- baseline：语义控制后的连续 module score 与当前任务 profile 行为指标的增量关联。
- 干预：沿同一连续轴分别施加正剂量和负剂量；尾部二分仅用于确定方向，不作为类别叙事。
- Safety 报告：生成 token 数、拒答/遵从/重定向、政策语言、缓和表达、有害细节、拒答起始位置。
- MathQA 报告：正确/可解析率、严格 `####` 格式正确/可解析率、推理长度、行/数字/运算符/方程行数和推理结构。
- UltraChat 报告：token/行/段落数量，列表/标题/代码块/提问，解释/示例/自我修正，以及 hedging/certainty/politeness 和人称使用。

旧离散聚类、within-class、质心/旧 OT、输入词缀、BAcc 导出门槛及旧绘图入口
均已从活动代码目录移除。完整旧代码位于：

`code_backup_20260810_before_bridge_cleanup/`

Runtime bridge 的 decoder、synthetic、teacher、KNN、ridge 与 inverse-comparison
实验也已移出活动代码。当前唯一的 code-to-hidden 机制是在目标层隐藏状态上直接
优化 code 方向，同时约束 E1、purified semantic 和 hidden norm。
