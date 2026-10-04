---
course: "fine-tuning-adapting-large-language-models"
chapter: "optimization-deployment-considerations"
lesson: "monitoring-fine-tuned-models"
sourceId: 3098
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/monitoring-fine-tuned-models"
title: "生产环境中微调模型的监控"
description: "部署后跟踪性能、数据漂移和潜在问题的策略。"
order: 8
plots: ["plots/3098-0.json"]
sourceHash: "199df8df76df0c871c1b8ff5ba6bdbe0405b8cf758d8cfab4dc025b8c2c000d7"
sourceCorrections: []
---

部署您微调 (fine-tuning)后的大型语言模型是一个重要步骤，但这并非终点。持续监控是必要环节，可确保模型在生产环境中可靠、高效且安全地运行。正如您为部署优化了模型一样，现在您也需要机制来观察其行为并随时间保持其有效性。如果没有监控，您将面临性能下降、意外成本和潜在故障的风险，这些都可能削弱您调整后模型所提供的价值。

### 监控微调 (fine-tuning)LLM的必要性

预训练 (pre-training)模型提供了一般性基础，但微调使其适应特定的分布和任务。然而，生产环境是动态的。输入数据的特性可能发生变化，用户期望可能演变，模型学到的内在规律也可能改变。这种现象，被称为**漂移**，是进行监控的主要原因。

- **数据漂移 (Data Drift)：** 输入数据的统计特性发生变化。例如，一个在产品发布期间普遍存在的问题上微调的客户支持聊天机器人，可能在数月后随着用户遇到新功能或问题而出现不同类型的查询。
- **概念漂移 (Concept Drift)：** 输入与期望输出之间的关系发生变化。一个微调用于总结新闻文章的模型，如果新闻报道的风格或重点随时间发生显著改变，其准确性可能会降低。

漂移监控处理了多项运营需求：

- **性能跟踪 (Performance Tracking)：** 持续衡量任务特定指标（例如，总结质量、分类准确率、指令遵循情况）以发现性能退步。
- **运行状况 (Operational Health)：** 监控推理 (inference)延迟、吞吐量 (throughput)、错误率以及资源消耗（CPU/GPU、内存），以确保部署符合服务水平目标（SLOs）并控制在预算内。
- **故障模式识别 (Identifying Failure Modes)：** 生产流量通常会显现出在离线评估中不明显的边缘情况或弱点。监控有助于发现故障模式，例如无意义文本（幻觉 (hallucination)）生成增加或针对某些输入类型产生有偏见的输出。
- **迭代反馈 (Feedback for Iteration)：** 日志和性能指标为诊断问题、指导再训练的数据收集以及改进未来版本的模型提供了有价值的数据。

### 生产监控指标

全面的监控策略会跟踪性能、运营、数据特性和负责任AI维度上的各项指标。

#### 性能指标

虽然标准NLP指标（如用于总结的ROUGE或用于分类的F1）对生成模型有局限性（如第六章所讨论），但当持续应用于有代表性的样本或评估集时，它们通常可作为有用的基准指标。可补充以下指标：

- **指令遵循准确率 (Instruction Following Accuracy)：** 抽样生产输入并评估模型遵循显式或隐式指令的程度。这可能需要人工审核或专用评估模型。
- **幻觉 (hallucination)率 (Hallucination Rate)：** 实施检查以发现与提供上下文 (context)或已知事实不符的输出。这可以涉及自动化方法（例如，NLI模型检查蕴涵）或定期人工审计。
- **事实一致性 (Factual Consistency)：** 对于需要事实准确性的任务，抽样输出并根据可信知识源进行验证。
- **用户反馈 (User Feedback)：** 纳入直接用户信号（例如，评分、标记 (token)）作为感知质量的代理。

#### 运行指标

这些是任何生产服务的基本指标：

- **延迟 (Latency)：** 跟踪每次推理 (inference)请求所需时间（例如，平均值、p95、p99）。峰值可能表示基础设施问题或有问题的输入。
- **吞吐量 (Throughput)：** 监控每单位时间处理的请求数量（例如，每秒请求数）。
- **资源利用率 (Resource Utilization)：** 关注CPU、GPU（利用率、内存使用）和系统内存消耗，以优化资源分配并防止瓶颈。
- **错误率 (Error Rates)：** 记录和监控系统级错误（例如，超时、处理失败）和模型特定错误（例如，无法生成响应）。
- **成本 (Cost)：** 跟踪每次推理或每时间段的成本，尤其是在云环境中。

#### 数据漂移指标

检测数据分布变化对保持模型性能很重要：

- **输入分布 (Input Distribution)：** 监控传入提示或数据的统计特性。
  - *文本统计 (Text Statistics)：* 跟踪平均长度、词汇使用（例如，与微调 (fine-tuning)集相比的词汇表 (vocabulary)外率）、相对于参考模型的困惑度。
  - *主题/意图分布 (Topic/Intent Distribution)：* 如果适用，使用更简单的模型或关键词分析来跟踪呈现给LLM的主题或意图的变化。
  - *统计度量 (Statistical Measures)：* 对于嵌入 (embedding)或派生特征，使用 Kullback-Leibler (KL) 散度或总体稳定性指数（PSI）等指标，将当前数据分布与参考（例如，微调数据集分布）进行比较。
- **输出分布 (Output Distribution)：** 监控生成文本的属性。
  - *文本统计 (Text Statistics)：* 跟踪输出长度、情感分布、特定安全相关关键词的存在。

可视化漂移会有帮助。例如，跟踪输入提示长度随时间的变化分布：



![输入提示长度随时间变化](plots/3098-0.json)



> 平均提示长度与微调期间观察到的参考长度进行比较。持续增加可能表明数据漂移。

#### 公平性和偏见指标

监控公平性需要仔细考虑敏感属性以及在微调过程中引入或放大的潜在偏见：

- **群体特定性能 (Group-Specific Performance)：** 如果可能且符合伦理，可分解相关人口群体的性能指标以检查差异。
- **偏见探测 (Bias Probes)：** 定期在旨在引出特定社会偏见的精选数据集（例如 StereoSet、Bias Benchmark）上运行模型，并跟踪偏见分数的变化。
- **用户报告 (User Reports)：** 专门监控用户反馈渠道，以获取关于偏见、不公平或有害输出的报告。

### 监控策略与实施

有效监控结合自动化系统与人工监督。

- **全面日志记录 (Comprehensive Logging)：** 记录输入、输出、时间戳、延迟、资源使用情况以及任何内部模型得分（如概率或置信度，如果可用）。确保日志结构化以便于查询和分析。
- **定向采样 (Targeted Sampling)：** 评估每一个预测通常不可行。实施智能采样策略：
  - *随机采样 (Random Sampling)：* 简单基线，适用于一般趋势。
  - *分层采样 (Stratified Sampling)：* 确保不同输入类型或用户群体的代表性。
  - *低置信度采样 (Low-Confidence Sampling)：* 将评估精力集中在模型不确定的预测上（如果置信度分数可用且经过校准）。
- **自动化健康检查 (Automated Health Checks)：** 定期安排任务，根据预定义评估集（“黄金数据集”）运行模型，以快速发现核心功能的退步。
- **人工在环（HITL）(Human-in-the-Loop (HITL))：** 设置人工审核员工作流程，定期评估生产流量样本。这对于评估连贯性、有用性、安全性以及对复杂指令的遵循等主观质量不可或缺。用户反馈机制（例如点赞/点踩）提供了一个可扩展但有噪声的信号。
- **监控平台 (Monitoring Platforms)：** 利用MLOps和可观测性平台。Prometheus和Grafana等工具非常适合运营指标。云服务提供商（AWS CloudWatch、Google Cloud Monitoring、Azure Monitor）提供集成解决方案。专用LLM可观测性平台（例如 Arize AI、WhyLabs、Truera、Langfuse）提供专门为跟踪嵌入 (embedding)漂移、评估生成质量和管理LLM特定指标而设计的功能。
- **警报系统 (Alerting Systems)：** 根据预定义阈值或异常检测配置警报。针对以下情况触发警报：
  - 性能指标显著下降（例如准确率 < 90%）。
  - 延迟超出SLO（例如 p99 延迟 > 2 秒）。
  - 错误率飙升（例如 5 分钟内错误率 > 1%）。
  - 检测到的数据漂移超过阈值（例如 PSI > 0.2）。
  - 资源消耗或成本突然增加。

典型的监控反馈环可视化如下：

> 一个监控循环，显示了数据从用户输入和模型输出到日志记录、仪表板、警报、分析以及潜在的再训练触发器的数据流。

### 响应监控信息

警报和观察到的偏差需要采取行动：

- **分类与诊断 (Triage and Diagnosis)：** 调查警报以确定根本原因。是临时基础设施故障、有问题的输入模式、真实数据漂移，还是模型退步？分析日志、指标趋势和具体的失败案例。
- **模型回滚 (Model Rollback)：** 如果新部署的模型版本显示严重问题，应有一个明确流程来快速回滚到先前已知的良好版本。
- **再训练/重新微调 (fine-tuning) (Retraining/Re-fine-tuning)：** 如果监控表明持续的性能下降或显著的数据漂移，则安排再训练。决定是针对新数据进行增量微调，还是在组合数据集上执行完整微调。利用故障分析的成果来指导数据增强或选择。
- **数据过滤/预处理更新 (Data Filtering/Preprocessing Updates)：** 有时，通过改进输入验证或预处理步骤而不是重新训练模型，可以缓解问题。
- **系统调优 (System Tuning)：** 通过调整扩展参数 (parameter)、优化推理 (inference)代码或升级基础设施来解决运营问题。

### LLM特有的监控难题

监控微调 (fine-tuning)LLM带来独特的难题：

- **评估的主观性与成本 (Subjectivity and Cost of Evaluation)：** 定义和持续衡量“有用性”或“创造力”等质量很困难。人工评估是黄金标准，但速度慢、成本高且难以扩展。自动化指标通常是不完美的替代品。
- **高维度与非结构化数据 (High Dimensionality and Unstructured Data)：** 在高维嵌入 (embedding)空间或非结构化文本中检测漂移比在传统表格数据监控中更复杂。
- **规模与速度 (Scale and Velocity)：** LLM通常处理大量流量，使得全面日志记录和分析计算密集。
- **延迟敏感性 (Latency Sensitivity)：** 许多LLM应用需要低延迟，这给添加到推理 (inference)路径中的任何监控组件施加了限制。
- **故障解释 (Interpreting Failures)：** 由于模型复杂性，理解LLM为何产生特定的不理想输出可能具有挑战性。

在生产环境中有效监控您的微调LLM是一个持续过程，它需要结合自动化工具、仔细的指标选择、人工监督和明确的响应流程。它是LLM的MLOps生命周期的组成部分，确保您精心调整的模型能够随着时间安全可靠地持续提供价值。

## 参考资料

- [Machine Learning Engineering](https://www.oreilly.com/library/view/machine-learning-engineering/9781098105312/) — Andriy Burkov, Xavier Bouthillier, Gabriele Cesa, Viktor Kerkez, Egor Mineev, Adam Proie, Karim Sayed, Saishruthi Swaminathan, Hubert Szücs, Sharan Narang (2022)
  Publisher: O'Reilly Media
  本书提供了部署和维护机器学习模型的全面指南，包括关于生产环境中监控、数据漂移和操作指标的章节。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); Volume: 162; Pages: 12866-12891; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  本文介绍了 HELM 框架，这是一个系统的语言模型评估基准，涵盖了准确性、公平性、鲁棒性和效率，为生产中 LLM 的持续监控提供了信息。
