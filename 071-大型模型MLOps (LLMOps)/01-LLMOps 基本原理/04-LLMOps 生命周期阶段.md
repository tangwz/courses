# LLMOps 生命周期阶段

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-1-foundations-llmops/llmops-lifecycle-stages)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管与传统 MLOps 有相似之处，但大型语言模型（LLMs）的管理生命周期涉及不同的阶段，这些阶段因模型的规模和特殊要求而凸显。理解这个生命周期是构建高效可靠的 LLMOps 流程线的根本。它并非严格线性；反馈循环和迭代很常见，体现了管理复杂 ML 系统的持续性特点。

让我们分析一下涉及的典型阶段：

### 1. 大规模数据管理与准备

这一初始阶段侧重于获取、处理和管理训练或微调 (fine-tuning) LLM 所需的庞大数据集。与标准 ML 项目不同，数据量可以轻易达到 PB 级。

- **数据获取与整理：** 收集大量文本、代码或多模态 (multimodal)数据。需投入大量精力进行清洗、去重和过滤，以确保数据质量并减少固有偏见。
- **预处理与分词 (tokenization)：** 将原始数据转换为适合模型的格式。这涉及针对大词汇量和长序列优化的专门分词策略。可扩展的数据流程线（例如，使用 Spark、Ray Data）非常重要。
- **数据版本管理：** 实施能够对多 TB 数据集进行版本管理的系统（如 DVC、Git LFS 扩展或专门数据平台），对于保证可复现性和可追溯性是必需的。

### 2. 模型开发：训练与微调 (fine-tuning)

此阶段涵盖了从头开始训练基础模型（在大型研究实验室或公司之外较不常见），或更常见的是，针对特定任务或领域微调现有的预训练 (pre-training) LLM 的计算密集型过程。

- **基础模型选择：** 根据模型大小、能力、许可和成本选择合适的 LLM 基础模型。
- **分布式训练：** 训练具有数十亿 ($P \gg 10^9$) 甚至数万亿参数 (parameter)的模型，需要通过大规模加速器集群（GPU/TPU）进行分布式计算。采用数据并行、张量并行和流水线并行等技术，通常由诸如 DeepSpeed 或 Megatron-LM 的框架管理（第3章介绍）。
- **微调策略：** 调整预训练模型，使用的方法从全量微调（资源密集型）到参数高效微调（PEFT）方法，如 LoRA 或 Adapters（第3章讨论）。PEFT 大幅减少了模型适应的计算需求。
- **实验追踪：** 记录参数、指标、代码版本和模型产物变得更为重要，因为 LLM 训练运行的成本和时长都很高。工具需要应对分布式任务的规模和复杂性。
- **检查点设置：** 在长时间训练运行中保存模型状态的机制非常重要，用于故障容忍和恢复训练。

### 3. 模型评估与验证

评估 LLM 需要评估语言质量、事实准确性、安全性以及特定任务表现。

- **量化 (quantization)指标：** 使用 SuperGLUE、HELM 等基准测试，或任务特定指标（例如，ROUGE 用于摘要，CodeBLEU 用于代码生成）。困惑度是一个常见的内在度量。
- **定性分析与人工评估：** 评估连贯性、相关性、毒性、偏见和幻觉 (hallucination)等属性通常需要人工审查或复杂的自动化检查。
- **红队测试：** 主动测试模型的漏洞、有害输出和失效模式。
- **资源消耗基准测试：** 在预期部署条件下评估模型的推理 (inference)延迟、吞吐量 (throughput)和内存占用。

### 4. 部署与服务优化

将大型 LLM 部署到生产环境带来重大的工程挑战，主要体现在模型大小、计算成本和延迟要求方面。

- **模型打包：** 将模型权重 (weight)（通常数百 GB）和依赖项打包成可部署的格式，通常使用容器（例如 Docker）。
- **推理 (inference)优化：** 应用量化 (quantization)（例如，将数值精度降低到 INT8 或 FP4）、剪枝或知识蒸馏 (knowledge distillation)等技术，以缩小模型大小并加速推理速度。常使用专门的推理服务器（例如，NVIDIA Triton 配备 TensorRT-LLM，vLLM）（第4章详细介绍）。
- **服务基础设施：** 将模型部署到可扩展的基础设施上，通常涉及 GPU 加速实例。考虑因素包括请求批处理、管理多个模型副本和自动扩缩容。
- **部署策略：** 使用金丝雀发布或 A/B 测试等模式（有时比较不同的提示或微调 (fine-tuning)模型版本），以安全地推出更新。

### 5. 监控、可观测性与维护

部署后，需要持续监控以确保性能、可靠性、成本效益和负责任的使用。LLM 监控与传统模型相比，具有独特之处。

- **性能监控：** 追踪推理 (inference)延迟、吞吐量 (throughput)（每秒令牌数）和系统资源利用率（GPU 负载、内存）。
- **成本监控：** 密切关注与服务中的 GPU 实例小时数相关的高昂成本。
- **输出质量监控：** 检测模型预测漂移，识别幻觉 (hallucination)、毒性或偏见的增加，并监控用户反馈或下游任务表现。这通常涉及对输出进行抽样，并应用评估指标或人工审查流程（第5章介绍）。
- **基础设施健康：** 对服务基础设施、网络和依赖项（例如用于 RAG 系统的向量 (vector)数据库）的常规监控。

### 6. 反馈循环与迭代

LLMOps 本身就是迭代的。从监控中获得的洞察会反馈到早期阶段，以推动改进。

- **持续评估：** 定期根据新数据或基准测试重新评估生产模型。
- **再训练/持续微调 (fine-tuning)：** 根据检测到的性能下降、数据漂移，或新整理数据或用户反馈的可用性，触发再训练或微调过程。自动化在这里很重要。
- **提示工程 (prompt engineering)更新：** 迭代地改进和更新与 LLM 一起使用的提示，通常通过版本控制和 A/B 测试进行管理（第6章讨论）。
- **RAG 系统更新：** 对于检索增强生成 (RAG)系统，这涉及使用新信息更新向量 (vector)数据库并管理其生命周期（第6章）。

这个生命周期可以被看作一个持续的循环，而非线性进展：

> LLMOps 生命周期调整了传统 MLOps 阶段，侧重于大规模数据处理、分布式计算、专门的评估与监控、部署优化以及由 LLM 独有特性所驱动的持续反馈循环。

这些阶段中的每一个都涉及特定的工具、技术和操作考量，它们与标准 MLOps 实践有很大不同，构成了本课程通篇探讨的主要内容。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了 LoRA，一种参数高效的微调技术，显著减少了适应大语言模型所需的可训练参数数量。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  介绍了 HELM，一个用于在多个维度（包括安全性、偏见和性能）上评估大语言模型的综合框架。

---

[上一节](03-%E5%A4%A7%E5%9E%8B%E6%A8%A1%E5%9E%8B%E6%89%80%E9%9C%80%E7%9A%84%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD.md) · [下一节](05-LLMOps%20%E5%B7%A5%E5%85%B7%E9%80%89%E7%94%A8%E8%A6%81%E7%82%B9.md)
