---
course: "llm-constitutional-ai-rlaif"
chapter: "advanced-rlaif-implementation"
lesson: "rlaif-data-collection"
sourceId: 4510
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/rlaif-data-collection"
title: "偏好数据收集与管理"
description: "高效收集、过滤和管理AI生成偏好数据的策略。"
order: 2
plots: []
sourceHash: "66769df5e7097419a56b91f4d1352d4425c2c0dd20ae4285c0f5c22688b37d86"
sourceCorrections: []
---

从AI反馈中进行有效强化学习 (reinforcement learning)，其成效取决于用于训练奖励模型的偏好数据的质量和结构。拥有AI偏好标注器后，下一步是系统地生成回复对、获取AI偏好，并组织这些数据以实现高效训练。此过程需要仔细权衡提示选择、回复生成、数据结构化和质量控制。

### 提示生成与选择策略

偏好数据集的核心是用于引出模型回复的提示集合（$x$）。这些提示的特点会很大程度上影响最终的对齐 (alignment)效果。

- **来源多样性：** 力求获得覆盖目标应用相关各类主题、任务、语气和潜在安全问题的多样化提示集。来源可包括现有评估数据集、用户日志（匿名化并整理过）、为测试特定行为而合成生成的提示（例如，红队测试提示），或由其他大型语言模型生成的提示。
- **分布匹配：** 理想情况下，提示分布应反映最终对齐模型的预期使用模式。然而，为了确保偏好模型在具有挑战性或对安全有高要求的提示方面能有效学习，通常需要对这些提示进行过采样。
- **避免偏见：** 注意提示来源中存在的偏见。如果提示主要代表某些观点或人群，则最终的偏好模型可能无意中继承这些偏见。分层抽样或基于提示元数据的明确平衡等方法可以减轻这种情况。

### 生成回复对

对于每个选定的提示 $x$，您需要生成两个或更多可比较的回复（$y_1, y_2, ...$）。目标是创建能让AI标注器做出有意义偏好判断的配对。常见策略包括：

- **从策略模型采样：** 使用不同的采样参数 (parameter)（例如，调整温度、top-k、top-p）从当前训练中的语言模型（策略模型）生成多个回复。较高的温度通常会导致更多样化、有时质量较低的回复，而较低的温度则产生更确定性的输出。比较使用不同参数生成的回复可以提供有用的偏好信号。
- **比较模型检查点：** 从当前模型检查点生成一个回复，从之前的检查点生成另一个。这有助于偏好模型学习偏好训练期间的改进。
- **使用不同模型：** 比较正在训练的模型与不同基准模型（可能是早期版本，或已知具有某些优缺点）的回复。

生成策略的选择会影响学习到的偏好类型。采样主要教授模型当前能力范围内的风格变化和质量偏好，而比较检查点或模型则教授方向性改进。

### AI标注流程

有了提示和回复对（$x, y_1, y_2$），AI偏好标注器便会发挥作用。

1. **输入格式化：** 为标注器准备输入。这通常涉及清晰地格式化提示和两个回复，通常使用标注器模型训练时所用的特定分隔符或模板。例如：`Prompt: [prompt text]\n\nResponse A: [y1 text]\n\nResponse B: [y2 text]\n\nWhich response is better?`
2. **批量处理：** 为了最大化效率，批量处理标注请求。这需要仔细管理API调用或推理 (inference)任务，尤其是当使用外部标注模型或大型内部模型时。
3. **输出解析：** 解析标注器的输出以提取偏好判断（例如，“回复A更好”，“回复B更好”）。处理标注器可能表示不确定或相等的情况，尽管通常目标是强制选择。将原始输出与解析后的标签一同存储，以备调试。

> AI偏好收集的数据流：选定提示，使用策略LLM生成回复对，AI标注器提供偏好，结果存储在偏好数据集中供模型训练。

### 数据模式、存储与版本控制

一个结构良好的数据集对训练和分析必不可少。考虑以下模式元素：

- `prompt_id`：提示的唯一标识符。
- `prompt_text`：提示的实际文本（$x$）。
- `response_1_text`：第一个回复的文本（$y_1$）。
- `response_2_text`：第二个回复的文本（$y_2$）。
- `chosen_response_id`：指示哪个回复被偏好的标识符（例如，'response\_1' 或 'response\_2'）。也可以用数字（0 或 1）表示。
- `rejected_response_id`：非偏好回复的标识符。
- `labeler_model_id`：所用特定AI偏好标注器的标识符。
- `generation_metadata`：（可选）用于生成回复的参数 (parameter)（例如，温度、模型检查点ID）。
- `labeler_confidence`：（可选）标注器提供的置信度分数（如果有）。
- `timestamp`：标签生成的时间戳。
- `prompt_metadata`：（可选）与提示关联的标签或类别（例如，'安全'、'编程'、'创意写作'）。

**存储方案：**
对于较小的数据集，JSON Lines 或 CSV 等简单格式可能足够。随着数据集增长到数百万或数十亿偏好，更具扩展性的方案变得必要：

- **数据库：** 关系型（例如，PostgreSQL）或NoSQL数据库提供查询能力，但对于大型文本字段可能需要优化。
- **数据湖/数据仓库：** Amazon S3、Google Cloud Storage 或 Azure Blob Storage 等平台结合查询引擎（例如，Spark SQL、Presto、BigQuery、Snowflake）为海量数据集提供扩展性。使用Parquet或ORC等结构化数据格式以提高效率。
- **机器学习 (machine learning)数据平台：** 专用平台（例如，Databricks Delta Lake、Hugging Face Datasets库）提供为机器学习工作负载定制的功能，包括版本控制和高效加载。

**版本控制：** 实行严格的数据版本控制。追踪创建每个版本偏好数据集所用的提示、回复生成方法、标注器模型和过滤步骤。这对于重现性、调试训练不稳定性和随时间分析模型性能变化必不可少。DVC（数据版本控制）等工具或机器学习平台内的功能可以帮助管理这一点。

### 质量控制与过滤

原始AI生成偏好可能包含噪声或不理想的瑕疵。实行过滤和质量控制步骤：

- **基本完整性检查：** 筛选掉一个或两个回复明显格式不正确的配对（例如，空、过度重复、只包含错误消息）。
- **标注器一致性：** 如果可能，偶尔将相同（或略有扰动）的提示/回复对多次发送给标注器，或使用多个标注器来检查一致性。高度不一致可能表明提示、回复或标注器本身存在问题。
- **处理不确定性：** 决定如何处理AI标注器表示低置信度或接近相等的情况。选项包括丢弃这些配对、在训练期间降低其权重 (weight)，或专门使用它们来校准偏好模型。
- **多样性采样：** 分析提示和偏好判断的分布。如果数据集严重倾斜（例如，绝大多数偏好来自特定模型版本的回复），考虑在训练期间采用采样策略，以确保偏好模型不会过度拟合简单的启发式方法。
- **移除琐碎配对：** 筛选掉偏好判断简单的配对（例如，将格式良好的答案与完全胡言乱语进行比较），因为这些可能不会为对齐 (alignment)提供强烈信号。但是，确保保留足够涵盖基本能力的例子。

有效的偏好数据收集与管理是一个持续的过程。随着您的策略模型演进以及对潜在故障模式的认知更加清晰，您可能需要改进您的提示选择、回复生成和过滤策略，以持续改进输入到RLAIF训练循环中的对齐信号。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems; Volume: 35; Pages: 27735-27756; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  一篇基础性论文，描述了InstructGPT模型以及收集人类偏好数据以训练指令遵循奖励模型的方法。
