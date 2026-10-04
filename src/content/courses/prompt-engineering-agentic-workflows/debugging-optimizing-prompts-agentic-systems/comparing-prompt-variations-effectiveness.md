---
course: "prompt-engineering-agentic-workflows"
chapter: "debugging-optimizing-prompts-agentic-systems"
lesson: "comparing-prompt-variations-effectiveness"
sourceId: 6753
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-6-debugging-optimizing-prompts-agentic-systems/comparing-prompt-variations-effectiveness"
title: "比较提示变体以提升智能体效果"
description: "利用A/B测试及其他比较方法，确定针对特定智能体任务最有效的提示变体。"
order: 5
plots: ["plots/6753-0.json"]
sourceHash: "41c50477b96d4499d0aa597a4c3c8e4874123d0fb8d66b62ba3d582af132a3cc"
sourceCorrections: []
---

一旦初始提示就绪，下一步通常是进行优化。仅仅凭直觉调整提示可能只会带来微小的改进，甚至导致退步。为了持续提升智能体效果，需要一种更有条理的方法来比较提示变体。这涉及隔离更改并衡量它们对智能体行为和任务成功的影响。

## 比较测试的原理

当你修改一个提示时，你引入了一个你假设会提升智能体表现的变化。比较测试就是严格评估这个假设。目的不仅仅是看一个新的提示是否*有效*，而是理解它为何比之前的版本或替代设计表现更好（或更差）。这种系统化方法有助于我们更好地理解提示结构如何影响智能体行为。

有效的比较取决于隔离变量。如果你同时改变提示中的五项内容，而表现有所改善，那么是哪一项改变导致了改善？还是它们的组合？不隔离变量，你主要是在猜测。

## 智能体提示的A/B测试

A/B测试是网页设计和市场营销中常用的一种技术，非常适用于提示工程 (prompt engineering)。在这种情况下，你比较一个现有提示版本（变体A，对照组）和包含特定修改的新版本（变体B，挑战组），以判断哪个版本在定义的衡量指标上表现更优。

提示A/B测试的主要构成包括：

1. **单一变量修改**：这是一项核心原则。在变体A和变体B之间，只改变提示的一个方面。例如：

   - 重新措辞某个特定指令。
   - 添加或删除少量示例。
   - 改变信息顺序。
   - 改变分配给智能体的人格。
   - 修改预期输出的格式。
     如果你改变了工具选择的指令*并*添加了新示例，你将无法得知是哪个改变导致了观察到的性能差异。
2. **明确的效果衡量指标**：你需要可量化 (quantization)的衡量标准来判断哪个提示“更优”。这些指标应与智能体的目标一致，可包括：

   - **任务完成率**：尝试完成指定任务的成功百分比是多少？
   - **输出的准确性/质量**：检索到的信息有多准确，或生成内容多大程度上符合要求？这可能需要评分标准或人工评估。
   - **工具使用效率**：智能体是否选择了正确的工具？是否使用正确的参数 (parameter)恰当地使用了工具？
   - **步骤/轮数**：某个提示是否能带来更简洁的解决方案？
   - **错误率**：智能体多久遇到一次错误或需要自我修正？
   - **资源消耗**：如果适用，考虑API成本或处理时间，如果某个提示总是导致更复杂的LLM调用。
     请参考第一章中讨论的“评估智能体表现的方法”，以获取更广泛的潜在衡量指标列表。
3. **充足且多样的测试用例**：对有代表性的一组输入场景或任务运行两个提示变体。单个测试用例不足以得出可靠结论。测试集应覆盖常见情况以及潜在的边缘情况。
4. **一致的测试环境**：确保在测试过程中所有其他因素保持不变。这包括使用相同的LLM、相同的模型参数（如温度）、相同的可用工具，以及如果任务涉及信息检索，则使用相同的基础数据源。

### 示例：任务分解提示的A/B测试

假设一个智能体负责规划营销活动。

- **变体A（对照组）提示片段**：“将目标‘推出新产品X’分解为更小的步骤。”
- **变体B（挑战组）提示片段**：“你的目标是分解高层目标：‘推出新产品X’。识别至少5个实现此目标所需的明确、可操作的子任务。清晰列出每个子任务。”

你将对两个提示使用相同的目标（“推出新产品X”）运行多次，或使用几个类似的高层目标。衡量指标可以包括：生成的子任务数量、子任务的清晰度（需要一些人工判断），以及子任务是否逻辑上促进主目标。如果变体B持续生成更全面、更可操作的计划，那么它在智能体规划能力的这一特定方面被认为更有效。

## 组织提示比较实验

为了使A/B测试及其他比较方法易于管理且有效，请考虑以下几点：

### 1. 建立基线

在开始实验变体之前，请确保你有一个稳定的基线提示（你的初始变体A）。全面衡量其在测试用例中的表现。这个基线将提供一个参照点，所有未来的迭代都将与它进行比较。

### 2. 隔离更改

如前所述，在创建新变体时，一次只改变一个元素。如果你想测试措辞指令的三种不同方式以及两组不同的少量示例，这意味着需要创建几个明确的变体，每个变体与基线只在一个特定方面不同。

### 3. 使用测试框架

手动运行测试可能繁琐且容易出错。如果可能，开发一个简单的脚本或框架（一个“测试”），它可以：

- 接受一个提示变体和一组测试输入。
- 执行智能体工作流程。
- 捕获智能体的输出和相关的中间步骤（例如工具调用或内部思考，如果你的智能体架构暴露了这些）。
- 自动计算一些你预定义的衡量指标（例如，它是否调用了正确的API？）。

这种自动化允许对多个变体进行更快速、更可靠的测试。

### 4. 评估方法

- **定量衡量指标**：对于成功/失败、步骤数量或特定工具的选择等，你通常可以自动确定结果。
- **定性衡量指标**：对于解释的清晰度、检索信息的关联性或回复的语气等方面，人工评估通常是必要的。为了使其更客观：
  - **盲测比较**：评估者不应知道哪个提示产生了哪个输出。
  - **标准化评分标准**：定义清晰的准则，说明何为“良好”、“可接受”或“差”的质量。
  - **多人评分**：让不止一人评估输出有助于减少个人偏见。

### 5. 记录所有内容

仔细记录：

- 测试过的每个提示变体（使用版本控制，如后面章节所述）。
- 每个变体相对于其前身或基线所做的具体更改。
- 使用的测试用例。
- 每个变体在每个测试用例上的原始结果。
- 你所选衡量指标的汇总统计数据。
- 任何定性观察结果。

这份文档对于理解趋势、避免重复测试失败的想法以及建立关于哪些方法适用于你的特定智能体和任务的知识库而言，价值巨大。

## 可视化比较数据

简单的可视化通常可以让你更容易看出哪个提示变体表现更优。例如，条形图可以有效比较成功率或平均分数。



![提示变体的效果](plots/6753-0.json)



> 对照提示与两个变体的任务成功率比较，其中一个变体优化了指令以提高清晰度，另一个变体添加了少量示例。

## 两种变体：多变体和因子测试

虽然A/B测试（比较两个版本）是一个良好的起点，但有时你可能希望同时比较多个替代方案（A/B/n测试）。

对于更复杂的场景，特别是当你怀疑不同提示元素之间存在相互作用时，你可以考虑因子设计。在这种设计中，你测试多个因子（例如，指令类型、示例是否存在）在多个水平（例如，指令类型1与类型2；示例存在与否）下的表现。这不仅能让你看到每个因子的主要效应，还能看到它们如何相互作用。然而，这些设计需要显著更多的测试运行和更复杂的分析。对于大多数提示优化任务，迭代A/B测试是一种实用且有效的方法。

通过系统地比较提示变体，你将从临时调整转向数据驱动的优化过程。这种有条理的方法对于可靠地提升智能体效果以及构建更强大、更可预测的智能体系统是必要的。所获见解也有助于你全面理解如何最好地与LLM进行复杂任务沟通。

## 参考资料

- [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/9781108724265) — Ron Kohavi, Diane Tang, Ya Xu (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781009148383](https://doi.org/10.1017/9781009148383)
  提供了设计和解释A/B测试的基础知识，适用于提示词比较。
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) — Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang (2023)
  Journal: arXiv preprint arXiv:2308.03688; DOI: [10.48550/arXiv.2308.03688](https://doi.org/10.48550/arXiv.2308.03688)
  详细介绍了评估大型语言模型在代理场景中的基准和方法，有助于定义代理效能指标。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个评估语言模型在各种指标上的全面框架，为选择相关性能指标提供了信息。
