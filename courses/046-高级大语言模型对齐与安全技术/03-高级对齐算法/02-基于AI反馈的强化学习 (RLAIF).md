# 基于AI反馈的强化学习 (RLAIF)

来源：[原文](https://apxml.com/zh/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/reinforcement-learning-ai-feedback-rlaif)

[返回章节目录](README.md) · [返回课程目录](../README.md)

人类反馈强化学习 (reinforcement learning)（RLHF）是大型语言模型（LLMs）对齐 (alignment)的一项重要技术。然而，收集人类偏好数据的过程在时间、成本和规模方面常常是一个主要瓶颈。基于AI反馈的强化学习（RLAIF）提供了一种替代方法，其设计宗旨是专为通过用AI生成的偏好替代人类偏好来应对这一可扩展性难题。

### RLAIF机制：用AI评判者替代人类

RLAIF的核心思想简单直接：我们不再要求人类比较成对的模型输出并指明他们的偏好，而是指派另一个AI模型（通常被称为“偏好模型”或“评判模型”）来执行这种比较。整个工作流程与RLHF的流程相似，但取代了人工标注步骤。

1. **提示词 (prompt)抽样：** 选择一组多样化的提示词 $x$。
2. **回复生成：** 使用当前正在训练的LLM策略 $\pi_\theta$，为每个提示词 $x$ 生成两个或更多回复 $(y_1, y_2, ...)$。
3. **AI偏好生成：** 将提示词 $x$ 和成对的回复 $(y_i, y_j)$ 输入给一个单独的、通常更强大的AI偏好模型。这个模型会根据特定标准（例如，是否有用、无害、遵守某些原则）来评估哪个回复“更好”。它输出一个偏好标签，表明 $y_{preferred}$ 和 $y_{rejected}$。
4. **偏好数据集汇编：** 将这些AI生成的偏好对 $(x, y_{preferred}, y_{rejected})$ 汇编成一个数据集。
5. **奖励模型训练：** 训练一个奖励模型 $r_\phi(x, y)$ 来预测偏好模型的判断。目标通常是对于给定的 $x$，为 $y_{preferred}$ 分配比 $y_{rejected}$ 更高的分数，使用类似于RLHF中的损失函数 (loss function)，例如成对排序损失：
   
   $$
   L(\phi) = -E_{(x, y_p, y_r) \sim D_{AI}}[\log(\sigma(r_\phi(x, y_p) - r_\phi(x, y_r)))]
   $$
   
   其中 $D_{AI}$ 是AI偏好数据集，$\sigma$ 是sigmoid函数。
6. **策略优化：** 使用强化学习 (reinforcement learning)（通常是PPO）微调 (fine-tuning)原始的LLM策略 $\pi_\theta$，其中学习到的奖励模型 $r_\phi$ 提供奖励信号。目标是最大化 $r_\phi$ 所分配的预期奖励。

主要区别在于步骤3，其中耗时且成本高昂的人工标注被自动化AI判断取代。

> RLHF和RLAIF工作流程的比较，强调了在RLAIF中用AI偏好模型取代了人工标注员。

### 偏好AI模型

RLAIF的有效性完全取决于AI偏好模型的质量和性质。通常，这是一个功能强大的LLM，甚至可能比正在训练的模型更先进。偏好模型通常由一套明确的原则或指令指导，这些原则有时源自“章程”（将RLAIF与接下来讨论的宪法AI联系起来）。例如，它可能被指示偏好那些更有帮助、诚实且无害的回复，或者避免特定类型的不良内容。

使用AI评判者提供了潜在优势，如一致性以及系统地应用复杂规则集的能力。然而，它也带来了风险，即目标LLM可能只是学会与特定AI评判者的偏好、偏见和潜在特殊性保持一致，而不是与更广泛的人类价值观保持一致。

### RLAIF的优势

- **可扩展性：** RLAIF能够以比人工标注更大的规模和更快的速度生成偏好标签。这使得在更大的偏好数据集上进行训练成为可能。
- **成本降低：** 尽管运行偏好模型需要大量的计算资源，但RLAIF比雇用大量人工标注团队更具成本效益，尤其是在生成数百万个偏好标签时。
- **一致性：** AI偏好模型，特别是那些由明确规则指导的，可以提供比不同的人工标注群体更一致的判断，这可能使得奖励模型训练更稳定。
- **定向改进：** RLAIF可以通过相应地设计偏好模型的标准来改进特定模型能力，例如，提高编码能力或遵守特定角色设定。

### 缺点与挑战

- **对齐 (alignment)准确性：** 主要问题是AI偏好模型是否准确反映所需的人类价值观。与AI代理对齐可能不完全等同于与真实人类意图对齐。目标模型可能非常善于取悦AI评判者，但这可能无法转化为在一般人类互动中的理想行为。
- **偏见传播：** 偏好模型固有的任何偏见（从其自身的训练数据中学到或编码在其指导原则中的）都容易转移并可能在通过RLAIF训练的LLM中被放大。
- **规则漏洞利用：** 正在训练的LLM可能会找到利用偏好模型的逻辑或标准的方法，以获得高奖励分数，而没有真正改进其期望的行为。例如，如果AI评判者隐含地奖励这种行为，它可能会学会使用过于冗长或奉承的语言。
- **评估复杂性：** 评估AI生成的偏好本身的质量成为一个重大挑战。您如何验证AI评判者是否根据预期标准做出合理可靠的比较？这通常需要补充性的人工评估或检查。
- **计算成本：** 在大型AI偏好模型上对数百万次比较运行推理 (inference)，会给训练流程增加相当大的计算负担。

### RLAIF 与 宪法AI

区分RLAIF与宪法AI（CAI）很有用，尽管它们经常一起使用。

- **RLAIF** 是使用AI生成的反馈（偏好标签）来训练奖励模型以进行基于RL的微调 (fine-tuning)的*机制*。
- **宪法AI** 主要指使用一套明确的原则（“章程”）来指导模型行为。这种指导可以通过多种方式实现：
  - 在监督式微调期间，通过根据章程生成被评论和修订的回复示例。
  - 作为RLAIF流程中AI偏好模型的指导原则（一种常见实现方式）。

因此，RLAIF可以被视为落实章程中所列原则的一种方式，利用AI规模进行反馈过程。

总而言之，RLAIF通过自动化反馈生成过程，为扩展LLM对齐 (alignment)能力提供了一个有吸引力的方法。尽管它克服了人工标注速度和成本的局限性，但它带来了与AI偏好准确性、偏见传播以及AI评判者本身评估相关的新挑战。它代表了先进对齐工具箱中的一个重要工具，当期望行为的标准能够被清晰地阐明并由另一个AI系统评估时，它尤其有效。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  提出了宪法式AI，一种利用一套原则指导AI模型提供反馈并精炼自身输出的方法，是使用AI评审的理论基础。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  一篇基础性论文，展示了人类反馈强化学习（RLHF）在将大型语言模型与人类对帮助性和无害性的偏好对齐方面的有效性。

---

[上一节](01-%E5%AE%AA%E6%B3%95%E5%BC%8FAI%EF%BC%9A%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E8%B7%B5.md) · [下一节](03-%E7%9B%B4%E6%8E%A5%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%20%28DPO%29.md)
