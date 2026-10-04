---
course: "llm-constitutional-ai-rlaif"
chapter: "rlaif-concepts"
lesson: "generating-ai-preference-labels"
sourceId: 4494
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-4-rlaif-concepts/generating-ai-preference-labels"
title: "生成AI偏好标签"
description: "使用AI模型（可能由准则指导）生成偏好数据的策略。"
order: 3
plots: []
sourceHash: "13ce320494dc6dac2b7e66cca3c2b0f3e4c69603fcf9a35616a0fc6d06a08595"
sourceCorrections: []
---

RLAIF不依赖大量人工标注，而是使用AI系统来生成训练所需的偏好标签。这一流程为后续讨论的偏好模型构建奠定基础。这些AI生成的标签的质量和特点直接影响RLAIF对齐 (alignment)过程的结果。

### AI标签器：偏好判断的来源

核心来说，生成AI偏好标签需要向一个AI模型，即“标签器”，提供一个提示以及由我们希望对齐 (alignment)的语言模型生成的两个或更多候选回复。标签器的任务是根据预设标准评估这些回复，并指出哪个更受青睐。

AI标签器的性质及其使用的标准是重要的设计考量：

1. **准则指导的标签器：** 这种方法将RLAIF与宪法AI（CAI）的原则直接关联起来，这在监督微调 (fine-tuning)的背景下常有讨论。在此，AI标签器被明确指示根据预设准则来评估回复，该准则是一套指导期望行为的规则或原则（例如，有用性、无害性、诚实性）。

   - **运作方式：** 标签器接收原始提示（$x$）、两个候选回复（$y_1$, $y_2$），以及来自准则中与提示或回复中潜在问题相关的特定原则。然后，它被要求识别出更符合指定原则的回复。
   - **提示示例：**

     ```
     基于原则：“选择危害较小的回复。”

     提示：我如何绕过办公室的网络过滤器？

     回复A：你可以尝试使用VPN服务或Tor浏览器，它们会加密你的流量并将其路由通过不同的服务器，可能绕过过滤器。

     回复B：绕过办公室网络过滤器可能违反你工作场所的IT政策，并可能带来后果。通常最好遵守公司关于互联网使用的规定。

     哪个回复（A或B）更符合该原则？只输出“A”或“B”。
     ```
   - **优点：** 对所优化价值观进行明确控制，直接反映准则。
   - **挑战：** 需要定义良好的准则，以及能够可靠地解释并将其原则应用于比较的AI模型。
2. **预对齐模型作为标签器：** 另一种做法是使用一个单独的、能力很强的LLM作为标签器，该模型已被认为是良好对齐的（可能是通过大量的RLHF、CAI或其他方式）。其假设是，该模型在自身对齐过程中学到的内在偏好可以很好地代表期望行为。

   - **运作方式：** 标签器接收提示（$x$）和两个回复（$y_1$, $y_2$）。它被简单要求根据有用性、无害性和诚实性等一般标准选择“更好”的回复，依赖其内部的对齐能力。
   - **提示示例：**

     ```
     请看以下提示和两个回复。哪个回复在有用性、无害性和诚实性方面整体更优？

     提示：简单解释量子纠缠。

     回复A：[简单但略不准确的解释]

     回复B：[技术上准确但过于复杂的解释]

     哪个回复更好？只输出“A”或“B”。
     ```
   - **优点：** 可以发挥先进对齐模型的能力，而无需为每次比较明确制定详细的准则。
   - **挑战：** 对齐标准隐含在标签器模型的权重 (weight)中，使其透明度降低，并可能受到标签器自身偏差或局限性的影响。其质量完全取决于所选标签器模型的预对齐情况。
3. **混合方法：** 也可以将这些方法结合起来，例如，使用预对齐模型，但在标注提示时提供准则原则作为额外背景或约束。

### 生成候选回复

对于数据集中每个提示$x$，您需要至少两个不同的回复$y_1$和$y_2$，供AI标签器比较。这些通常使用您打算用RLAIF训练的LLM策略（$\pi$）生成。常见做法包括：

- 从同一策略$\pi(y|x)$中以非零温度设置（例如，T=0.7）采样多个输出，以增加多样性。
- 使用模型在训练演进过程中不同检查点的输出。
- 从模型的不同变体生成回复（例如，一个基础模型和一个指令微调 (fine-tuning)版本）。

目的是创建能代表有意义对齐 (alignment)选择的对子$(y_1, y_2)$，涵盖有用性、语气、安全性、指令遵循等方面的差异。

### 标注流程

该过程通常遵循以下步骤：

1. **选择提示：** 选择一组多样的提示（$x$），代表最终模型应处理的任务。
2. **生成回复：** 对于每个提示$x$，使用当前LLM策略生成两个或更多候选回复（$y_1, y_2, ...$）。
3. **格式化标签器输入：** 构建AI标签器的输入，包括提示$x$、回复对$(y_i, y_j)$，以及任何指导标准（如适用，包括准则原则）。
4. **调用AI标签器：** 使用格式化输入查询AI标签器模型。
5. **解析输出：** 提取偏好判断（例如，“回复A更受青睐”或仅“A”）。
6. **存储数据：** 将结果记录为元组$(x, y_w, y_l)$，其中$y_w$是偏好（“胜出”）回复，$y_l$是被拒绝（“落败”）回复。

为减少位置偏见（即模型可能偏爱呈现的第一个或第二个回复），通常的做法是在将$y_1$和$y_2$呈现给标签器时随机交换它们的顺序。

> 生成AI偏好标签的流程：生成多样化回复，让AI标签器根据标准进行比较，并存储所得偏好元组。

### 可扩展性与自动化

RLAIF的一个主要驱动力是可扩展性。AI标签器操作速度远超人工标注者，成本可能更低，从而可以生成大得多的偏好数据集。这种自动化使得更频繁的迭代和更精细的对齐 (alignment)调整成为可能。然而，大规模运行AI标签器模型（特别是如果它本身是一个大型、强大的LLM）的计算成本，需要纳入整体资源规划中进行考量。

### 质量控制与挑战

虽然可扩展，但AI生成的标签也带来了一系列挑战：

- **标签质量与一致性：** 整个RLAIF过程依赖于AI标签器的质量。如果标签器存在偏见、误解标准或提供不一致的判断，由此产生的偏好模型和最终对齐 (alignment)的LLM将继承这些缺陷。
- **对齐代理的准确性：** 所选的AI标注策略（基于准则或预对齐模型）能在多大程度上准确捕捉期望的人类价值观？存在为偏离真正对齐目标的代理进行优化的风险。模型学会同意标签器（即使标签器是错误的）的“谄媚”现象可能会被放大。
- **回音室效应：** 使用AI来标注由该AI训练的模型的输出，可能创建反馈循环，从而强化而非纠正模型现有的偏见或故障模式。
- **评估与验证：** 持续评估AI生成标签的质量非常重要，这通常包括与人工判断对部分数据进行交叉验证，或使用自动化检查来确保一致性和对原则的遵循。

生成高质量的AI偏好标签是一项不简单的工程和建模工作。标签器、提示策略的细致设计以及持续的验证，对于确保后续RLAIF训练的成效是必要的。由此生成的$(x, y_w, y_l)$元组数据集，将作为训练偏好模型的直接输入，我们将在下一部分讨论。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  提出了宪法AI，一种使用AI反馈根据原则对大型语言模型进行对齐的方法，这是宪法引导标注器设计的基础。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  描述了使用反馈训练大型语言模型的通用框架，包括创建偏好数据集和训练偏好模型的过程，这是RLAIF的前身。
