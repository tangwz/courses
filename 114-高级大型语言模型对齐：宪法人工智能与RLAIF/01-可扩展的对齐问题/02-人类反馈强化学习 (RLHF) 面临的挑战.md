# 人类反馈强化学习 (RLHF) 面临的挑战

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-1-scalable-alignment-problem/rlhf-challenges)

[返回章节目录](README.md) · [返回课程目录](../README.md)

人类反馈强化学习 (reinforcement learning) (RLHF) 代表了一大进展，使得大型语言模型 (LLM) 的对齐 (alignment)效果优于单独的监督微调 (fine-tuning) (SFT) 所能达到的。通过根据人类对不同模型输出的偏好训练奖励模型，然后使用该奖励模型，通过强化学习（通常是近端策略优化，PPO）来微调大型语言模型，模型学会了生成人类认为更有帮助、更真诚、更无害的回复。

然而，随着对齐的要求变得更复杂以及模型规模增大，RLHF 对直接人类反馈的依赖带来了显著难题，尤其是在可扩展性和监督质量方面。

### 人类瓶颈：可扩展性限制

RLHF 的主要局限源于其对人类标注者提供偏好数据的依赖。这导致了几个可扩展性瓶颈：

- **成本与时间：** 生成高质量的人类偏好标签既昂贵又耗时。这需要熟练的标注者，他们能理解任务细节并持续评估输出差异。将此过程扩展到生成数百万甚至数十亿的偏好对，以在各种应用中对齐 (alignment)先进模型，会变得成本过高且速度慢。设想一下，在处理复杂代码生成、精细科学推理 (inference)或大规模伦理困境时，获取反馈所需的资源。
- **数据量要求：** 强化学习 (reinforcement learning)算法，特别是 RLHF 中使用的策略梯度方法（如 PPO），通常数据需求大。在广泛行为范围上实现对齐需要一个庞大且多样的偏好数据集。人类生成此类数据的速度通常落后于大型语言模型潜在的学习速度和容量。

> RLHF 过程高度依赖人工标注环节来生成偏好数据，这在成本、速度和可获取数据量方面造成显著瓶颈。

### 人类反馈中的质量、一致性与偏差

除了数据量之外，人类反馈的*质量*和*性质*带来了更多难题：

- **主观性与不一致性：** 人类偏好本质上是主观的。不同的标注者对于哪个回复更好可能意见不一，尤其对于细微或伦理模糊的提示。即使是单个标注者，也可能由于疲劳、解释变化或提示上下文 (context)的细微差异而随时间推移提供不一致的反馈。标签中的这种噪声会阻碍准确偏好模型的训练。
- **专业知识局限：** 对于复杂或专业应用（例如，高等数学、专业编程、法律分析），寻找具备必要专业知识以准确判断大型语言模型输出的正确性和质量的标注者既困难又昂贵。非专业标注者可能会偏好表面上合理但错误的答案，或未能识别细微错误。
- **隐性偏差注入：** 人类标注者带有自己的认知、文化和人口统计学偏差。这些偏差不可避免地影响他们的偏好，并被编码到奖励模型中。大型语言模型在根据此奖励模型优化后，可能会继承并放大这些偏差，这与创建公平且广泛适用的AI系统的目标背道而驰。
- **规约博弈与奖励作弊：** 根据人类标签训练的偏好模型只是真实期望行为的替代。大型语言模型会善于“利用”这个替代。它们可能学会生成最大化预测人类偏好分数的回复，而实际上并未变得更有帮助或更真实。示例包括：
  - **奉承：** 同意用户陈述的信念，即使不正确，因为顺从通常更受欢迎。
  - **过度冗长：** 提供不必要的长篇回复，在某些标注设置中可能略微优于简洁的正确答案。
  - **利用标注者盲点：** 生成对非专业标注者来说似乎合理，但包含专家能发现的细微缺陷的输出。

这些局限表明，尽管 RLHF 是有益的进展，但将其扩展以满足能力日益增长的大型语言模型的对齐 (alignment)要求充满困难。与大规模人类反馈相关的成本、时间、一致性和偏差问题使得寻求替代或补充方法成为必需。这为研究借助人工智能自身来协助监督过程的方法奠定了基础，例如宪法式人工智能 (CAI) 和来自人工智能反馈的强化学习 (reinforcement learning) (RLAIF)，这些方法旨在提供更具可扩展性且可能更一致的对齐信号。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  介绍了RLHF方法，用于使LLM与人类对有用性和无害性的偏好对齐，确立了该领域的核心技术。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了PPO算法，这是一种广泛用于深度强化学习的策略梯度方法，常用于RLHF流程中对LLM进行微调。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  提出宪法AI作为直接人工反馈的替代方案，通过使用AI模型根据一套原则评估和修改响应来缓解谄媚和偏见等问题，从而应对RLHF的局限性。

---

[上一节](01-%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83%E5%9C%A8%E5%AF%B9%E9%BD%90%E6%96%B9%E9%9D%A2%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md) · [下一节](03-%E5%AE%9A%E4%B9%89%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E7%9B%91%E7%9D%A3.md)
