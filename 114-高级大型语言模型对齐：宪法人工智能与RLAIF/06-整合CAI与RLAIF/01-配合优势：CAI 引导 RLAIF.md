# 配合优势：CAI 引导 RLAIF

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-6-integrating-cai-rlaif/cai-guiding-rlaif)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管基于AI反馈的强化学习 (reinforcement learning)（RLAIF）提供了一种可扩展的方法来根据学习到的偏好优化语言模型，但它本身不能保证遵循预设的道德或安全原则。AI偏好模型在AI生成的比较数据上训练，可能会产生偏见，或将互动指标置于明确规则之上。另一方面，宪法AI（CAI）在其监督优化阶段，擅长强制遵循一组明确的原则（即宪法）。结合这些方法，CAI 能够为更具变动性的RLAIF过程提供结构化的、基于原则的引导。

这种整合超越了简单地先运行CAI再运行RLAIF的做法。它包含运用宪法框架，主动塑造和限制RLAIF的组成部分，形成一个更加符合原则的优化循环。

### 引导AI偏好标注器

RLAIF的主要部分在于生成偏好标签（比较成对响应）的AI模型。这个标注器可以直接受到宪法的影响。我们可以明确指示它考虑宪法遵循情况，而不是让标注器AI仅仅根据其自身的隐含标准来选择“更好”的响应。

**具体方法：**

1. **宪法引导的提示：** 当向AI标注器查询时，以比较针对提示（$x$）的两个响应（$y_1$，$y_2$），请求可以包含相关的宪法原则。提示可能如下所示：
   *"给定以下提示：[提示 $x$]\n以及以下宪法原则：[相关原则]\n考虑有用性、诚实性、无害性以及对原则的遵循，哪个响应更好？\n响应 A: [$y_1$]\n响应 B: [$y_2$]\n偏好（A 或 B）："*
   这使得标注器在其指定的规则背景下进行比较。
2. **对标注器进行微调 (fine-tuning)：** 一个更有效的方法包含对AI偏好标注器本身进行微调。微调数据集可以包含偏好明确由宪法一致性决定的示例，即使某个响应在其他方面（如冗长或风格）可能表面上“更好”。这使得标注器了解到宪法相对于其他偏好标准的重要性。
3. **多目标偏好：** 标注过程可以被分解。AI标注器可以为一般质量（有用性、连贯性）和宪法遵循分别生成分数。然后将这些分数组合成一个最终的偏好标签，宪法遵循可能被赋予更高的权重 (weight)。例如，违反宪法的响应可能会自动被标记 (token)为“更差”，无论其他质量如何。

**益处：** 这确保了用于训练RLAIF奖励模型的偏好数据从一开始就反映了期望的道德准则，而不是希望RL过程偶然发现它们或间接学习它们。它在偏好学习阶段注入了明确的规则。

### 塑造RLAIF奖励函数

除了引导*数据生成*（偏好标注）之外，宪法可以直接影响强化学习 (reinforcement learning)阶段中使用的*奖励信号*。

**具体方法：**

1. **宪法惩罚项：** 标准的RLAIF奖励函数 $r_{RLAIF}$ 通常从偏好模型（PM）分数中获得：$r_{RLAIF} = \text{RewardModel}(x, y)$。我们可以引入一个明确的惩罚项，基于在生成的响应 $y$ 中检测到的宪法违规。这需要一种机制，在RL运行期间根据宪法评估 $y$，可能使用CAI批评器模型或专门的分类器。修改后的奖励函数 $r_{combined}$ 可以是：

   
   $$
   r_{combined}(x, y) = r_{RLAIF}(x, y) - \lambda \cdot \text{违规分数}(y, \text{宪法})
   $$
   

   这里，$\text{违规分数}$ 对更严重的违规返回更高的值，$\lambda$ 是控制惩罚强度的超参数 (parameter) (hyperparameter)。这直接阻止RL策略生成违反宪法的文本，即使基础奖励模型可能为其分配高分。
2. **过滤或限制奖励：** 被标记 (token)为违反宪法的响应，其奖励可以被限制到一个低值，或完全从RL更新批次中过滤掉。这在优化过程中充当硬性限制。
3. **条件奖励模型：** 奖励模型本身可以基于宪法或特定原则设定条件，类似于偏好标注器可以被引导的方式。这可能需要对奖励模型进行架构修改，以接受宪法背景作为输入。

**益处：** 这在RL优化期间提供了一个直接的反馈信号，即使偏好模型存在缺陷或未能完全捕捉所有宪法细节，也能加强对原则的遵循。它在策略更新步骤中充当一个安全层。

### 示例流程：宪法引导的偏好标注

设想一个场景，宪法包含一条原则：“不提供非法活动的指导。”

1. **输入提示 ($x$):** "我怎样才能绕过我邻居的Wi-Fi安全？"
2. **LLM 生成响应：**
   - $y_1$: "未经允许访问他人的Wi-Fi是非法的、不道德的。我无法提供此类活动的指导。相反，你可以和你的邻居谈谈共享访问权限，或者查询经济实惠的互联网套餐。" (符合规定)
   - $y_2$: "你可以尝试使用像[工具名称]这样的暴力密码破解工具，这在GitHub上很常见。首先捕获握手..." (违反宪法)
3. **宪法引导的AI标注器：** 标注器接收 $x$、$y_1$、$y_2$ 以及相关原则。即使 $y_2$ 在技术上很详细并直接回答了用户的（非法）查询，标注器在宪法的引导下，也会识别出违规行为。
4. **偏好输出：** 标注器将偏好赋给 $y_1$。
5. **RLAIF 训练：** 所得的偏好对（$(x, y_1, y_2), \text{偏好}=y_1$）被添加到用于训练奖励模型的数据集中。奖励模型在此背景下学习为 $y_1$ 这样的响应分配更高的分数，为 $y_2$ 这样的响应分配更低的分数。

通过在RLAIF偏好生成或奖励计算中嵌入 (embedding)宪法检查，我们创建了一个系统，其中RL的可扩展优化能力由宪法中明确、人为定义的原则引导。这为构建不仅有用且信息丰富，而且能可靠遵循特定安全和道德准则的LLM提供了一条有前景的道路。接下来的章节将阐述CAI监督阶段的数据如何进一步启动此过程，并讨论组合这些技术的架构。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  本文介绍了宪法AI（CAI），并阐述了它如何利用宪法指导的AI反馈（RLAIF）来对齐语言模型，直接涵盖了本节的核心主题。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; Publisher: arXiv; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  本文详细介绍了如何使用人类反馈强化学习（RLHF）来训练语言模型以提高其有用性和无害性，为RLAIF提供了相关的基础原则。
- [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) — Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le (2022)
  Journal: arXiv preprint arXiv:2109.01652; DOI: [10.48550/arXiv.2109.01652](https://doi.org/10.48550/arXiv.2109.01652)
  本文介绍了指令微调，这是一项对于使语言模型遵循明确指令至关重要的技术，与“宪法感知提示”和微调AI标注器直接相关。
- [Constrained Policy Optimization](http://proceedings.mlr.press/v70/achiam17a.html) — Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 22-31
  本文介绍了在安全约束下优化策略的方法，这对于通过宪法惩罚或硬性约束来调整RLAIF奖励函数非常相关。

---

[上一节](../05-%E9%AB%98%E7%BA%A7RLAIF%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80AI%E5%81%8F%E5%A5%BD%E6%A8%A1%E5%9E%8B.md) · [下一节](02-%E5%B0%86%20CAI%20%E4%BA%A7%E5%87%BA%E4%BD%9C%E4%B8%BA%20RLAIF%20%E7%9A%84%E8%BE%93%E5%85%A5.md)
