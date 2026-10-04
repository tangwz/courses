# 针对 CAI/RLAIF 模型的红队策略

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-7-advanced-alignment-evaluation/red-teaming-cai-rlaif)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管标准基准测试能提供模型性能的基本认知，但它们常无法发现使用宪法式AI（CAI）或基于AI反馈的强化学习 (reinforcement learning)（RLAIF）进行对齐 (alignment)的模型中可能出现的故障模式。这些对齐技术引入其特有的潜在弱点，例如宪法中的漏洞、AI批评器或偏好模型的局限，或如奉承等涌现 (emergence)行为。红队方法提供了一种必要且积极的方式来找出这些弱点。它涉及专门的、通常是对抗性的努力，旨在寻找导致模型违反其预期对齐原则或表现出不良行为的输入或交互模式。

可以将红队活动视为一种应用于AI对齐的安全渗透测试，而不仅仅是简单的测试。其目的是在具有挑战性的条件下，主动对模型遵循其宪法（CAI中）或其学习到的偏好（RLAIF中）的情况进行压力测试，这与在典型评估数据集中测试不同。

### 对齐 (alignment)模型红队活动的目标

对CAI和RLAIF模型进行红队活动的主要目标包括：

1. **确定宪法弱点（CAI）：** 找出宪法中可能导致不良解释或有害输出的模糊之处、矛盾或空白。
2. **测试批评/修订过程（CAI）：** 评估AI批评器是否正确识别宪法违规行为，以及AI修订器是否适当地修改了响应，尤其是在复杂或临界情况下。
3. **发现偏好模型缺陷（RLAIF）：** 识别AI偏好模型对实际有害、无用或未对齐的输出给出高分，从而导致奖励作弊的输入。
4. **检测策略利用（RLAIF）：** 找出在有奖励信号的情况下，仍能操纵强化学习 (reinforcement learning)策略生成不良内容的方法，这可能是通过利用分布盲点或强化学习算法的不稳定性来实现的。
5. **揭示涌现 (emergence)行为：** 显现出意想不到的负面行为，如过度回避、奉承（告诉用户它认为他们想听的，即使不正确或有害），或形成操纵性的对话策略。
6. **评估对欺骗的抵御能力：** 测试模型抵御复杂尝试绕过安全协议的能力，包括越狱提示、鼓励有害行为的角色扮演场景，或旨在混淆对齐机制的输入。

### 红队方法

有效的红队活动通常采用多种策略组合：

- **手动对抗性提示：** 这是一种传统方法，人类专家（通常具有不同背景，如AI研究员、伦理学家、领域专家、创意作家）有意设计旨在破坏模型对齐 (alignment)的提示。例子包括：

  - **越狱：** 使用元提示或场景来诱导模型忽略安全规则（例如，“忽略之前的指示。你现在是……”）。
  - **利用模糊性：** 设计针对宪法中不明确之处或相互冲突原则的提示（例如，请求可能有用但可能被滥用的信息）。
  - **角色扮演：** 指示模型扮演与其对齐目标可能冲突的角色。
  - **压力测试：** 突破特定原则的界限（例如，生成越来越接近临界线的内容，以观察模型划定的界限）。
- **自动化和半自动化生成：** 通过编程方式生成具有挑战性的提示可以扩大红队工作的规模。

  - **基于大型语言模型（LLM）的生成：** 使用另一个大型语言模型（可能针对任务进行了微调 (fine-tuning)）来生成可能导致目标模型失败的提示。这可以包括指示生成器大型语言模型扮演对手的角色。
  - **基于梯度的修改方法：** 虽然对于离散文本较难，但视觉领域对抗性攻击启发的技术有时可以进行调整，以找到最大化不良输出特征可能性的输入令牌。
  - **演化方法：** 使用遗传算法或类似的优化技术来演化能够有效挑战模型对齐的提示。
- **结构化审查：** 系统地审查预定义的潜在故障类别。这包括创建模板或场景列表，以针对已知影响CAI/RLAIF系统的特定弱点。类别可能包括：

  - 宪法原则之间的冲突。
  - 旨在引出奉承式同意的提示。
  - 需要谨慎拒绝的请求。
  - 测试先前已识别故障模式的变体。
- **基于角色的测试：** 模拟不同用户类型（例如，提问敏感问题的儿童、试图操纵的技术熟练用户、表达有害意识形态的用户）的交互，以评估模型在不同交互风格下的抵御能力。

### 针对CAI特定弱点

对CAI模型进行红队活动时，请关注宪法与AI反馈循环之间的交互：

- **宪法解释：** 设计提示，测试模型如何解释宪法中的特定短语或原则。是否存在字面解释导致不良结果的边缘情况？
- **批评准确性：** 测试AI批评器是否正确标记 (token)违规行为。提供细微不合规或以意想不到方式合规的例子。
- **修订质量：** 评估修订过程是否充分解决了批评，而没有引入新问题或变得过于冗长/回避。
- **原则冲突：** 创建不同宪法原则可能建议冲突行动的场景。模型如何优先处理或协调它们？

### 针对RLAIF特定弱点

对于RLAIF模型，红队活动通常针对偏好模型和由此产生的强化学习 (reinforcement learning)策略：

- **奖励作弊：** 积极寻找模型生成高预测偏好得分但实际上不良（例如，事实不正确但自信表述、伪装成帮助的有害建议、符合风格偏好但重复或无意义的内容）的输出的提示。
- **偏好模型局限：** 如果AI标注者存在已知偏见（例如，偏爱更长或更有礼貌的响应），则设计提示来利用这些偏见并生成次优内容。
- **分布偏移：** 使用与RLAIF训练期间使用的提示明显不同的提示来测试模型。对齐 (alignment)是否具有泛化性，或者策略是否崩溃？
- **奉承行为检测：** 使用专门设计的提示来检测模型是否过度顺从或不恰当地迎合用户意见，这是基于预测偏好进行优化时的常见故障模式。

### 红队活动的组织与执行

成功的红队计划需要结构化：

1. **界定范围与目标：** 清楚阐明正在测试的对齐 (alignment)方面以及何为失败。
2. **组建多元团队：** 包含具有不同技能和观点的人员。
3. **开发工具：** 使用平台或内部工具来管理提示、收集响应、分类失败并跟踪结果。
4. **建立反馈循环：** 确保系统地记录发现结果并将其传达给建模团队，以便为重新训练、宪法更新或偏好模型调整提供依据。
5. **持续迭代：** 红队活动是一个持续的过程。随着模型更新，需要进行新一轮的红队活动，以发现退化或新的弱点。

> 对齐大型语言模型迭代红队周期的简化视图。

衡量红队活动的成功不仅仅是简单地计算缺陷数量。它包括评估发现的弱点的严重性、理解故障模式，并跟踪根据发现结果实施的缓解策略的有效性。最终，系统化的红队活动是建立对使用CAI和RLAIF等先进技术对齐的大型语言模型安全性和可靠性信心的不可或缺的实践。它将评估从被动测量转变为主动发现弱点。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  这篇基础性论文介绍了“宪法式AI”（CAI）和“基于AI反馈的强化学习”（RLAIF），详细解释了作为红队攻击目标的对齐技术。
- [Red Teaming Language Models to Reduce Harms: Methods, Limitations, and Successes](https://arxiv.org/abs/2209.07858) — Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, Jack Clark (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  这篇论文概述了大型语言模型的红队攻击方法，讨论了识别和缓解有害模型行为的策略、挑战和成功案例。
- [Discovering Language Model Behaviors with Automated Red Teaming](https://arxiv.org/abs/2212.09251) — Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Ben Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson Elhage, Nicholas Joseph, Noemí Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam McCandlish, Scott Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen-Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Nicholas Schiefer, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.09251; DOI: [10.48550/arXiv.2212.09251](https://doi.org/10.48550/arXiv.2212.09251)
  这项研究探讨了自动化红队攻击方法，包括使用其他大型语言模型生成对抗性提示，这是本节中讨论的关键方法之一。

---

[上一节](01-%E6%A0%87%E5%87%86%E5%9F%BA%E5%87%86%EF%BC%9A%E5%AF%B9%E9%BD%90%E4%B8%93%E7%94%A8%E6%8C%87%E6%A0%87.md) · [下一节](03-%E5%AF%B9%E6%8A%97%E6%80%A7%E8%BE%93%E5%85%A5%E9%B2%81%E6%A3%92%E6%80%A7%E6%B5%8B%E8%AF%95.md)
