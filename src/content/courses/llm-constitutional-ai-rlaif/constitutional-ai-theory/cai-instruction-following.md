---
course: "llm-constitutional-ai-rlaif"
chapter: "constitutional-ai-theory"
lesson: "cai-instruction-following"
sourceId: 4455
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-2-constitutional-ai-theory/cai-instruction-following"
title: "与指令遵循的关联"
description: "CAI与标准指令微调技术进行比较和区分。"
order: 5
plots: []
sourceHash: "25850ff487fb756fc1af84e6d161fbb24b52193dbf37ce90c7d59095b72560ee"
sourceCorrections: []
---

尽管宪法AI（CAI）和标准指令微调 (fine-tuning)（IFT）都旨在塑造大型语言模型（LLM）的行为，但它们通过不同的机制运行，并针对模型对齐 (alignment)的不同方面。了解它们之间的联系有助于认识CAI在LLM开发工具集中的具体作用。

IFT主要侧重于教导LLM准确有效地遵循明确指令。它是一种监督学习 (supervised learning)形式，模型在通常由对组成的数据集上训练：`（指令，期望输出）`。目标是使模型生成的响应与给定`指令`所提供的`期望输出`之间的差异最小化。这种方法对于赋予模型特定技能非常有效，使其能够执行诸如摘要、翻译、问答等任务，或按指示采取特定角色或格式。监督信号是直接的：对于*此*输入指令，生成*此*输出。

宪法AI，尤其是在本章中讨论的其初始监督学习（SL）阶段，采用一种不同方法。CAI不是优化对人类提供的特定输入-输出示例的遵循，而是优化对一套明确书面原则（即宪法）的遵循。目标不仅仅是任务执行，而是由这些原则指导的行为调整。

以下是主要区别的细分：

### 目标与监督来源

- **IFT:** 目标是任务能力和指令遵循。监督来自于精心组织或人类生成的正确指令执行示例。模型学习从指令提示到期望响应格式或内容的直接映射。
- **CAI (SL Phase):** 目标是原则遵循。监督来自于AI驱动的自我修正过程。模型首先对提示生成一个初始响应。然后，在宪法指导下，一个AI系统（或可能是同一LLM在适当提示下）根据这些原则批判该响应，并随后进行修订。微调 (fine-tuning)数据集包含源自此过程的示例，通常训练模型根据初始提示预测*修订后*的响应。学习信号是间接的，通过在批判和修订步骤中应用宪法原则来调节。

### 控制的性质

- **IFT:** 对模型应如何响应训练数据中存在的特定类型指令提供直接控制。它擅长教导模型在明确的场景中*做什么*。
- **CAI:** 提供一种更具通用性的行为控制形式。它旨在灌输总体规则或约束（例如“无害”或“避免偏见性言论”），应适用于广泛的输入，即使是训练期间未曾遇到的输入。它更侧重于模型通常应*如何*表现，而不是为每条可能的指令指定确切的输出。

### 数据生成过程

- **IFT:** 数据生成通常涉及大量人工投入，用于编写指令和相应的**高质量输出**，或采用复杂方法合成此类对。这些对的质量控制是一个核心问题。
- **CAI:** 数据生成依赖于AI批判和修订模型的效力，这些模型在宪法指导下运作。人工投入转向设计全面且有效的宪法，并可能监督或改进AI反馈过程本身。此过程可以生成大量训练数据，这些数据体现了期望的原则，而无需人类为每个提示指定“正确”的输出。

### 泛化特性

- **IFT:** 泛化能力在很大程度上取决于指令数据集的广度和质量。对于与训练分布差异很大的提示，模型可能难以遵循指令或展现期望行为。
- **CAI:** 旨在基于原则实现更广泛的行为泛化。如果宪法能有效捕捉到期望的行为特征，模型理想情况下即使面对新情况或提示也应展现这些特征，因为底层原则指导其响应生成过程。

### 互补性

需要认识到IFT和CAI并非相互排斥；它们可以是互补的技术。LLM可以首先进行广泛的指令微调 (fine-tuning)，以获得多种技能和遵循指示的能力。随后，CAI方法（包括此处描述的监督阶段和稍后讨论的强化学习 (reinforcement learning)阶段）可用于使模型的行为与宪法中概述的特定伦理或安全原则保持一致，从而改进其执行这些指令的*方式*。这种分层能够构建既有能力又符合期望规范的模型。

从根本上说，IFT教导模型遵循指令，而CAI（在其SL阶段）开始教导模型一套内部规则来指导其行动，这些规则源自所提供的宪法。这种区别阐明了CAI在可扩展LLM对齐 (alignment)的难题中发挥的独特作用。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073 [cs.CL]; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法式AI，阐述了其通过监督学习和强化学习阶段使大型语言模型与原则保持一致的方法。
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS) 30; DOI: [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741)
  一篇基础性论文，介绍了通过优化源自人类偏好的目标函数来训练强化学习代理的方法。
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint arXiv:2210.11416; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  讨论了指令微调的扩展对大型语言模型性能和泛化能力的影响。
