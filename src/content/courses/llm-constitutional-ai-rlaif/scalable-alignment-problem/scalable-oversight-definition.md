---
course: "llm-constitutional-ai-rlaif"
chapter: "scalable-alignment-problem"
lesson: "scalable-oversight-definition"
sourceId: 4434
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-1-scalable-alignment-problem/scalable-oversight-definition"
title: "定义可扩展的监督"
description: "阐述可扩展监督的理念，作为先进对齐的一项要求。"
order: 3
plots: ["plots/4434-0.json"]
sourceHash: "3fb5cbef1ac9cf3cf3b8e622d98855016ae7bbab9ac7a0b82ae21ba551d1a7c8"
sourceCorrections: []
---

用于对齐 (alignment)的监督微调 (fine-tuning)（SFT）和来自人类反馈的强化学习 (reinforcement learning)（RLHF）都面临着重要的规模化难题。SFT 需要大量高质量、由人类制作的数据，涵盖无数场景，随着模型能力和所需对齐范围的增长，这变得难以处理。RLHF 尽管有效，但依赖于持续的人类偏好标注，形成了一个瓶颈，限制了反馈的数量和多样性，可能引入偏见，并且无法跟上模型交互的体量。

这使我们认识到**可扩展的监督**的必要性：指导和监督AI行为的机制，其中所需的人力投入增长速度显著慢于AI运作的规模或其承担任务的复杂性。理想情况下，人力投入应与AI交互或决策的数量呈次线性关系增长，或者主要集中在系统设计、定期评估和更新上，而不是持续的逐例监督。

设想一个大型语言模型（LLM）每天处理数十亿次交互。即使是其中极小一部分的直接人工监督也是不可行的。即使是进行交互采样的RLHF，也需要大量持续的人工标注工作（$E_{RLHF} \propto k \times T_{label}$，其中 $k$ 是标注样本的数量，$T_{label}$ 是每个标注的平均时间）。如果所需对齐的复杂性需要更详细的比较（$T_{label}$ 增加）或更广的覆盖（$k$ 增加），人力成本很快就会变得过高。

因此，可扩展的监督与要求对模型大部分输出或行为进行直接人工判断的方法形成鲜明对比。相反，它意味着系统应具备以下特点：

1. **运用人类输入：** 人类提供高层次的指导、原则（如章程）、目标或评估标准，而非针对单个数据点的详细标注。
2. **采用自动化：** 自动化流程，可能借助AI自身，用于大规模检查是否符合准则、生成反馈或识别有问题输出。
3. **侧重转向系统级设计：** 人力投入集中于设计、完善和审计监督*系统*（例如，章程、AI反馈模型、评估协议），而非直接对模型输出执行监督任务。



![监督投入与模型交互规模对比](plots/4434-0.json)



> 比较了在不同监督模式下，随着模型交互或复杂性的增加，人力监督投入可能如何变化。可扩展监督的目标是显著减缓人力投入的增长。

实现可扩展监督对于开发安全可靠的先进AI系统非常重要。它克服了直接人工监督的限制，创造出可能应对高能力LLM复杂性的方法。接下来的章节将研究如宪法AI（CAI）和来自AI反馈的强化学习（RLAIF）等技术，这些技术被清晰地设计为尝试实现此类可扩展监督机制。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Karthik Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, S.K. Sutskever, Amanda Askell, Sarita Char, Janelle Shane, Brian Mcmahan, Noah Fiedel, Paul Christiano, Geoff Irving, Kyle Scott (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  介绍了基于人类反馈的强化学习（RLHF），本节指出其因持续依赖人类偏好标注而面临扩展挑战。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Roslyn Campbell, Anna Chen, Dawn Drain, Deep Ganguli, Andy Jones, Nicholas Joseph, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tom Henighan, Brian Hutchinson, Rita Johnston, Abhishek Karkhanis, Jeremy Kim, Carol Chen, Kristóf T. Garamvölgyi, Sam McCandlish, Chris Olah, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Samuel R. Bowman, Kevin Scott, Shauna Gordon-McKeon, Lauren Hume, Michael Johnston, Ben Mann, Amanda Ngo, Arvind Neelakantan, Long Ouyang, Catherine Perez, Nicholas Schiefer, Justin Shlegeris, Stephanie Sclafani, Gabe Selsky, Sam Ringer, Mike Smith, Jordan Schneider, Noah Shinn, Brooke Smyth, Stephen McAleer, Andrew Trask, Jon Uesato, Jeff Wu, Danny Wu, Steven T. Young, Evan Hubinger (2022)
  Journal: arXiv preprint arXiv:2212.08073; Publisher: arXiv; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  直接提出了宪法式AI，作为一种可扩展对齐方法，专门利用AI反馈来减少对大量人工监督的需求。
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI-Generated Feedback](https://arxiv.org/pdf/2309.00267.pdf) — Sungdong Lee, Seonghyeok Park, Hyeonseung Lee, et al. (2023)
  Journal: arXiv preprint arXiv:2309.00267
  关注于基于AI反馈的强化学习（RLAIF），该方法旨在通过使用AI生成的偏好来训练模型以实现对齐，直接解决了人类反馈的瓶颈问题。
