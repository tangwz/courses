---
course: "llm-alignment-safety"
chapter: "evaluating-llm-safety-alignment"
lesson: "automated-evaluation-benchmarks"
sourceId: 3168
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-4-evaluating-llm-safety-alignment/automated-evaluation-benchmarks"
title: "自动化评估基准 (HELM, TruthfulQA)"
description: "使用HELM和TruthfulQA等标准自动化基准进行评估。"
order: 2
plots: ["plots/3168-0.json", "plots/3168-1.json"]
sourceHash: "920b4442f0e7f8701bb2f07316da2a75819bb3e9aa581293b70d92aa968f5eb4"
sourceCorrections: []
---

尽管人工检查和传闻证据能提供初步信息，但确定大型语言模型（LLM）的安全性和对齐 (alignment)情况，需要系统化、可复现和可扩展的评估方法。自动化基准作为标准化的测试场所，让我们能够针对预设任务测试模型特定行为，并比较不同模型或对齐技术之间的表现。它们是多方面评估策略的一个主要组成部分，提供与定性人工评估相辅相成的量化 (quantization)数据。

"然而，认识到这些基准是工具而非神谕，这一点很重要。它们衡量模型在特定、预设任务上的表现，这可能无法完全反映实际交互的复杂性和不可预测性。在基准测试中获得高分表明模型在所测方面表现熟练，但不能保证普遍的安全或对齐。"

两个典型的例子说明了这类基准的用处和侧重：HELM 和 TruthfulQA。

### 大型语言模型的全面评估 (HELM)

由斯坦福大学CRFM开发，大型语言模型的全面评估（HELM）框架旨在进行广泛的覆盖评估。HELM没有侧重于单一指标或任务，而是使用多种指标（例如，准确性、校准、韧性、公平性、偏差、效率）评估LLM在各种情况（例如，问答、摘要、情感分析、毒性检测）下的表现。

核心思想是“多指标测量”：认识到没有任何单一数字能完全反映LLM的质量，HELM明确规定了标准化的场景、数据源、适应方法（如何提示或微调 (fine-tuning)模型以适应场景）和指标。这种标准化使得模型之间的比较更具意义。

HELM中与安全性和对齐 (alignment)有关的要点包括：

- **针对有害行为的场景：** HELM包含明确设计用于衡量不良行为的场景，例如生成有害言论、传播虚假信息或表现出社会偏见。
- **公平性和偏差指标：** 它包含了旨在量化 (quantization)不同人口群体之间性能差异的指标，有助于发现潜在的公平性问题。
- **韧性测试：** 评估模型性能在输入数据分布扰动或变化下的表现，这对于在各种条件下保持安全行为很重要。



![HELM指标覆盖范围](plots/3168-0.json)



> 不同评估方面如何在广泛基准（如HELM）的各种任务类型中分配权重 (weight)的示意性分类。特定于安全的任务自然地侧重于毒性和偏差等指标。

HELM的优势在于其广度及其结构化方法，提供模型能力和潜在不足的全面视图。然而，运行完整的测试套件可能计算量很大，并且像任何基准一样，它可能遗漏其预设场景和指标未涵盖的细微故障模式。

### TruthfulQA

HELM旨在实现广度，而TruthfulQA则侧重于对齐 (alignment)的一个特定方面：诚实或真实性。由Google和Anthropic的研究人员开发，该基准衡量语言模型是否避免生成虚假或误导性信息，尤其是网上常见的错误观念。

TruthfulQA向模型提出旨在引出模仿性虚假信息的问句。这些问题的答案，基于大量的互联网训练数据，统计上可能是最常见的，但实际上可能不正确或具有误导性。该基准根据两个主要标准评估回答：

1. **真实性：** 答案是否与已知事实相符？
2. **信息量：** 如果存在真实且有信息量的答案，模型是否提供了有用信息？

评估通常结合使用自动化评分（例如，使用像BLEURT这样的微调 (fine-tuning)模型或与参考的真/假答案进行比较）以及人工判断来评估答案的真实性和有用性。模型可能在技术上是真实的但没有帮助（例如，总是说“数据有冲突”），或者它可能自信地陈述一个常见的错误。



![TruthfulQA性能比较](plots/3168-1.json)



> 比较了不同模型或对齐阶段在TruthfulQA上的表现，根据真实性和信息量细分了回答。有效的对齐旨在增加“真实且有信息量”的部分，同时减少虚假信息。

TruthfulQA因其直接侧重于HHH框架中的“诚实”部分而有价值。其对抗性问题设计使其能有效揭示生成听起来合理但具有误导性信息的倾向。它的主要局限性在于，与HELM等全面基准相比，它的侧重范围较窄。模型可能被专门微调，以在TruthfulQA的问题风格上表现良好，但不一定能提升它们在不同对话情境中的普遍诚实性。

### 将基准测试整合到评估中

像HELM和TruthfulQA这样的自动化基准，当在更广阔的评估背景中适当使用时，是强大的工具：

- **互补性：** 使用侧重不同的多个基准（例如，HELM用于广度，TruthfulQA用于诚实性，以及特定的偏差基准）以获得更全面的评估结果。
- **情境化解读：** 在模型预期用途和已知基准局限性的背景下分析分数。如果另一个更重要的安全指标有显著提升，特定指标的下降可能是可以接受的。
- **跟踪进展：** 在开发期间和部署后定期运行基准测试，以跟踪对齐 (alignment)进展，检测退步，并理解对齐技术或模型更新的影响。
- **指导干预：** 在特定基准场景中的表现不佳能突出需要进一步对齐工作的方面，例如有针对性的数据过滤、RLHF中奖励模型的改进或安全提示的调整。

自动化基准对于可扩展和可复现的评估必不可少，但它们主要测试已知的故障模式和预设的能力。它们必须与人工评估和红队测试等技术结合使用（这些将在后续讨论），以发现未知问题并在更开放、对抗的环境中评估安全性。

## 参考资料

- [HELM: Holistic Evaluation of Language Models (Official Website)](https://crfm.stanford.edu/helm/latest/) — Stanford CRFM (2022)
  Publisher: Stanford Center for Research on Foundation Models (CRFM)
  HELM项目的官方网站，提供详细的文档、评估结果和基准测试的当前状态。
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) — Stephanie Lin, Jacob Hilton, Owain Evans (2021)
  Journal: ACL 2022; DOI: [10.48550/arXiv.2109.07958](https://doi.org/10.48550/arXiv.2109.07958)
  介绍了TruthfulQA基准测试，专门用于衡量语言模型在回答容易引发常见人类误解的问题时生成真实答案的能力。
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) — Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, Jack Clark (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  讨论了系统对抗性测试方法（红队测试），用于识别和减轻语言模型中的有害行为，这种做法是对自动化基准测试的补充。
