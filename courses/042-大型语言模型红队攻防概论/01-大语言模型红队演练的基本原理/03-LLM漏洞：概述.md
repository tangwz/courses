# LLM漏洞：概述

来源：[原文](https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-1-foundations-llm-red-teaming/llm-vulnerabilities-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

红队行动是一种积极主动的方法，用于测试和完善防御措施。对于大型语言模型（LLMs）来说，这种做法尤为重要，因为尽管它们能力先进，LLMs并非没有缺陷。它们的复杂性、对数据集的依赖以及其处理人类语言的特性，都可能导致潜在的弱点。了解这些漏洞是构建更安全LLM系统的第一步。

LLMs在多个阶段都可能存在漏洞：从其训练数据，到模型架构本身，它们如何进行微调 (fine-tuning)，它们的部署方式，以及用户与它们的交互方式。我们来概览一下常见的LLM漏洞类别。后续章节将更详细地介绍具体的攻击途径和技术。

### 规避指令或安全协议

LLM漏洞中一个讨论较多的方面，是使模型以开发者不期望的方式行事。

- **提示注入 (Prompt Injection)：** 攻击者可以精心设计输入（提示），使LLM忽略其原有指令，转而执行新的恶意指令。例如，一个提示可能会指令一个专为客户服务设计的LLM，转而泄露内部系统信息。
- **越狱 (Jailbreaking)：** 这是一种提示工程 (prompt engineering)形式，用户试图绕过或禁用LLM中设定的安全功能和道德准则。目标通常是引出模型通常设计为拒绝的回复，例如生成有害内容或仇恨言论。

这些漏洞的出现是因为LLMs通过相同的输入通道处理指令和数据，使得严格区分可信的系统指令与不可信的用户输入变得困难。

### 生成不期望的内容

LLMs有时会生成有问题的内容，即使没有用户恶意意图。

- **偏见放大 (Bias Amplification)：** 如果训练数据包含社会偏见（例如关于性别、种族或国籍），LLM可能会学习甚至在其回复中放大这些偏见。这可能导致不公平或歧视性结果。
- **幻觉 (hallucination)与错误信息 (Hallucinations and Misinformation)：** LLMs可能会生成听起来合理但事实不准确或毫无意义的文本。如果用户未经核实就信任LLM的输出，这些“幻觉”会尤为危险，导致错误信息的传播。
- **有害内容 (Harmful Content)：** 尽管有安全过滤器，LLMs仍可能被操纵或遇到边缘情况，从而生成冒犯性、仇恨或其它不当内容。

识别此类输出的倾向是红队行动的一个重要侧重。

### 信息泄露

LLMs处理并存储来自其训练数据的信息，在某些情况下，还来自正在进行的对话。这可能导致隐私问题。

- **敏感数据暴露 (Sensitive Data Exposure)：** LLM可能无意中泄露其在训练阶段接触到的敏感信息。如果此类数据存在于训练语料库中且未得到充分匿名化或过滤，这可能包括个人身份信息（PII）、机密商业数据或专有代码。
- **上下文 (context)窗口利用 (Context Window Exploitation)：** 如果应用程序环境不安全或模型受到损害，用户在当前对话中提供的信息（在LLM的上下文窗口内）可能会被泄露给其他用户或系统。

防范信息泄露对于维护用户信任和遵守数据隐私法规很重要。

### 易受数据投毒影响

LLM行为的完整性在很大程度上依赖于其训练数据的质量和完整性。

- **训练数据攻击 (Training Data Attacks)：** 恶意行为者可能试图“投毒”用于预训练 (pre-training)或微调 (fine-tuning)LLM的数据集。通过注入精心设计的例子，他们可能植入隐藏后门，制造特定偏见，或降低模型在某些任务上的性能。在模型训练后检测此类投毒会极其困难。

这类漏洞凸显了在整个MLOps生命周期中数据溯源和安全的重要性。

### 系统接口与基础设施的滥用

LLMs并非孤立运行；它们是大型系统的一部分，并且通常通过API或其他接口访问。

- **API滥用 (API Abuse)：** 如果LLM的API未得到妥善保护，它可能会易受常见Web应用程序攻击的影响，例如未经授权的访问、针对API处理逻辑的注入攻击或拒绝服务。
- **拒绝服务 (DoS)：** 攻击者可能试图通过发送大量复杂或资源密集型查询来使LLM系统不堪重负，从而使合法用户无法使用该服务。这对于公开可用的LLM服务尤为相关。
- **资源耗尽 (Resource Exhaustion)：** 某些提示可能导致LLM消耗不成比例的计算资源，可能导致性能下降或崩溃。

这些漏洞通常处于LLM本身与周围软件和硬件基础设施的交汇处。下图描绘了LLM系统中的几个点，攻击者可能针对这些漏洞进行攻击。

> 攻击者可以针对LLM系统中的多个漏洞点。

### 过度依赖与误解

虽然这并非LLM代码的直接技术缺陷，但人们如何使用和解释LLM输出的社会影响是一个重要问题，红队行动可以帮助评估。

- **过度信任 (Excessive Trust)：** 用户可能过度信任LLM生成内容的准确性和中立性，即使其中包含细微错误、偏见或虚构信息。
- **自动化偏见 (Automation Bias)：** 这指的是一种倾向，即过度依赖自动化系统，并在没有足够批判性审查的情况下信任其建议或输出。

红队演练可以模拟过度依赖导致负面后果的情景，帮助组织了解这些风险，并制定用户教育和负责任AI部署的策略。

本章对LLM漏洞的介绍为此后的详细讨论做了铺垫。作为一名红队成员，您的职责是像攻击者一样思考，预测这些漏洞可能如何被利用，并严格测试LLM的防御能力。这种了解非常必要，之后我们才能进入LLM红队行动生命周期以及如何规划您的行动。

## 参考资料

- [Poisoning Attacks on Large Language Models](https://arxiv.org/abs/2305.00392) — Eric Wallace, Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Nicholas Roberts, Ryan Staudemeyer, Danqi Chen (2023)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2305.00392](https://doi.org/10.48550/arXiv.2305.00392)
  探讨了恶意行为者如何通过污染LLM训练数据来引入后门或降低模型性能。
- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/10.1145/3442188.3445922) — Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell (2021)
  Journal: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency; Publisher: Association for Computing Machinery; Pages: 610–623; DOI: [10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922)
  一篇讨论大型语言模型伦理影响、偏见和环境成本的基础性论文，包括有害内容的生成和过度依赖的风险。

---

[上一节](02-%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E5%AF%B9%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%BA%E4%BD%95%E5%A6%82%E6%AD%A4%E9%87%8D%E8%A6%81.md) · [下一节](04-LLM%20%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F.md)
