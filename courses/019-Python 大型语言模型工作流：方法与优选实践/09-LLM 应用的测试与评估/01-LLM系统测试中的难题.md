# LLM系统测试中的难题

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/challenges-testing-llm-systems)

[返回章节目录](README.md) · [返回课程目录](../README.md)

测试使用大型语言模型（LLM）构建的应用，与测试传统软件相比，带来了一系列不同的问题。传统测试常依赖可预测的输入产生确定性输出（例如，$2 + 2$ 总是等于 $4$），而LLM则引入了变动性和复杂性，需要新的方法。了解这些特有的难题是构建可靠评估策略的第一步。

### 非确定性输出

最根本的难题源于许多LLM的非确定性。即使多次给予完全相同的输入提示，LLM也可能产生略有不同，有时甚至显著不同的回应。这种变动性由以下几个因素引起：

1. **采样策略：** 像温度缩放和top-p采样这样的技术在生成过程中的token选择中引入了随机性。温度大于0允许模型尝试不太可能出现的词语选择，从而导致多样的输出。
2. **模型内部状态：** 尽管对于无状态的API调用较少见，但模型架构中复杂的内部机制和潜在的优化可能导致细微的差异。
3. **模型更新：** 对于基于API的模型（如OpenAI、Anthropic、Google的模型），底层模型通常由提供方更新，而最终用户看不到明确的版本信息。一次更新可能会改变模型的行为，潜在地修改对先前测试过的提示的响应。

这种非确定性意味着简单的断言测试（`assert output == expected_output`）通常不足够。你不能总是定义一个LLM*必须*产生的单一、精确的字符串。

> 传统函数的确定性与LLM对相同输入可能产生多个有效输出的对比。

### “正确性”的模糊性

LLM的“正确”响应是什么样的？与计算总和或排序列表不同，摘要、翻译或创意写作等任务通常没有唯一的标准答案。

- 一份摘要可以简洁，也可以详细，两者都可能有效。
- 一份译文可能使用不同的同义词，但仍保持含义。
- 生成的代码可以使用不同的算法达到相同的结果。

评估正确性通常转向评估*质量*属性，例如相关性、连贯性、有用性、无害性以及事实准确性（尤其在RAG系统中非常重要）。这些质量往往是主观的，难以自动测量。定义客观的通过/失败标准变得更加困难。

### 对输入变动的敏感性

LLM对输入提示中的微小改动可能表现出令人惊讶的敏感性。改变标点符号，轻微改写问题，或改变信息顺序，有时会导致截然不同的结果。这种敏感性使得确保全面的测试覆盖成为一项难题。测试少数提示变体可能无法显现由略微不同用户输入引发的边缘情况或意外故障模式。

### 隐式知识与潜在偏见

LLM在庞大的数据集上训练，吸取了大量信息，但也继承了数据中存在的偏见。这些训练数据是不透明的；我们无法完全了解模型“知道”的一切或它可能表现出的偏见。因此，测试必须考虑到可能产生的以下内容：

- **事实不准确（幻觉 (hallucination)）：** 生成看似合理但不正确的信息。
- **有偏见输出：** 反映训练数据中存在的社会偏见。
- **不良内容：** 生成有害、不道德或不适当的响应，即使提示是无害的。

预测这些问题何时何地发生很困难，因此对它们进行详尽测试几乎不可能。缓解这些问题通常需要仔细的提示工程 (prompt engineering)、过滤和评估，而不仅仅是传统的基于断言的测试。

### 定义自动化指标的难度

虽然传统软件测试严重依赖自动化指标（例如代码覆盖率、断言通过/失败率），但自动评估LLM质量是一个持续进行的研究方向。像BLEU或ROUGE（常用于机器翻译和摘要）这样的指标可以衡量表层文本相似性，但可能与人类对质量、流畅性或事实正确性的判断不完全吻合。针对RAG的指标，如忠实性（答案是否与检索到的文档矛盾？）和答案相关性，正在出现，但仍需要仔细的实施和解释。通常，自动化指标和人工评估的结合是必需的。

### 资源消耗与成本

彻底测试一个LLM应用可能会消耗大量资源。

- **计算成本：** 反复查询强大的LLM API会产生直接成本并耗费时间。
- **人工评估：** 评估生成内容的质量通常需要领域专家进行人工审查，这既耗时又昂贵。

在严格评估的需求与这些资源限制之间取得平衡是LLM开发中的一个实际难题。

这些难题并非意味着测试LLM应用不可能，但它们要求我们调整策略。接下来的部分将介绍单元测试、集成测试、评估框架和监控实践的方法，旨在解决这些特定问题。

## 参考资料

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出一个评估大型语言模型（LLMs）的框架，涵盖多项指标，强调了在定义和衡量质量方面的挑战。
- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://doi.org/10.1145/3442188.3445922) — Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell (2021)
  Journal: FAccT '21: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency; Publisher: ACM; Pages: 610-623; DOI: [10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922)
  探讨大型语言模型的伦理问题、偏见以及其训练数据可能导致的有害内容生成。
- [Aligning Language Models with Human Preferences through Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  详细介绍使大型语言模型输出与人类偏好对齐的方法，强调了人类反馈在定义和实现理想模型行为中的作用。

---

[上一节](../08-Python%20%E4%B8%AD%E7%9A%84%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%80%E5%8F%91%E5%92%8C%E6%B5%8B%E8%AF%95%E6%8F%90%E7%A4%BA.md) · [下一节](02-%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E7%BB%84%E4%BB%B6.md)
