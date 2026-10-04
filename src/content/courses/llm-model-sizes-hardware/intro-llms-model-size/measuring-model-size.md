---
course: "llm-model-sizes-hardware"
chapter: "intro-llms-model-size"
lesson: "measuring-model-size"
sourceId: 4197
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-1-intro-llms-model-size/measuring-model-size"
title: "模型大小如何衡量"
description: "说明衡量大型语言模型尺寸的常用指标：十亿个参数。"
order: 3
plots: ["plots/4197-0.json"]
sourceHash: "c90e7c437b663f88beabd4cf92273eb410ccae793f89f45288d09e3f48c87602"
sourceCorrections: []
---

既然我们明白了参数 (parameter)是大型语言模型在训练期间学习到的可调整数值，那么接下来的问题就是：我们如何量化 (quantization)这些巨型模型的“大小”？

衡量大型语言模型大小最常用和标准的方式是计算其可训练参数的总数。就像我们之前谈到的，可以把这些参数看作模型调整的内部旋钮和刻度盘，以此学习其训练所用语言数据中的模式与联系。

为什么使用参数数量？它直接表明了模型的潜在复杂性和容量。参数较多的模型通常能存储更多信息，捕捉语言中更精细的差异，并在复杂任务上表现得更好，相比之下，参数较少的模型则不然。它是衡量模型能力的一个有用的替代指标，尽管不完美。

你会经常听到大型语言模型的尺寸以一个特定单位来讨论：**十亿个参数**。这通常缩写为“$B$”。例如：

- 一个“7B模型”表示一个拥有大约70亿 ($7 \times 10^9$) 参数的模型。
- 一个“70B模型”拥有大约700亿参数。
- 一些研究模型达到数千亿参数，例如175B，甚至能达到万亿 ($T$) 参数。

如此庞大的规模正是大型语言模型中“大型”一词的由来。这些数字比许多早期类型的机器学习 (machine learning)模型中的数字要大得多。



![大型语言模型参数数量示例](plots/4197-0.json)



> 不同大型语言模型尺寸类别的近似参数数量。垂直轴上的对数刻度有助于呈现这些尺寸之间巨大的量级差异。

参数数量这一衡量方式对本课程很重要，因为它直接影响所需的计算资源。我们会看到，更多的参数通常意味着更高的内存（例如GPU中的显存 (VRAM)）和处理能力要求。了解参数数量是评估您可能需要哪些硬件的第一步。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  介绍了GPT-3，一个拥有1750亿参数的语言模型，展示了规模显著增加的模型的能力。这项工作为LLM中的“大型”设定了新基准，并显示了参数数量对少样本学习的影响。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  本文系统地研究了神经语言模型性能如何随模型大小（参数）、数据集大小和训练计算资源而扩展。它为参数数量与模型容量、性能和资源需求之间的关系提供了理论和实证见解。
