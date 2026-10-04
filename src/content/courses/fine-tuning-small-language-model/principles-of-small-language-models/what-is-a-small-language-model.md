---
course: "fine-tuning-small-language-model"
chapter: "principles-of-small-language-models"
lesson: "what-is-a-small-language-model"
sourceId: 8497
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-1-principles-of-small-language-models/what-is-a-small-language-model"
title: "什么是小语言模型"
description: "定义小语言模型，并了解它们与大型模型的区别。"
order: 1
plots: []
sourceHash: "2ca3622ea05d39a978127830c976db5bdf1b301eacbecc058c5762b6098fcb1e"
sourceCorrections: []
---

小语言模型代表了自然语言处理从单纯追求规模向目标效率的转变。虽然拥有数千亿参数 (parameter)的模型在通用任务中占据主导地位，但小语言模型的参数量通常在 10 亿到 80 亿之间。它们与大型模型采用相同的底层 Transformer 架构，但经过专门设计，可在受限的硬件环境中运行，例如消费级 GPU 或本地企业服务器。

要理解两者的差异，必须观察参数本身。在神经网络 (neural network)中，参数是训练阶段学习到的权重 (weight)和偏置 (bias)。当我们说一个模型有 70 亿参数时，是在量化 (quantization)处理输入词元 (token)的矩阵大小。将模型加载到显存 (VRAM)中所需的内存完全由参数数量和存储它们所使用的数值精度决定。

设 $P$ 为参数数量，$B$ 为每个参数占用的字节数。总内存 $M$（以 GB 为单位）的计算公式为：

$M = \frac{P \times B}{10^9}$

如果使用 16 位浮点精度（每个参数需要 2 个字节）加载一个 70 亿参数的模型，基础内存占用约为 14 GB。而一个 700 亿参数的模型则需要 140 GB，这超出了标准本地硬件的处理范围。通过保持较低的参数量，小语言模型使得单 GPU 环境和边缘设备运行模型成为可能。

这些模型并不是仅通过缩小网络规模来获得高性能，而是高度依赖训练数据的质量。近期的工程方法表明，在经过高度筛选的高质量数据集上训练较小的网络，可以使其推理 (inference)能力接近更大的模型。这种方法减少了模型必须记忆的噪声，将其有限的容量集中在核心语言结构和逻辑模式上。

> 强调数据质量而非参数数量的训练与运行流程。

明确这些较小架构的长处和短板非常重要。一个拥有 30 亿参数的模型缺乏存储海量百科知识的容量。如果你询问冷门的历史事实，它可能会产生幻觉 (hallucination)或完全失败。但如果你提供特定文本并要求其进行摘要、提取实体或格式化 JSON，它的表现会非常出色。它们的作用更像是推理引擎，而非静态知识库。

由于其记忆容量有限，监督微调 (fine-tuning)对它们大有裨益。与其指望模型开箱即用并了解一切，不如更新其权重使其专精于某个窄领域。你可以将模型的通用语言理解能力调整到符合你特定格式和逻辑要求的状态。这使得它们在注重数据隐私的私有应用中非常有效，因为模型可以完全在本地、安全的系统中进行微调和部署，无需将敏感数据传输到外部 API。

## 参考资料

- [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) — Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu (2024)
  Journal: arXiv preprint arXiv:2401.02385; DOI: [10.48550/arXiv.2401.02385](https://doi.org/10.48550/arXiv.2401.02385)
  关于1.1B参数模型开发的工程报告，展示了在大规模数据集上训练小型架构的有效性。
- [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219) — Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah (2024)
  Journal: arXiv; DOI: [10.48550/arXiv.2404.14219](https://doi.org/10.48550/arXiv.2404.14219)
  详细介绍了使用精选合成数据训练高性能小语言模型的方法，使其推理能力可媲美大型模型。
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) — Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, Yuanzhi Li (2023)
  Journal: arXiv preprint arXiv:2306.11644; DOI: [10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644)
  介绍了phi-1模型，并证明了高质量的“教科书级”数据如何大幅减少特定任务所需的参数量。
- [Llama 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) — AI@Meta (2024)
  Llama 3系列官方文档，提供了常用于本地高效部署的8B参数版本的规范说明。
