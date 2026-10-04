---
course: "llm-model-sizes-hardware"
chapter: "intro-llms-model-size"
lesson: "model-size-examples"
sourceId: 4200
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-1-intro-llms-model-size/model-size-examples"
title: "不同模型大小的例子"
description: "查看按参数数量分类的大型语言模型示例（小型、中型、大型）。"
order: 4
plots: ["plots/4200-0.json"]
sourceHash: "ab43441020a37d50c812f556423707630445bb5289cb2fafb191e5bf675ca2a5"
sourceCorrections: []
---

既然我们知道大型语言模型是根据参数 (parameter)数量来衡量的，接下来我们来看一些例子。这会让您对我们所讨论的规模有更直观的感受，以及模型通常是如何分类的，即使这些分类并非正式的。请记住，相关技术发展迅速，所以今天被认为是“大”的模型，明天可能就会显得中等。

模型通常可以按其参数数量进行分组。尽管没有严格的界限，但以下是一种常见的思考方式：

### 小型模型（小于10亿到约70亿参数 (parameter)）

这些模型在计算需求方面最容易获得。

- **参数数量：** 通常从几亿到约70亿（$7B$）不等。
- **例子：**
  - **BERT-Large：** 尽管在现代意义上并非总是被称为“LLM”（它是一个编码器模型），但它拥有约3.4亿参数，并且具有基础作用。
  - **DistilBERT：** BERT的一个更小、精简版本，拥有约6600万参数，旨在提高效率。
  - **Phi-2：** 微软的一个模型，拥有27亿参数，以其相对于自身大小的强大性能而闻名。
  - **Gemma 2B：** 谷歌的一个较小型模型，拥有20亿参数。
- **特点：**
  - 运行速度相对较快（推断）。
  - 内存需求较低；有些可以在CPU或消费级GPU上运行，且显存 (VRAM)有限。
  - 通常在经过训练或微调 (fine-tuning)的特定任务上表现良好。
  - 可能缺乏大型模型所具有的广泛知识或复杂推理 (inference)能力。
  - 适用于部署在边缘设备或资源受限的应用中。

### 中型模型（约70亿到约700亿参数 (parameter)）

这一类别代表了一个受欢迎的中间选择，在能力与资源需求之间取得平衡。许多广泛使用的开源模型都属于这个范围。

- **参数数量：** 大约从70亿（$7B$）到约700亿（$70B$）。
- **例子：**
  - **Llama 2 (7B, 13B, 70B)：** Meta 公司推出的一系列流行开源模型，在此范围内有多种尺寸可用。
  - **Mistral 7B：** 一个备受好评的开源模型，拥有70亿参数，以其高效性而闻名。
  - **Falcon (7B, 40B)：** 另一系列开源模型。
  - **Gemma 7B：** 谷歌的70亿参数开源模型。
- **特点：**
  - 在更广泛的任务（文本生成、摘要、翻译、基本推理 (inference)）上提供比小型模型明显更强的能力。
  - 需要配备大量显存 (VRAM)（例如16GB、24GB或更多，特别是对于此范围中较大的模型）的强大GPU。
  - 对于许多研究和开发目的来说，代表着一个良好的平衡点。

### 大型模型（约700亿到几千亿参数 (parameter)）

这些模型展现出强大的能力，但伴随着庞大的硬件需求。

- **参数数量：** 通常从约700亿（$70B$）开始，最高可达几千亿参数。
- **例子：**
  - **GPT-3 (175B)：** OpenAI 的模型（特指 `davinci` 版本）拥有1750亿参数，具有非常大的影响力。
  - **Falcon 180B：** 一个非常大的开源模型。
  - **BLOOM (176B)：** 一个开源多语言模型。
- **特点：**
  - 在语言理解、生成和推理 (inference)方面表现出强大的性能。
  - 硬件需求非常大，通常需要多个高端GPU共同运行。
  - 个人在本地运行较不常见；通常通过云平台或API访问。

### 超大型/前沿模型（通常大于5000亿参数 (parameter)，具体数量常未公开）

这些是顶尖模型，通常由大型研究实验室和科技公司开发。它们的具体参数数量通常不公开，但人们普遍认为它们比“大型”类别明显更大，可能达到或超过万亿参数。

- **参数数量：** 估计在数千亿到万亿以上（$1T+$）。具体数字通常未得到开发者确认。
- **例子：**
  - **GPT-4：** OpenAI 的GPT-3后续模型。参数数量不公开，但普遍认为非常大，可能采用“专家混合”（MoE）架构，这使得单一参数计数变得复杂。
  - **Google Gemini (Ultra, Pro, Nano)：** 谷歌的一系列高级模型。尺寸未按参数指定。
  - **Claude 3 (Opus, Sonnet, Haiku)：** Anthropic 的模型系列。同样，参数数量不公开。
- **特点：**
  - 代表人工智能能力的尖端水平，擅长复杂推理 (inference)、创造力和理解。
  - 训练和推断都需要庞大的计算资源（大型GPU集群或TPU等专用硬件）。
  - 由于其庞大的资源需求，主要通过API或托管云服务来访问。

为了帮助您直观感受规模差异，请参考这些近似的参数数量：



![近似模型大小（十亿参数）](plots/4200-0.json)



> 所示为选定大型语言模型在对数刻度下的近似参数数量，以显示不同类别间的相对大小。请注意，GPT-3（175B）比Llama 2（70B）大得多，后者又远大于Mistral（7B）和Phi-2（2.7B）。

了解这些大致的类别有助于我们对不同大型语言模型进行背景化讨论。正如您所见，参数数量直接影响模型的潜在能力，但也会极大地改变运行模型所需的硬件。在接下来的章节中，我们将讨论所涉及的具体硬件组件，以及它们的规格如何与处理这些不同大小的模型相关联，主要关注 *推断*（运行模型）的需求。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.18653/v1/N19-1423) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  介绍了BERT，一种用于上下文嵌入的基础模型，包括其“大型”变体，它是现代标准下较小型LLM的一个例子。
- [Language Models are Few-Shot Learners](https://doi.org/10.48550/arXiv.2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: arXiv; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  描述了GPT-3的架构和规模，这是一个具有1750亿参数的开创性大型语言模型，展示了其小样本学习能力。
- [Mistral 7B](https://doi.org/10.48550/arXiv.2310.06825) — Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2310.06825](https://doi.org/10.48550/arXiv.2310.06825)
  介绍了Mistral 7B模型，这是一个在较小到中等参数范围内表现高效且能力出众的模型示例。
