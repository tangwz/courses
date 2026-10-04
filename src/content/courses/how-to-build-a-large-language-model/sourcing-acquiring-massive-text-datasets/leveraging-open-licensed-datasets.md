---
course: "how-to-build-a-large-language-model"
chapter: "sourcing-acquiring-massive-text-datasets"
lesson: "leveraging-open-licensed-datasets"
sourceId: 5977
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-6-sourcing-acquiring-massive-text-datasets/leveraging-open-licensed-datasets"
title: "使用开放许可数据集"
description: "识别和使用公开可用、许可宽松的文本数据集（例如The Pile）。"
order: 4
plots: []
sourceHash: "b5646d342440c1225f537f0fe76e4d4a7d011d0c7e1c1edd21f692dc2b1b4624"
sourceCorrections: []
---

虽然自行构建大规模网络爬虫或处理Common Crawl等原始档案能够提供最大程度的控制，但这需要大量的工程投入和基础设施。幸运的是，研究社群和各类组织已经整理并发布了许多大型文本数据集，这些数据集均采用开放许可，为获取预训练 (pre-training)数据提供了有用的捷径或补充。

使用这些现有数据集可以节省大量与抓取、初步清洗和格式化相关的时间和资源。它们通常附带详细说明其来源和预处理步骤的文档，不过仍需仔细核查。

### 理解开放许可

在使用任何数据集之前，了解其许可条款非常重要。“开放”并非意味着“不受限制地免费用于任何用途”。许可规定了数据如何使用、修改和分发。常见例子包括：

- **知识共享 (CC)：** 常见的变体包括CC0（公共领域奉献）、CC-BY（要求署名）和CC-BY-SA（要求署名并以相同许可共享修改）。
- **开放数据公共领域 (ODC)：** 像ODbL（开放数据库许可）或ODC-By（类似于CC-BY）这样的许可专门用于数据库。
- **自定义许可：** 某些数据集可能拥有由发布组织制定的独有许可。

务必仔细审查数据集*及其各个组成部分*所附带的特定许可。有些汇编数据混合了来自不同原始许可来源的数据。不遵守许可条款可能导致法律问题，尤其是在商业环境中。

### 值得关注的开放许可数据集

几个大型文本语料库已成为LLM预训练 (pre-training)的标准资源。以下是一些著名例子：

1. **The Pile：** 由EleutherAI开发，The Pile是一个825 GiB的英文文本数据集，由22个不同的小型数据集汇编而成。其目的是创建一个广泛且多样的语料库，适用于通用语言模型。来源包含学术论文（PubMed Central、arXiv）、网络文本（Common Crawl子集）、书籍（Project Gutenberg、Books3）、代码（GitHub）、对话（Stack Exchange）等。虽然该汇编旨在采用宽松许可，但底层许可因来源而异，用户需要针对其具体用例检查合规性。
2. **C4 (Colossal Cleaned Common Crawl)：** C4最初是为T5模型创建的，它来自Common Crawl网络档案。它经过了大量的筛选和清洗，包括删除样板文本、使用黑名单排除冒犯性语言、文档去重以及主要保留英文文本。生成的数据集约为750GB，并以ODC-By许可发布。它对清洗的侧重使其成为一个受欢迎的起点，尽管清洗过程本身可能会过滤掉某些有用的文本类型，或反映清洗启发式方法的偏见。
3. **ROOTS (Responsible Open-science Open-collaboration Text Sources)：** 这个1.6TB的多语言语料库是为训练BLOOM模型而创建的。它汇集了来自59种语言的498个数据源。在文档化来源和许可方面投入了大量精力，以负责任的方式进行。它为训练多语言模型提供了宝贵的资源。
4. **其他来源：** 除了这些大型汇编，还有许多更小或更专业的开放数据集存在：

   - **维基百科：** 定期发布的数据快照可用，提供多种语言的高质量百科全书文本。常被包含在像The Pile这样的大型数据集中。
   - **古腾堡计划：** 提供数万本公共领域书籍。
   - **arXiv：** 一个科学预印本库，主要涉及物理、数学、计算机科学等。是技术语言的很好来源。
   - **GitHub：** 虽然有效过滤具有挑战性，但它是各种编程语言代码的大量来源。常被处理并包含在像The Pile这样的数据集中。

### 以编程方式访问数据集

Hugging Face的`datasets`等库简化了访问和使用许多流行的开放数据集，包括The Pile、C4和ROOTS。该库处理下载、缓存、处理和流式传输，与PyTorch等深度学习 (deep learning)框架结合得很好。

以下是使用`datasets`加载和查看C4数据集一小部分的简单示例：

```python
import torch
from datasets import load_dataset

# 以流式模式加载C4的“en”子集，以避免下载所有数据
# 注意：某些数据集可能需要通过Hugging Face Hub进行身份验证
try:
    c4_dataset = load_dataset(
        'allenai/c4',
        'en',
        split='train',
        streaming=True
    )
except Exception as e:
    print(
        f"加载数据集出错。您可能需要通过"
        f"`huggingface-cli login`登录。错误：{e}"
    )
    # 适当处理错误，可能退出或使用备用方案
    c4_dataset = None # 或某个占位符

if c4_dataset:
    # 从流中获取一小部分样本
    sample_size = 5
    sampled_data = list(c4_dataset.take(sample_size))

    print(f"从C4中抽取了{len(sampled_data)}个示例：")
    for i, example in enumerate(sampled_data):
        print(f"\n--- 示例 {i+1} ---")
        print(f"URL: {example.get('url', 'N/A')}") # 使用.get以确保安全
        # 打印文本的前300个字符
        text_snippet = example.get('text', '')[:300]
        print(f"文本片段: {text_snippet}...")
        print(f"时间戳: {example.get('timestamp', 'N/A')}")

# 将小批量数据转换为PyTorch张量的示例（如果不是流式传输）
# 这需要下载数据，因此在大型数据集上使用时需谨慎
# 实际训练中，请使用流式传输或带torch转换的map函数
# try:
#     c4_small_batch = load_dataset(
#         'allenai/c4',
#         'en',
#         split='train[:10]'
#     ) # 加载前10个示例
#     c4_small_batch.set_format(
#         type='torch',
#         columns=['text_length']
#     ) # 示例：格式化一列
#     # 访问张量：tensors = c4_small_batch['text_length']
# except Exception as e:
#     print(f"加载小批量出错: {e}")
```

对于多TB级别的数据集，强烈推荐使用`streaming=True`。它允许您迭代数据，而无需在本地下载和存储整个数据集，而是按需获取数据块。`datasets`库提供了强大的映射函数（`.map()`），可以在流式传输期间即时应用分词 (tokenization)和其他预处理步骤。

### 使用开放数据集时的注意事项

虽然方便，但使用现有数据集需要仔细考虑：

- **许可合规性：** 仔细检查您预期用途的许可条款（研究用途还是商业用途，分发要求）。对于像The Pile这样的汇编，请核实您所依赖的特定组件的许可。
- **数据质量和偏见：** 创建者应用的预处理步骤（例如C4的清洗）会引入其自身的偏见，并且可能与您的需求不完全一致。务必检查数据并考虑是否需要额外的过滤或清洗。请注意，这些数据集反映了其原始来源中固有的偏见。
- **数据集重叠：** 许多大型数据集都来源于Common Crawl或维基百科等常见来源。在比较使用不同语料库训练的模型时，或构建评估集以避免数据污染时，这种重叠值得注意。
- **任务适用性：** 数据集的组成会影响生成模型的能力。确保所选数据集与您的LLM的目标领域或任务相符。您可能需要组合多个开放数据集或使用自定义数据进行补充，以达到所需的性能特征。这与第九章讨论的数据混合策略直接关联。

总而言之，开放许可数据集是LLM开发的重要资源。与从头开始相比，它们提供了大量文本数据，并可能降低工程开销。然而，负责任的使用需要仔细关注许可、数据质量、潜在偏见以及对预期应用的适用性。像Hugging Face的`datasets`这样的库使得访问和处理这些资源的实际操作变得更加容易。

## 参考资料

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy (2020)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2101.00027](https://doi.org/10.48550/arXiv.2101.00027)
  介绍了The Pile数据集的构建、组成和特性，是LLM预训练的重要资源。
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2020)
  Journal: JMLR; Volume: 21; Pages: 1-67; DOI: [10.5555/3455716.3455823](https://doi.org/10.5555/3455716.3455823)
  描述了C4数据集的创建过程，作为T5模型研究的一部分，详细介绍了其从Common Crawl中清洗和过滤的步骤。
- [🤗 Datasets Documentation](https://huggingface.co/docs/datasets/index) — Hugging Face (2024)
  Publisher: Hugging Face
  提供了`datasets`库的完整指南和API参考，对于高效访问和处理用于LLM的开放数据集至关重要。
- [Open Data Handbook - Licensing Open Data](https://opendatahandbook.org/guide/en/how-to-open-up-data/licensing-open-data/) — Open Knowledge Foundation (2010)
  Publisher: Open Knowledge Foundation
  提供了关于理解和选择数据开放许可的实用指南，阐明了Creative Commons和Open Data Commons等术语。
