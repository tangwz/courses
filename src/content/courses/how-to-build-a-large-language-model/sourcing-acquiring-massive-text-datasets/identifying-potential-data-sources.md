---
course: "how-to-build-a-large-language-model"
chapter: "sourcing-acquiring-massive-text-datasets"
lesson: "identifying-potential-data-sources"
sourceId: 5974
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-6-sourcing-acquiring-massive-text-datasets/identifying-potential-data-sources"
title: "确定潜在数据来源"
description: "查看适用于预训练的各种文本数据来源（网络、图书、代码）。"
order: 1
plots: ["plots/5974-0.json"]
sourceHash: "7cc2192646241236955952d5bd6d9dccd0e3524d09ec46f32620f446dadf1fe4"
sourceCorrections: []
---

任何强大大型语言模型的根本在于其预训练 (pre-training)期间所使用的数据。如前所述，我们说的不是兆字节或吉字节；我们处理的是数太字节，有时甚至是数拍字节的文本数据。找出这类庞大数量数据的来源是实际的第一步。数据的构成在很大程度上决定了最终模型的能力、偏见和整体表现。因此，了解不同数据来源的特点、优点和缺点非常必要。

让我们分析一下大型语言模型预训练常用的文本数据主要类别。

### 互联网

互联网是目前最大且最多样化的文本数据来源。它包含几乎所有能想到的主题信息，以无数种风格和语言书写。

- **数据量庞大：** 网络爬取可以获取数百太字节的原始HTML内容。像Common Crawl这样的项目提供了可公开访问的网络快照，构成了许多广泛使用的大型语言模型数据集（例如C4、OSCAR）的根本。我们将在下一节详细讨论Common Crawl数据的处理。
- **多样性：** 网络文本涵盖新闻、博客、论坛、评论、百科文章等等。这种多样性有助于模型学习广泛的语言模式和知识。
- **挑战：** 原始网络数据出了名的嘈杂。它包含大量“样板内容”（导航菜单、广告、法律免责声明）、重复内容、低质量写作、机器翻译文本，以及可能有害或带有偏见的语言。提取核心文本内容和进行质量过滤是第7章中介绍的重要工程任务。有针对性的网络抓取可以补充广泛的爬取，但这需要在道德和网站服务条款方面仔细执行（本章稍后会讨论）。

### 图书

数字化图书提供了高质量、长篇文本的来源。

- **质量与连贯性：** 图书通常经过编辑过程，因此语言语法正确、结构良好。它们使模型接触到叙事流畅、复杂句式以及连贯的论证或故事，这些在普通网络文本中较难找到。
- **来源：** 像古腾堡项目这样的公共领域藏书很有价值。BookCorpus等其他数据集过去也曾被使用，但由于版权限制，获取大量、多样化且合法合规的图书数据集仍然存在挑战。
- **挑战：** 版权是主要障碍。即使对于可获取的图书，光学字符识别（OCR）错误也可能引入噪声，并且移除页码、页眉和页脚等格式化痕迹需要特定的预处理步骤。

### 代码库

对于旨在理解或生成计算机代码的模型，在预训练 (pre-training)数据中包含源代码是不可或缺的。

- **结构与逻辑：** 代码是高度结构化的文本，具有正式的语法和语义。对代码进行训练有助于模型学习逻辑推理 (inference)和算法模式。
- **来源：** GitHub等平台上的公共代码库是主要来源。像“The Stack”这样的聚合数据集提供了大量来自多种语言的许可宽松的代码集合。
- **挑战：** 处理各种编程语言及其特定语法是必要的。过滤掉非代码元素（例如，代码库中混杂的构建文件、文档、问题讨论）也很重要。软件许可证差异很大，需要仔细过滤以符合使用条款。

### 学术论文与科学文献

像arXiv、PubMed Central和机构知识库这样的来源包含大量的科学知识。

- **专业知识：** 这些数据充分覆盖了特定的科学和技术方面。
- **正式语言：** 学术写作通常正式且精确，使模型接触到专业词汇和复杂的推理 (inference)结构。
- **挑战：** 从PDF文件中提取干净文本是一个重大障碍，因为PDF文件通常具有复杂的多栏布局、图表、表格和公式。付费墙和限制性许可限制了对许多已发表文献的访问。

### 对话数据

来自社交媒体、论坛和其他互动平台的文本记录了非正式的语言使用。

- **非正式风格与对话：** 这些数据反映了人们的自然交流方式，包括俚语、缩写和对话轮次。它对于构建聊天机器人或用于互动应用程序的模型很有价值。
- **来源：** 像Reddit这样的平台已被使用（例如Pushshift Reddit数据集），但API变更、服务条款以及用户隐私方面的道德考量是主要因素。
- **挑战：** 高度噪声、重复内容、毒性言论和个人身份信息（PII）很常见。数据可能碎片化，缺乏更广泛的背景。道德获取和严格的过滤/匿名化极其重要。

### 新闻文章

新闻语料库提供了访问时事和事实报道风格的途径。

- **时事信息：** 新闻文本使模型了解最新事件、人物和地点。
- **事实性风格：** 新闻写作通常目标是客观和事实呈现（尽管偏见仍然是一个问题）。
- **来源：** 存在大量新闻档案，有时可通过API或特定数据集（例如RealNews）访问。
- **挑战：** 付费墙限制了对许多付费新闻来源的访问。检测和减轻观点偏见很困难。新闻的快速变化意味着数据可能很快过时，并且类似事件经常在不同来源中重复报道。

### 专门语料库

在这些通用类别之外，还存在许多特定领域的数据集。

- **专业知识：** 法律文件（法院裁决、合同）、医学文献（研究论文、临床笔记——如果经过匿名化和符合道德规范的来源）或财务报告可以被纳入以构建具有专业知识的模型。
- **挑战：** 这些数据集通常小于网络规模的语料库。由于隐私、商业敏感性或知识产权，访问可能会受到限制。处理可能需要特定领域的知识。

### 组合数据来源

单一来源的数据不足以满足需求。最先进的大型语言模型几乎总是基于对这些类别中多个数据源的精心策划的*混合*进行训练。具体的组合是一个重要的设计选择，影响着模型的优点和缺点。例如，一个主要在网络文本上训练的模型可能擅长日常对话，但在正式推理 (inference)方面可能表现不佳，而一个侧重于学术论文的模型则可能表现出相反的特点。C4、The Pile或精炼的专有数据集所使用的具体比例通常是根据下游任务的表现进行调整的。我们将在第9章进一步讨论数据混合策略。



![大型语言模型预训练数据构成](plots/5974-0.json)



> 这是一个示例分布，说明了通用大型语言模型的预训练 (pre-training)数据集中不同数据来源可能如何加权。实际比例在不同模型之间差异很大。

访问和处理这些多样化的来源通常需要使用为大型数据集设计的库和工具。例如，Hugging Face的`datasets`库提供了对许多预处理数据集的便捷访问，包括大型网络爬取数据的子集或特定语料库。

```python
# 示例：使用Hugging Face datasets库加载大型数据集的子集
# 注意：这需要安装 'datasets' 库
# (`pip install datasets`)

from datasets import load_dataset

try:
    # 加载OSCAR数据集的一小部分（从Common Crawl获取的未打乱的多语言数据集
    # 使用 streaming=True 避免下载整个庞大数据集
    # 'en' 指定英文子集，
    # 'unshuffled_deduplicated_en' 是一种特定的配置
    # split='train[:1%]' 获取训练集的第一个1%
    # 用于演示
    oscar_subset = load_dataset(
        "oscar",
        "unshuffled_deduplicated_en",
        split='train[:1%]',
        streaming=True
    )

    # 遍历前几个示例
    print("OSCAR子集的前5个示例：")
    count = 0
    for example in oscar_subset:
        print(f"示例 {count + 1}：")
        # 打印文本的前200个字符
        print(example['text'][:200] + "...")
        print("-" * 20)
        count += 1
        if count >= 5:
            break

except Exception as e:
    print(
        "尝试加载数据集时发生错误： "
        f"{e}"
    )
    print(
        "请确保您有互联网连接并且 'datasets' 库已安装。"
    )
    print("某些数据集可能需要特定的配置或权限。")
```

此代码片段说明了如何以编程方式访问大型标准化数据集的一部分。虽然方便，但请记住，使用现有数据集意味着要依赖其创建者所做的预处理选择。构建独特、高质量的数据集通常需要回到原始来源（如Common Crawl存档或直接网络抓取），并实施自定义的清洗和过滤流程，我们将在后续部分和第7章中介绍这些内容。

## 参考资料

- [Common Crawl: Open Datasets for Web-Scale Language Models](https://commoncrawl.org/) — Common Crawl Foundation (2025)
  介绍了该项目、数据格式以及如何访问构成许多大型语言模型数据集基础的海量网络爬取存档。
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2020)
  Journal: Journal of Machine Learning Research; Volume: 21; Pages: 1-67; DOI: [10.5555/3455716.3455823](https://doi.org/10.5555/3455716.3455823)
  介绍了C4数据集，这是广泛使用的、经过清洗的Common Crawl数据版本，作为大型语言模型预训练的基准。
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy (2020)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2101.00027](https://doi.org/10.48550/arXiv.2101.00027)
  描述了一个为训练大型语言模型而设计的大规模、多样化、高质量数据集，包含了本节中讨论的许多数据源。
- [Hugging Face Datasets Library Documentation](https://huggingface.co/docs/datasets/en/main/) — Hugging Face (2024)
  使用`datasets`库访问、处理和共享机器学习数据集（包括许多大型语言模型预训练语料库）的官方指南。
