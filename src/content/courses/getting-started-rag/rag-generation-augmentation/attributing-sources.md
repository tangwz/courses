---
course: "getting-started-rag"
chapter: "rag-generation-augmentation"
lesson: "attributing-sources"
sourceId: 3657
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation/attributing-sources"
title: "生成内容中的来源归属"
description: "讨论标示用于生成回应的来源文档的方法，并使用块元数据。"
order: 6
plots: []
sourceHash: "e6a7ad741cd37d07aa27fc1c4f300aebcb737a36fdf1a877b14873098f429954"
sourceCorrections: []
---

使用检索增强生成 (RAG)的一个重要优点是，相比单独使用标准大型语言模型，它能提升透明度和可信度。因为大型语言模型的回应是基于特定的检索文档，我们有机会引用这些来源，让用户（或开发者）能核实信息并了解其出处。这个过程通常被称为**来源归属**。

来源归属不仅仅是建立用户信任；它也是调试和了解RAG系统行为的一个有价值的工具。如果生成的输出不准确或异常，追溯到影响它的具体来源文档，可以帮助明确问题是出在检索阶段（找到了不相关的文档）还是生成阶段（误解了提供的背景信息）。

### 元数据在归属中的应用

回想第三章（“准备用于检索的数据”），在准备您的知识库时，您最好为每个文档块关联元数据。这些元数据通常包含以下细节：

- 原始文档的文件名或标识符（例如，`report_q3_2023.pdf`）。
- 文档中的具体位置（例如，页码、章节标题）。
- 如果来源是网页，则为URL。
- 创建或修改日期。

这些元数据对归属来说非常重要。当检索器根据用户查询识别出相关块时，它不仅应返回块的文本，还应返回其关联的元数据。这包信息（块文本 + 元数据）随后会传递给生成阶段。

### 实现归属的方法

有几种方法可以将来源信息呈现给最终用户或开发者：

1. **内联引用：** 可以通过提示工程 (prompt engineering)明确指示大型语言模型在其生成的回复中直接包含引用。增强后的提示将包含检索到的块及其元数据，指令可能如下所示：

   > “使用下方提供的背景信息回答问题。引用您使用的信息的来源文档和页码（在元数据中提供，例如 `[source: 'doc_name', page: 5]`）。例如：‘系统需要X配置 [source: config\_guide.pdf, page: 12]’。

   大型语言模型随后会尝试将这些引用编织到文本中。这为特定信息段的来源提供了即时背景。然而，这依赖于大型语言模型准确遵循指令的能力，有时可能会使回复显得混乱。
2. **附加来源列表：** 一种更简洁的方法通常是生成不带内联引用的主要回复，然后附加一份提供给大型语言模型作为背景信息的所有来源文档列表。这种方法实现起来更简单，因为它不需要复杂的提示工程来进行内联放置。您可以从增强提示中使用的所有检索到的块中收集元数据，并将其格式化为一个列表（例如，“参考来源：”，后跟项目符号或编号引用）。缺点是它不能将回复中的特定语句直接链接到具体的来源。
3. **用户界面集成：** 许多应用程序在用户界面层面实现归属。RAG流程返回生成的答案*以及*来源块的元数据。用户界面随后显示答案，并可能提供可点击的链接、图标或可展开的区域，以显示来源文档乃至使用的特定文本段落。这使得大型语言模型的主要输出保持整洁，同时仍提供完全的透明度。

   > 一个简化的流程，展示了与文本块一起检索到的来源元数据如何通过RAG流程传递，最终被用户界面用于独立于主要生成回应显示来源信息。

### 归属中的挑战

尽管功能强大，RAG系统中的归属也并非没有挑战：

- **综合与提取：** 大型语言模型通常会从多个检索到的块中综合信息，而不是简单地按原文提取文本。准确判断输出中的哪个句子对应哪个具体来源块可能很困难或不可能。归属通常指向所使用的文档*集合*，而不是提供句子级别的对应关系。
- **大型语言模型的忠实性：** 即使提供了来源文档，大型语言模型也可能无法准确呈现其中的信息。它仍然可能误解、轻微改变或以偏离原意的方式组合信息。这意味着引用不能保证引用信息在输出中得到完美反映。评估忠实性是一项独立的、重要的工作（在第六章中会进一步讨论）。
- **信息密度：** 有时，生成输出中的一个句子可能基于从多个不同块或文档中汇集的信息。通过简单的引用清晰地表达这种复杂关系可能很困难。

尽管存在这些挑战，实现来源归属是构建更可靠、更透明的人工智能系统的重要一步。通过在数据准备和检索过程中仔细管理元数据，并选择合适的方法呈现这些信息，您可以为用户和开发者提供有价值的见解，了解RAG系统是如何得出其答案的。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了基础的检索增强生成（RAG）框架，该框架将大型语言模型（LLM）的响应基于外部文档。
- [Retrieval Augmented Generation (RAG) Concepts](https://python.langchain.com/docs/concepts/#retrieval-augmented-generation-rag) — LangChain (2024)
  提供了实现RAG系统的实用指南，包括管理和在应用程序中展示源信息。
