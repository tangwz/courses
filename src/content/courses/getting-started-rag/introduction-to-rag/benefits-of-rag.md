---
course: "getting-started-rag"
chapter: "introduction-to-rag"
lesson: "benefits-of-rag"
sourceId: 3637
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-1-introduction-to-rag/benefits-of-rag"
title: "使用RAG的优势"
description: "讨论RAG的实现优势，例如提高准确性、获取最新信息和来源归属。"
order: 5
plots: []
sourceHash: "6aad2443b184a7f5540826e7bdf6cf1710efeff5948eb93296ce443f7e32919e"
sourceCorrections: []
---

标准大型语言模型（LLM）面临着知识截止和生成不准确信息（即“幻觉 (hallucination)”）的潜在问题。检索增强生成（RAG）通过将外部知识整合到生成过程中，提供了一个有效的解决方案。这种方法为构建更高效和可靠的AI应用带来了多项重要优势。

### 提高准确性并减少幻觉 (hallucination)

RAG最直接的好处也许是它能将LLM的回答建立在从指定知识源检索到的事实数据之上。LLM不再仅仅依赖训练期间学到的可能过时或不完整的模式，而是会收到与用户查询特别相关的文本段落。通过基于这些检索到的内容进行生成，RAG系统更不容易“凭空捏造”或编造细节。如果所需信息存在于知识源中，检索器会找到它，生成器会使用它，从而产生事实更准确的输出。这种基础性处理很大程度上减轻了独立LLM中常见的幻觉问题。

### 获取最新信息

LLM通常在大量数据集上训练，但这个训练过程耗时且计算成本高昂。因此，它们的内部知识库会变得静态，反映出其上次训练日期的信息。RAG巧妙地规避了这一限制。检索器访问的外部知识源可以独立且频繁地更新，无需重新训练核心LLM。如果您需要一个应用来回答有关近期事件、不断变化的产品规范或最新发布的研究的问题，RAG能让您只需将相关文档添加到您的知识库中。LLM随后便可通过检索步骤运用这些当前信息，提供及时且相关的回答。

### 提高专属性和领域适应能力

使LLM在小众或专业领域（例如法律分析、医疗信息检索、公司内部政策）表现良好，可以通过微调 (fine-tuning)实现，但这需要精心策划的数据集和大量计算资源。RAG为领域适应提供了一种更轻量级的替代方案。通过用目标领域特有的文档（例如法律合同、医学教科书、公司文档）填充知识源，RAG系统可以检索并将高度相关、特定领域的内容整合到其回答中。这使得通用LLM无需修改其底层参数 (parameter)，便能在专业领域生成有见识的回答。

### 透明度和来源归属

理解LLM生成特定回答的*原因*可能不透明。RAG系统本身提供一定程度的透明度，因为生成过程与检索到的文档明确关联。通常可以设计RAG系统来引用用于构建回答的具体源文本段落。这种可追溯性对于以下方面很有用：

- **事实核查：** 用户或审计人员可以通过查阅原始资料轻松核实信息。
- **建立信任：** 引用来源会增加用户对系统可靠性的信心。
- **调试：** 如果生成了不正确的答案，识别问题是出在检索步骤（获取了错误的文档）还是生成步骤（LLM错误解释了内容）会变得简单得多。

我们将在课程的后面部分了解与数据块关联的元数据如何实现这种归属。

### 知识更新的成本与效率考量

虽然微调 (fine-tuning)LLM需要大量计算工作进行再训练，但更新RAG系统可用的知识通常效率要高得多。添加新信息涉及处理新文档并将其嵌入 (embedding)到检索索引中，这个过程通常远低于完整模型再训练的需求。这使得RAG特别适用于底层知识库频繁变化的应用。虽然RAG会带来其自身的运行成本（嵌入、向量 (vector)存储、检索计算），但与定期微调周期相比，它通常提供一种更灵活、更具成本效益的方式来保持LLM知识的最新性。

总之，RAG提供了一种有效的机制来增强LLM，使它们更准确、及时、具备领域知识并更透明，提供了一种实用方法来克服标准生成模型的固有局限性。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇奠基性论文介绍了 RAG 模型，详细阐述了其架构，并展示了其在知识密集型 NLP 任务中提高事实准确性、减少幻觉的初步效果。
- [CS224N: Natural Language Processing with Deep Learning - Lecture Notes](https://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  这所顶尖大学课程的讲义提供了关于高级 NLP 主题的结构化教学解释，包括 LLM 面临的挑战以及 RAG 等技术如何解决这些挑战。
