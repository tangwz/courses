---
course: "nlp-fundamentals"
chapter: "nlp-text-processing-techniques"
lesson: "advanced-tokenization-methods"
sourceId: 2475
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-1-nlp-text-processing-techniques/advanced-tokenization-methods"
title: "高级分词方法"
description: "了解并运用BPE和WordPiece等子词分词技术。"
order: 2
plots: ["plots/2475-0.json"]
sourceHash: "5cc01cf2a21676ca0fa529b8de3b42525106984175acfa96ecb2bae851837250"
sourceCorrections: []
---

虽然基于空格或标点符号的简单分词 (tokenization)方法是基本的首要步骤，但它们在处理语言的复杂性时常常显得不足。请考虑以下难题：

- **词汇量剧增：** 在大型文本集合中，独特词汇的数量可能变得非常庞大，导致特征空间维度极高，并带来计算效率低下。
- **词外词（OOV）问题：** 标准分词器 (tokenizer)难以处理训练中未出现的词语，例如生僻词、错别字、新词或专业术语。这些词外词通常被映射到一个通用的“未知”标记 (token)，丢失有价值的信息。
- **词形变化：** 语言常通过词形变化（例如，“run”、“running”、“ran”、“runner”）来表达语法差异。简单的分词将这些视为完全独立的实体，未能捕获其潜在的语义关联 (semantic relationship)。

为了更有效地处理文本，基于**子词 (subword)单元**的更复杂分词策略已成为现代NLP深度学习 (deep learning)模型中的标准做法。子词分词不把整个词作为基本单元，而是将词语拆分为更小的、有意义的片段。这种方法巧妙地平衡了字符级标记的细粒度与词级标记的语义表示。两种主要技术是字节对编码（BPE）和WordPiece。

### 字节对编码（BPE）

字节对编码（BPE）最初是作为一种数据压缩算法开发的，后被应用于NLP分词 (tokenization)，以有效管理词汇量并处理词形变化。其核心思想是迭代合并频繁出现的字符序列。

**BPE工作原理：**

1. **初始化：** 从包含训练语料库中所有单个字符的词汇表 (vocabulary)开始。在语料库中每个词的末尾添加一个特殊的词尾符号（如`</w>`或`</s>`）以区分词的边界。
2. **迭代：** 重复计算语料库中所有相邻符号对（字符或已合并的子词 (subword)）的出现频率。
3. **合并：** 找出最频繁出现的相邻对（例如，`'t'`后跟`'h'`）。将这对合并成一个新的符号（例如，`'th'`）。
4. **词汇表更新：** 将这个新合并的符号添加到词汇表。
5. **重复：** 继续迭代（计算符号对并合并最频繁的对），直到达到预设的合并操作次数，或达到预期的词汇表大小。

**示例：**

让我们追踪语料“hug hugger huge hugs”上的一个简化BPE过程，添加`</w>`来标记 (token)词尾：

- 初始语料表示：`h u g </w> h u g g e r </w> h u g e </w> h u g s </w>`
- 初始词汇表：`{h, u, g, </w>, e, r, s}`
- 字符计数：`h:4, u:4, g:5, </w>:4, e:2, r:1, s:1`
- 初始对频率：`(h u):4, (u g):4, (g </w>):1, (g g):1, (g e):2, (e r):1, (r </w>):1, (e </w>):1, (g s):1, (s </w>):1`
- **合并1：** `(h u)`和`(u g)`对最频繁（4次）。我们将`u`和`g`合并成`ug`。
  - 新词汇表：`{h, u, g, </w>, e, r, s, ug}`
  - 语料：`h ug </w> h ug g e r </w> h ug e </w> h ug s </w>`
- **合并2：** 现在再次计算对的频率。`(h ug)`出现4次。合并`h`和`ug`成`hug`。
  - 新词汇表：`{h, u, g, </w>, e, r, s, ug, hug}`
  - 语料：`hug </w> hug g e r </w> hug e </w> hug s </w>`
- **合并3：** 计算对的频率：`(hug </w>):1, (hug g):1, (g e):2, (e r):1, (r </w>):1, (hug e):1, (e </w>):1, (hug s):1, (s </w>):1`。最频繁的对是`(g e)`（2次）。合并`g`和`e`成`ge`。
  - 新词汇表：`{h, u, g, </w>, e, r, s, ug, hug, ge}`
  - 语料：`hug </w> hug g er </w> hug ge </w> hug s </w>`

这个过程会继续。像“er”这样的频繁序列可能会被合并。最终，词汇表将包含常用字符和频繁的子词单元。

**新文本分词：** 要对一个新词（例如，“hugging”）进行分词，BPE会贪婪地应用学到的合并规则。它会首先将其拆分为字符`h, u, g, g, i, n, g, </w>`。然后它会按照学习的顺序应用学到的合并规则。如果先学到了`ug`，然后是`hug`，再是`ing`（假设），结果可能是`['hug', 'g', 'ing', '</w>']`。一个未知词如“snuggles”可能会被拆解为`['s', 'n', 'ug', 'gle', 's', '</w>']`，如果这些子词存在于词汇表中，通过使用已知部分来表示，从而有效处理词外词问题。

OpenAI的GPT系列等模型使用BPE。

### WordPiece

WordPiece是另一种流行的子词 (subword)分词 (tokenization)算法，与BPE相似，但合并准则不同。Google的BERT等模型使用它。

**WordPiece工作原理：**

1. **初始化：** 从包含训练语料库中所有单个字符的基本词汇表 (vocabulary)开始。
2. **迭代与似然：** 与BPE类似，它通过合并单元迭代构建词汇表。然而，WordPiece不合并最*频繁*的对，而是合并在被选择时能最大化训练数据*似然*值的对。这种似然通常根据该对的计数以及构成该对的单个单元的计数来计算。对合并A和B的分数，一个简化的思考方式与其相关：
   
   $$
   \text{得分}(A, B) = \frac{\text{频率}(AB)}{\text{频率}(A) \times \text{频率}(B)}
   $$
   
   相对其各自频率而言，频繁共同出现的对会获得优先。这倾向于优先合并形成更“像词”的单元。
3. **词汇表更新：** 得分最高的对被合并，新单元被添加到词汇表。
4. **重复：** 这个过程重复进行，直到达到目标词汇表大小。

**分词策略：** 对新文本进行分词时，WordPiece尝试找到词汇表中可能的最长子词，它与当前词的开头匹配。如果一个子词单元不表示原始词的开头，它通常会加上一个特殊前缀符号，通常是`##`（例如，“tokenization”可能变为`['token', '##ization']`）。这种明确的标记 (token)有助于模型理解`##ization`是前一部分的延续。

**对比：**

| 特征 | BPE | WordPiece |
| --- | --- | --- |
| 合并准则 | 相邻对的最高频率 | 数据似然的最大提升 |
| 实现方式 | 更简单，纯粹基于计数 | 更复杂，涉及似然计算 |
| 子词标记 | 通常使用词尾符号（`</w>`） | 常使用前缀标记（`##`）表示延续 |
| 常用模型 | GPT, RoBERTa | BERT, DistilBERT |

BPE和WordPiece都能有效减少词汇量，与词级标记相比，它们通过将词外词拆分为已知子词来处理词外词问题，并捕获词形关系（例如，“run”、“running”可能共享“run”这个子词）。



![词汇量对比](plots/2475-0.json)



> 子词分词提供了一个中间方案，在提供可管理的词汇量的同时，比字符级标记保留更多语义信息，并比词级标记更能处理词外词。请注意Y轴上的对数刻度。

### SentencePiece

值得一提的是，Google开发的**SentencePiece**。它将输入文本视为一个原始的Unicode字符流，包括空格。这意味着它无需依赖预分词 (tokenization)（如按空格拆分）就能学习分词，并能更容易地处理多种语言。SentencePiece可配置为使用BPE或unigram语言建模方法来构建其子词 (subword)词汇表 (vocabulary)。一个重要优势是其完全可逆的分词；原始文本可以从标记 (token)中完美重建，因为空格被视为子词单元的一部分（通常由一个元符号如表示）。

### 选择和使用高级分词 (tokenization)器 (tokenizer)

BPE、WordPiece或SentencePiece之间的选择通常取决于您打算使用的特定预训练 (pre-training)模型，因为模型通常会与其对应的已训练分词器一起发布。Hugging Face的`transformers`等库抽象化了许多实现细节，使您能够轻松加载和使用给定模型的相应分词器（例如，`BertTokenizer`使用WordPiece，`GPT2Tokenizer`使用BPE）。

训练这些分词器需要大量代表您所从事领域的数据。然而，对于许多应用来说，使用与预训练语言模型关联的预训练分词器是最实际的方法。

理解这些高级分词方法很重要，因为文本的拆分方式从根本上影响着下游模型如何解释和处理语言数据。它们标志着一个重要进展，使现代NLP模型能够更有效地处理多样的词汇和词形。

## 参考资料

- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, and Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 1715-1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  介绍了用于开放词汇神经机器翻译的字节对编码（BPE），是NLP子词分词的一项工作。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  虽然主要介绍BERT模型，但本文详细阐述了所使用的WordPiece分词策略，使其成为理解WordPiece的来源。
- [Tokenizers](https://huggingface.co/course/chapter2/5?fw=pt) — Hugging Face (2023)
  Publisher: Hugging Face
  提供了现代分词方法（包括BPE、WordPiece和SentencePiece）的实用指导和概述，并附有实现示例。
