---
course: "nlp-fundamentals"
chapter: "nlp-feature-engineering"
lesson: "calculating-tf-idf"
sourceId: 2482
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/calculating-tf-idf"
title: "计算 TF-IDF 分数"
description: "学习计算词频-逆文档频率所涉及的公式和步骤。"
order: 2
plots: []
sourceHash: "8af8c03c966df8cbf743769a50ad8f26fd3636d38bddefa4fd339e104a45e699"
sourceCorrections: []
---

简单地计数词语（例如在词袋模型中）会将所有词项一视同仁，无论其预测能力或独特性如何。像“the”这样的词可能在许多文档中频繁出现，但很少能说明任何单个文档的具体主题。为了生成更有意义的特征，需要一种方法来量化 (quantization)一个词在一个更大的文本集合（语料库）中对特定文档的重要性。词频-逆文档频率（TF-IDF）精确地实现了这一点。它根据词项在文档中的出现频率及其在所有文档中的稀有程度来赋予权重 (weight)。

让我们分解一下这个分数是如何计算的。它包含两个主要部分：词频（TF）和逆文档频率（IDF）。

### 词频（TF）

词频衡量词项 $t$ 在特定文档 $d$ 中出现的频率。其基本思想很简单：一个词在文档中出现多次，可能比只出现一次的词与该文档的内容更相关。

然而，原始计数可能具有误导性。一个比另一个文档长两倍的文档，对于相同的词项自然会有更高的计数，即使该词的相对重要性相似。为了考虑文档长度，词频通常会进行归一化 (normalization)处理。最常见的归一化方法是将词的原始计数除以文档中的总词数：


$$
TF(t, d) = \frac{\text{词项 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的词项总数}}
$$


**示例：**
考虑文档1：“The cat sat on the mat.”
总词项数 = 6。
词项“the”出现 2 次。
词项“cat”出现 1 次。

$TF(\text{"the"}, \text{Doc1}) = \frac{2}{6} \approx 0.333$
$TF(\text{"cat"}, \text{Doc1}) = \frac{1}{6} \approx 0.167$

TF还有其他一些变体，例如直接使用原始计数、使用布尔频率（如果词项存在则为1，否则为0），或者应用对数缩放（例如，$1 + \log(\text{原始计数})$）以减弱非常高词项计数的影响。上述所示的归一化版本被广泛使用，因为它避免了长文档仅仅因为其长度而主导词项权重 (weight)。

### 逆文档频率（IDF）

逆文档频率衡量词项 $t$ 的信息量。它评估词项在整个语料库 $D$（所有文档的集合）中的稀有性。这个思想是，出现在许多文档中的词项（如常用词或停用词）的信息量，不如仅出现在少数文档中的词项。这些稀有的词项有助于区分特定文档与其余文档。

IDF的标准公式是：


$$
IDF(t, D) = \log\left(\frac{\text{文档总数 } |D|}{\text{包含词项 } t \text{ 的文档数}}\right)
$$


让我们检查这些组成部分：

- $|D|$：语料库中的文档总数。
- 包含词项 $t$ 的文档数：这通常表示为 $df(t)$，即词项 $t$ 的文档频率。
- $\log$：对数（通常是自然对数，$\ln$）用于缩小IDF值。这避免了非常稀有的词项（只出现在一两个文档中的词项）完全主导权重 (weight)，使评分对异常词频的敏感度降低。

请注意，如果一个词项出现在 *所有* 文档中，对数内的比率将变为 $\frac{|D|}{|D|} = 1$，并且 $\log(1) = 0$。这意味着常用词项会得到非常低的IDF分数，有效地降低了它们的权重。反之，如果一个词项出现在很少的文档中，比率会很大，从而得到高的IDF分数。

**IDF 平滑：**
如果一个词项出现在每个文档中，就会出现一个潜在问题，即导致 $IDF = 0$。更麻烦的是，如果我们的词汇表 (vocabulary)中的某个词项（可能在测试时遇到）从未出现在用于计算IDF的训练语料库中，那么分母 $df(t)$ 将为0，从而导致除以零。

为了避免这些问题，一种常见的修改，通常称为“IDF平滑”，是通过在分母中加1来应用：


$$
IDF_{smooth}(t, D) = \log\left(\frac{|D|}{1 + df(t)}\right)
$$


许多实现，包括scikit-learn的默认设置，更进一步，还在对数 *结果* 上加1（在分母加1之后），确保所有IDF值都是正数，并且即使对于出现在所有文档中的词项也赋予一定的权重：


$$
IDF_{sklearn}(t, D) = \log\left(\frac{|D|}{1 + df(t)}\right) + 1
$$


在实践中通常建议使用其中一种平滑版本。

**示例（使用 $IDF_{smooth}$）：**
考虑一个包含100个文档的语料库（$|D| = 100$）。

- 词项“the”出现在 95 个文档中（$df(\text{“the”}) = 95$）。
- 词项“cat”出现在 10 个文档中（$df(\text{“cat”}) = 10$）。
- 词项“VAE”出现在 2 个文档中（$df(\text{“VAE”}) = 2$）。

$IDF_{smooth}(\text{"the"}, D) = \log\left(\frac{100}{1 + 95}\right) = \log\left(\frac{100}{96}\right) \approx \log(1.04) \approx 0.04$
$IDF_{smooth}(\text{"cat"}, D) = \log\left(\frac{100}{1 + 10}\right) = \log\left(\frac{100}{11}\right) \approx \log(9.09) \approx 2.21$
$IDF_{smooth}(\text{"VAE"}, D) = \log\left(\frac{100}{1 + 2}\right) = \log\left(\frac{100}{3}\right) \approx \log(33.33) \approx 3.51$

正如预期的那样，常用词项“the”获得了非常低的IDF分数，而稀有词项“VAE”则获得了高得多的分数。

### 结合TF和IDF：TF-IDF分数

词项 $t$ 在语料库 $D$ 中文档 $d$ 内的最终TF-IDF分数，是通过将其词频（TF）与其逆文档频率（IDF）相乘来计算的：


$$
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$


这个分数既反映了词项在文档内的局部重要性（TF），也反映了其在语料库中的全局重要性（IDF）。

- **高TF-IDF分数：** 授予那些在特定文档中频繁出现（$TF$高）但在整个语料库中不常出现（$IDF$高）的词项。这些词项是文档具体内容的良好指标。
- **低TF-IDF分数：** 发生在文档中不常见（$TF$低）或在语料库中非常常见（$IDF$低），或两者兼有的词项。

**示例（结合之前的计算）：**
使用文档1：“The cat sat on the mat.”以及我们的示例语料库（$|D|=100$, $df(\text{“the”})=95$, $df(\text{“cat”})=10$），并使用平滑后的IDF：

- $TF(\text{"the"}, \text{Doc1}) \approx 0.333$
- $IDF_{smooth}(\text{"the"}, D) \approx 0.04$
- $TF\text{-}IDF(\text{"the"}, \text{Doc1}, D) \approx 0.333 \times 0.04 \approx 0.013$
- $TF(\text{"cat"}, \text{Doc1}) \approx 0.167$
- $IDF_{smooth}(\text{"cat"}, D) \approx 2.21$
- $TF\text{-}IDF(\text{"cat"}, \text{Doc1}, D) \approx 0.167 \times 2.21 \approx 0.369$

尽管“the”在文档1中出现更频繁，但其非常低的IDF分数导致最终的TF-IDF权重 (weight)远低于“cat”，后者在此特定文档中不那么频繁但总体上稀有得多。

### 生成 TF-IDF 向量 (vector)

TF-IDF计算对语料库中每个文档的词汇表 (vocabulary)中的每个词项进行。结果通常表示为一个矩阵，其中行对应文档，列对应词项（来自词汇表）。该矩阵中的每个单元格 $(d, t)$ 包含词项 $t$ 在文档 $d$ 中的TF-IDF分数。

```
          词项 1 | 词项 2 | ... | 词项 V
文档 1   | tfidf_11 | tfidf_12 | ... | tfidf_1V |
文档 2   | tfidf_21 | tfidf_22 | ... | tfidf_2V |
...     | ...      | ...      | ... | ...      |
文档 N   | tfidf_N1 | tfidf_N2 | ... | tfidf_NV |
```

该矩阵的每一行都是一个TF-IDF向量，表示相应的文档。这些向量捕捉了词项的加权重 (weight)要性，并作为机器学习 (machine learning)算法的数值输入特征。通常，这些向量会进一步归一化 (normalization)（例如，使用L2归一化），以便所有文档向量都具有单位长度，这可以提高对特征大小敏感的算法（如基于距离的分类器）的性能。

通过计算TF-IDF分数，我们将原始文本转换为有意义的数值向量，这些向量反映了词项在局部和全局的重要性，为许多自然语言处理任务提供了坚实的基础。

## 参考资料

- [A statistical interpretation of term specificity and its application in retrieval](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-97.pdf) — Karen Spärck Jones (1972)
  Journal: Journal of Documentation; Publisher: Emerald Publishing; Volume: 28; Pages: 11-21; DOI: [10.1108/eb026526](https://doi.org/10.1108/eb026526)
  介绍逆文档频率（IDF）的原始论文。提出了词项加权的统计学基础方法。
- [A Vector Space Model for Automatic Indexing](https://doi.acm.org/10.1145/361219.361220) — Gerard Salton, A. Wong, C. S. Yang (1975)
  Journal: Communications of the ACM; Publisher: ACM Press; Volume: 18; Pages: 613-620; DOI: [10.1145/361219.361220](https://doi.org/10.1145/361219.361220)
  介绍了用于信息检索的向量空间模型，其中包含了构成TF-IDF基础的词项加权方案。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  标准教科书，全面解释了TF-IDF、其变体以及在信息检索和文本处理中的应用。
- [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — scikit-learn developers (2023)
  TfidfVectorizer的官方文档，详细说明了其参数、行为以及scikit-learn中使用的特定TF-IDF和IDF平滑计算方法。
