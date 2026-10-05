# 评估合成文本：困惑度、BLEU分数

来源：[原文](https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-5-specialized-model-specific-metrics/evaluating-synthetic-text)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估生成文本的质量带来了独特的挑战，因为一般的统计指标虽然能提供合成数据评估的基准，但在文本方面却常力有不逮。自然语言拥有复杂的结构、依赖关系和语义细节，简单的分布比较可能无法体现。评估合成文本需要专为语言数据设计的指标，关注流畅性、连贯性以及与人类书写文本的相似度等。困惑度(Perplexity)和BLEU分数是此领域两个被广泛采用的指标。

### 困惑度：衡量语言模型的适合程度

困惑度与语言模型紧密相关。它量化 (quantization)了概率模型预测样本的能力。在评估合成文本时，我们通常使用预训练 (pre-training)的语言模型（理想情况下，它代表了“好”或“真实”文本的特点）来对生成文本进行评分。较低的困惑度分数表明语言模型认为合成文本序列更具可能性，这表示根据该模型，文本具有更好的流畅性和语法正确性。

从数学角度看，困惑度(PPL)是序列根据语言模型的平均负对数似然的指数化结果。对于一个符号序列 $W = w_1, w_2, ..., w_N$，其困惑度计算公式为：


$$
PPL(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log p(w_i | w_1, ..., w_{i-1}) \right)
$$


另外，它也常被计算为生成文本分布与语言模型所代表的目标分布之间的交叉熵损失的指数化结果。

**解释：**

- **低困惑度：** 模型对序列的“意外”程度较低；文本与评估语言模型学习到的模式非常吻合。这通常与更好的流畅性和语法结构相关。
- **高困惑度：** 模型认为该序列不太可能出现；相对于评估模型的训练数据，文本可能包含不自然的措辞、语法错误或意想不到的词语组合。

**在合成数据中的应用：**

您可以使用标准语言模型（例如GPT-2、BERT的掩码语言模型头或更简单的N-gram模型）来计算您的合成文本语料库的困惑度。比较合成数据集与真实数据集的平均困惑度（使用*相同*的语言模型评估），可以衡量语言的保真度。如果合成文本的困惑度分数接近真实文本，这表明生成器捕捉到了相似的语言模式。

**局限性：**

- **依赖评估模型：** 困惑度分数是相对于所选语言模型的。不同的模型可能会产生不同的分数。
- **对词汇和分词 (tokenization)的敏感性：** 当评估模型与被评估文本之间的分词和词汇保持一致时，比较才最有意义。超出词汇表 (vocabulary)的词语会显著影响分数。
- **不保证事实准确性或语义：** 文本可以流畅且语法正确（低困惑度），但可能毫无意义或事实不准确。
- **不直接衡量多样性：** 生成重复但流畅文本的生成器可能获得低困惑度。

### BLEU分数：评估翻译和生成质量

BLEU（双语评估替补）分数起源于机器翻译，用于衡量机器翻译文本与高质量人工参考译文之间的相似度。它已被调整以评估其他文本生成任务，包括合成文本生成，这类任务的目标通常是生成与参考语料库相似的文本。

BLEU通过衡量N-gram（连续的N个词序列）的重叠程度，将生成文本与一个或多个参考文本进行比较。其主要组成部分是：

1. **修正的N-gram精确率($p_n$)：** 计算生成文本（候选文本）中在*任何*参考文本中也出现的N-gram的比例。它之所以“修正”，是因为每个参考N-gram在每个候选句子中只被匹配一次，这防止了重复但相关的词语导致分数虚高。精确率针对不同N值（通常为1到4）进行计算。
2. **简短惩罚(BP)：** 对比其相应参考文本明显更短的生成文本进行惩罚。这可以避免模型通过简单输出非常短、安全的句子来获得高精确率分数。
   
   $$
   BP = \begin{cases} 1 & \text{如果 } c > r \\ e^{(1 - r/c)} & \text{如果 } c \le r \end{cases}
   $$
   
   其中 $c$ 是候选语料库的总长度，$r$ 是有效参考语料库长度（通常是与其最接近的参考句子的长度之和）。

最终的BLEU分数通常计算为各个N-gram精确率的几何平均值，再乘以简短惩罚：


$$
BLEU = BP \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)
$$


通常使用均匀权重 (weight)（$w_n = 1/N$），且 $N$ 通常设为4（BLEU-4）。

**解释：**

- **更高的BLEU分数（接近1）：** 表示生成文本与参考文本在N-gram重叠方面具有更大的相似度。
- **更低的BLEU分数（接近0）：** 表示相似度较低。

**在合成数据中的应用：**

要使用BLEU评估一般的合成文本，您将*真实*数据集中的样本视为“参考”。然后，您针对真实文本样本集合，计算每个合成文本样本的BLEU分数。更高的平均BLEU分数表明合成文本与真实数据共享更多连续的词语序列。如果合成数据需要紧密模仿原始数据的风格或内容模式，这一点就显得尤为重要。

**局限性：**

- **需要参考文本：** BLEU本质上需要参考文本进行比较。其解释很大程度上依赖于这些参考文本的质量和相关性。
- **侧重精确率：** 它主要衡量词语和短语的重叠（精确率），可能遗漏召回率（是否涵盖了参考文本的所有方面）。
- **对语义不敏感：** 含义相似但措辞不同的文本会获得较低的BLEU分数。同义词或意译不会天然获得奖励。
- **形态丰富性方面的挑战：** 具有复杂形态的语言对精确的N-gram匹配构成挑战。
- **短文本问题：** 对于非常短的文本或比较单个句子时，可靠性可能较低。

### 了解困惑度和BLEU

尽管困惑度和BLEU很常用，其他指标也提供了不同的视角：

- **ROUGE（面向召回率的摘要评估辅助指标）：** 常用于文本摘要，ROUGE侧重于N-gram的*召回率*（参考文本中有多少N-gram出现在候选文本中），并有ROUGE-L（最长公共子序列）等变体。
- **METEOR（带显式排序的翻译评估指标）：** 考虑精确词语匹配、词干提取和同义词，在评分前对候选句和参考句进行对齐 (alignment)。通常与人类判断的相关性优于BLEU。
- **基于嵌入 (embedding)的指标：** 计算合成文本与真实文本的向量 (vector)表示（嵌入）之间的相似度（例如，使用Sentence-BERT）。这些指标比N-gram重叠更能捕捉语义相似性，评估即使措辞不同，*含义*是否仍得到保持。例子包括平均余弦相似度或嵌入距离指标。

### 实际操作

像`nltk`、Hugging Face的`evaluate`和`torchtext`这样的库提供了计算困惑度（通常需要与语言模型集成）以及BLEU/ROUGE/METEOR分数的实现。

```python
# 使用Hugging Face的evaluate库计算BLEU的示例
# 注意：需要安装：pip install evaluate sacrebleu

import evaluate

# 示例合成数据和真实数据（参考）
predictions = ["the cat sat on the mat", "this is a generated sentence"]
references = [
    ["the cat was on the mat", "a cat sat on the mat"], # 第一个预测的参考
    ["this is the reference text", "this is reference sentence number two"] # 第二个预测的参考
]

# 加载BLEU指标
bleu_metric = evaluate.load("bleu")

# 计算分数
results = bleu_metric.compute(predictions=predictions, references=references)

print(f"BLEU Score: {results['bleu']:.4f}")
# 输出可能如下：BLEU Score: 0.3905（值取决于具体的实现细节）
# 各个N-gram精确率通常也在'results'中提供。

# 困惑度示例（使用evaluate，需要模型）
# perplexity_metric = evaluate.load("perplexity", module_type="metric")
# model_id = "gpt2" # Example model
# synthetic_texts = ["generated sentence one.", "another generated sentence."]
# ppl_results = perplexity_metric.compute(model_id=model_id,
#                                         add_start_token=False, # 模型特有
#                                         data=synthetic_texts)
# print(f"Mean Perplexity: {ppl_results['mean_perplexity']:.2f}")
# 注意：实际实现可能因模型和库的具体情况而异。
```

### 选择合适的指标

困惑度、BLEU、ROUGE、METEOR或嵌入 (embedding)指标之间的选择，取决于合成文本生成的具体目标：

- **流畅性/连贯性：** 困惑度是一个有力的指标。
- **模仿风格/内容：** 如果期望与真实数据有高N-gram重叠，BLEU或ROUGE会很有用。
- **语义相似性/含义保留：** 基于嵌入的指标通常更合适。
- **特定NLP任务的效用：** 根据下游任务进行评估（例如，如果文本用于训练分类器，则评估分类准确性）。

通常，这些指标的组合能够提供比单一分数更全面的评估。评估合成文本不仅包括理解统计相似性，还涉及语言质量和语义有效性，这使得这些专业指标成为不可或缺的工具。

## 参考资料

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311–318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍了BLEU分数，这是一种广泛用于评估机器翻译和文本生成质量的指标。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Pearson
  一本全面的自然语言处理教材，涵盖了语言建模和困惑度等基本概念。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81; DOI: [10.3115/1621251.1621280](https://doi.org/10.3115/1621251.1621280)
  提出了ROUGE指标，常用于基于n-gram召回率评估摘要和文本生成。
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1904.09675](https://doi.org/10.48550/arXiv.1904.09675)
  介绍了BERTScore，一种基于预训练上下文嵌入的指标，用于更语义化地评估文本生成质量。

---

[上一节](01-%E8%AF%84%E4%BC%B0%E5%90%88%E6%88%90%E5%9B%BE%E5%83%8F%EF%BC%9AFID%E3%80%81IS%E3%80%81%E7%B2%BE%E7%A1%AE%E7%8E%87%E3%80%81%E5%8F%AC%E5%9B%9E%E7%8E%87.md) · [下一节](03-%E8%AF%84%E4%BC%B0%E5%90%88%E6%88%90%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE.md)
