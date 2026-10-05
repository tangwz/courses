# N-gram 语言模型

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/n-gram-language-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

深度学习 (deep learning)模型擅长从原始特征中学习，但它们主要处理声学信息。为了帮助它们做出符合语言学原理的判断，我们引入了统计语言模型。最基本的一种是N-gram模型，它通过对词语之间如何相互关联做出简化假设，来计算词语序列$W$的概率。

N-gram模型不尝试计算给定词语在所有前序词语完整历史背景下的概率（这会是计算密集且数据需求大的任务），而是假设给定词语的概率仅取决于前$n-1$个词。这是马尔可夫假设的一种运用。

### 语言中的马尔可夫假设

词语序列$W = (w_1, w_2, \dots, w_k)$的完整概率可以使用概率链式法则表示为：


$$
P(W) = P(w_1) P(w_2|w_1) P(w_3|w_1, w_2) \dots P(w_k|w_1, \dots, w_{k-1})
$$


N-gram模型通过限制上下文 (context)窗口来近似此计算。对于给定的$n$，词语$w_i$的概率近似为：


$$
P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \dots, w_{i-1})
$$


下面我们考察一下这对于常见的$n$值是如何运作的。

#### 一元模型 (n=1)

最简单的模型，即一元模型，假设每个词都独立于其他所有词。一个词的概率就是它在训练文本中出现的频率。

- **假设：** $P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i)$
- **序列概率：** $P(\text{"wreck a nice beach"}) = P(\text{"wreck"}) \times P(\text{"a"}) \times P(\text{"nice"}) \times P(\text{"beach"})$

这是一种“词袋”方法。它完全忽略了词序和上下文，使其过于简单，无法生成连贯的句子。它无法区分“recognize speech”和“speech recognize”。

#### 二元模型 (n=2)

二元模型是一个重要的进步。它假设一个词的概率仅取决于前面一个词。

- **假设：** $P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-1})$
- **序列概率：** $P(W) \approx \prod_{i=1}^{k} P(w_i | w_{i-1})$ （其中$w_0$是特殊的句首标记 (token)符）。

使用我们当前的例子，二元模型将评估：

- $P(\text{speech} | \text{recognize})$
- $P(\text{beach} | \text{nice})$

在一个大型英语文本语料库中，“recognize speech”这个短语远比“wreck a nice”常见，因此模型会正确地为第一个转录结果分配更高的概率。

#### 三元模型 (n=3)

三元模型会考虑前两个词，从而获取更多上下文信息。

- **假设：** $P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})$

这使得模型能够学习更长距离的依赖关系，比如“a nice beach”这个短语比“a nice speech”更可能出现。随着$n$的增加，模型可以捕获更多的上下文，但这会带来代价。高阶N-gram模型需要更多的训练数据，并且更容易受到我们即将讨论的问题的影响。在实际应用中，3元和4元模型历来为ASR系统提供了良好的平衡。

### 从文本语料库估计

N-gram概率直接从称为语料库的大量文本中估计得出。计算是最大似然估计（MLE）的直接运用，它涉及计数出现次数。对于二元模型，条件概率$P(w_i | w_{i-1})$估计如下：


$$
P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}
$$


例如，要计算$P(\text{speech} | \text{recognize})$，您需要统计语料库中“recognize speech”这对词出现的次数，并将其除以“recognize”出现的总次数。



[交互图表：'recognize'后的二元概率示例](https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/n-gram-language-models#plot-13kjflg)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "'recognize'后的二元概率示例",
    "xaxis": {
      "title": "词语"
    },
    "yaxis": {
      "title": "概率 P(词语 | 'recognize')",
      "type": "log",
      "tickformat": ".5f"
    },
    "barmode": "stack",
    "bargap": 0.4
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "speech",
        "the",
        "a",
        "beach"
      ],
      "y": [
        0.08,
        0.04,
        0.01,
        0.0001
      ],
      "marker": {
        "color": [
          "#20c997",
          "#adb5bd",
          "#adb5bd",
          "#fa5252"
        ],
        "line": {
          "width": 0
        }
      }
    }
  ]
}
```

</details>



> 基于文本语料库，“recognize”后面可能出现的不同词语的条件概率。训练良好的N-gram模型会为语言上合理的二元词“recognize speech”分配明显高于不合理的“recognize beach”的分数。

### 零频问题与平滑处理

N-gram模型面临的一个主要难题是数据稀疏性。如果像“transcribe audio”这样完全有效的二元词从未在您的训练语料库中出现，会发生什么？根据公式，它的计数为零，因此其概率为零。


$$
P(\text{audio} | \text{transcribe}) = \frac{\text{count}(\text{"transcribe audio"})}{\text{count}(\text{"transcribe"})} = \frac{0}{\text{count}(\text{"transcribe"})} = 0
$$


这是个问题。一个未曾出现的N-gram会导致整个句子的概率变为零，迫使解码器将其丢弃，即使句子其余部分的概率很高。

解决方案是**平滑处理**。平滑技术从已见过的N-gram中取少量概率质量，并将其重新分配给未见过的N-gram。这确保了没有N-gram的概率会恰好为零。

一种简单的方法是**拉普拉斯（或加一）平滑**，即我们对所有N-gram计数加一。然而，实践中使用了更复杂的技术。现代工具包，例如我们将在下一节使用的KenLM，采用**Kneser-Ney平滑**等高级平滑算法，这些算法为低频和未出现的N-gram的概率提供了更好的估计。

通过理解这些统计基础，您就能明白N-gram语言模型如何提供所需的语言学约束来引导解码器。在下一节中，我们将从理论转向实践，使用标准工具包构建我们自己的N-gram模型。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson; Pages: Chapter 3: N-gram Language Models
  一本内容全面的教科书，详细解释了N-gram语言模型、其估计方法、马尔可夫假设以及各种平滑技术。
- [Improved backing-off for M-gram language models](https://ieeexplore.ieee.org/document/488975) — Reinhard Kneser and Hermann Ney (1995)
  Journal: Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP); Pages: 181-184; DOI: [10.1109/ICASSP.1995.488975](https://doi.org/10.1109/ICASSP.1995.488975)
  介绍Kneser-Ney平滑的原始论文，这是一种在N-gram模型中用于概率估计的有效且广泛使用的方法，解决了零频率问题。
- [KenLM: Faster and smaller language model queries](https://aclanthology.org/W11-2139) — Kenneth Heafield (2011)
  Journal: Proceedings of the Sixth Workshop on Statistical Machine Translation; Publisher: Association for Computational Linguistics; Pages: 187-197
  描述了KenLM工具包，这是一个用于构建和查询N-gram语言模型的高性能库，常用于语音识别系统。

---

[上一节](01-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9C%A8%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20KenLM%20%E6%9E%84%E5%BB%BA%20N-gram%20%E6%A8%A1%E5%9E%8B.md)
