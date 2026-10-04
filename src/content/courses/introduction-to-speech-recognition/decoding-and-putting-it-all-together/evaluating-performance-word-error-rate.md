---
course: "introduction-to-speech-recognition"
chapter: "decoding-and-putting-it-all-together"
lesson: "evaluating-performance-word-error-rate"
sourceId: 7134
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together/evaluating-performance-word-error-rate"
title: "评估表现：词错率 (WER)"
description: "了解词错率 (WER)，这是衡量语音识别系统性能的行业通用指标。"
order: 6
plots: []
sourceHash: "c968dcf4801c63f1e406531c4daa5c4e66516598d409edcb1c8e897f07e5a14c"
sourceCorrections: []
---

当语音识别系统通过结合声学模型和语言模型输出生成最终转录文本后，一个基本问题浮现：结果有多准确？仅仅查看文本可能只会给您一个大概印象，但为了提升我们的系统或将其与其他系统比较，我们需要一个统一且量化 (quantization)的方法来衡量其准确性。此时，词错率（WER）就派上用场了。

### 什么是词错率 (WER)？

词错率是衡量语音识别系统性能的通用指标。它将自动语音识别 (ASR) 系统输出的文本（即“假设文本”）与正确的人工转录文本（即“参考文本”或“真实文本”）进行比较。最终的词错率是一个百分比，它表示系统相对于参考文本长度所犯的错误数量。词错率越低，系统性能越好。

其计算基于将假设文本转换为参考文本所需的最少修改次数。这些修改分为三类：

1. **替换 (S)：** 当系统错误转录了一个词时（例如，参考文本是“on”，但假设文本是“off”）。
2. **删除 (D)：** 当系统遗漏了参考文本中的一个词时（例如，参考文本是“sat on the”，但假设文本是“sat the”）。
3. **插入 (I)：** 当系统添加了参考文本中没有的词时（例如，参考文本是“the cat”，但假设文本是“the a cat”）。

### 词错率公式

为了计算词错率，您需要将这三类错误的数量相加，然后除以参考文本中的总词数 ($N$)。


$$
WER = \frac{S + D + I}{N}
$$


其中：

- $S$ = 替换次数
- $D$ = 删除次数
- $I$ = 插入次数
- $N$ = 参考文本中的词数

这个公式实质上计算了被错误替换、删除或插入的词语所占的百分比。

> 这是一个词错率计算图示，显示了替换、删除和插入如何组合形成总错误数，然后该总错误数再通过参考文本的长度进行归一化 (normalization)处理。

### 实际示例

让我们通过一个示例来了解它的运作方式。在计算词错率之前，我们需要对齐 (alignment)参考文本和假设文本，以找出最少的错误数量。

- **参考文本 (N=6)：** `the cat sat on the mat`
- **假设文本：** `the cat on a mat`

为了对齐这些文本，我们可以这样来表示：

```
Reference:   THE   CAT   SAT   ON   THE   MAT
Hypothesis:  THE   CAT   ---   ON    A    MAT
Error:       C     C     D     C     S     C
```

这里，`C` 代表正确（Correct），`D` 代表删除（Deletion），`S` 代表替换（Substitution）。让我们统计一下：

- **替换 (S)：** 1（词语 `THE` 被 `A` 替换了）
- **删除 (D)：** 1（词语 `SAT` 完全遗漏了）
- **插入 (I)：** 0（没有添加额外词语）
- **参考文本中的词数 (N)：** 6

现在，我们将这些数字代入公式：


$$
WER = \frac{1 + 1 + 0}{6} = \frac{2}{6} = 0.333
$$


因此，词错率为 33.3%。

### 词错率能高于 100% 吗？

是的，完全可以。当错误数量 ($S + D + I$) 大于参考文本中的词数 ($N$) 时，就会发生这种情况。这最常发生在自动语音识别 (ASR) 系统生成大量额外词语（插入）时。

例如，考虑这个情况：

- **参考文本 (N=2)：** `recognize speech`
- **假设文本：** `wreck a nice beach`

在这里，最佳对齐 (alignment)结果是两次替换（`recognize` -> `wreck`，`speech` -> `nice`）和两次插入（`a`，`beach`）。

- **替换 (S)：** 2
- **删除 (D)：** 0
- **插入 (I)：** 2
- **参考文本中的词数 (N)：** 2


$$
WER = \frac{2 + 0 + 2}{2} = \frac{4}{2} = 2.0
$$


这使得词错率达到 200%。超过 100% 的分数明确表明，系统的输出比原始音频明显更长且更容易出错。

### 解读词错率分数

- **词错率 0%** 表示完美转录。
- **词错率越低越好。** 一个词错率为 10% 的系统比词错率为 25% 的系统表现更好。
- **“好”的词错率取决于具体情况。** 对于转录清晰、录制良好的音频（如播客），词错率低于 5-10% 被认为是很好的。对于嘈杂的呼叫中心音频，其中包含多位说话者和浓重口音，30-40% 的词错率可能就是可接受甚至是很不错的表现了。

### 词错率的局限性

尽管词错率是行业通用指标，但它并非衡量质量的完美方法。它有一些您应该了解的局限性：

- **所有错误均等。** 词错率对将 `a` 替换为 `the` 的惩罚程度与将 `start` 替换为 `stop` 的惩罚程度相同。在语音命令系统中，第二种错误要严重得多，但词错率却将它们同等对待。
- **它忽略了语义。** “How is the weather today?” 和 “What is the weather like today?” 这两句话意思相同，但比较时会得出非零的词错率。
- **标点符号和大小写通常会被忽略。** 为确保公平比较，参考文本和假设文本在计算词错率之前通常会进行标准化处理。这包括将所有文本转换为小写、移除标点符号，有时还会将数字映射为词语（例如，`4` 转换为 `four`）。

尽管存在这些局限性，词错率仍然是一个重要的工具。它提供了一个简单、标准化的分数，以便以统一的方式跟踪改进和比较不同自动语音识别 (ASR) 系统的性能。

## 参考资料

- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Pages: Chapter 10, Section 3
  自然语言处理和语音的标准教材，全面详细地解释了语音识别评估，包括词错误率（WER）的定义、计算及其细微之处。
- [Automatic Speech Recognition: A Deep Learning Approach](https://link.springer.com/book/10.1007/978-1-4471-5779-3) — Dong Yu, Li Deng (2014)
  Publisher: Springer; Pages: Chapter 16, Section 2; DOI: [10.1007/978-1-4471-5779-3](https://doi.org/10.1007/978-1-4471-5779-3)
  一本关于现代自动语音识别的基础书籍，在深度学习架构的背景下，讨论了WER作为评估ASR系统性能的主要指标。
- [NIST Speech Recognition Scoring Toolkit (SCTK) Documentation](https://github.com/usnistgov/SCTK) — National Institute of Standards and Technology (NIST) (2021)
  Publisher: National Institute of Standards and Technology (NIST)
  提供了计算语音识别中词错误率（WER）的权威方法和工具（如sclite），概述了对齐、错误计数和标准化的标准实践。
