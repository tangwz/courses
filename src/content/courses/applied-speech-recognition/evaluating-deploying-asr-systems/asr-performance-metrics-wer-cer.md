---
course: "applied-speech-recognition"
chapter: "evaluating-deploying-asr-systems"
lesson: "asr-performance-metrics-wer-cer"
sourceId: 7157
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems/asr-performance-metrics-wer-cer"
title: "ASR 性能评估指标：词错误率 (WER) 和字符错误率 (CER)"
description: "了解评估语音识别系统的标准指标：词错误率 (WER) 和字符错误率 (CER)。"
order: 1
plots: []
sourceHash: "8906c0d74dea4aa78fb3f51a5a57267d1afee98dbeea995f9dc9baad75448b91"
sourceCorrections: []
---

评估声学模型的性能是一个重要步骤。虽然手动检查少量转录结果可能可行，但这种方法主观且无法扩展。为了准确评估系统，你需要客观、可重复的评估指标。词错误率 (WER) 和字符错误率 (CER) 是语音识别技术中的两种标准评估指标。这些指标衡量模型预测的转录（假设）与正确的人工验证转录（参考）之间的差异。

### 词错误率 (WER)

词错误率是 ASR 系统最常用的评估指标。它衡量词级别的错误数量，这与人类对大多数语言转录准确性的感知方式很好地契合。其计算基于 Levenshtein 距离，该距离用于寻找将一个序列转换为另一个序列所需的最小编辑次数。对于 WER，这些编辑类型包括：

- **替换 ($S$):** 参考文本中的一个词被假设文本中的另一个词替换。（例如，`weather` 变为 `feather`）。
- **删除 ($D$):** 参考文本中的一个词在假设文本中缺失。（例如，`turn left` 变为 `turn`）。
- **插入 ($I$):** 假设文本中存在一个词，但在参考文本中没有。（例如，`go` 变为 `uh go`）。

然后，这些计数会根据参考转录中的总词数 ($N$) 进行归一化 (normalization)，以计算最终的 WER。正如章节引言中提到的，公式如下：


$$
\text{词错误率} = \frac{S + D + I}{N}
$$


WER 值越低表示模型表现越好，WER 为 0 则表示完美转录。

我们来看一个实例。

- **参考:** `SHOW ME THE WEATHER` ($N=4$)
- **假设:** `SHOW THE WEATHER NOW`

为了计算错误，我们需要对齐 (alignment)这两个句子，以找到最小的编辑次数。

1. `SHOW` -> `SHOW` (正确)
2. `ME` ->  (删除, $D=1$)
3. `THE` -> `THE` (正确)
4. `WEATHER` -> `WEATHER` (正确)
5. -> `NOW` (插入, $I=1$)

在此示例中，我们有 $S=0$, $D=1$, $I=1$。参考文本中的总词数 $N$ 为 4。


$$
\text{词错误率} = \frac{0 + 1 + 1}{4} = \frac{2}{4} = 0.5 \text{ 或 } 50\%
$$


下图展示了这种对齐过程和相应的错误。

> 参考文本与假设文本的对齐，展示了一处删除和一处插入。总错误数为 2，导致 WER 为 50%。

### 字符错误率 (CER)

虽然 WER 是默认指标，但它不太适用于不以空格分词 (tokenization)的语言，例如中文或日语。对于需要单个字符准确度的任务，例如转录专有名词或字母数字代码，WER 也可能产生误导。对于这些情形，字符错误率 (CER) 是一个更优的选择。

其计算原理与 WER 相同，但操作对象是字符而非词语。


$$
\text{字符错误率} = \frac{S_{char} + D_{char} + I_{char}}{N_{char}}
$$


这里，$S_{char}$、$D_{char}$ 和 $I_{char}$ 分别是字符级别的替换、删除和插入，而 $N_{char}$ 是参考文本中的总字符数。

例如：

- **参考:** `HELLO` ($N_{char}=5$)
- **假设:** `HALLOW`

对齐 (alignment)结果将显示一处插入（`A`）和一处替换（`O` -> `W`）。

- $S_{char}=1$, $D_{char}=0$, $I_{char}=1$。
- $CER = (1 + 0 + 1) / 5 = 0.4$ 或 $40\%$。

### 错误率的理解与使用

一个常见的困惑点是 WER 可以超过 100%。当错误数量，特别是插入错误，大于参考文本中的词数时，就会出现这种情况。设想一个模型为一个短音频片段生成了非常长且不准确的转录。

- **参考:** `GO` ($N=1$)
- **假设:** `PLEASE NO DON'T GO`
- 这里，$S=0$, $D=0$, $I=3$。
- $WER = (0 + 0 + 3) / 1 = 3.0$ 或 $300\%$。

还需要记住，WER 和 CER 纯粹是词汇级别的指标。它们不考虑语义。一次简单的替换就可能彻底改变句子的意图，然而 WER 的惩罚与一个微不足道的错误相同。

- `turn left` -> `turn right` (1 替换，巨大的语义错误)
- `turn left` -> `turn uh left` (1 插入，轻微的语义错误)

尽管存在这些局限性，WER 仍然是比较 ASR 系统的普遍标准，因为它简单、可计算，并为整体性能提供了可靠的基准。

### 在 Python 中计算 WER 和 CER

你很少需要自己实现对齐 (alignment)算法。有几个 Python 库可以高效地计算这些指标。`jiwer` 库是一个常用且易于使用的选择。

要使用它，首先用 pip 安装：

```bash
pip install jiwer
```

然后，你可以用它来计算 WER 并获得不同错误类型的详细分类。

```python
import jiwer

# 参考和假设字符串
reference = "show me the weather"
hypothesis = "show the weather now"

# jiwer.wer 函数直接给出最终的 WER 分数
error_rate = jiwer.wer(reference, hypothesis)
print(f"Word Error Rate: {error_rate:.2%}")

# compute_measures 函数返回一个包含所有计数的字典
measures = jiwer.compute_measures(reference, hypothesis)

print(f"Substitutions: {measures['substitutions']}")
print(f"Deletions:     {measures['deletions']}")
print(f"Insertions:    {measures['insentions']}")
print(f"Reference words (N): {measures['truth']}")
```

运行此代码会产生我们手动计算的结果：

```
Word Error Rate: 50.00%
Substitutions: 0
Deletions:     1
Insertions:    1
Reference words (N): 4
```

要计算 CER，你可以使用相同的函数。如果你想明确指定，请确保传入字符分隔的字符串，尽管 `jiwer` 在你告知后会自动处理。

```python
# 通过处理字符来计算 CER
reference_chars = " ".join(list("hello")) # "h e l l o"
hypothesis_chars = " ".join(list("hallow")) # "h a l l o w"

cer_measures = jiwer.compute_measures(reference_chars, hypothesis_chars)
print(f"\nCharacter Error Rate: {cer_measures['wer']:.2%}")
```

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  全面介绍了语音识别基础知识，包括词错误率（WER）、Levenshtein距离和ASR评估方法的详细解释。
- [jiwer Documentation](https://jitsi.github.io/jiwer) — Nik Vaessen (2025)
  Publisher: 8x8
  本节中用于计算词错误率（WER）和字符错误率（CER）的Python库的官方文档，提供了实际使用示例。
- [Automatic Speech Recognition: A Deep Learning Approach](https://link.springer.com/book/10.1007/978-1-4471-5779-3) — Dong Yu and Li Deng (2014)
  Publisher: Springer; Pages: 321; DOI: [10.1007/978-1-4471-5779-3](https://doi.org/10.1007/978-1-4471-5779-3)
  一本详细的教科书，提供了基于深度学习的现代ASR系统见解，包括在此背景下对WER等评估指标的讨论。
