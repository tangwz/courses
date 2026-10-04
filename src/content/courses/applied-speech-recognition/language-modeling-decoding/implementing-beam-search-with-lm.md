---
course: "applied-speech-recognition"
chapter: "language-modeling-decoding"
lesson: "implementing-beam-search-with-lm"
sourceId: 7155
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/implementing-beam-search-with-lm"
title: "结合语言模型实现束搜索"
description: "了解结合了外部语言模型分数的束搜索解码器的实现细节。"
order: 6
plots: []
sourceHash: "b86e6799c5cbd1e0ab1d132f165a0ef0518e1adde14403f5b3c1f9dd56208914"
sourceCorrections: []
---

虽然标准的束搜索解码器通过保留多个候选转录结果改进了贪婪搜索，但其决策仅基于声学证据。为了生成符合语言学逻辑的输出，我们必须将语言模型的分数直接整合到束搜索过程中。这通过修改在每一步中用于对假设进行排序的评分函数来实现。

解码器的目标从找到声学上最可能的路径转变为找到同时最好地满足声学模型和语言模型的路径。束中每个假设的分数使用前面介绍的组合公式进行更新：

要生成语言连贯的输出，需要将语言模型的分数直接整合到束搜索过程中。这通过修改用于在每一步中评估假设的评分函数来实现。修改后的评分函数如下：


$$
\text{分数}(W) = \text{声学分数} + \alpha \cdot \text{语言模型分数} + \beta \cdot \text{词数}
$$


让我们分解一下每个部分是如何纳入的：

- **$\text{声学分数}$**: 这是字符序列的运行对数概率，从CTC输出计算。这是标准束搜索解码器会使用的分数。
- **$\text{语言模型分数}$**: 这是由外部n-gram语言模型计算的完整词序列的对数概率。此分数仅在形成完整词时应用。
- **$\alpha$ (阿尔法)**: 语言模型权重 (weight)。这个超参数 (parameter) (hyperparameter)控制语言模型的影响。较高的`α`会优先考虑语法正确性，有时会牺牲声学准确性，而较低的`α`则使解码器更信任声学模型。
- **$\beta$ (贝塔)**: 词插入奖励。这一项为假设中的每个词添加一个小的奖励。它抵消了概率模型偏爱较短序列的自然倾向（因为将更多小于1的概率相乘会导致更小的数值）。它有助于确保解码器不会不公平地惩罚较长、正确的句子。

### 评分机制的实际运作

语言模型的整合在解码过程中发生在特定时刻。束搜索算法按时间步进行，用新字符扩展每个假设。声学分数随着每个新字符的出现而更新。然而，语言模型分数只在识别到词边界时才计算并添加，这通常是在假设中附加空格字符时。

考虑束中两个相互竞争的假设：“recognize”和“wreck a nice”。当音频的下一部分听起来像“speech”时，声学模型可能会为字符`s-p-ee-ch`生成高概率。

1. **声学更新：** 解码器将“recognize”用一个空格扩展，然后是`s`、`p`等。声学分数在每次字符扩展时更新。
2. **语言模型查询：** 一旦假设变为“recognize ”，解码器就会识别出一个完整的词。然后它查询语言模型，获取“speech”跟在“recognize”后面的概率。语言模型将为此序列返回一个相对较高的概率。
3. **分数组合：** 加权的语言模型分数被添加到这个假设的总分中，使其获得显著提升。

同时，“wreck a nice”假设被扩展为“wreck a nice beach”。当“wreck a nice ”确定后，将查询语言模型以获取“beach”跟在“wreck a nice”后面的概率。尽管这个短语在语音上是可信的，但它不太常见，因此语言模型会为其分配较低的分数。结果，“recognize speech”假设可能会获得更高的总分并保留在束中，而“wreck a nice beach”可能会被剪枝。

> 上图显示了两个假设是如何评分的。即使“wreck a beach”的声学分数略高，语言模型为“recognize speech”分配的高概率也可以提高其最终分数，使其成为优选的转录结果。

### 伪代码实现

为了使此过程更具体，以下是一个包含n-gram语言模型的CTC束搜索解码器的高级算法。

```python
# 算法的简化表示

def ctc_beam_search_decoder(acoustic_probs, beam_width, lm, alpha, beta):
    # beam = [(前缀文本, 声学分数, 语言模型状态)]
    # lm_state 存储假设的n-gram历史

    initial_lm_state = lm.initial_state()
    beam = [("", 0.0, initial_lm_state)]

    for timestep_probs in acoustic_probs:
        new_beam = []
        for prefix, acoustic_score, lm_state in beam:
            for char, char_prob in timestep_probs.items():

                # 处理空白符和重复字符的标准CTC逻辑
                # (此部分为清晰起见已省略)
                new_prefix = prefix + char
                new_acoustic_score = acoustic_score + log(char_prob)

                # 检查是否刚完成一个词
                if char == ' ':
                    # 使用语言模型对完成的词进行评分
                    word = get_last_word(new_prefix)
                    lm_score = lm.score(word, state=lm_state)

                    # 获取下一个词的新的语言模型状态
                    new_lm_state = lm.update_state(word, lm_state)

                    # 用语言模型分数和词奖励更新总分
                    total_score = new_acoustic_score + alpha * lm_score + beta

                    new_beam.append((new_prefix, total_score, new_lm_state))
                else:
                    # 不是词边界，只需沿用语言模型状态
                    total_score = new_acoustic_score
                    new_beam.append((new_prefix, total_score, lm_state))

        # 剪枝束：按分数排序并保留前 `beam_width` 个假设
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    # 从最终束中最佳假设返回文本
    sorted(beam, key=lambda x: x[1], reverse=True)[0]
    return best_hypothesis[0]
```

### 实际考量

- **状态管理：** 如伪代码所示，束中的每个假设都必须维护自己的语言模型状态。对于n-gram模型，此状态就是假设的最后`n-1`个词。这确保了*下一个*词的概率是基于其正确语境计算的。
- **调整`α`和`β`：** `α`和`β`的值并非固定不变。它们是重要的超参数 (parameter) (hyperparameter)，必须在验证集上进行调整，以找到最适合您的特定声学模型和语言模型的最佳平衡。这通常通过使用不同值运行解码并选择产生最低词错误率（WER）的组合来完成。
- **性能：** 为每个假设中的每个新词查询外部语言模型会增加计算开销。像KenLM这样高效的语言模型工具包旨在实现快速查询，使得即使对于大型束，此过程也是可行的。

通过将语言模型整合到解码搜索中，您将ASR系统从一个简单的声学到音素转录器转变为一个更能理解和应用语言规则的智能系统，大幅提高了最终转录的质量和可读性。

## 参考资料

- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/1143844.1143891) — Alex Graves, Santiago Fernández, Faustina Gomez, and Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML '06); Publisher: ACM; Pages: 369-376; DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
  介绍连接时序分类（CTC），它是集束搜索中声学得分的来源，并讨论了基本的CTC集束搜索。
- [Automatic Speech Recognition: A Deep Learning Approach](https://link.springer.com/book/10.1007/978-1-4471-5779-3) — Dong Yu, Li Deng (2014)
  Publisher: Springer; DOI: [10.1007/978-1-4471-5779-3](https://doi.org/10.1007/978-1-4471-5779-3)
  全面介绍现代语音识别，其中第7章详细阐述了用于大词汇量连续语音识别（LVCSR）的解码策略，包括语言模型集成。
