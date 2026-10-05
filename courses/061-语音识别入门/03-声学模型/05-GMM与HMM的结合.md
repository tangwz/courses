# GMM与HMM的结合

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/combining-gmms-and-hmms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

高斯混合模型（GMM）可以对单个音素的音频特征分布进行建模，而隐马尔可夫模型（HMM）可以表示序列。它们各自独立使用时，每个模型都有一个重要的局限。GMM对时间或序列没有感知；它只能告诉您单个音频帧与某个音素声音特征的匹配程度如何。另一方面，HMM能处理序列，但没有固有的方式将其状态与音频信号的连续、复杂数据关联起来。

解决方案是将它们结合成一个单一、功能更强的架构：GMM-HMM。这种混合模型多年来一直是语音识别领域的标准方法，为理解ASR系统的工作方式提供了坚实的基础。

### GMM作为HMM的“发射”机制

可以将HMM看作是一个状态机，其中每个状态代表一个音素。为使系统正常运行，它需要在每一步回答两个问题：

1. **转移：** 从当前音素状态转移到下一个状态的概率是多少？（这是HMM的任务）。
2. **发射：** 假设我们处于某个音素状态，我们当前观察到的音频帧是由该音素生成的概率是多少？

第二个问题就是GMM发挥作用的地方。在GMM-HMM中，HMM的每个状态都与其自己的GMM相关联。特定音素（如`/t/`）的GMM仅针对与/t/音对应的音频帧进行训练。

当ASR系统评估一段音频时，HMM会提出一个状态（音素）序列。对于序列中的每个状态，它会请求相应的GMM计算观察到的音频特征的概率。这个概率被称为**发射概率**。

因此，HMM处理序列（$P(\mathrm{下一个状态} \mid \mathrm{当前状态})$），而GMM处理每个状态下的观测似然（$P(\mathrm{音频特征} \mid \mathrm{状态})$）。

### GMM-HMM在实际中如何工作

让我们追踪一下GMM-HMM将如何处理单词“cat”（`/k/` `/æ/` `/t/`）的音频。系统使用一个HMM，其状态与这些音素相对应。

1. **状态`/k/`：** 最初的几帧音频（对应“k”音）被输入到系统中。HMM处于`/k/`状态。专门为`/k/`音素训练的GMM评估这些帧。它计算出一个高概率，确认这些音频特征与`/k/`音良好匹配。其他音素（如`/æ/`或`/s/`）的GMM对这些相同帧将返回非常低的概率。
2. **转移到`/æ/`：** HMM从其训练数据中得知，从`/k/`到`/æ/`的转移在英语中很常见。它转移到`/æ/`状态。
3. **状态`/æ/`：** 下一组音频帧（对应“a”音）现在由与`/æ/`状态关联的GMM评估。这个GMM找到强烈的匹配并输出高发射概率。
4. **转移和状态`/t/`：** 该过程重复进行。HMM转移到`/t/`状态，并且`/t/`的GMM成功验证了单词的最终音频帧。

路径`/k/` -> `/æ/` -> `/t/`的总概率是通过将沿途的转移概率和发射概率相乘计算得出的。解码器（您将在后面学习到它）负责寻找具有最高总概率的状态序列。

下图描绘了这种关系。每个HMM状态代表一个音素，其中包含一个GMM，负责计算该时刻观测到音频特征的概率。

> HMM确定音素状态的可能序列，而每个状态内的GMM计算观测到的音频特征与特定音素匹配的概率。

通过结合这两个模型，GMM-HMM系统能够有效建模单个语音的统计特性以及语言的序列化、时间依赖性。这种架构被证明极其有效，在端到端深度学习 (deep learning)方法兴起之前，成为几十年来语音识别领域的主力。

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本在线草稿教科书章节介绍了GMM-HMM架构作为语音识别中的标准声学模型。
- [Fundamentals of Speech Recognition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFapcU7Y4tqOhhpkhUO_ivWhiZQUE9AVvt6sHYpSY5rRlWNktuKHPPjil8ZZkhjwsxUjuT0ZfohMcd6W8YB1NAn9LL_ymyjH5PeUlj_FziHJRLs_OXsknXR7M5XV_9vvD34Ft1ac9xDi9FLG3htmUZpLbUkRGnOGZ9kl3B0wRXJSK80xOqbhnI1yC3Nw4vGTL4LyKsUy_v8) — Lawrence R. Rabiner, Biing-Hwang Juang (1993)
  Publisher: Prentice Hall
  一本经典教科书，全面解释了HMM、GMM及其在语音识别中的结合。
- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://ieeexplore.ieee.org/document/18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  一篇开创性论文，介绍了隐马尔可夫模型及其在语音识别中的早期应用。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738; DOI: [9780387310732](https://doi.org/9780387310732)
  本书对高斯混合模型和隐马尔可夫模型进行了数学和概率性分析。

---

[上一节](04-%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B%20%28HMM%29%20%E5%9C%A8%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md) · [下一节](06-%E5%9F%BA%E4%BA%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%BB%8D.md)
