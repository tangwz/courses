# 第 17 章：LLMs的优化算法

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms)

[返回课程目录](../README.md)

训练大型语言模型面临着特有的优化难题，因为它们规模庞大且计算需求高。标准的优化方法通常需要进行调整，以便有效地训练LLMs。本章将讲解常用到的优化算法和策略。

我们会先简要回顾梯度下降方法，然后把重点放在自适应优化器上，例如Adam和AdamW，并解释解耦权重衰减的原理。您将学习如何实现包含预热和衰减阶段的常见学习率调度策略 (例如：$$ \eta_{t} = \text{schedule}(\text{step}) $$)。我们还会介绍梯度裁剪，这是一种常用来防止梯度爆炸并提升训练稳定性的方法，通常通过重新缩放那些范数超出阈值$c$的梯度来实现：$$ \mathbf{g} \leftarrow \frac{c}{\|\mathbf{g}\|} \mathbf{g} \text{ if } \|\mathbf{g}\| > c $$ 最后，我们将谈到选择优化器重要超参数的实用指导，例如学习率$\eta$、Adam的动量项($\beta_1, \beta_2$)、数值稳定性项$\epsilon$以及权重衰减系数$\lambda$。

## 小节

- 1. [梯度下降算法变体回顾 (SGD, 动量)](01-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95%E5%8F%98%E4%BD%93%E5%9B%9E%E9%A1%BE%20%28SGD%2C%20%E5%8A%A8%E9%87%8F%29.md)
- 2. [自适应优化器：Adam和AdamW](02-%E8%87%AA%E9%80%82%E5%BA%94%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%9AAdam%E5%92%8CAdamW.md)
- 3. [学习率调度策略](03-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6%E7%AD%96%E7%95%A5.md)
- 4. [梯度裁剪方法](04-%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E6%96%B9%E6%B3%95.md)
- 5. [选择优化器超参数 (lr, betas, eps, weight_decay)](05-%E9%80%89%E6%8B%A9%E4%BC%98%E5%8C%96%E5%99%A8%E8%B6%85%E5%8F%82%E6%95%B0%20%28lr%2C%20betas%2C%20eps%2C%20weight_decay%29.md)
