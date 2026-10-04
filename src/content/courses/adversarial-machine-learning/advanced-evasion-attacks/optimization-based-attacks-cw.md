---
course: "adversarial-machine-learning"
chapter: "advanced-evasion-attacks"
lesson: "optimization-based-attacks-cw"
sourceId: 4126
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-2-advanced-evasion-attacks/optimization-based-attacks-cw"
title: "基于优化的攻击：Carlini & Wagner 方法"
description: "了解 Carlini 和 Wagner 开发的强大基于优化的攻击。"
order: 2
plots: []
sourceHash: "b7bebf2175fbb69a803664ececa5135046ddb5792d5db6505471fbfab3485879"
sourceCorrections: []
---

虽然像投影梯度下降 (gradient descent)（PGD）这样的基于梯度的攻击为生成对抗性样本提供了坚实的基础，但它们主要侧重于在预定义的扰动预算（$\epsilon$-球）内找到*任何*对抗性样本。它们不一定能找到导致错误分类的*最小*扰动样本。对于扰动感知度降到最低非常重要，或衡量模型承受极限的情况，基于优化的攻击提供了一种更直接的方法。

由 Nicholas Carlini 和 David Wagner 提出的 Carlini & Wagner (C&W) 攻击，代表了一类强大的基于优化的规避攻击。C&W 方法并非仅在约束条件下找到*一个*对抗性样本，而是将搜索问题构建为一个正式的优化问题：找到最小的失真 $\delta$，使得 $x_{adv} = x + \delta$ 被错误分类并满足输入约束（例如，像素值保持有效）。

### 优化问题的构建

主要思想是最小化扰动大小（通过 $L_p$ 范数衡量），条件是模型对扰动后的输入进行错误分类。设 $x$ 为原始输入，$x'$ 为对抗性输入（$x_{adv}$），$f$ 为模型（通常由其在最终 softmax 前的 logits $Z(x')$ 表示）。目标是找到 $x'$ 来解决以下问题：


$$
\text{最小化 } ||x' - x||_p \quad \text{满足条件 } f(x') \text{ 被错误分类且 } x' \in [0, 1]^n
$$


在标准优化框架中，直接强制执行“错误分类”约束很困难。C&W 巧妙地用一个特殊设计的损失函数 (loss function) $g(x')$ 替换了此约束，当模型将 $x'$ 错误分类时，该损失函数达到最小值。优化问题于是变为：


$$
\text{最小化 } ||x' - x||_p + c \cdot g(x')
$$


此处：

- $||x' - x||_p$ 是失真项，使用 $L_p$ 范数（通常是 $L_0$、$L_2$ 或 $L_\infty$）衡量原始输入和对抗性输入之间的距离。
- $g(x')$ 是分类损失函数。
- $c$ 是一个平衡常数，经过仔细选择（通常通过二分查找）以保证在满足分类约束的同时保持低失真。更大的 $c$ 会更侧重于实现错误分类。

### C&W 分类损失

C&W 攻击的巧妙之处在于 $g(x')$ 的设计。对于目标攻击（目标是使模型将 $x'$ 分类为特定目标类别 $t$），一个常见的损失函数 (loss function)是：


$$
g(x') = \max( \max_{i \neq t} Z(x')_i - Z(x')_t, -\kappa )
$$


我们来分析一下：

- $Z(x')_i$ 是类别 $i$ 的对数几率值。
- $\max_{i \neq t} Z(x')_i$ 找到*除了*目标类别 $t$ 之外所有类别中最高的对数几率值。
- $Z(x')_t$ 是目标类别 $t$ 的对数几率值。
- 差值 $\max_{i \neq t} Z(x')_i - Z(x')_t$ 衡量了最有可能的错误类别与目标类别相比“强”了多少。我们希望这个差值是负数且绝对值较大，以便目标类别 $t$ 成为最有可能的预测。
- 外部的 $\max(\dots, -\kappa)$ 设定了一个置信度裕度 $\kappa \ge 0$。只有当 $Z(x')_t$ 比所有其他对数几率至少大 $\kappa$ 时，损失才变为负数。最小化此损失会促使目标对数几率超过次高对数几率达此裕度，从而提升错误分类的置信度。将 $\kappa = 0$ 设置为仅要求目标类别对数几率最高即可。

对于非目标攻击（目的只是使模型将 $x'$ 错误分类，而非其原始类别 $y_{true}$），损失可以定义为：


$$
g(x') = \max( Z(x')_{y_{true}} - \max_{i \neq y_{true}} Z(x')_i, -\kappa )
$$


此处，目的是通过使真实类别对数几率 $Z(x')_{y_{true}}$ 比任何其他类别中最高的对数几率至少小 $\kappa$ 来最小化损失。

### 问题的解决

C&W 攻击通常使用标准的基于梯度的优化器（如 Adam）来最小化组合目标函数 $||x' - x||_p + c \cdot g(x')$。然而，存在一个技术问题：输入 $x'$ 必须经常满足边界约束（例如，像素值在 $[0, 1]$ 之间）。直接应用梯度下降 (gradient descent)并在每一步裁剪 $x'$ 会阻碍收敛。

C&W 建议采用变量替换。例如，他们不直接优化 $x'$，而是优化一个变量 $w$，其中 $x' = \frac{1}{2} (\tanh(w) + 1)$。由于 $\tanh$ 的输出总是在 $(-1, 1)$ 之间，$x'$ 自然地保证在 $(0, 1)$ 之间，从而满足了边界约束。优化是针对 $w$ 进行的，失真 $||x' - x||_p$ 和分类损失 $g(x')$ 都是基于转换后的 $x'$ 计算的。

### C&W 攻击变体 ($L_2$, $L_\infty$, $L_0$)

C&W 框架是为不同的 $L_p$ 范数而开发的：

- **C&W $L_2$ 攻击**：这是最常见的变体。它最小化欧几里得距离 $||x' - x||_2$。它通常使用上述损失函数 (loss function)和 $\tanh$ 变量替换技巧。它以找到极低失真的 $L_2$ 对抗性样本而闻名，在扰动幅度方面通常优于 PGD，尽管速度通常较慢。
- **C&W $L_\infty$ 攻击**：此变体目标是最小化任何单个特征的最大变化，即 $||x' - x||_\infty$。它使用略微修改的优化方法，通常涉及类似于 PGD 的迭代步骤，但结合了 C&W 损失函数。其目标是找到能使 $L_\infty$ 攻击成功的最小 $\epsilon$。
- **C&W $L_0$ 攻击**：这种攻击最小化改变的像素数量，即 $||x' - x||_0$。它的计算难度更大。该方法通常涉及迭代地固定对损失贡献不大的像素，而只优化其余像素。

> PGD 和 C&W ($L_2$) 方法的区别。PGD 迭代地进行步长并投影，以保持在原始输入 $x$ 周围固定大小的 $L_p$ 球体内，寻找任何错误分类。C&W 则直接最小化与 $x$ 的 $L_2$ 距离，同时满足错误分类损失，从而找到跨越决策边界的最近点。

### 优势与权衡

**优势：**

- **有效性**：C&W 攻击，特别是 $L_2$ 变体，非常有效，对于给定模型，通常能找到扰动更小的对抗性样本，优于 PGD。
- **绕过梯度掩蔽**：因为 C&W 使用依赖于 logits 并包含失真项的优化目标，它有时可以绕过试图掩蔽或混淆梯度的防御措施，这可能会欺骗较简单的基于梯度的方法。
- **灵活性**：该框架可以适用于目标/非目标攻击和不同的 $L_p$ 范数。

**权衡：**

- **计算成本**：C&W 攻击比 FGSM 或 PGD 慢得多。它们涉及迭代优化过程，并且找到最优常数 $c$ 通常需要二分查找，多次运行攻击。
- **复杂性**：其实现比基本的基于梯度的方法更复杂，需要仔细处理优化目标和约束。

尽管计算成本较高，C&W 攻击仍是衡量机器学习 (machine learning)模型和防御抵御能力的重要衡量基准。它们能够找到接近最小的扰动，这使它们成为安全评估的标准工具。如果防御措施能够抵御强大的 C&W 攻击，与仅针对 FGSM 或 PGD 等更快速的方法进行测试相比，它能为其有效性提供更大的信心。

## 参考资料

- [Towards Evaluating the Robustness of Neural Networks](https://ieeexplore.ieee.org/document/7942998) — Nicholas Carlini, David Wagner (2017)
  Journal: 2017 IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 39-57; DOI: [10.1109/SP.2017.49](https://doi.org/10.1109/SP.2017.49)
  描述了开创性的Carlini & Wagner攻击（L0、L2、L∞）及其寻找最小扰动的优化框架。
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  虽然主要以PGD和对抗训练闻名，但该论文及其相关资源为理解强梯度攻击和鲁棒性提供了重要背景，这对于评估C&W作为基准的作用至关重要。
- [Adversarial Attacks and Defenses in Machine Learning: A Survey](https://dl.acm.org/doi/10.1145/3371921) — Xinyun Chen, Changxi Zheng, Li Erran Li, Dawn Song (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 53; Pages: Article 32; DOI: [10.1145/3371921](https://doi.org/10.1145/3371921)
  一项全面的综述，回顾了包括Carlini & Wagner在内的各种对抗性攻击技术，并讨论了它们的影响及相应的防御措施。
- [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Attacks](https://arxiv.org/abs/2003.01690) — Francesco Croce, Matthias Hein (2020)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2003.01690](https://doi.org/10.48550/arXiv.2003.01690)
  引入了AutoAttack，这是一个强大的基准测试套件，将C&W L2作为核心组件，用于全面评估模型抵抗强对抗性样本的鲁棒性。
