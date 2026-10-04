---
course: "probability-statistics-essentials-ml"
chapter: "probability-foundations-revisited"
lesson: "bayes-theorem-explained"
sourceId: 1350
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-1-probability-foundations-revisited/bayes-theorem-explained"
title: "贝叶斯定理讲解"
description: "学习贝叶斯定理的公式和在概率更新方面的应用。"
order: 3
plots: []
sourceHash: "49a60edebf63f54dd9123465d787e5f5411b02b48e17c077dbd0853e8dfe9b7d"
sourceCorrections: []
---

条件概率 $P(A|B)$ 告诉我们事件 B 已经发生的情况下，事件 A 发生的概率。但是，如果我们知道 $P(A|B)$ 而想求 $P(B|A)$ 怎么办？这在数据分析和机器学习 (machine learning)中是一种常见情形。例如，我们可能知道在患有某种疾病的情况下，出现某些症状的概率，但我们想计算在观察到某些症状的情况下，患有该疾病的概率。这正是贝叶斯定理发挥作用的地方。

该定理以托马斯·贝叶斯牧师的名字命名，它提供了一种合理方法，用于根据新证据更新我们的信念（概率）。它是贝叶斯统计的理论依据，在从医疗诊断到垃圾邮件过滤和模型参数 (parameter)估计的诸多方面都有应用。

### 公式

贝叶斯定理的数学表达式为：

$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$

我们来分析每个组成部分：

- **$P(B|A)$: 后验概率。** 这是事件 B 在我们观察到事件 A *之后*发生的概率。它代表我们对 B 的更新后的信念。这通常是我们想要计算的。
- **$P(A|B)$: 似然。** 这是在事件 B 为真 *的情况下*，观察到事件 A 的概率。如果我们的假设 B 是正确的，证据 A 的可能性有多大？在许多机器学习 (machine learning)问题中，这对应于给定模型或其参数 (parameter)的数据似然。
- **$P(B)$: 先验概率。** 这是在我们观察到任何证据 A *之前*，对事件 B 概率的初始信念。它反映了先前的知识或假定。
- **$P(A)$: 证据（或边际似然）。** 这是无论 B 如何，观察到事件 A 的总体概率。它充当一个归一化 (normalization)常数，确保所得的后验概率 $P(B|A)$ 是一个介于 0 和 1 之间的有效概率。

从本质上讲，贝叶斯定理告诉我们如何通过纳入在假设 B 下观察到证据 A 的似然 $P(A|B)$，并按证据 $P(A)$ 的总体概率进行缩放，从而将我们的先验信念 $P(B)$ 更新为后验信念 $P(B|A)$。

### 如何推导

这个定理并非凭空出现；它直接源于条件概率的定义。回顾一下：

1. $P(A|B) = \frac{P(A \cap B)}{P(B)}$
2. $P(B|A) = \frac{P(B \cap A)}{P(A)}$

由于交集是对称的（$P(A \cap B) = P(B \cap A)$），我们可以重新排列公式 (1) 得到：
$P(A \cap B) = P(A|B)P(B)$

现在，将 $P(A \cap B)$（与 $P(B \cap A)$ 相同）的这个表达式代入公式 (2)：

$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$

这样就推导出来了。

### 计算证据 $P(A)$

有时，证据 $P(A)$ 的概率不能直接获得。我们通常可以使用全概率定律来计算它。如果 B 既可能发生也可能不发生（设 $B^c$ 表示补集，即“非 B”），那么事件 A 既可以在 B 发生时发生，也可以在 B 不发生时发生。我们可以将 $P(A)$ 表示为：

$P(A) = P(A|B)P(B) + P(A|B^c)P(B^c)$

这种展开形式很有用，因为我们通常知道在不同假设（B 和非 B）下证据的似然以及这些假设的先验概率。将其代入分母即可得到贝叶斯定理的展开形式：

$P(B|A) = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A|B^c)P(B^c)}$

### 贝叶斯定理为何对机器学习 (machine learning)很重要？

贝叶斯定理有以下几个重要原因：

1. **信念更新：** 它提供了一种正式的机制，用于在新数据到达时更新模型或参数 (parameter)。这对于贝叶斯机器学习和在线学习系统很重要。
2. **分类模型：** 它是朴素贝叶斯分类器的依据。这些模型运用贝叶斯定理（并采用特征独立性的简化“朴素”假定）来计算给定特征下数据点属于某个类别的概率，$P(\text{类别} | \text{特征})$。
3. **不确定性下的推理 (inference)：** 它使我们能够将先前的知识与观察到的数据结合起来进行推断，这在处理有噪声或不完整信息时很重要。

### 一个简单例子：疾病诊断

我们用一个常见例子来说明。假设有一种疾病 (D) 影响了 1% 的人口。有一种针对这种疾病的检测 (T)。

- 该检测正确识别疾病（真阳性）的概率为 95%。$P(T|D) = 0.95$ (敏感度)
- 当一个人健康时（$\neg D$），检测错误指示疾病（假阳性）的概率为 5%。$P(T|\neg D) = 0.05$

我们已知：

- 患病先验概率：$P(D) = 0.01$
- 未患病先验概率：$P(\neg D) = 1 - P(D) = 0.99$
- 患病时检测呈阳性的似然：$P(T|D) = 0.95$
- 未患病时检测呈阳性的似然：$P(T|\neg D) = 0.05$

现在，有人检测结果呈阳性（事件 T）。他们实际患病的概率 $P(D|T)$ 是多少？我们使用贝叶斯定理：

$P(D|T) = \frac{P(T|D)P(D)}{P(T)}$

首先，我们需要分母 $P(T)$，即检测呈阳性的总体概率。我们使用全概率定律：

$P(T) = P(T|D)P(D) + P(T|\neg D)P(\neg D)$
$P(T) = (0.95 \times 0.01) + (0.05 \times 0.99)$
$P(T) = 0.0095 + 0.0495$
$P(T) = 0.059$

现在我们可以计算后验概率了：

$P(D|T) = \frac{0.95 \times 0.01}{0.059}$
$P(D|T) = \frac{0.0095}{0.059} \approx 0.161$

所以，即使检测结果呈阳性，实际患病的概率也只有约 16.1%。这可能看起来违反直觉，但它突显了低先验概率（$P(D)=0.01$）和非零假阳性率的影响。相对大量的健康人群意味着即使是很小的假阳性率，也会比来自少量患病人群的真阳性产生更多的假阳性。

> 疾病诊断示例中使用贝叶斯定理的计算流程。先验和似然组合形成证据，然后证据对似然与先验的乘积进行归一化 (normalization)，从而得到后验概率。

贝叶斯定理提供了一个结构化的框架，用于概率推理 (inference)，并随着我们收集更多数据而更新我们的认知。它的应用范围远不止简单例子，为有效处理不确定性的复杂机器学习 (machine learning)算法提供了依据。在后面的章节中，您将看到 SciPy 等库如何提供帮助，但理解其核心定理很重要。

## 参考资料

- [Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/9780262018029/machine-learning/) — Kevin P. Murphy (2012)
  Publisher: MIT Press
  对于理解机器学习的概率基础至关重要，其中贝叶斯定理应用于许多模型和算法。
- [Bayesian Data Analysis](https://www.crcpress.com/Bayesian-Data-Analysis-Third-Edition/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin (2013)
  Publisher: CRC Press
  贝叶斯统计方法的标准参考文献，详细介绍了贝叶斯定理在高级数据分析中的理论和实际应用。
- [Probabilistic Systems Analysis and Applied Probability](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) — John Tsitsiklis (2013)
  Publisher: MIT OpenCourseWare
  提供了麻省理工学院课程的高质量讲义和习题，为理解贝叶斯定理提供了详细且实用的方法。
