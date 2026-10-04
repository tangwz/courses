---
course: "probability-statistics-fundamentals-ml"
chapter: "basic-probability"
lesson: "conditional-probability"
sourceId: 2434
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-3-basic-probability/conditional-probability"
title: "条件概率讲解"
description: "理解条件概率并学习如何计算P(A|B)。"
order: 4
plots: []
sourceHash: "6c6828ff7b884968b2b706ca583324c32824ee23e6bf4cd82b79d7fa9f700e38"
sourceCorrections: []
---

通常，我们考虑概率时，计算的是某个事件在所有可能结果中发生的可能性。但如果已经有了一些已知信息呢？如果我们已知有其他事件发生了呢？这就引出了**条件概率**。

条件概率衡量的是在事件B已经发生的情况下，事件A发生的可能性。可以把它看作是根据新信息来更新我们的概率估计。我们将其记作$P(A|B)$，读作“在B发生的条件下A的概率”。

### 直观理解：缩小样本空间

“给定B”这个条件很重要。当我们知道事件B已经发生时，原始的样本空间（所有可能结果的集合）就不再完全适用。我们可以有效忽略B未发生的任何结果。我们的注意力转移到一个新的、缩小的样本空间，该空间只包含B已经发生的结果。条件概率此时会问：在这个新的、更小的、B为真的背景下，A也发生的可能性是多少？

### 条件概率的计算

条件概率的正式定义和公式是：


$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$


我们来细分一下：

1. **$P(A \cap B)$ （分子）：** 这是事件A和事件B同时发生的概率。请记住，在集合论中，交集符号（$\cap$）表示“和”。我们需要A和B共有的结果。
2. **$P(B)$ （分母）：** 这是事件B发生的概率（我们的已知信息）。这起到了归一化 (normalization)因子的作用，表示我们缩小后样本空间的大小。
3. **条件：** 这个公式只有在$P(B) > 0$时才有意义。我们不能以一个发生概率为零的事件为条件。

本质上，这个公式计算了在B发生的结果中，A发生的结果所占的比例。

### 案例1：掷骰子

我们用一个标准的六面骰子。
样本空间$S = \{1, 2, 3, 4, 5, 6\}$。

定义两个事件：

- 事件A：掷出4。$A = \{4\}$。
- 事件B：掷出偶数。$B = \{2, 4, 6\}$。

首先，我们找出所需的概率：

- $P(B)$：在6个可能结果中有3个偶数，所以$P(B) = 3/6 = 1/2$。
- $P(A \cap B)$：这是掷出4*并且*掷出偶数的概率。唯一同时满足这两个条件的结果是掷出4。所以，$A \cap B = \{4\}$，且$P(A \cap B) = 1/6$。

现在，我们来计算条件概率$P(A|B)$：“已知掷出的是偶数，掷出4的概率是多少？”

使用公式：


$$
P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{1/6}{1/2} = \frac{1}{6} \times \frac{2}{1} = \frac{2}{6} = \frac{1}{3}
$$


这符合我们的直观判断吗？如果我们知道掷出的是偶数，我们缩小后的样本空间就是$\{2, 4, 6\}$。在这3个等可能的结果中，只有一个结果是4。因此，概率确实是1/3。条件概率$P(A|B) = 1/3$高于原始概率$P(A) = 1/6$，因为知道掷出的是偶数，增加了它是4的可能性。

### 案例2：使用调查数据

假设一项对100人进行的关于宠物饲养和住房类型的简短调查。结果总结在下表中：

|  | 有猫 | 无猫 | 总计 |
| --- | --- | --- | --- |
| 公寓 | 15 | 35 | 50 |
| 房屋 | 25 | 25 | 50 |
| **总计** | **40** | **60** | **100** |

我们来定义事件：

- 事件A：随机选择的人有猫。
- 事件B：随机选择的人住公寓。

我们要计算$P(A|B)$：“已知一个人住公寓，他有猫的概率是多少？”

首先，从表格中找出必要的概率（假设我们从100人中均匀随机选择一人）：

- $P(B) = P(\text{住公寓})$：100人中有50人住公寓。$P(B) = 50/100 = 0.5$。
- $P(A \cap B) = P(\text{有猫且住公寓})$：有15人既有猫又住公寓。$P(A \cap B) = 15/100 = 0.15$。

现在，应用条件概率公式：


$$
P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{15/100}{50/100} = \frac{0.15}{0.50} = \frac{15}{50} = 0.3
$$


因此，已知一个人住公寓，他有猫的概率是0.3或30%。

再次，我们来检验直观理解。如果我们*已知*这个人住公寓，我们只需关注表格的第一行。在该行中，总共有50人，其中15人有猫。比例是$15/50 = 0.3$。公式也给出了相同的结果。

### 与机器学习 (machine learning)的关系

条件概率是机器学习许多方面的基本原理。例如：

- **分类：** 模型通常会估计在给定输入特征的情况下，该输入属于某个类别的概率。$P(\text{类别} | \text{特征})$。例如，$P(\text{垃圾邮件} | \text{邮件包含“免费金钱”})$。
- **贝叶斯方法：** 贝叶斯定理，我们接下来会介绍，直接建立在条件概率之上，并广泛用于对不确定性进行建模和更新认知。

认识到一个事件的概率如何根据另一个事件的发生而变化，对于分析数据和构建从数据中学习的模型来说很有用。它使我们能够量化 (quantization)不同信息之间的关联。

## 参考资料

- [All of Statistics: A Concise Course in Statistical Inference](https://books.google.com/books/about/All_of_Statistics.html?id=l6K4BQAAQBAJ) — Larry Wasserman (2004)
  Publisher: Springer; Pages: Chapter 1; DOI: [10.1007/978-0-387-21765-7](https://doi.org/10.1007/978-0-387-21765-7)
  该教材为与现代数据科学相关的统计学提供了精炼的数学基础。
- [Conditional Probability (Lecture Notes and Video)](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/video-lectures/lecture-2-conditioning-and-bayes-rule/) — John N. Tsitsiklis, Dimitri P. Bertsekas (2000)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  大学课程材料，解释了基本的概率概念。
