---
course: "intro-synthetic-data-ml"
chapter: "generating-synthetic-tabular-data"
lesson: "row-sampling-techniques"
sourceId: 5661
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-3-generating-synthetic-tabular-data/row-sampling-techniques"
title: "行采样方法"
description: "了解简单的技术，例如从现有数据集中采样行，可能带有修改。"
order: 2
plots: []
sourceHash: "f44493051f6455e652670506b4d624fc204461f863bcedd081b0d6553f359bbc"
sourceCorrections: []
---

我们来了解一种生成合成表格数据最直接的方法：直接从现有数据集中采样行。设想你有一个真实数据表，例如客户信息或传感器读数。行采样方法以这个现有表格为蓝图，构建一个新的合成表格。

这种方法尤其有用，当你需要一个紧密反映原始数据整体结构和统计属性的数据集时，可能用于软件测试、创建用于共享的匿名版本，或者仅仅是生成一个具有相似特征的更大规模数据集。

### 基本行采样：有放回采样

最简单的方法是**有放回采样**。将原始数据集想象成一个装满编号球的大袋子，每个球代表一行。

1. 你伸进袋子里，随机取出一个球（一行）。
2. 你记录下它的编号（将该行复制到你的新合成数据集中）。
3. 重要的一点是，你把球*放回*袋子里。
4. 你重复这个过程，直到你的合成数据集达到所需的大小。

在合成数据生成中，有放回采样是一种方法，它允许原始数据集中的同一行在合成数据集中出现多次。当原始数据集包含 $N$ 行，并且目标是创建一个包含 $M$ 行的合成数据集时，该过程涉及重复随机抽取并放回行的操作 $M$ 次。

这里有一个小示例：

**原始数据（3 行）：**

| ID | 特征A | 特征B |
| --- | --- | --- |
| 1 | 10 | 红色 |
| 2 | 15 | 蓝色 |
| 3 | 12 | 红色 |

**可能的合成数据（有放回采样，大小为 4）：**

| ID | 特征A | 特征B |
| --- | --- | --- |
| 2 | 15 | 蓝色 |
| 1 | 10 | 红色 |
| 2 | 15 | 蓝色 |
| 3 | 12 | 红色 |

注意第2行被选中了两次。

**优点：**

- **简单：** 非常容易理解和实现。
- **保持行完整性：** 合成数据集中的每一行都是原始数据中存在的真实行。这意味着单行*内*的复杂关系得到完美保持。
- **保持分布：** 合成数据集中每个独立列的数值分布（边际分布）将与原始数据非常相似，特别是当合成数据集较大时。

**缺点：**

- **无新数据：** 你并没有创建真正*新*的数据点或特征组合。你只是重复使用现有的。
- **隐私有限：** 由于复制的是整行，如果原始数据包含敏感信息，这种方法本身提供的隐私保护很少。
- **不增加多样性：** 如果你的原始数据集很小或缺乏多样性，简单地重采样行不会增加任何新的模式。

### 无放回采样

另一种选择是**无放回采样**。这次，当你从袋子里取出一个球（一行）时，你*不*把它放回去。这意味着每一条原始行在你的合成数据集中最多只能出现一次。

这种方法通常用于创建原始数据的一个更小、随机的子集。例如，你可能会用它来为机器学习 (machine learning)模型创建训练集或测试集划分。它通常不用于创建*大于*原始数据的合成数据集，因为你将没有足够的唯一行可供采样。

### 分层采样

如果你的数据集有重要的特定子组，但有些子组比其他子组小得多怎么办？例如，设想一个客户数据集，其中95%是普通客户，5%是高价值客户。如果你使用简单的随机采样（有放回），一个小的合成样本可能会不小心遗漏高价值客户，或者包含很少的高价值客户，无法反映它们真实的比例或重要性。

**分层采样**解决了这个问题。其过程包括：

1. **划分：** 根据一个或多个列中的值（例如，根据“客户类型”列）将原始数据集分成不同的子组（层）。
2. **层内采样：** *在每个子组内独立地*执行随机采样（如果想保持或增加大小，通常采用有放回采样）。你决定从每个层中抽取多少样本。通常，你会按比例抽取样本以保持原始组大小，但如果需要，你也可以对稀有组进行过采样。
3. **合并：** 将从每个层中抽取的样本组合起来，形成最终的合成数据集。

这保证了合成数据集保持原始数据中已定义子组的相对比例。

> 分层采样的流程，保证了不同客户类型的代表性。

### 带有修改的行采样（展望）

虽然上述方法复制的是整行，但一个稍微复杂一些的方法是对一行进行采样，然后对其某些值进行轻微*修改*。例如，你可以：

- 采样一行。
- 保持大多数分类特征不变。
- 向数值特征添加少量随机噪声（例如，从均值为0的正态分布中抽取）。

这开始创建了原始数据集中不存在的数据点，可能提供更好的隐私和新颖性。然而，进行这些修改需要仔细考虑，以确保生成的数据保持真实和有用。这项技术是通向更复杂生成模型的桥梁，这些模型学习数据的潜在模式，而不是仅仅复制行。

### 纯行采样的局限性

记住主要的局限性很重要：基本的行采样方法（不带修改）主要复制你已有的数据。它们复制现有行，无论是精确复制还是通过保持组比例来实现。它们不生成原始数据中不存在的根本性的新模式或关系。如果你的目标是生成的数据能够呈现源数据中未明确存在的可能性，你将需要后续讨论的更高级方法。

然而，对于创建平衡测试集、在保持其基本结构的同时快速扩展数据集，或生成简单的匿名版本等任务，行采样提供了一个易于上手且有效的方法。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHk60FrGk3ayaxnCT33GiMjut8swwT9mmSr8I0T3KuJJBICuHxFEgh2yxKE4R8GewnokCpQr4eNXIAfoIotyxFW2NfqMZ_lsa5tdsGJzzffqOOyDH26BaqGzkPXIgOVAA_XGsLe4Ksv3dbSGRXmac5EjDCm1ypgWSrn9uQQsVRhKftTEH3BavfMe0k5zhaoVnmmfuNKguFUzrhEYSEn1t0irlM0zoXQX6MpuU_6kvH3y2RUhGiCtmk1vd_Q-6zJNTU=) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  介绍统计抽样技术，包括随机抽样和分层抽样方法，适用于数据科学和机器学习。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  统计学习的标准教材。书中包含重采样方法，例如自助法（有放回抽样）和数据划分，对机器学习中的行抽样有益。
- [sklearn.model_selection.StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) — scikit-learn developers (2023)
  Publisher: scikit-learn
  机器学习中分层抽样的官方文档。展示了在数据划分时保持类别比例的实际实现方法。
- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Morgan Kaufmann; DOI: [10.1016/C2009-0-61819-5](https://doi.org/10.1016/C2009-0-61819-5)
  数据挖掘教材。讨论数据预处理和数据缩减，其中包含基本抽样技术（例如随机抽样、分层抽样），对数据准备和合成数据集创建有用。
