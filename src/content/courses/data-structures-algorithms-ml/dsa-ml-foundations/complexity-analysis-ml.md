---
course: "data-structures-algorithms-ml"
chapter: "dsa-ml-foundations"
lesson: "complexity-analysis-ml"
sourceId: 5231
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-1-dsa-ml-foundations/complexity-analysis-ml"
title: "机器学习实践者的复杂性分析"
description: "学习应用大O表示法来分析机器学习算法和数据操作的效率。"
order: 2
plots: ["plots/5231-0.json"]
sourceHash: "77922c9f41c119e049a31915a9807895ca43616571f8bc097ae32e1cd9ad1203"
sourceCorrections: []
---

机器学习 (machine learning)通常涉及处理大量数据或进行计算需求高的训练步骤。仅仅选择*一种*数据结构或算法是不够的；机器学习实践者需要了解它表现*如何*，特别是当问题规模扩大时。为了量化 (quantization)这种表现，复杂性分析，特别是大O表示法，成为机器学习实践者一项必不可少的工具。

### 理解大O表示法

大O表示法是计算机科学中一种数学符号，用来描述函数当自变量趋向特定值或无穷大时的极限行为。在我们这里，它描述了算法的运行时间（时间复杂度）或内存占用（空间复杂度）如何随输入数据的大小而变化。

假设输入大小为$n$（例如，数据点的数量、特征的数量或图中节点的数量）。大O表示法告诉我们所需资源的数量级，它关注描述资源使用情况的函数中最主要的项，并忽略常数因子和低阶项。

为什么忽略常数和低阶项？因为大O表示法关注的是**可扩展性**。当$n$变得非常大时（这在机器学习 (machine learning)中很常见），最高阶项决定了增长速度，而常数或较小项的影响变得可以忽略不计。一个需要$5n^2 + 100n + 5000$次操作的算法，在$n$很大时，其主要由$n^2$项决定。我们将其简化为$O(n^2)$。这意味着如果我们把输入大小增加一倍，运行时间将大约增加四倍。

### 机器学习 (machine learning)中常见的复杂度分类

在处理机器学习算法和数据结构时，您会经常遇到以下大O复杂度：

- **$O(1)$ - 常数时间：** 算法运行时间与输入大小$n$无关。
  - *机器学习示例：* 访问NumPy数组中指定索引的元素，平均情况下在哈希表（字典）中查找值。这是许多操作的理想情况。
- **$O(\log n)$ - 对数时间：** 时间随输入大小对数增加。这非常高效，因为所需时间即使对于大的$n$也增长缓慢。
  - *机器学习示例：* 在平衡二叉搜索树中查找项，二分查找已排序的特征值列表。
- **$O(n)$ - 线性时间：** 时间随输入大小$n$线性增长。
  - *机器学习示例：* 遍历数据集中的所有$n$个样本，计算两个大小为$n$的向量 (vector)的点积，对每个特征进行简单转换，使用预训练 (pre-training)的线性模型或决策树进行预测。
- **$O(n \log n)$ - 线性对数时间：** 时间与$n$乘以$\log n$成比例增长。这在高效排序算法中很常见。
  - *机器学习示例：* 归并排序或堆排序等高效排序算法（库内部常使用），构建某些类型的树索引。
- **$O(n^2)$ - 平方时间：** 时间与输入大小的平方成比例增长。随着$n$增加，这会很快变慢。
  - *机器学习示例：* 使用嵌套循环计算$n$个点之间的成对距离矩阵，某些优化程度较低的矩阵操作。
- **$O(n^k)$ - 多项式时间：** 时间与$n$的某个常数幂$k$成比例增长。$O(n^2)$和$O(n^3)$是常见例子。
- **$O(2^n)$ - 指数时间：** 时间随输入大小的每次增加而翻倍（或更多）。具有这种复杂度的算法通常被认为对于除非常小输入以外的所有情况都无法处理。
  - *机器学习示例：* 某些暴力搜索算法，在没有特定技术的情况下解决某些组合优化问题。

### 可视化增长速度

随着输入大小($n$)的增加，这些增长速度之间的差异变得非常明显。考虑所需的资源（例如时间）可能会增长多快：



![常见复杂度分类的增长速度](plots/5231-0.json)



> 常见大O复杂度的对比。请注意，即使对于中等大小的输入，$O(n^2)$的增长速度也比线性$O(n)$或对数$O(\log n)$时间快得多。指数$O(2^n)$的增长会更加剧烈，为显示效果而省略。

### 关联复杂性与机器学习 (machine learning)任务

掌握大O表示法帮助您分析机器学习流程中的性能瓶颈：

- **数据加载与预处理：** 从文件读取$n$行、$m$个特征的数据通常是$O(n \cdot m)$。简单的特征缩放可能是$O(n \cdot m)$，而更复杂的转换则可能不同。使用像NumPy数组这样的高效结构，由于向量 (vector)化，元素级操作实际上是$O(n \cdot m)$，这在实践中比显式Python循环快得多。
- **模型训练：** 复杂度因模型而异。
  - *k-Nearest Neighbors (KNN)：* $k$近邻算法。朴素预测涉及计算到所有$n$个训练点的距离，每个点有$m$个特征，每次预测需要$O(n \cdot m)$。使用专门的数据结构（如KD树，稍后会提到）可以改善这种平均情况性能，通常更接近$O(\log n)$。
  - *Linear Regression (Normal Equation)：* 线性回归（正规方程）。涉及矩阵运算，如求逆，这大约是$O(m^3)$（如果$n > m$，则为$O(n \cdot m^2)$），主要由特征数量$m$决定。
  - *Decision Trees：* 决策树。构建树通常涉及在每个节点对特征进行排序或查找。对于像CART这样的算法，常见复杂度大约是$O(n \cdot m \log n)$。
  - *Gradient Descent (for many models)：* 梯度下降 (gradient descent)（对于许多模型）。每次迭代通常涉及处理整个数据集（$n$个样本，$m$个特征），导致每个epoch的复杂度大约为$O(n \cdot m)$，可能乘以模型复杂度因子（如神经网络 (neural network)中的层/神经元）。总时间取决于收敛所需的epoch数量。
- **模型推理 (inference)（预测）：** 通常比训练快得多。
  - *线性模型：* 通常每次预测$O(m)$。
  - *决策树：* 与树的深度成比例，对于平衡树理想情况下是$O(\log n)$，但对于倾斜树在最坏情况下可以是$O(n)$。
  - *哈希表查找（例如，用于特征嵌入 (embedding)）：* 平均情况$O(1)$。

### 时间复杂度与空间复杂度

请记住，大O表示法适用于时间要求和空间（内存）要求。有时，您可以进行二者之间的权衡。

- **例子：** 创建一个哈希映射（字典）来存储预计算结果会占用额外空间（$O(k)$，其中$k$是存储项的数量），但可以实现更快的查找（平均$O(1)$时间），与重复计算相比可能节省时间。这是一个常见的时间-空间权衡。在机器学习 (machine learning)中，存储中间结果或构建索引（如在KNN或数据库中）通常会占用更多空间以加快查询或计算速度。

### 对机器学习 (machine learning)实践者的实际意义

为什么要在这些方面投入时间？掌握大O表示法可以帮助您：

1. **选择合适工具：** 根据预期数据量选择合理可扩展的数据结构和算法。对于只有100个数据点的原型，$O(n^2)$算法可能没问题，但对于100万个点则可能非常糟糕。
2. **评估性能：** 大致了解随着数据集增长，训练可能需要多长时间或需要多少内存。
3. **发现瓶颈：** 如果您的流程运行缓慢，复杂性分析有助于指出哪些部分最有可能导致运行时间过长（例如，导致$O(n^2)$行为的嵌套循环）。
4. **构建可扩展系统：** 对架构和实现做出明智决定，以应对未来数据量或复杂度的增长。

### 提醒

大O表示法提供了渐近行为的宏观视图。它并不能说明全部情况：

- **常数因子很重要：** 对于实际输入大小，一个常数因子很大的$O(n)$算法可能比一个常数因子小的$O(n \log n)$算法要慢。
- **实际表现：** 硬件（CPU缓存、内存带宽）、特定库实现和输入数据特点严重影响实际运行时间。

因此，尽管大O分析是分析效率的基本出发点，但它应始终辅以对代表性数据进行实际代码的**性能分析**和**基准测试**。这正是我们将在本章后面的实践部分中涉及的内容。

既然我们有了分析性能的框架，接下来我们来看看Python的基本数据结构，特别是NumPy和Pandas等专门库在常见机器学习 (machine learning)任务的效率方面表现如何。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press; DOI: [10.7551/mitpress/14022.001.0001](https://doi.org/10.7551/mitpress/14022.001.0001)
  一本标准教科书，全面涵盖算法分析，包括大O符号和常见复杂度类别。
- [Introduction to Algorithms (SMA 5503) Lecture Notes and Study Materials](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/pages/lecture-notes/) — Erik Demaine, Charles Leiserson (2011)
  Publisher: MIT OpenCourseWare
  提供关于基础算法分析（包括大O符号）的易于理解的讲义和视频材料。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQElhKwlXRap_pDVALGRJ2345SXWP9f6K2S97PFIb5gGcdTJYKcM-4y-n3uSxSpSuxqZO04N103RneBr0I_vq3I7nLZ8oiP3Hc56L04bdTW3gJUasMijlAjOpNloCo4aIrYtN-TNjB_doCOGxNlIoWMjLUbWrN_obvF9Bgs2C_BEA51JaZ1wjZe_biweMXBH4kXZUj0kzYHtgNgyIvc=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供机器学习算法的实践应用，经常讨论其计算效率和扩展行为，尤其是针对提及的常见模型。
