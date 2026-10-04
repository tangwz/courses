---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "vector-magnitude-direction"
sourceId: 1448
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/vector-magnitude-direction"
title: "向量大小与方向"
description: "计算向量的大小（长度）并理解向量的方向。"
order: 3
plots: ["plots/1448-0.json"]
sourceHash: "883fe539d0c04e5ecdc929669c0255c2642f7d3e72e206db7e652fe75624a3f3"
sourceCorrections: []
---

向量 (vector)不仅仅是数值的有序列表。它们具有固有的几何属性：**大小**（长度）和**方向**。理解这些属性对于在机器学习 (machine learning)中解读向量很重要，因为向量常用来表示数据点或特征属性。

### 向量 (vector)大小：长度的衡量

向量的大小，常称作其**范数**，表示它的长度或尺寸。把向量想象成从坐标系原点出发指向特定位置的箭头。大小就是这个箭头的长度。

向量具有固有的几何属性，包括大小（长度）和方向。衡量向量的长度最常见的方式是使用**欧几里得范数**，也称为**L2范数**。对于n维空间中的向量$\mathbf{v}$，表示为$\mathbf{v} = [v_1, v_2, \dots, v_n]$，其L2范数（大小）使用推广到n维的勾股定理计算：

$||\mathbf{v}|| = ||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$

以一个简单的二维向量$\mathbf{v} = [3, 4]$为例。这个向量沿着第一个轴指向3个单位，沿着第二个轴指向4个单位。它的大小是：

$||\mathbf{v}|| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

因此，向量$\mathbf{v} = [3, 4]$的长度是5个单位。



![向量大小 (L2 范数)](plots/1448-0.json)



> 二维空间中的向量$\mathbf{v} = [3, 4]$。其大小计算为$\sqrt{3^2 + 4^2}$，结果是5。这对应于由其分量构成的直角三角形的斜边长度。

在机器学习 (machine learning)中，特征向量的大小有时能反映所表示数据点或特征集的强度或重要程度。例如，在自然语言处理中，文档向量的大小可能与文档的长度有关。但是，大小也可能受特征尺度影响。通常，会应用特征缩放技术（如归一化 (normalization)）来调整向量大小，以便模型不会被固有值较大的特征过度影响。我们将在“向量范数”一节中讨论衡量向量长度的其他方式（如L1范数）。

### 向量 (vector)方向

大小告诉我们向量“多长”，而**方向**则告诉我们它在向量空间中“指向何处”。方向与大小无关。例如，向量$[1, 1]$和$[3, 3]$指向同一方向，但第二个向量的长度是第一个的三倍。

方向对于理解向量之间的关系（如相似性或对齐 (alignment)）很基础。我们如何从数学上表示方向，同时将其与大小分离？我们使用**单位向量**。

**单位向量**就是大小为1的向量。任何非零向量$\mathbf{v}$都可以通过将其除以自身大小来转换为指向相同方向的单位向量$\mathbf{\hat{v}}$（通常用“帽子”符号表示）：

$\mathbf{\hat{v}} = \frac{\mathbf{v}}{||\mathbf{v}||}$

这个过程叫做**归一化 (normalization)**。

让我们对示例向量$\mathbf{v} = [3, 4]$进行归一化。我们发现它的大小是$||\mathbf{v}|| = 5$。对应的单位向量是：

$\mathbf{\hat{v}} = \frac{[3, 4]}{5} = \left[ \frac{3}{5}, \frac{4}{5} \right] = [0.6, 0.8]$

让我们验证一下$\mathbf{\hat{v}}$的大小是否为1：

$||\mathbf{\hat{v}}|| = \sqrt{(0.6)^2 + (0.8)^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1$

这个单位向量$[0.6, 0.8]$捕捉了原始向量$[3, 4]$的纯粹方向。

> 向量$\mathbf{v}$及其对应的单位向量$\mathbf{\hat{v}}$。两者都从同一点出发，指向同一方向，但$\mathbf{\hat{v}}$的大小（长度）为1。

在机器学习 (machine learning)算法中，关注方向通常很重要。例如，当比较以向量表示的文档的语义相似性时，向量之间的角度（与方向相关）通常比它们的大小更能提供信息。在这类情况下，将向量归一化为单位长度是常见的预处理步骤。

计算大小和归一化向量是常规操作。后续章节将展示像NumPy这样的库如何使这些计算变得简单高效。目前，请关注它们的不同之处：大小是长度，方向是方位，单位向量代表纯粹的方向。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  线性代数经典教材，涵盖向量性质、模和方向等内容。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; DOI: [10.7551/mitpress/9780262035613.001.0001](https://doi.org/10.7551/mitpress/9780262035613.001.0001)
  第二章提供了数学背景，包括对向量、范数及其在机器学习中应用的说明。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  解释了向量和范数等基础数学概念在实际机器学习算法中的应用。
- [Linear Algebra (18.06SC)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  视频讲座系列，提供了线性代数的入门课程，包括向量、长度和单位向量。
