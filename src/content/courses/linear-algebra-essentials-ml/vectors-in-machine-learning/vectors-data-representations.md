---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "vectors-data-representations"
sourceId: 1445
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/vectors-data-representations"
title: "向量作为数据表示形式"
description: "了解向量如何用于表示机器学习模型中的数据点和特征。"
order: 1
plots: ["plots/1445-0.json"]
sourceHash: "817bcf181d95fdcb7fbca742e612deaaa6c21b6640f98fe6eecfef41bec69866"
sourceCorrections: []
---

在机器学习 (machine learning)中，我们处理数据。大量的数据。这些数据可以是客户购买记录、传感器读数，也可以是图像和文本文档。为了将数学算法应用于这些各类数据，我们首先需要一种一致的方法来表示它们。这就是向量 (vector)的用武之地。

可以将向量视为一个有序的数字列表。此列表中的每个数字代表单个数据点的特定属性或*特征*。通过将这些特征组织成向量，我们创建了一个算法可以处理的数学对象。

### 从数据点到特征向量 (vector)

设想您有关于房屋的数据，并且您想预测它们的价格。对于每个房屋，您可能会记录几个特征：

1. 面积 (平方英尺)
2. 卧室数量
3. 房龄 (年)
4. 到最近学校的距离 (英里)

机器学习 (machine learning)中的数据需要一种一致的表示方式。例如，一个特定的房屋（如房屋 A）可能具有以下特点：1500 平方英尺，3 间卧室，20 年房龄，距离学校 0.5 英里。为了对这类数据应用数学算法，房屋 A 的这些特点可以表示为一个向量：

$\mathbf{v}_A = \begin{bmatrix} 1500 \\ 3 \\ 20 \\ 0.5 \end{bmatrix}$

这个列向量$\mathbf{v}_A$常被称为**特征向量**。每个元素都按预定顺序对应一个特定的特征。另一个房屋，房屋 B（2100 平方英尺，4 间卧室，5 年房龄，距离学校 1.2 英里），将由另一个向量表示：

$\mathbf{v}_B = \begin{bmatrix} 2100 \\ 4 \\ 5 \\ 1.2 \end{bmatrix}$

这种表示方法极其灵活：

- **图像：** 灰度图像可以表示为向量，其中每个元素是像素的亮度值。一个 100x100 像素的图像变成一个具有 10,000 个元素的向量（通常通过将二维像素网格“扁平化”为长的一维列表）。彩色图像每个像素可能有三个值（红、绿、蓝），从而得到更大的向量。
- **文本：** 文档可以使用词袋模型（其中每个元素统计词汇表 (vocabulary)中特定单词的出现次数）或更高级的方法（如词嵌入 (embedding)，其中单词或文档被映射到捕捉语义的密集向量 (dense vector)）转换为向量。
- **传感器数据：** 来自传感器的时间序列数据可以表示为向量，其中每个元素是特定时间点的读数，或者是一个时间点上来自不同传感器的读数集合。

### 几何视角：特征空间中的点

将数据表示为向量 (vector)使我们能够从几何角度思考。如果一个数据点有两个特征（例如身高和体重），我们可以将其表示为二维向量。我们可以将此向量可视化为从原点 (0, 0) 开始，终止于由特征值（身高，体重）在二维平面中定义的点处的箭头。



![二维特征空间中作为向量的数据点](plots/1445-0.json)



> 两个数据点 A=(2, 5) 和 B=(4.5, 3)，表示为从二维特征空间原点出发的向量。

如果我们有三个特征，每个数据点就成为三维空间中的向量。对于具有 $n$ 个特征的数据（如我们的房屋示例中 $n=4$），每个数据点对应于 $n$ 维空间中的一个向量，这个空间常被称为**特征空间**。虽然我们无法轻易可视化超过三维的空间，但数学原理保持不变。

这种向量表示是大多数机器学习 (machine learning)算法所基于的形式。它提供了一种结构化形式，使我们能够：

1. **应用数学运算：** 我们可以向量相加、缩放它们，并测量它们之间的角度和距离，这些在诸如 k-近邻、支持向量机和线性回归等算法中都是重要的运算。
2. **处理高维数据：** 线性代数提供了工具，即使向量具有数千或数百万维度（特征）时，也能高效处理。
3. **执行数据变换：** 正如我们稍后将看到的，矩阵可用于变换这些向量，支持诸如降维（PCA）或特征工程等技术。

在接下来的章节中，我们将了解您可以在这些向量上执行的基本运算，以及这些运算与机器学习任务的关系。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第二章清晰地解释了线性代数基础知识，阐明了向量在表示各种数据类型以用于深度学习模型中的作用。
- [CS229 Lecture Notes: Linear Algebra Review and Newton's Method](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHXidvPLH5MWjhhTxB2Mh-Hpj1xX7mcDECfV0LOiA-_7YXWtpIthp8HjrjGp-ZJ6jLuvO2yqscAWFvk03UCIpd0rsSzZGgSKBkiaST4KHZzhMuzFha6s-yPdcXuHX503CMe0sk=) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  这些来自知名机器学习课程的讲义概述了线性代数，包括使用向量来构造和处理数据。
