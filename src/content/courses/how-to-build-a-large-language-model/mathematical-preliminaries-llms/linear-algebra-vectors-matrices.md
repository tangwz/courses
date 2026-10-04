---
course: "how-to-build-a-large-language-model"
chapter: "mathematical-preliminaries-llms"
lesson: "linear-algebra-vectors-matrices"
sourceId: 5952
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-2-mathematical-preliminaries-llms/linear-algebra-vectors-matrices"
title: "线性代数回顾：向量与矩阵"
description: "回顾向量空间、矩阵运算以及与神经网络相关的分解。"
order: 1
plots: []
sourceHash: "0d5da1d936223340adaeda5578542371ffcc913ad553faf1d7fab71f5ae072e0"
sourceCorrections: []
---

线性代数提供了处理大型语言模型中常见高维数据的数学语言。向量 (vector)和矩阵是我们用于表示神经网络 (neural network)中的输入数据、模型参数 (parameter)和中间激活的主要对象。把握它们的属性和运算方式，对于明白信息在这些模型中如何流动和转变来说，是十分重要的。

### 向量 (vector)：表示数据点

本质上，向量是一个有序的数字列表，通常表示多维空间中的一个点或方向。在大型语言模型的背景下，向量通常表示：

- **词嵌入 (embedding)：** 捕获语义的密集表示，其中每个维度对应一个潜在特征。例如，词汇表 (vocabulary)中的一个词可以映射到一个300维向量。
- **隐藏状态：** 网络层内的中间表示（如RNN隐藏状态或Transformer层输出），捕获截至某一特定点的输入序列的上下文 (context)信息。
- **梯度：** 表示损失函数 (loss function)相对于模型参数 (parameter)或激活值的最陡峭上升方向和大小的向量。

$n$维欧几里得空间中的向量$v$表示为：


$$
v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$


向量加法和标量乘法等基本运算使我们能够组合或缩放这些表示。例如，将输入嵌入添加到位置编码 (positional encoding)向量会组合语义和位置信息。

在PyTorch中，向量表示为一维张量。

```python
import torch

# 示例：一个5维向量
vector_a = torch.tensor([1.0, 2.5, -0.8, 4.0, 0.0])

# 示例：标量乘法
scaled_vector = 2.0 * vector_a

# 示例：向量加法
vector_b = torch.tensor([-0.5, 1.0, 1.2, -2.0, 1.5])
summed_vector = vector_a + vector_b

print(f"原始向量 A: {vector_a}")
print(f"缩放后的向量: {scaled_vector}")
print(f"相加后的向量: {summed_vector}")
print(f"向量维度 (阶): {vector_a.ndim}")
print(f"向量形状: {vector_a.shape}")
```

### 矩阵：表示变换和参数 (parameter)

矩阵是数字的矩形排列，将向量 (vector)扩展到二维。它们在深度学习 (deep learning)中最重要的作用是表示向量空间之间的线性变换。

- **权重 (weight)矩阵：** 线性层（或全连接层）的参数存储在矩阵中。应用一个层涉及将输入向量（或输入向量的矩阵）乘以该层的权重矩阵。
- **注意力分数：** 矩阵可以表示序列中不同位置之间计算出的注意力权重。
- **数据批次：** 输入向量的集合（例如，序列中多个词的嵌入 (embedding)或批次中多个序列）通常排列成矩阵。

一个有$m$行$n$列的矩阵$A$（$A \in \mathbb{R}^{m \times n}$）是：


$$
A = \begin{bmatrix}
A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\
A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
$$


### 神经网络 (neural network)的核心运算

某些线性代数运算在神经网络计算中无处不在：

#### 矩阵-向量 (vector)乘法

这个运算将矩阵$W$定义的线性变换应用于向量$x$。如果$W \in \mathbb{R}^{m \times n}$且$x \in \mathbb{R}^n$，则结果$y = Wx$是一个在$\\mathbb{R}^m$中的向量。这是神经网络中一个密集层（无偏置 (bias)）内的基本计算，它将一个$n$维输入表示转换为一个$m$维输出表示。


$$
y_i = \sum_{j=1}^{n} W_{i,j} x_j
$$


```python
import torch

# 定义一个权重矩阵（例如，用于将4个特征映射到3个特征的层）
W = torch.randn(3, 4) # 形状：(输出维度, 输入维度)

# 定义一个输入向量（4个特征）
x = torch.tensor([1.0, 0.5, -1.0, 2.0]) # 形状：(输入维度,)

# 执行矩阵-向量乘法
# 注意：torch.matmul 会适当地处理维度
y = torch.matmul(W, x) # 或者 W @ x

print(f"权重矩阵 W (形状 {W.shape}):\n{W}")
print(f"\n输入向量 x (形状 {x.shape}): {x}")
print(f"\n输出向量 y (形状 {y.shape}): {y}")
```

> 矩阵 W 将向量 x 从4维空间变换到3维空间。

#### 矩阵-矩阵乘法

两个矩阵 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$ 相乘得到矩阵 $C = AB \in \mathbb{R}^{m \times p}$。这在批量处理数据时被广泛使用，其中输入 $X$ 可能是一个矩阵，每行是一个输入向量，或者在组合多个线性变换时使用。例如，Transformer的前馈网络中的计算通常涉及多次矩阵乘法。


$$
C_{i,k} = \sum_{j=1}^{n} A_{i,j} B_{j,k}
$$


```python
import torch

# 输入批次（例如，2个序列/样本，每个有4个特征）
X = torch.randn(2, 4) # 形状：(批次大小, 输入维度)

# 来自前面示例的权重矩阵
W = torch.randn(3, 4) # 形状：(输出维度, 输入维度)

# 对批次应用变换
# 我们需要对 W 进行转置，以匹配标准矩阵乘法的维度约定
# Y = X @ W.T 的结果是 (2, 4) @ (4, 3) -> (2, 3)
Y = torch.matmul(X, W.T) # 形状：(批次大小, 输出维度)

print(f"输入批次 X (形状 {X.shape}):\n{X}")
print(f"\n权重矩阵 W 转置 (形状 {W.T.shape}):\n{W.T}")
print(f"\n输出批次 Y (形状 {Y.shape}):\n{Y}")
```

#### 逐元素运算（哈达玛积）

这涉及到将两个相同形状的矩阵（或向量）的对应元素相乘。表示为 $A \odot B$，结果 $C$ 的 $C_{i,j} = A_{i,j} \times B_{i,j}$。这与矩阵乘法不同，并出现在各种神经网络组成部分中，例如逐元素应用激活函数 (activation function)或在LSTM或GRU中实现门控机制。

```python
import torch

A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[0.5, 1.], [-1., 2.]])

# 逐元素乘法
C = A * B # 或者 torch.multiply(A, B)

print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"逐元素乘积 C:\n{C}")
```

### 点积

一个基本运算是两个向量 (vector) $v, w \in \mathbb{R}^n$ 之间的点积（或内积）。它计算为 $v \cdot w = \sum_{i=1}^n v_i w_i$。在几何上，它与一个向量在另一个向量上的投影相关（$v \cdot w = \|v\| \|w\| \cos \theta$，其中 $\theta$ 是它们之间的夹角）。

如果第一个向量被视为行向量，第二个被视为列向量，则点积在计算上等同于矩阵乘法：$v^T w$。


$$
v^T w = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix} = \sum_{i=1}^n v_i w_i
$$


在大型语言模型中，点积是**注意力机制 (attention mechanism)**的核心。缩放点积注意力使用点积来计算查询（$Q$）和键（$K$）向量之间的相关性，以确定对输入序列不同部分的关注程度。

```python
import torch

v = torch.tensor([1.0, 2.0, -1.0])
w = torch.tensor([3.0, -1.0, 0.5])

# 计算点积
dot_product_val = torch.dot(v, w)

print(f"向量 v: {v}")
print(f"向量 w: {w}")
print(
    f"点积: {dot_product_val}"
) # 预期: (1*3) + (2*-1) + (-1*0.5) = 3 - 2 - 0.5 = 0.5
```

### 范数：测量向量 (vector)大小

范数是一个函数，它为向量空间中的每个向量赋予一个严格为正的长度或大小（零向量除外，其长度为零）。机器学习 (machine learning)中最常见的范数是：

- **L2范数（欧几里得范数）：** $\|v\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$。这对应于到原点的标准欧几里得距离。它常用于正则化 (regularization)（L2正则化或权重 (weight)衰减），以惩罚大的参数 (parameter)值并防止过拟合 (overfitting)。
- **L1范数（曼哈顿范数）：** $\|v\|_1 = \sum_{i=1}^n |v_i|$。它测量分量的绝对值之和。L1正则化鼓励稀疏性（将一些参数精确地驱动到零）。

范数也用于归一化 (normalization)技术，如层归一化，这通常涉及根据激活值的L2范数对其进行缩放。

```python
import torch

v = torch.tensor([3.0, -4.0, 0.0])

l2_norm = torch.linalg.norm(v, ord=2) # 或者直接 torch.linalg.norm(v)
l1_norm = torch.linalg.norm(v, ord=1)

print(f"向量 v: {v}")
# 预期: sqrt(3^2 + (-4)^2 + 0^2) = sqrt(9 + 16) = sqrt(25) = 5.0
print(f"L2 范数: {l2_norm}")
print(f"L1 范数: {l1_norm}") # 预期: |3| + |-4| + |0| = 3 + 4 + 0 = 7.0
```

### 维度与张量

尽管我们主要讨论了向量 (vector)（1维）和矩阵（2维），但深度学习 (deep learning)非常依赖于**张量**，它是对更高维度的推广。例如：

- 大型语言模型的输入可能是一个3D张量：`（批次大小，序列长度，嵌入维度）`。
- 卷积层（在某些架构中使用）的权重 (weight)可能是一个4D张量。

跟踪张量形状非常重要，以确保运算兼容。维度不匹配是深度学习代码中常见的错误来源。PyTorch及其他框架提供工具来检查和操作张量形状（`.shape`、`.reshape()`、`.permute()`等）。

本次回顾涵盖了最直接相关的线性代数内容。随着课程进展，特别是在讨论Transformer架构和注意力机制 (attention mechanism)时，矩阵乘法、点积以及管理张量维度的作用将变得越来越明显。牢固掌握这些运算对于有效理解和实现大型语言模型来说是不可或缺的。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press
  本书是一本广受好评的基础教材，涵盖了所有基本的线性代数概念，包括向量、矩阵、运算和范数。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第2章提供了简洁而全面的线性代数回顾，专门为理解深度学习模型及其基础数学而设计。
- [Linear Algebra (18.06SC)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  这门综合性在线课程提供视频讲座和配套材料，与讨论的基础线性代数主题保持一致，是视觉学习者的绝佳资源。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  第3章和第4章涵盖了向量、矩阵及其运算，为机器学习应用提供了强大的数学基础。
