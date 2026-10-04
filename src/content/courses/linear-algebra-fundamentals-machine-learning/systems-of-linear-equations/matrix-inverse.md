---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "systems-of-linear-equations"
lesson: "matrix-inverse"
sourceId: 7823
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-4-systems-of-linear-equations/matrix-inverse"
title: "矩阵的逆"
description: "介绍矩阵的逆以及它如何被用来解方程组。"
order: 3
plots: []
sourceHash: "5b4307df1c2d062d593337a68240676699bacf0a2b7a80dff4c242654a7c6f1e"
sourceCorrections: []
---

解像 $5x = 20$ 这样的方程在基础代数中是很直接的。为了使未知变量 $x$ 独立，你将方程两边同乘以 5 的乘法逆元，即 $1/5$ 或 $5^{-1}$。这会得到 $(5^{-1}) \cdot 5x = (5^{-1}) \cdot 20$，简化后为 $1x = 4$，或者就是 $x = 4$。

我们可以将非常类似的思路应用于矩阵方程 $Ax = b$。我们希望使向量 (vector) $x$ 独立。然而，你不能对矩阵进行“除法”。相反，我们需要找出矩阵形式的乘法逆元：**矩阵的逆**。

方阵 $A$ 的逆矩阵表示为 $A^{-1}$。当你将一个矩阵与其逆矩阵相乘时，你会得到**单位矩阵** $I$。单位矩阵是相当于数字 1 的矩阵形式，我们将在下一节中更详细地查看它。它是一个主对角线上是 1，其他所有位置是 0 的方阵。

逆矩阵的定义性质是：


$$
A A^{-1} = A^{-1} A = I
$$


### 使用逆矩阵求解

有了矩阵逆 $A^{-1}$，我们有了一种直接方法来解我们的方程组。让我们通过将其应用到我们最初的问题 $Ax = b$ 来看看它是如何工作的。

1. **从方程开始：**

   
   $$
   Ax = b
   $$
   
2. **在方程两边预乘以逆矩阵 $A^{-1}$：** 要记住在方程两边都从左侧相乘，因为矩阵乘法不满足交换律（意思是 $AB$ 不总是等于 $BA$）。

   
   $$
   A^{-1} (Ax) = A^{-1} b
   $$
   
3. **矩阵分组：** 由于矩阵乘法满足结合律，我们可以重新组合左侧的项。

   
   $$
   (A^{-1} A) x = A^{-1} b
   $$
   
4. **使用逆矩阵性质进行简化：** 我们知道 $A^{-1} A$ 简化为单位矩阵 $I$。

   
   $$
   I x = A^{-1} b
   $$
   
5. **使 $x$ 独立：** 将任何向量 (vector)或矩阵乘以单位矩阵 $I$ 都会使其保持不变（就像将一个数乘以 1 一样）。因此，$Ix$ 简单来说就是 $x$。

   
   $$
   x = A^{-1} b
   $$
   

这个最终方程为我们提供了一个直接公式来找到未知向量 $x$。如果我们可以找到矩阵 $A$ 的逆，我们只需将 $A^{-1}$ 乘以向量 $b$ 即可解该系统。

> 该图说明了逆矩阵 $A^{-1}$ 如何被用来将问题 $Ax = b$ 转换为直接解出 $x$。

### 逆矩阵何时存在？

在我们过于乐观之前，有一个主要条件：**并非所有矩阵都有逆矩阵**。一个矩阵 $A$ 要拥有逆矩阵，它必须满足两个条件：

1. **矩阵必须是方阵。** 逆矩阵只对行数和列数相等的矩阵有定义（例如 2x2, 3x3）。这与以下观点一致：对于线性方程组的唯一解，通常需要与未知变量数量相等的方程数。
2. **矩阵必须是非奇异的。** 这是一个你将在下一节中查看的新术语。目前，可以把奇异矩阵看作是会使数据“塌缩”的矩阵。例如，它可能将一个二维平面转换为一条单线。如果一个矩阵这样做，信息会丢失，并且无法逆转这种转换。一个非奇异的，或者说**可逆**的矩阵不会丢失信息。

让我们看一个简单的例子。考虑矩阵 $A$：


$$
A = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}
$$


这个矩阵的逆（可以通过我们稍后会看到的方法找到）是：


$$
A^{-1} = \begin{bmatrix} 1.5 & -0.5 \\ -2 & 1 \end{bmatrix}
$$


为了验证这确实是逆矩阵，让我们将它们相乘，看看是否得到单位矩阵：


$$
A A^{-1} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix} \begin{bmatrix} 1.5 & -0.5 \\ -2 & 1 \end{bmatrix} = \begin{bmatrix} (2 \cdot 1.5 + 1 \cdot -2) & (2 \cdot -0.5 + 1 \cdot 1) \\ (4 \cdot 1.5 + 3 \cdot -2) & (4 \cdot -0.5 + 3 \cdot 1) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I
$$


这行得通。如果我们有一个方程组，其中 $A$ 是我们的系数矩阵，我们就可以使用这个 $A^{-1}$ 来找到解。

矩阵逆是解线性系统的基本工具。但这引出了两个值得思考的问题：我们如何知道一个矩阵是否可逆，而无需只是尝试去找到它的逆矩阵？以及我们如何高效地计算逆矩阵或解系统？我们将在关于行列式的下一节中开始回答这些问题。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本基础教材，全面介绍了线性代数的基础知识，包括矩阵运算、逆矩阵和方程组求解。
- [18.06SC Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  提供了线性代数的视频讲座和全面的课程材料，直接涵盖了矩阵逆及其在解$Ax=b$中的作用。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 37-67
  第二章“线性代数”从应用角度介绍了机器学习中必需的概念，包括矩阵逆和方程组求解。
