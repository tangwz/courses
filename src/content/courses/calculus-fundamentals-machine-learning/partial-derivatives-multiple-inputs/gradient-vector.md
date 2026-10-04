---
course: "calculus-fundamentals-machine-learning"
chapter: "partial-derivatives-multiple-inputs"
lesson: "gradient-vector"
sourceId: 2294
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs/gradient-vector"
title: "梯度向量"
description: "将梯度定义为包含函数所有偏导数的向量。"
order: 5
plots: []
sourceHash: "7c31bd6946f241c646cf8a7520a433d4cf733c71e85f3e8c5f1bd5f2cdbe313b"
sourceCorrections: []
---

偏导数是理解具有多个输入（例如 $f(x, y)$）的函数如何变化的基础。例如，$\frac{\partial f}{\partial x}$ 描述了当 $x$ 变化时 $f$ 如何变化（保持 $y$ 不变）。同样地，$\frac{\partial f}{\partial y}$ 表示当 $y$ 变化时 $f$ 如何变化（保持 $x$ 不变）。

但是，如果我们想用一个单独的对象来表示对所有输入变量的同时变化率，该怎么办？这就是**梯度向量 (vector)**的作用。

函数 $f$ 的梯度就是一个向量，其每个分量都是 $f$ 的一个偏导数。如果我们的函数有两个输入 $x$ 和 $y$，它的梯度就是一个二维向量。如果它有 $n$ 个输入 $x_1, x_2, \dots, x_n$，它的梯度就是一个 $n$ 维向量。

我们通常用 nabla 符号 $\nabla$ 来表示梯度。对于函数 $f(x, y)$，梯度写作 $\nabla f$ 或 $\nabla f(x, y)$，定义为：


$$
\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}
$$


有时你也会看到它用尖括号水平书写：$\nabla f(x, y) = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \rangle$。

对于一个有 $n$ 个变量的函数 $f(x_1, x_2, \dots, x_n)$，梯度是：


$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$


简单来说，梯度将所有一阶偏导数打包成一个方便的向量。

### 例子：计算梯度

让我们以函数 $f(x, y) = x^2 + 5xy$ 为例。

1. **求偏导数：**

   - 将 $y$ 视为常数来求 $\frac{\partial f}{\partial x}$：
     
     $$
     \frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2) + \frac{\partial}{\partial x}(5xy) = 2x + 5y
     $$
     
   - 将 $x$ 视为常数来求 $\frac{\partial f}{\partial y}$：
     
     $$
     \frac{\partial f}{\partial y} = \frac{\partial}{\partial y}(x^2) + \frac{\partial}{\partial y}(5xy) = 0 + 5x = 5x
     $$
     
2. **组合梯度向量 (vector)：**

   
   $$
   \nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x + 5y \\ 5x \end{bmatrix}
   $$
   

这个梯度 $\nabla f(x, y)$ 给出了一个向量，它取决于我们所考虑的点 $(x, y)$。例如，在点 $(1, 2)$ 处：


$$
\nabla f(1, 2) = \begin{bmatrix} 2(1) + 5(2) \\ 5(1) \end{bmatrix} = \begin{bmatrix} 2 + 10 \\ 5 \end{bmatrix} = \begin{bmatrix} 12 \\ 5 \end{bmatrix}
$$


在点 $(-1, 0)$ 处：


$$
\nabla f(-1, 0) = \begin{bmatrix} 2(-1) + 5(0) \\ 5(-1) \end{bmatrix} = \begin{bmatrix} -2 \\ -5 \end{bmatrix}
$$


因此，梯度本身是一个函数，它以点 $(x, y)$ 为输入并输出一个向量。这个向量包含有关函数 $f$ 在该特定点附近行为的重要信息，我们将在下一节中进行更详细的讨论。在机器学习 (machine learning)中，成本函数的梯度告诉我们如何调整模型参数 (parameter)（如输入 $x_1, x_2, \dots, x_n$）以改变成本。

## 参考资料

- [Calculus: Early Transcendentals](https://www.cengage.com/c/calculus-early-transcendentals-8e-stewart/9781285741550/) — James Stewart (2015)
  Publisher: Cengage Learning
  一本广泛使用的本科教材，涵盖多变量微积分，包括偏导数和梯度向量。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679989](https://doi.org/10.1017/9781108679989)
  一本专门为机器学习提供数学基础的教材，解释了向量微积分和梯度。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教材，展示了梯度在训练神经网络的优化算法中的实际应用。
- [Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) — Denis Auroux (2010)
  Publisher: MIT OpenCourseWare
  麻省理工学院开放课程资源，提供多变量微积分课程的讲座视频、笔记和习题集，涵盖梯度向量。
