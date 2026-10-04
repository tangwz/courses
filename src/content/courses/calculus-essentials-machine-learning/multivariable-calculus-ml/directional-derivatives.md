---
course: "calculus-essentials-machine-learning"
chapter: "multivariable-calculus-ml"
lesson: "directional-derivatives"
sourceId: 1376
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-3-multivariable-calculus-ml/directional-derivatives"
title: "方向导数"
description: "了解如何计算函数在特定方向上的变化率。"
order: 4
plots: ["plots/1376-0.json"]
sourceHash: "56382463dfeda3817e58e11f4d23c57f796813907470d3b32636eb20ada87dcd"
sourceCorrections: []
---

偏导数衡量函数沿着坐标轴（如同在地图上纯粹地向东或向北移动）的变化率。梯度向量 (vector) $\nabla f$ 指向最陡峭的上升方向（即最快爬坡的方向）。

但如果你想知道地形在*另一个*特定方向上有多陡峭呢？或许你想向东北移动，或沿着一个由向量表示的任意路径移动。这正是**方向导数**使我们能够计算的。它量化 (quantization)了多变量函数 $f$ 在特定点 $\mathbf{a}$ 沿由**单位向量** $\mathbf{u}$ 指定的方向移动时的变化率。

### 定义方向导数

函数 $f$ 在点 $\mathbf{a}$ 沿单位向量 (vector) $\mathbf{u}$ 方向的方向导数记作 $D_{\mathbf{u}}f(\mathbf{a})$。它通过该点的梯度与方向向量的点积来计算：


$$
D_{\mathbf{u}}f(\mathbf{a}) = \nabla f(\mathbf{a}) \cdot \mathbf{u}
$$


我们来详细说明一下：

1. **梯度 $\nabla f(\mathbf{a})$:** 如我们所知，这个向量包含了 $f$ 在 $\mathbf{a}$ 处求值的所有偏导数。对于函数 $f(x, y)$，$\nabla f(\mathbf{a}) = \langle \frac{\partial f}{\partial x}(\mathbf{a}), \frac{\partial f}{\partial y}(\mathbf{a}) \rangle$。它概括了函数在点 $\mathbf{a}$ 处所有坐标轴方向上的变化率信息。
2. **单位向量 $\mathbf{u}$:** 这个向量指定了所关注的方向。重要的是 $\mathbf{u}$ 是一个*单位*向量，这意味着它的长度或模为 1 ($||\mathbf{u}|| = 1$)。为什么？因为我们只想捕捉由方向本身引起的变化，而不是由方向向量的长度缩放后的变化。如果你有一个由向量 $\mathbf{v}$ 指定的方向，而它*不是*一个单位向量，你必须首先通过除以其模来标准化它：$\mathbf{u} = \frac{\mathbf{v}}{||\mathbf{v}||}$。
3. **点积 ($\cdot$):** 点积有效地衡量了一个向量与另一个向量“方向一致”的程度。

### 几何解释：投影

回顾一下，两个向量 (vector) $\mathbf{v}_1$ 和 $\mathbf{v}_2$ 的点积也可以表示为 $\mathbf{v}_1 \cdot \mathbf{v}_2 = ||\mathbf{v}_1|| ||\mathbf{v}_2|| \cos \theta$，其中 $\theta$ 是它们之间的夹角。

将此应用于我们的方向导数公式，并且知道 $||\mathbf{u}|| = 1$，我们得到：


$$
D_{\mathbf{u}}f(\mathbf{a}) = \nabla f(\mathbf{a}) \cdot \mathbf{u} = ||\nabla f(\mathbf{a})|| \, ||\mathbf{u}|| \cos \theta = ||\nabla f(\mathbf{a})|| \cos \theta
$$


在这里，$\theta$ 是梯度向量 $\nabla f(\mathbf{a})$ 和方向向量 $\mathbf{u}$ 之间的夹角。这个公式提供了一个有益的观点：方向导数是梯度向量在方向向量 $\mathbf{u}$ 上的*标量投影*。这就像在问：“梯度的模有多少指向 $\mathbf{u}$ 方向？”



![∇f 在 u 上的投影作为方向导数](plots/1376-0.json)



> 方向导数 $D_{\mathbf{u}}f$ 是梯度向量 $\nabla f$ 在单位方向向量 $\mathbf{u}$ 上的标量投影。它衡量了梯度在方向 $\mathbf{u}$ 上起作用的分量。

这种投影视角有助于理解梯度与方向导数之间的关系：

- **最大变化：** 当 $\mathbf{u}$ 指向与 $\nabla f(\mathbf{a})$ 相同的方向时，夹角 $\theta$ 为 0，$\cos \theta = 1$，并且 $D_{\mathbf{u}}f(\mathbf{a}) = ||\nabla f(\mathbf{a})||$。方向导数达到最大值，并等于梯度的模。这证实了梯度指向最陡峭的上升方向。
- **最小变化（最陡峭下降）：** 当 $\mathbf{u}$ 指向与 $\nabla f(\mathbf{a})$ 正相反的方向时，夹角 $\theta$ 为 $\pi$（180 度），$\cos \theta = -1$，并且 $D_{\mathbf{u}}f(\mathbf{a}) = -||\nabla f(\mathbf{a})||$。这是最陡峭的下降方向。
- **零变化：** 当 $\mathbf{u}$ 与 $\nabla f(\mathbf{a})$ 正交（垂直）时，夹角 $\theta$ 为 $\pi/2$（90 度），$\cos \theta = 0$，并且 $D_{\mathbf{u}}f(\mathbf{a}) = 0$。沿此方向移动会导致函数值的瞬时变化为零。从几何角度看，你正沿着函数曲面上的等高线或等值线移动。

### 计算示例

我们考虑函数 $f(x, y) = x^2 + y^2$，它描述了一个以原点为中心的抛物面碗形。我们想找到在点 $\mathbf{a} = (1, 1)$ 处沿向量 (vector) $\mathbf{v} = \langle 1, 2 \rangle$ 方向的变化率。

1. **计算梯度：**
   $\nabla f(x, y) = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \rangle = \langle 2x, 2y \rangle$。
2. **在 $\mathbf{a} = (1, 1)$ 处评估梯度：**
   $\nabla f(1, 1) = \langle 2(1), 2(1) \rangle = \langle 2, 2 \rangle$。这个向量直接远离原点，是这种碗形的最陡峭上升方向。
3. **找到方向 $\mathbf{v} = \langle 1, 2 \rangle$ 的单位向量 $\mathbf{u}$：**
   - 模：$||\mathbf{v}|| = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5}$。
   - 标准化：$\mathbf{u} = \frac{\mathbf{v}}{||\mathbf{v}||} = \langle \frac{1}{\sqrt{5}}, \frac{2}{\sqrt{5}} \rangle$。
4. **使用点积计算方向导数：**
   $D_{\mathbf{u}}f(1, 1) = \nabla f(1, 1) \cdot \mathbf{u} = \langle 2, 2 \rangle \cdot \langle \frac{1}{\sqrt{5}}, \frac{2}{\sqrt{5}} \rangle$
   $D_{\mathbf{u}}f(1, 1) = (2)(\frac{1}{\sqrt{5}}) + (2)(\frac{2}{\sqrt{5}}) = \frac{2}{\sqrt{5}} + \frac{4}{\sqrt{5}} = \frac{6}{\sqrt{5}}$。

因此，在点 $(1, 1)$ 处，如果我们沿方向 $\langle 1, 2 \rangle$ 移动，函数 $f(x, y)$ 以大约 $\frac{6}{\sqrt{5}} \approx 2.68$ 单位每单位移动距离的速度增加。请注意，这小于梯度的模，$||\nabla f(1, 1)|| = ||\langle 2, 2 \rangle|| = \sqrt{2^2 + 2^2} = \sqrt{8} \approx 2.83$，后者是最陡峭方向 $\langle 2, 2 \rangle$ 上的变化率。

### 在机器学习 (machine learning)中的意义

在机器学习优化中，特别是梯度下降 (gradient descent)，我们主要关注最陡峭的*下降*方向，即 $-\nabla f$。然而，理解方向导数能提供关于参数 (parameter)空间的有益背景。它帮助我们思考为何沿着负梯度方向移动是（局部）最小化损失函数 (loss function)最有效的一步。虽然在标准梯度下降过程中通常不会明确计算，但它支持我们理解函数在模型训练期间所处的高维参数空间中如何表现。这加强了梯度在优化指导中的核心作用。

## 参考资料

- [Calculus: Early Transcendentals](https://www.cengage.com/c/calculus-early-transcendentals-8e-stewart/9781285741550/) — James Stewart (2015)
  Publisher: Cengage Learning
  一本广泛使用的多元微积分教材，提供了方向导数、梯度及其几何解释的完整处理。
- [Multivariable Calculus (18.02SC)](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) — Denis Auroux (2010)
  Publisher: MIT OpenCourseWare
  麻省理工学院的在线多元微积分课程，包括方向导数和梯度的讲座及习题。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  本书为机器学习提供了数学基础，其中一个部分专门讲解了理解基于梯度的优化所需的微积分知识。
