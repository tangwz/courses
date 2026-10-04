---
course: "calculus-fundamentals-machine-learning"
chapter: "optimization-with-derivatives"
lesson: "finding-max-min"
sourceId: 2269
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-3-optimization-with-derivatives/finding-max-min"
title: "寻找最大值和最小值点"
description: "了解导数为零如何帮助确定可能的最大值/最小值点。"
order: 1
plots: ["plots/2269-0.json"]
sourceHash: "5b601622c3b0f481b9598130486a2929896ccea0fa76d1764a34f590914a28fb"
sourceCorrections: []
---

函数 $f(x)$ 的导数 $f'(x)$ 描述了在任意给定点 $x$ 处的瞬时变化率，或切线的斜率。斜率的这个概念是识别函数图上特定点（如峰值和谷值）的基础。

想象一下沿着由函数 $f(x)$ 表示的丘陵小路行走。当你到达山顶（最大值点）或山谷底部（最小值点）时，会有一段短暂的时刻，脚下的路径是平坦的，也就是水平的。水平路径的斜率为零。

这个直觉直接对应于微积分。如果一个光滑函数 $f(x)$ 在点 $x = c$ 处达到局部最大值或局部最小值，并且在该点导数存在，那么在该点处的切线斜率必定为零。用数学表示就是：

$f'(c) = 0$

导数为零（或无定义，尽管我们目前主要关注零）的点被称为 **临界点**。这些临界点是局部最大值或最小值可能出现的 *候选点*。

### 可视化平坦的斜率

考虑函数 $f(x) = x^2 - 4x + 5$。这是一个抛物线。我们来用上一章学到的幂规则以及和/常数规则找到它的导数：

$f'(x) = \frac{d}{dx}(x^2 - 4x + 5) = 2x - 4 + 0 = 2x - 4$

现在，我们通过将导数设为零来找到斜率为零的位置：

$f'(x) = 0$
$2x - 4 = 0$
$2x = 4$
$x = 2$

这告诉我们，当 $x = 2$ 时，函数的斜率为零。我们来看图表：



![函数 f(x) = x^2 - 4x + 5 及其在最小值处的切线](plots/2269-0.json)



> 该图显示了抛物线 $f(x) = x^2 - 4x + 5$。红点标记 (token)了在 $(2, 1)$ 处的最小值点。红色虚线是该点的切线，它是水平的，表明斜率为 $f'(2) = 0$。

### 过程：寻找最大值和最小值的候选点

那么，为函数 $f(x)$ 寻找最大值或最小值（即临界点）的潜在位置的一般过程包含以下步骤：

1. **计算导数**：找到导函数 $f'(x)$。
2. **将导数设为零**：将导数设为零：$f'(x) = 0$。
3. **解出 x**：解出所得到的关于 $x$ 的方程。你找到的 $x$ 值就是临界点。

有必要记住，找到 $f'(x) = 0$ 只会给你 *候选点*。虽然这些点通常对应于局部最大值或最小值，但它们偶尔也可能是其他类型的点（例如曲线暂时变平然后继续朝同一大致方向的拐点）。对于我们在机器学习 (machine learning)优化中的目标，我们通常是在寻找最小值，而 $f'(x) = 0$ 这个条件是找到它的主要起点。

### 这对机器学习 (machine learning)有何作用？

寻找导数为零的这个步骤不只是一个数学练习。它构成了许多机器学习算法如何“训练”的根本。

设想你有一个机器学习模型，你定义了一个 **成本函数**（也称为损失函数 (loss function)）。这个函数衡量模型预测与实际数据相比的“好坏”。值高表示模型表现不佳；值低表示模型表现良好。

训练模型的目的是调整其内部参数 (parameter)，使成本函数尽可能小，理想情况下达到其最小值。如果我们可以将成本表示为模型参数的函数，我们就可以使用微积分来找到使此成本最小化的参数值。找到成本函数导数为零的位置，有助于我们定位这个最小误差状态，从而根据我们选择的成本衡量标准，获得最佳可能的模型性能。

在接下来的部分，我们将考察成本函数和特定算法——梯度下降 (gradient descent)，该算法迭代地使用导数来找到这些最小值点，即使直接求解 $f'(x)=0$ 困难或不可能。

## 参考资料

- [Calculus](https://www.cengage.com/c/calculus-9e-stewart-clegg-watson/9781337624183/) — James Stewart, Daniel K. Clegg, Saleem Watson (2020)
  Publisher: Cengage Learning
  一本标准的本科微积分教材，全面介绍了导数及其应用，包括寻找局部和全局最大值与最小值。
- [Mathematics for Machine Learning](https://mml-book.com/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679901](https://doi.org/10.1017/9781108679901)
  提供了机器学习必需的数学基础的严谨而易懂的介绍，包括专门的优化章节，清晰地解释了导数在寻找成本函数最小值中的作用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第四章“数值计算”涵盖了优化算法和导数在寻找成本函数最小值中的作用，为机器学习提供了相关基础知识。
- [Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — Prof. Dr. David Jerison (2010)
  Publisher: MIT OpenCourseWare
  这门来自麻省理工学院的免费在线课程提供了涵盖单变量微积分基础概念的讲座和材料，包括使用导数寻找最大值和最小值。
