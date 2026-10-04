---
course: "optimization-techniques-ml"
chapter: "ml-optimization-foundations-review"
lesson: "numerical-stability-optimization"
sourceId: 1237
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-1-ml-optimization-foundations-review/numerical-stability-optimization"
title: "数值稳定性考量"
description: "讨论浮点误差和病态问题等潜在的数值问题。"
order: 6
plots: ["plots/1237-0.json"]
sourceHash: "ccba1a7663b3a8dcc3002ab3570367292f02a1f0195a242f1e7f1ed6fd5f9f10"
sourceCorrections: []
---

尽管优化算法的理论保证常假定精确算术，实际执行却在有限精度的硬件上进行。理解计算机算术的局限性及其对优化的影响，对诊断问题和构建可靠的机器学习 (machine learning)模型十分要紧。

### 浮点算术的局限

现代计算机通常使用浮点格式表示实数，例如IEEE 754标准（常见的`float32`或`float64`）。这些格式使用固定数量的位来存储符号、指数和尾数（或有效数字）。这种有限的表示会带来一些影响：

1. **舍入误差：** 并非所有实数都能被精确表示。当计算结果落在可表示数之间时，必须进行舍入。在涉及数百万或数十亿次迭代操作的优化算法中，这些微小误差会累积，可能导致发散或收敛到次优解。计算得到的梯度 $\hat{g}(x)$ 可能与真实梯度 $g(x)$ 略有不同，从而影响更新步骤。
2. **灾难性抵消：** 两个几乎相等的数相减会导致相对精度的显著损失。例如，如果 $a \approx b$，计算 $a - b$ 得到的结果可能主要由 $a$ 和 $b$ 的舍入误差主导，而非它们的真实差值。这在通过有限差分计算梯度时，或在参数 (parameter)更新中步长相对于参数值变得非常小时，会引发问题。
3. **上溢与下溢：** 上溢发生在计算结果的绝对值大于最大可表示数时，常得到无穷大 (`inf`)。下溢发生在结果的绝对值小于最小正可表示数时，常被舍入为零。梯度爆炸（梯度变得过大）时可能出现上溢，而梯度消失或参数值/更新非常小的时候可能出现下溢。两者都可能中断或破坏训练过程的稳定性。例如，激活函数 (activation function)（如`exp`）或损失计算中的中间步骤，如果输入未适当缩放，有时会发生上溢或下溢。

### 病态问题及其影响

数值稳定性也与优化问题本身的数学特性，特别是其条件数，密切相关。如果输入（如参数 (parameter) $\theta$ 或数据 $x$）的微小变动会导致输出（如损失 $L(\theta)$ 或梯度 $\nabla L(\theta)$）不成比例的巨大变动，则该问题被认为是**病态的**。

在优化中，病态问题常与**Hessian矩阵** $H$ 有关联，该矩阵包含损失函数 (loss function)的二阶偏导数。Hessian矩阵的**条件数**，通常定义为其最大特征值 ($\lambda_{max}$) 与最小特征值 ($\lambda_{min}$) 之比，用于衡量这种敏感度：
$\kappa(H) = \frac{|\lambda_{max}|}{|\lambda_{min}|}$
极大的条件数 ($\kappa(H) \gg 1$) 表明问题是病态的。从几何上看，这意味着损失曲面在某些方向上远比其他方向陡峭，形似狭长山谷或沟壑。

病态问题对优化算法带来几类困难：

- **收敛缓慢：** 梯度下降 (gradient descent)等一阶方法倾向于在狭长山谷中摆动，同时沿着谷底进展缓慢，因为梯度方向并非直接指向最小值。
- **步长敏感：** 找到合适的学习率变得困难。适用于陡峭方向的速率对平坦方向可能过小，而适用于平坦方向的速率则可能在陡峭方向上导致不稳定或发散。
- **数值误差加剧：** 在病态问题中，浮点误差的影响会被放大，使得优化过程的可靠性降低。优化器采取的步骤可能严重受数值不准确性的影响，而非损失曲面的真实结构。



![病态二次函数上的梯度下降](plots/1237-0.json)



> 损失曲面 $L(x, y) = 0.1x^2 + 10y^2$ 上的梯度下降步骤（蓝色路径）。Y轴方向的高曲率和X轴方向的低曲率导致了病态问题 ($\kappa=100$)，使得优化器在狭长山谷（Y方向）上震荡，同时在谷底（X方向）进展缓慢。

### 缓解策略

虽然数值问题无法完全消除，但有几类策略可以帮助减轻其影响：

1. **使用更高精度：** 使用 `float64`（双精度）而非 `float32`（单精度）能提供更大的尾数和指数范围，从而减少舍入误差以及上溢/下溢的可能性。然而，这会增加内存占用（每个参数 (parameter)的存储空间加倍），并可能降低计算速度，尤其是在为 `float32` 优化过的硬件（如GPU）上。
2. **输入/输出缩放：** 规范化输入特征并确保目标变量处于合理范围，可以避免在中间计算中出现过大或过小的值。
3. **梯度裁剪：** 为防止梯度爆炸（可能导致上溢或不稳定更新），一种常用方法是在梯度范数超过特定阈值 $c$ 时对其进行裁剪。更新方式如下：
   $g \leftarrow \begin{cases} g & \text{若 } \|g\| \le c \\ \frac{c}{\|g\|} g & \text{若 } \|g\| > c \end{cases}$
   这会在梯度过大时将其范数缩放到 $c$，同时保持其方向。
4. **正则化 (regularization)：** L1或L2正则化等方法会根据参数大小向损失函数 (loss function)添加惩罚项。这有时能改善优化问题的条件数，使损失曲面更平滑。
5. **谨慎初始化：** 如后续章节（第六章）所述，适当初始化模型参数可以避免训练早期激活值和梯度变得过大或过小。
6. **数值稳定算法：** 一些算法本身就更为稳定。例如，像Adam这样的自适应方法，在分母中包含一个小的epsilon项 ($\epsilon$)，以防止平方梯度非常小时发生除以零的情况：
   $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
   类似地，L-BFGS等拟牛顿法避免直接求Hessian矩阵的逆，因为这通常在数值上不稳定。
7. **使用现有库：** 标准机器学习 (machine learning)库（如TensorFlow、PyTorch、JAX）和数值计算库（如NumPy、SciPy）在实现运算时非常注重数值稳定性，常采用融合操作或针对常见计算（例如softmax的对数和指数技巧）的数值稳定算法。依赖这些经过测试的实现通常比从头编写复杂的数值程序更安全。

了解这些潜在的数值问题，对于调试表现异常的训练过程（例如，损失变为 `NaN`、突然发散、收敛极其缓慢），以及选择适当的技巧和超参数 (hyperparameter)以实现稳定高效的优化，具有重要意义。

## 参考资料

- [Accuracy and Stability of Numerical Algorithms](https://epubs.siam.org/doi/book/10.1137/1.9780898718027) — Nicholas J. Higham (2002)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
  关于浮点算术、误差分析和数值方法稳定性的标准教科书。
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  优化方面的书籍，详细介绍了条件数、数值属性和算法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  包含数值计算、优化稳定性以及神经网络训练常见问题的讨论。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化器，一种广泛使用的自适应方法，包含数值稳定策略。
