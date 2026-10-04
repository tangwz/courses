# 第 4 章：处理多输入：偏导数

来源：[原章节](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs)

[返回课程目录](../README.md)

大多数机器学习模型处理带有多个特征的数据，或需要调整大量参数。与我们迄今研究的单变量函数 $f(x)$ 不同，实际模型通常更像是 $f(x_1, x_2, \dots, x_n)$。当一个函数依赖于多个输入时，我们需要方法弄明白它仅当*某个*输入变化时是如何变化的。

本章将介绍做到这一点的方法：**偏导数**。你将学到：

*   如何理解和处理多变量函数。
*   偏导数的定义和计算，使用诸如 $\frac{\partial f}{\partial x}$ 的符号。
*   如何将偏导数组合成**梯度向量** ($\nabla f$)。
*   梯度的几何意义，以及它在指明函数最陡峭变化方向方面的作用。

这些知识将我们对导数的理解扩展到机器学习优化中常见的多维情况。

## 小节

- 1. [多变量函数](01-%E5%A4%9A%E5%8F%98%E9%87%8F%E5%87%BD%E6%95%B0.md)
- 2. [偏导数：其思想](02-%E5%81%8F%E5%AF%BC%E6%95%B0%EF%BC%9A%E5%85%B6%E6%80%9D%E6%83%B3.md)
- 3. [计算偏导数](03-%E8%AE%A1%E7%AE%97%E5%81%8F%E5%AF%BC%E6%95%B0.md)
- 4. [偏导数记法](04-%E5%81%8F%E5%AF%BC%E6%95%B0%E8%AE%B0%E6%B3%95.md)
- 5. [梯度向量](05-%E6%A2%AF%E5%BA%A6%E5%90%91%E9%87%8F.md)
- 6. [梯度的几何含义](06-%E6%A2%AF%E5%BA%A6%E7%9A%84%E5%87%A0%E4%BD%95%E5%90%AB%E4%B9%89.md)
- 7. [练习：计算偏导数和梯度](07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E8%AE%A1%E7%AE%97%E5%81%8F%E5%AF%BC%E6%95%B0%E5%92%8C%E6%A2%AF%E5%BA%A6.md)

章节测验：[在线测验](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs/quiz)
