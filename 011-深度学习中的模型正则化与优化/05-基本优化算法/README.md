# 第 5 章：基本优化算法

来源：[原章节](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers)

[返回课程目录](../README.md)

在研究了通过正则化控制模型复杂度的方法之后，我们现在关注为深度学习模型寻找最优参数的过程。这个被称为优化的过程，是有效训练神经网络的核心。标准梯度下降提供了理论依据，但它应用于大型数据集时面临实际问题。

本章介绍深度学习中使用的基本优化算法。我们首先回顾标准梯度下降，并讨论它的局限性。然后您将了解：

*   **随机梯度下降（SGD）：** 一种计算高效的近似方法，使用单个数据点更新参数。
*   **小批量梯度下降：** 一种广泛使用的方法，平衡了SGD和批量梯度下降的优点。
*   **动量：** 一种加速SGD收敛的技术，特别是在梯度方向一致时，并抑制振荡。
*   **Nesterov加速梯度（NAG）：** 对动量的一种改进，通常能带来更快的收敛。

在本章结束时，您将理解这些核心算法的运作原理，它们各自的优缺点，以及它们在模型训练过程中如何应对复杂损失曲面带来的挑战。我们还将在实践中实现并比较这些优化器。

## 小节

- 1. [回顾梯度下降](01-%E5%9B%9E%E9%A1%BE%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 2. [标准梯度下降的难题](02-%E6%A0%87%E5%87%86%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 3. [随机梯度下降（SGD）](03-%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%EF%BC%88SGD%EF%BC%89.md)
- 4. [小批量梯度下降](04-%E5%B0%8F%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 5. [SGD的挑战：噪声与局部最小值](05-SGD%E7%9A%84%E6%8C%91%E6%88%98%EF%BC%9A%E5%99%AA%E5%A3%B0%E4%B8%8E%E5%B1%80%E9%83%A8%E6%9C%80%E5%B0%8F%E5%80%BC.md)
- 6. [带动量的随机梯度下降：加速收敛](06-%E5%B8%A6%E5%8A%A8%E9%87%8F%E7%9A%84%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%EF%BC%9A%E5%8A%A0%E9%80%9F%E6%94%B6%E6%95%9B.md)
- 7. [涅斯捷罗夫加速梯度 (NAG)](07-%E6%B6%85%E6%96%AF%E6%8D%B7%E7%BD%97%E5%A4%AB%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%20%28NAG%29.md)
- 8. [实现SGD和动量](08-%E5%AE%9E%E7%8E%B0SGD%E5%92%8C%E5%8A%A8%E9%87%8F.md)
- 9. [实践：比较梯度下降、随机梯度下降和动量法](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%AF%94%E8%BE%83%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E3%80%81%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%92%8C%E5%8A%A8%E9%87%8F%E6%B3%95.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/quiz)
