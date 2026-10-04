---
course: "fundamentals-quantum-machine-learning"
chapter: "variational-quantum-algorithms-ml"
lesson: "variational-principle-quantum-computation"
sourceId: 1055
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/variational-principle-quantum-computation"
title: "量子计算中的变分原理"
description: "理解变分原理作为变分量子算法的依据。"
order: 1
plots: []
sourceHash: "5a94fab3dc2cce5376f94ac631d6d39c7539f2c53f73a0d21ca7cd2450e54faf"
sourceCorrections: []
---

变分量子算法（VQAs）是一种强大的混合方法，它将量子计算的能力与经典优化技术相结合。如前所述，这种结构使得它们尤其适合在当前和近期的量子硬件上运行。这些算法的构建依据是量子力学中的变分原理。

### 变分原理：逼近的依据

在量子力学中，变分原理提供了一种估算由哈密顿量 $H$ 所描述的量子系统的基态能量 $E_0$ 的方法。该原理指出，对于任何行为良好的试探波函数 $|\psi\rangle$，哈密顿量对该状态的期望值为真实基态能量提供了一个上限。在数学上，这通过瑞利-里兹商表示为：


$$
E[\psi] = \frac{\langle \psi | H | \psi \rangle}{\langle \psi | \psi \rangle}
$$


如果试探状态 $|\psi\rangle$ 是归一化 (normalization)的（意味着 $\langle \psi | \psi \rangle = 1$），则简化为：


$$
E[\psi] = \langle \psi | H | \psi \rangle
$$


变分原理的核心表述是：


$$
E[\psi] \ge E_0
$$


当且仅当试探状态 $|\psi\rangle$ 恰好是与 $E_0$ 对应的基态本征向量 (vector) $|\psi_0\rangle$ 时，等式成立。这个原理非常有用：它意味着我们可以遍历一组可能的量子状态，计算每个状态的哈密顿量期望值，而产生最小期望值的状态将是我们对该组状态中真实基态（及其能量）的最佳逼近。我们的试探状态组越能代表实际的基态，我们得到的最小期望值就越接近 $E_0$。

### 使原理适用于量子计算

变分量子算法巧妙地将此原理应用于量子计算机。我们不再使用任意的试探波函数，而是使用可以通过量子计算机利用参数 (parameter)化量子电路（PQC）制备的量子状态 $|\psi(\vec{\theta})\rangle$，该电路通常表示为 $U(\vec{\theta})$。这里，$\vec{\theta} = (\theta_1, \theta_2, \dots, \theta_M)$ 表示一组可调参数，通常是电路中量子门内的旋转角度。PQC 作用于某个初始状态，通常是全零状态 $|0\dots 0\rangle$：


$$
|\psi(\vec{\theta})\rangle = U(\vec{\theta}) |0\dots 0\rangle
$$


哈密顿量 $H$ 现在表示我们旨在解决的问题。在量子化学中，它可能是我们寻求其基态能量的分子哈密顿量。在机器学习 (machine learning)中，$H$ 通常被构建，使其期望值对应于一个我们想要最小化的经典代价函数 $C(\vec{\theta})$（例如，分类误差，回归损失）。

那么任务就变成找到最优参数集 $\vec{\theta}^*$，使 $H$ 相对于 PQC 输出状态的期望值最小化：


$$
C(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle = \langle 0\dots 0 | U^\dagger(\vec{\theta}) H U(\vec{\theta}) | 0\dots 0 \rangle
$$


目标是找到：


$$
\vec{\theta}^* = \arg \min_{\vec{\theta}} C(\vec{\theta})
$$


### 混合量子-经典循环

这种最小化是通过一个迭代的混合循环实现的，该循环包含量子处理器和经典优化器：

1. **初始化：** 选择初始参数 (parameter)集 $\vec{\theta}_0$。
2. **量子执行：**
   - 使用当前参数 $\vec{\theta}_k$ 配置 PQC $U(\vec{\theta}_k)$。
   - 在量子计算机上制备状态 $|\psi(\vec{\theta}_k)\rangle = U(\vec{\theta}_k)|0\dots 0\rangle$。
   - 测量期望值 $\langle H \rangle_{\vec{\theta}_k} = \langle \psi(\vec{\theta}_k) | H | \psi(\vec{\theta}_k) \rangle$。这通常需要测量构成 $H$ 的单个泡利项，并对结果进行经典组合。这会给出代价函数值 $C(\vec{\theta}_k)$。
3. **经典优化：**
   - 将代价 $C(\vec{\theta}_k)$（以及可能其梯度，我们稍后将讨论）输入给经典优化算法。
   - 优化器提出一组新的参数 $\vec{\theta}_{k+1}$，旨在降低代价。
4. **迭代：** 重复步骤2和3，直到满足收敛标准（例如，代价的变化低于阈值，或达到最大迭代次数）。

> 变分量子算法特有的混合量子-经典循环示意图。量子处理器制备并测量参数化状态，而经典优化器根据测量结果调整参数，以最小化代价函数。

这种迭代过程运用量子计算机处理被认为经典困难的任务（制备和测量复杂的量子状态），同时依赖成熟的经典算法进行优化。变分原理保证，通过最小化可测量的代价函数 $C(\vec{\theta})$，我们能够找到我们所选 PQC 架构能够表示的问题解的最佳逼近。变分量子算法的有效性很大程度上取决于 PQC 的表达能力（其生成接近真实解状态的能力）以及经典优化过程的效率。我们将在后续章节中研究 PQC 设计、代价函数、梯度计算和优化策略。

## 参考资料

- [A variational eigenvalue solver on a photonic quantum processor](https://www.nature.com/articles/ncomms5213) — Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J. Love, Alán Aspuru-Guzik, and Jeremy L. O'Brien (2014)
  Journal: Nature Communications; Volume: 5; Pages: 4213; DOI: [10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213)
  这篇开创性论文介绍了变分量子本征求解器（VQE），展示了用于量子化学计算的混合量子-经典方法。
- [Variational Quantum Algorithms](https://www.nature.com/articles/s42254-021-00348-9) — M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, and Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Publisher: Nature Portfolio; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  一篇全面的综述，提供了VQA的最新概览，包括其理论、应用和挑战。
