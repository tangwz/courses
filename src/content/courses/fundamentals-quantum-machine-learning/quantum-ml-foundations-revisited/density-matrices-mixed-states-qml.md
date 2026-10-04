---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-ml-foundations-revisited"
lesson: "density-matrices-mixed-states-qml"
sourceId: 1036
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-1-quantum-ml-foundations-revisited/density-matrices-mixed-states-qml"
title: "QML中的密度矩阵与混合态"
description: "了解密度矩阵的形式体系，用于描述QML背景下的混合态和噪声。"
order: 4
plots: ["plots/1036-0.json"]
sourceHash: "9cd6e7d9b21b4e1972c96651d33009c69bd5f9184a537067991efd234d7e1f3d"
sourceCorrections: []
---

虽然纯态矢量 $|\psi\rangle$ 提供了孤立量子系统的根本描述，但它们代表了一种理想化的情形。在实际操作中，特别是当处理受环境作用影响的量子计算，或者我们只对系统制备有部分信息时，我们需要一个更普适的工具：密度矩阵。掌握密度矩阵对于准确模拟QML算法中的噪声以及解读其在真实硬件上的结果非常重要。

### 量子态的描述：态矢量

希尔伯特空间 $\mathcal{H}$ 中的纯量子态 $|\psi\rangle$ 可以用一个**密度算符**（或密度矩阵）$\rho$ 来描述，其定义为外积：


$$
\rho = |\psi\rangle\langle\psi|
$$


此算符投影到由 $|\psi\rangle$ 张成的子空间。它具有以下一些重要性质：

1. **厄米性：** $\rho^\dagger = \rho$。
2. **迹为1：** $\text{Tr}(\rho) = 1$。这反映了态矢量的归一化 (normalization)，即 $\langle\psi|\psi\rangle = 1$。
3. **半正定性：** 对于任意矢量 $|\phi\rangle$，$\langle\phi|\rho|\phi\rangle \ge 0$。
4. **幂等性（对于纯态）：** $\rho^2 = \rho$。这直接源于归一化：$(|\psi\rangle\langle\psi|)(|\psi\rangle\langle\psi|) = |\psi\rangle(\langle\psi|\psi\rangle)\langle\psi| = |\psi\rangle(1)\langle\psi| = \rho$。

密度矩阵形式体系的真正作用在处理**混合态**时显现出来。混合态代表了纯态的统计系综。设想一个源以概率 $p_i$ 制备量子态 $|\psi_i\rangle$，其中 $\sum_i p_i = 1$。这种系综通常不能用单个态矢量描述。此时，我们使用密度矩阵：


$$
\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|
$$


这是纯态密度矩阵的凸组合。这种更普适的密度矩阵仍然满足性质1、2和3（厄米性、迹为1、半正定性）。然而，它通常不是幂等的（$\rho^2 \neq \rho$）。

区分纯态和混合态的一个有用量度是**纯度**，定义为 $\text{Tr}(\rho^2)$。

- 对于纯态 $\rho = |\psi\rangle\langle\psi|$，$\text{Tr}(\rho^2) = \text{Tr}(|\psi\rangle\langle\psi||\psi\rangle\langle\psi|) = \text{Tr}(|\psi\rangle\langle\psi|) = 1$。
- 对于混合态，可以证明 $\text{Tr}(\rho^2) < 1$。最小值取决于希尔伯特空间的维度。一个最大混合态（例如，处于任意基态的概率相等）具有最低纯度。

对于单个量子比特，其状态可以在布洛赫球上进行可视化。纯态位于表面，而混合态占据内部。



![布洛赫球：纯态 vs. 混合态](plots/1036-0.json)



> 纯态（蓝色圆点）位于布洛赫球表面（$\text{Tr}(\rho^2)=1$），而混合态（红色菱形）位于球体内部（$\text{Tr}(\rho^2)<1$）。

### 期望值与动力学

计算由 $\rho$ 描述的系统可观测量 $M$（由厄米算符表示）的期望值很简单：


$$
\langle M \rangle = \text{Tr}(\rho M)
$$


你可以验证，如果 $\rho = |\psi\rangle\langle\psi|$，这会简化为我们熟悉的 $\text{Tr}(|\psi\rangle\langle\psi| M) = \langle\psi| M |\psi\rangle$。对于混合态 $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$，期望值为 $\text{Tr}(\sum_i p_i |\psi_i\rangle\langle\psi_i| M) = \sum_i p_i \text{Tr}(|\psi_i\rangle\langle\psi_i| M) = \sum_i p_i \langle\psi_i| M |\psi_i\rangle$，这正是系综中每个纯态期望值的加权平均值，如预期。

由哈密顿量 $H$ 支配的封闭系统，其密度矩阵的时间演化由**刘维尔-冯诺依曼方程**给出：


$$
i\hbar \frac{d\rho}{dt} = [H, \rho] = H\rho - \rho H
$$


这是密度矩阵对应的薛定谔方程。对于与环境产生作用的开放量子系统（会引入噪声和退相干），需要更复杂的主方程，如林德布拉德方程，来描述 $\rho$ 的演化。在讨论第7章的硬件噪声和错误缓解时，我们将遇到这些内容。

### 子系统与部分迹

在量子计算和QML中，我们经常处理复合系统（多个量子比特），但只对系统的一部分感兴趣。考虑一个由作用于希尔伯特空间 $\mathcal{H}_A \otimes \mathcal{H}_B$ 的联合密度矩阵 $\rho_{AB}$ 描述的双体系统AB。如果只描述子系统A的状态，忽略子系统B，我们使用对B的部分迹运算，记作 $\text{Tr}_B$：


$$
\rho_A = \text{Tr}_B(\rho_{AB})
$$


如果 $\{|b_j\rangle_B\}$ 是 $\mathcal{H}_B$ 的正交归一基，则部分迹计算如下：


$$
\rho_A = \sum_j ({_B\langle b_j|}) \rho_{AB} (|b_j\rangle_B)
$$


所得的 $\rho_A$ 是子系统A的**约化密度矩阵**。重要的是，即使复合系统AB处于纯纠缠态，约化密度矩阵 $\rho_A$ 和 $\rho_B$ 通常也会表示混合态。这表明纠缠如何导致关联，使得子系统的状态无法独立地以纯态描述。部分迹对描述量子比特子集上的测量结果或量子比特丢失到环境中的影响非常重要。

### 在量子机器学习 (machine learning)中的相关性

密度矩阵在QML中必不可少，原因有以下几点：

1. **噪声建模：** 近期量子计算机存在噪声。退相干、门操作不完美和测量误差导致纯态演化为混合态。密度矩阵模拟常用于实际预测QML算法在真实硬件上的表现。去极化噪声或振幅阻尼等特性，很自然地可以通过对密度矩阵的操作来表达。
2. **算法规范：** 有些QML算法本身就可能包含混合态。例如，量子生成模型的输出可能是一个表示不确定性的混合态，或者源自有噪声物理过程的输入数据最好由密度矩阵表示。
3. **理论分析：** 分析QML模型的性质，如它们的表达能力或可训练性，通常需要完整的密度矩阵形式体系，尤其是在考虑噪声或量子电路不同部分之间的纠缠时。
4. **信息几何：** 给定系统所有有效密度矩阵的集合形成一个几何空间。信息几何的工具，如量子费希尔信息度量，作用于密度矩阵并为变分量子算法（VQA）的参数 (parameter)空间提供理解，帮助我们理解优化难题，例如贫瘠高原（第4章涵盖）。

总而言之，虽然态矢量提供了最初的直觉，但掌握密度矩阵形式体系对推进量子机器学习的实际运用和理论深度是必需的。它提供了处理统计不确定性、环境噪声和子系统的语言，所有这些都是设计、实施和分析QML算法时常见考量。

## 参考资料

- [Quantum Computation and Quantum Information](https://www.cambridge.org/gb/academic/subjects/computer-science/programming-languages-and-applied-logic/quantum-computation-and-quantum-information-10th-anniversary-edition) — Michael A. Nielsen, Isaac L. Chuang (2010)
  Publisher: Cambridge University Press; DOI: [10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)
  一本涵盖量子计算和信息基础的综合性教科书，包括密度矩阵、开放量子系统及其应用。
- [Quantum Computation](http://www.theory.caltech.edu/~preskill/ph219/index.html) — John Preskill (2021)
  Publisher: California Institute of Technology
  备受推崇的讲义，对量子力学和量子信息进行严谨的阐述，包括对密度矩阵和混合态的详细讨论。
- [Machine Learning with Quantum Computers](https://doi.org/10.1007/978-3-030-83098-4) — Maria Schuld, Francesco Petruccione (2021)
  Publisher: Springer International Publishing; DOI: [10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)
  量子机器学习的基础教科书，阐述了密度矩阵形式主义在量子机器学习中的应用，尤其是在噪声建模和算法设计方面。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  一篇开创性论文，介绍了变分量子算法中贫瘠高原的概念，其中密度矩阵特性和信息几何在理解梯度消失方面发挥了作用。
