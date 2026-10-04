---
course: "fundamentals-quantum-machine-learning"
chapter: "variational-quantum-algorithms-ml"
lesson: "gradient-calculation-methods-vqa"
sourceId: 1058
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/gradient-calculation-methods-vqa"
title: "梯度计算方法"
description: "详细分析计算 PQC 期望值梯度的方法，例如参数位移规则。"
order: 4
plots: []
sourceHash: "b7f2306b98297dd1bac0c3f6ebbfe9ac3fa27675f5ed3137f31db11fbe473908"
sourceCorrections: []
---

训练变分量子算法（VQA）高度依赖于经典优化循环。如我们所论，VQA 包含一个参数 (parameter)化量子电路（PQC）$U(\theta)$ 和一个成本函数 $C(\theta)$，该函数通常定义为在 PQC 准备的输出态上测量的可观测量 $H$ 的期望值：


$$
C(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle = \langle 0 | U^\dagger(\theta) H U(\theta) | 0 \rangle
$$


为了使用基于梯度的办法（例如梯度下降 (gradient descent)及其变体）最小化此成本函数，我们需要计算梯度向量 (vector) $\nabla_\theta C(\theta)$，其分量为相对于电路中的每个参数 $\theta_j$ 的偏导数 $\frac{\partial C}{\partial \theta_j}$。由于成本函数评估涉及运行量子电路和执行测量，因此计算这些梯度需要专门为这种混合量子-经典配置设计的技术。我们来分析实践中使用的主要办法。

### 数值有限差分

最直接的办法直接借鉴于经典数值办法：有限差分。为了估算相对于参数 (parameter) $\theta_j$ 的偏导数，我们可以在轻微扰动的参数值处评估成本函数。中心差分公式常用：


$$
\frac{\partial C}{\partial \theta_j} \approx \frac{C(\theta + \epsilon e_j) - C(\theta - \epsilon e_j)}{2\epsilon}
$$


这里，$e_j$ 是一个在第 $j$ 个位置为 1、其余位置为 0 的单位向量 (vector)，$\epsilon$ 是一个小步长。此办法需要对量子电路进行两次评估（并估算期望值），以计算每个参数 $\theta_j$ 的偏导数。

尽管有限差分易于理解和实现，但在量子计算背景下，它存在显著缺点：

1. **$\epsilon$ 的选择：** 选择合适的 $\epsilon$ 比较困难。如果 $\epsilon$ 过大，近似就不准确（截断误差）。如果 $\epsilon$ 过小，$C(\theta + \epsilon e_j) - C(\theta - \epsilon e_j)$ 的差值可能被统计噪声（散粒噪声）所主导，这种噪声源于从有限次测量中估算期望值，导致梯度估算中出现较大误差（减法误差）。
2. **统计噪声：** 分子中的减法会放大散粒噪声的影响，尤其当 $C(\theta + \epsilon e_j)$ 和 $C(\theta - \epsilon e_j)$ 非常接近时。为了达到足够的精度，每次评估通常需要大量的测量（shots），从而增加了计算成本。

由于这些局限，尽管有限差分对于快速检查或简单问题有益，但对于训练 VQA，通常倾向于使用更专门的办法。

### 参数 (parameter)位移规则

一种更广为采用的计算 PQC 梯度的办法是参数位移规则。对于特定类别的参数化门，该规则提供了梯度的解析表达式，避免了与在有限差分中选择 $\epsilon$ 相关的数值不稳定性。

考虑 PQC 中形式为 $G_j(\theta_j) = e^{-i \frac{\theta_j}{2} P_j}$ 的一个门，其中 $P_j$ 是一个满足 $P_j^2 = I$ 的算符。这包含常见的单比特旋转门，例如 $R_X(\theta_j) = e^{-i \frac{\theta_j}{2} X}$、$R_Y(\theta_j) = e^{-i \frac{\theta_j}{2} Y}$ 和 $R_Z(\theta_j) = e^{-i \frac{\theta_j}{2} Z}$，因为 $X^2 = Y^2 = Z^2 = I$。

如果整个 PQC $U(\theta)$ 可以写为 $U(\theta) = V G_j(\theta_j) W$，其中 $V$ 和 $W$ 是与 $\theta_j$ 无关的其他量子电路，那么期望值 $C(\theta) = \langle 0 | U^\dagger(\theta) H U(\theta) | 0 \rangle$ 相对于 $\theta_j$ 的导数可以使用以下公式精确计算：


$$
\frac{\partial C(\theta)}{\partial \theta_j} = \frac{1}{2} \left[ C\left(\theta + \frac{\pi}{2} e_j\right) - C\left(\theta - \frac{\pi}{2} e_j\right) \right]
$$


这个显著的结论表明，精确导数与在参数 $\theta_j$ 向前和向后位移特定量 $s = \pi/2$ 后评估的成本函数差值成比例。

**其原理何在？**
我们概述一下思路。求导涉及对门 $G_j(\theta_j)$ 进行求导：
$\frac{\partial G_j(\theta_j)}{\partial \theta_j} = -\frac{i}{2} P_j e^{-i \frac{\theta_j}{2} P_j} = -\frac{i}{2} P_j G_j(\theta_j)$
将其代入 $C(\theta)$ 的导数表达式会导致包含 $P_j$ 的项。主要的认识是，对于 $P_j^2=I$ 的门，算符 $P_j$ 可以与原始门 $G_j(\theta_j)$ 的位移版本相关联。具体而言，可以证明：
$P_j G_j(\theta_j) = \frac{i}{2} \left[ G_j\left(\theta_j + \frac{\pi}{2}\right) - G_j\left(\theta_j - \frac{\pi}{2}\right) \right]$
将其代回 $C(\theta)$ 的导数表达式并简化，最终得到参数位移规则公式。

> 参数位移规则的示意图，用于计算梯度分量 $\partial C / \partial \theta_j$。期望值 $\langle H \rangle$ 通过运行电路，将参数 $\theta_j$ 位移 $+\pi/2$ 和 $-\pi/2$ 来估算。将结果组合以得到精确梯度。

**优点：**

- **解析精确：** 它提供了真实梯度，而非数值近似（在评估 $C$ 时会受到散粒噪声影响）。
- **无需调整步长：** 它避免了选择小 $\epsilon$ 的问题。位移量 $s = \pi/2$ 是固定的。
- **韧性：** 相较于有限差分，它对散粒噪声通常更具韧性，因为位移量较大，使得差值 $C(\theta + s e_j) - C(\theta - s e_j)$ 相对于噪声基底通常更大。

**成本：** 与中心有限差分类似，参数位移规则对每个参数 $\theta_j$ 需要两次电路评估。

**适用性：** 基本规则直接适用于由泡利算符（$X, Y, Z$）生成的门。QML 中常用的大多数 PQC 拟设主要是由这类门构建的，使参数位移规则得以广泛应用。

### 泛化与其它办法

- **泛化参数 (parameter)位移规则：** 参数位移的理念可以推广到由仅有两个独特特征值的算符 $G$ 生成的门，即使 $G^2 \neq I$。这包含在特定条件下的受控旋转门。公式中的位移值 $s$ 和系数可能会根据生成器 $G$ 而变化。
- **酉算符线性组合（LCU）：** 存在更先进的技术，例如基于将导数算符自身表示为酉运算的线性组合的办法。这些有时能提供优势，在特定情况下可能以更少的电路执行计算梯度，但通常涉及更复杂的电路构建或测量。
- **随机参数位移：** 对于涉及许多参数的梯度，存在一些变体，其中每个优化步骤只计算参数位移的一个子集，引入类似于经典小批量梯度下降 (gradient descent)的随机性。

### 来自测量的梯度

重要的是要记住，在实践中，无论是使用有限差分还是参数 (parameter)位移规则，成本函数值 $C(\theta \pm s e_j)$ 本身都是从量子计算机上有限数量的测量（shots）中*估算*出来的。因此，计算出的梯度 $\nabla_\theta C(\theta)$ 也是一个估算值。此梯度估算的准确性直接取决于每次期望值计算使用的测量次数。这种统计不确定性是 VQA 所固有的，是选择优化算法和解读训练动态时的主要考量，我们将在接下来讨论这一点。

这些梯度计算办法在量子电路评估和经典优化程序之间建立了联系，从而使得 VQA 可以用于机器学习 (machine learning)任务的训练。参数位移规则，尤其是一种在大多数量子软件框架（如 Qiskit、PennyLane 和 TensorFlow Quantum）中实现的标准且有效的技术。

## 参考资料

- [Evaluating analytic gradients of quantum circuits by parameter shift rules](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331) — Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran (2019)
  Journal: Physical Review A; Publisher: American Physical Society; Volume: 99; Pages: 032331; DOI: [10.1103/PhysRevA.99.032331](https://doi.org/10.1103/PhysRevA.99.032331)
  这篇学术论文介绍并规范了参数偏移规则，用于计算量子电路的精确梯度，是变分量子算法中的一项标准技术。
- [Variational quantum algorithms](https://www.nature.com/articles/s42254-021-00348-9) — M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Hanamichi Yamasaki, and Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  一篇关于变分量子算法的全面综述，其中包含对与其训练相关的优化技术和梯度计算方法的概述。
- [Gradients and optimization](https://docs.pennylane.ai/en/stable/introduction/quantum_gradients.html) — PennyLane Documentation Team (2024)
  Publisher: Xanadu
  官方文档，解释了在 PennyLane 量子机器学习框架中，如何实现和使用包括参数偏移规则在内的梯度计算方法。
