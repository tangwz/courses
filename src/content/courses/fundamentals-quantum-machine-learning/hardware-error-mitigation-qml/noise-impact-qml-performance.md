---
course: "fundamentals-quantum-machine-learning"
chapter: "hardware-error-mitigation-qml"
lesson: "noise-impact-qml-performance"
sourceId: 1079
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml/noise-impact-qml-performance"
title: "噪声对QML算法性能的影响"
description: "分析不同噪声信道如何影响各种QML算法的训练和性能。"
order: 2
plots: ["plots/1079-0.json"]
sourceHash: "c4da026016700a4d87ad33512b3fa85a02aac8fd5d28803b80067fa687dd12bf"
sourceCorrections: []
---

如前所述，NISQ时代的量子处理器本身存在噪声。这种噪声不只是一个小麻烦；它从根本上改变了在这些设备上执行的量子机器学习 (machine learning)算法的行为和性能。了解噪声如何影响计算对结果的解读和开发有效的QML应用十分重要。

回想一下，量子噪声通常表现为量子比特与其环境之间不必要的作用，或用于实现量子门的控制脉冲的缺陷。这些过程导致偏离预期的量子演化。常见的噪声模型有：

- **退极化噪声：** 以一定概率随机作用泡利算符（$X, Y, Z$），将状态有效地混合到最大混合态（$I/2^n$）。
- **振幅阻尼：** 模拟能量弛豫（例如，量子比特从$\vert 1 \rangle$衰变到$\vert 0 \rangle$）。
- **相位阻尼（退相干）：** 模拟$\vert 0 \rangle$和$\vert 1 \rangle$之间相位相干性的损失，不伴随能量交换。
- **门错误：** 单量子比特和多量子比特门实现中的缺陷，导致实际操作与理想酉变换不同。
- **读出错误：** 测量过程中的错误，记录的经典比特值与测量前量子比特的最终状态不匹配。

数学上，噪声将理想纯态$\vert \psi \rangle$变为混合态，由密度矩阵$\rho$准确描述。理想操作将状态转换为$\rho_{out} = U \rho_{in} U^\dagger$，其中 $U$ 是理想酉算符。然而，有噪声的操作通常由量子信道表示，常使用算子求和表示法表示：


$$
\mathcal{E}(\rho_{in}) = \sum_k E_k \rho_{in} E_k^\dagger
$$


其中 $E_k$ 是满足$\sum_k E_k^\dagger E_k = I$的克劳斯算符。这种变换通常会增加状态的熵并降低其纯度（$\text{Tr}(\rho^2) < 1$）。

### 噪声如何降低QML算法各部分的性能

噪声影响典型QML工作流的每个阶段，从数据编码到测量，累积错误并降低最终结果的质量。

#### 对数据编码和特征映射的影响

量子特征映射将经典数据 $x$ 编码为量子态$\vert \phi(x) \rangle$。编码电路$\mathcal{U}_{\phi(x)}$执行期间产生的噪声意味着实际产生的状态是一个混合态$\rho_x = \mathcal{E}_{enc}(\vert 0 \rangle \langle 0 \vert)$，其中$\mathcal{E}_{enc}$代表有噪声的编码信道。

这直接导致以下结果：

1. **特征空间几何失真：** 编码态之间的内积，$|\langle \phi(x_i) | \phi(x_j) \rangle|^2$，构成了量子核方法的基础。噪声会改变这些内积：$\text{Tr}(\rho_{x_i} \rho_{x_j})$将与理想值不同。这种失真改变了希尔伯特空间中数据表示的几何形状，可能导致不同的数据点显得更接近，或相似的数据点显得比预期更远。类别的可分离性可能会下降。
2. **表达能力下降：** 噪声倾向于缩小特征映射可到达的状态空间体积，可能限制QML模型能学习的函数的复杂度。

#### 对变分电路（PQC）和训练的影响

变分算法是许多QML方法的核心。它们涉及优化参数 (parameter)$\theta$的参数化量子电路（PQC），$U(\theta)$，以最小化成本函数$C(\theta)$，该函数通常来源于可观测量的期望值$\langle O \rangle_{\theta} = \text{Tr}(O \, U(\theta) \rho_{in} U(\theta)^\dagger)$。噪声严重破坏了这一过程：

1. **期望值不准确：** PQC执行期间的噪声意味着测量前的最终状态是一个有噪声的混合态$\rho_{out}(\theta) = \mathcal{E}_{PQC}(U(\theta) \rho_{in} U(\theta)^\dagger)$。测得的期望值变为$\langle O \rangle_{\theta, noisy} = \text{Tr}(O \, \rho_{out}(\theta))$。该值偏离了理想的$\langle O \rangle_{\theta, ideal}$。因此，正在评估的成本函数$C_{noisy}(\theta)$与目标$C_{ideal}(\theta)$不同。优化可能在一个失真的成本函数上进行。
2. **梯度错误：** 基于梯度的优化依赖于准确估计$\nabla_{\theta} C(\theta)$。像参数移位规则这样的方法通过在移位参数下评估的期望值差异来计算梯度。噪声会将误差引入这些期望值，导致不准确的梯度估计$\nabla_{\theta} C_{noisy}(\theta)$。有噪声的梯度会减慢收敛速度，导致优化器陷入与理想成本函数最小值无关的局部最小值，甚至完全阻止收敛。
3. **噪声导致的贫瘠高原（NIBPs）：** 尽管贫瘠高原（梯度消失）即使在理想的深层PQC中也可能出现，但噪声即使在浅层电路中也能引起类似现象。噪声有效地平均了梯度信息，尤其是在全局定义的成本函数中，这使得优化变得极为困难。
4. **训练动态改变：** 整个优化路径可能会改变。与无噪声情况相比，模型可能收敛到一组不同的参数，通常导致在未见数据上的泛化性能下降。



![噪声对VQC训练收敛的影响](plots/1079-0.json)



> 简单变分量子分类器在理想（无噪声）条件下与包含中等退极化噪声的模拟训练收敛比较。噪声明显阻碍了优化，导致最终成本值更高。

#### 对测量和读出的影响

即使量子态在测量前已正确准备，读出错误也会引入最后一层损坏。当真实状态是$\vert 1 \rangle$时，经典比特可能记录为'0'，反之亦然。这可以通过混淆矩阵或概率赋值$P(\text{读取 } i | \text{真实 } j)$来建模。读出错误直接偏倚测量结果的统计数据，影响期望值的估计，进而影响成本函数和梯度。

### 特定QML算法的结果

- **量子支持向量 (vector)机（QSVM）：** QSVM的性能完全取决于量子核矩阵$K_{ij} = f(\text{Tr}(\rho_{x_i} \rho_{x_j}))$的质量，其中$f$通常是$|\cdot|^2$或相关的保真度量。噪声会扭曲这些核项。如果失真明显，核矩阵可能会失去其所需属性（如良好表示数据结构），甚至不再是半正定，导致后续的经典SVM优化无效或效果不佳。噪声还会加剧核浓度问题，使不同的数据点看起来均匀相似。
- **变分量子分类器（VQC）：** 如上所述，有噪声的执行会导致不准确的成本评估和梯度。这使得VQC难以有效训练。与理想情况相比，特征空间中学习到的决策边界可能不是最优的，从而导致分类准确性下降。
- **量子生成模型（QGANs, QCBMs）：** 噪声从根本上限制了这些模型准确学习和表示目标概率分布的能力。对于QCBMs，从有噪声电路中采样的输出分布将偏离目标分布。对于QGANs，训练动态变得更加复杂和不稳定，因为量子生成器和潜在的量子判别器都受到噪声影响，使其更难达到满意的平衡。生成的样本质量（例如，通过保真度或散度指标衡量）会更低。

> 理想与有噪声QML工作流的比较。噪声源（门错误、退相干、读出错误）影响每个阶段，导致计算偏离预期并降低输出质量。

总之，量子硬件噪声是QML的一个重要瓶颈。它扭曲了数据表示，通过破坏成本函数和梯度来阻碍优化，使测量结果产生偏差，并最终降低QML模型的预测或生成性能。识别这些具体影响是设计和应用策略来抵消它们的第一步，这也是后续关于错误缓解部分所关注的。

## 参考资料

- [Quantum Computation and Quantum Information](https://doi.org/10.1017/CBO9780511976667) — Michael A. Nielsen and Isaac L. Chuang (2010)
  Publisher: Cambridge University Press; DOI: [10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)
  一本经典的教科书，介绍了量子信息和计算，详细解释了量子信道、噪声模型和密度矩阵形式。
- [Noise-induced barren plateaus in variational quantum algorithms](https://doi.org/10.1038/s41467-021-27045-6) — Samson Wang, Enrico Fontana, Marco Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, Patrick J. Coles (2021)
  Journal: Nature Communications; Publisher: Nature Publishing Group; Volume: 12; Pages: 6961; DOI: [10.1038/s41467-021-27045-6](https://doi.org/10.1038/s41467-021-27045-6)
  这篇研究论文详细介绍了噪声如何导致变分量子算法中的贫瘠高原现象，对优化过程提出挑战。
- [Variational Quantum Algorithms](https://doi.org/10.1038/s42254-021-00348-9) — M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, and Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  一篇关于变分量子算法的综述，涵盖了其理论基础、应用和问题，包括噪声影响及提出的缓解策略。
- [Effect of noise on quantum machine learning algorithms](https://doi.org/10.1088/2058-9565/aba060) — Hong-Ye Hu, Chen-Yi Li, and Xiao-Ding Yuan (2020)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 5; Pages: 035003; DOI: [10.1088/2058-9565/aba060](https://doi.org/10.1088/2058-9565/aba060)
  这篇论文审视了不同类型的噪声如何影响各种量子机器学习算法，包括量子核方法。
