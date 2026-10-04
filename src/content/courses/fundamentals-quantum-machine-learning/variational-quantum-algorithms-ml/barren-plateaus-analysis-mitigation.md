---
course: "fundamentals-quantum-machine-learning"
chapter: "variational-quantum-algorithms-ml"
lesson: "barren-plateaus-analysis-mitigation"
sourceId: 1061
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/barren-plateaus-analysis-mitigation"
title: "荒原高原的分析与缓解"
description: "了解深度 PQC 中荒原高原现象及其对训练影响的缓解策略。"
order: 7
plots: ["plots/1061-0.json"]
sourceHash: "7bbb29e4d4a22ffeea35c7bc14a13c4ee207db784dd4e01b4bc804ed0b71bfb2"
sourceCorrections: []
---

变分量子算法 (VQA) 的训练涉及迭代调整一个参数 (parameter)化量子电路 (PQC) $U(\boldsymbol{\theta})$ 的参数 $\boldsymbol{\theta}$，以最小化成本函数 $C(\boldsymbol{\theta})$。这种优化主要依赖于计算梯度 $\nabla C(\boldsymbol{\theta})$。然而，在许多 VQA 配置中出现了一个重要难题：称为**荒原高原**的现象。

设想一下，您正在尝试在一个广袤、极其平坦的沙漠中找到最低点。您的步长（梯度更新）变得微乎其微，找到谷底似乎几乎不可能。这类似于 VQA 中的荒原高原问题。它指的是成本函数的梯度随量子比特数 $n$ 呈指数衰减的情况，这使得大型系统的优化变得极其困难，甚至实际上无法进行。

具体来说，对于许多 PQC 和成本函数的选择，成本函数相对于电路参数 $\theta_k$ 的偏导数方差随系统规模增长而迅速衰减：

$\text{方差}\left[\frac{\partial C}{\partial \theta_k}\right] \in O\left(\frac{1}{e^{cn}}\right)$

其中 $c$ 为某个正数常数。这种指数衰减意味着即使对于中等规模的量子系统，梯度也变得与零无法区分，在达到最小值之前就停止了优化过程。了解荒原高原的起因和可能解决方案对于开发可扩展的量子机器学习 (machine learning)算法非常重要。

## 荒原高原的起因

荒原高原并非单一现象，而是可能来自不同的原因，通常与 PQC 的结构、成本函数的选择以及噪声的存在有关。

### 全局成本函数和纠缠

一个主要原因在于足够深或具有纠缠能力的 PQC 与**全局成本函数**的结合。全局成本函数依赖于跨大量量子比特的测量，通常涉及测量对大多数或所有 $n$ 个量子比特都产生非平凡作用的可观测值 $O$。

当一个 PQC 具有足够的表现力（通常与深度和纠缠能力相关联）时，使用随机初始参数 (parameter)应用它，往往会产生可由**Haar随机态**很好地近似的量子态。这些态本质上在希尔伯特空间中均匀分布。如果成本函数涉及测量一个全局可观测值 $O$，使得 $\text{Tr}(O) = 0$，对于Haar随机态 $|\psi(\boldsymbol{\theta})\rangle$，期望值 $\langle \psi(\boldsymbol{\theta}) | O | \psi(\boldsymbol{\theta}) \rangle$ 会急剧集中在零附近。因此，梯度分量（它们也依赖于相关算符的期望值）也会集中在零附近，导致方差消失。

> 全局成本函数依赖于作用于许多量子比特的可观测值，增加了对荒原高原的敏感性。局部成本函数则涉及作用于少量量子比特的可观测值的和。

### 噪声引起的荒原高原 (NIBP)

量子硬件本身就有噪声。噪声过程，特别是影响所有量子比特的全局噪声，如**去极化噪声**，也会引发荒原高原，即使对于那些原本可能可训练的相对较浅的电路也是如此。

全局去极化噪声有效地将输出态与最大混合态 $I/2^n$ 混合。随着噪声水平的增加或电路深度的增加（累积更多噪声），输出态 $\rho(\boldsymbol{\theta})$ 趋近于 $I/2^n$。对于无迹可观测值 $O$，成本函数 $C(\boldsymbol{\theta}) = \text{Tr}(\rho(\boldsymbol{\theta}) O)$ 趋近于 $\text{Tr}( (I/2^n) O ) = (1/2^n) \text{Tr}(O) = 0$。成本函数曲面因噪声而全局平坦化，导致梯度消失，且与特定参数 (parameter) $\boldsymbol{\theta}$ 无关。这对于噪声显著的近期设备尤其不利。

### 表达能力与可训练性之间的权衡

通常存在一种明显的权衡：高度表达能力的 PQC（能够表示希尔伯特空间的大部分内容），当随机初始化时，往往最容易受到荒原高原的影响。相反，限制 PQC 的结构以避免荒原高原，可能会限制其表示所需解态或函数的能力。



![梯度方差随量子比特数衰减](plots/1061-0.json)



> 梯度方差衰减的示意性比较。具有全局成本的深度 PQC 通常表现出指数衰减（红色），而浅层电路或具有局部成本的电路可能显示多项式衰减（蓝色），从而缓解荒原高原问题。

## 缓解策略

尽管荒原高原构成了一个重要挑战，但已提出多种策略来减轻其影响：

1. **使用局部成本函数：** 如前所述，定义为局部可观测值（作用于 $k \ll n$ 个量子比特）之和的成本函数通常表现出梯度方差仅随 $n$ 多项式衰减，即 $O(1/\text{poly}(n))$。与与全局成本函数相关的指数衰减相比，这显著改善了可训练性。对于许多问题，可以使用局部项重新表述目标。
2. **结构化和问题启发式 Ansatz：** 并非使用通用、高度纠缠的 PQC，根据问题的结构或对称性设计 Ansatz 会有帮助。例如，具有特定连接的硬件高效 Ansatz，或受系统物理学启发（如用于量子化学的幺正耦合簇）的 Ansatz，会隐式限制搜索空间，可能避免导致荒原高原的区域。量子卷积神经网络 (neural network) (CNN) (QCNN) 等架构由于其分层结构表现出一定的抗性。
3. **参数 (parameter)初始化策略：**

   - **恒等初始化：** 初始化参数使初始 PQC 接近恒等操作，有时会有帮助，特别是对于浅层电路。
   - **逐层训练：** 逐层训练 PQC 的参数。在训练后续层时冻结前面层的参数。
   - **迁移学习 (transfer learning)/预训练 (pre-training)：** 根据较小或相关问题的解决方案初始化参数。
4. **关联参数：** 通过引入关联或将参数关联起来减少独立参数的数量，有时可以缓解荒原高原问题，尽管这可能也会限制表达能力。
5. **自适应优化方法：** 尽管标准优化器难以应对消失的梯度，但量子自然梯度 (QNG) 等方法考虑了量子态空间的几何结构。QNG 根据量子费舍尔信息度量重新缩放梯度，即使标准梯度很小，也可能允许采取更有效的步骤。然而，QNG 的计算成本更高，并且并未从根本上消除方差消失的问题。
6. **错误缓解 (针对 NIBP)：** 应用量子错误缓解技术（第 7 章讨论）可以减少噪声的影响，从而减轻噪声引起的荒原高原的严重程度。

## 影响与展望

荒原高原的存在对 VQA 的实际应用具有深远影响，特别是在近期、有噪声的量子计算机上。这表明，仅仅通过增加量子比特数和电路深度，使用通用 Ansatz，不太可能产生可扩展的算法。

成功的 VQA 实现可能需要仔细协同设计 PQC 结构、成本函数和初始化策略，通常会加入关于目标问题的领域知识。缓解策略，特别是使用局部成本函数和结构化 Ansatz，是重要的手段。然而，荒原高原仍然是一个活跃的研究课题，克服它们是量子机器学习 (machine learning)在更大问题实例上展示实际优势的重要一步。解决复杂问题所需的表达能力与荒原高原施加的可训练性限制之间的权衡，仍然是 VQA 发展中的一个核心主题。

## 参考资料

- [Barren Plateaus in Quantum Neural Networks](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  奠基性论文，引入了贫瘠高原的概念，并证明了其在具有全局成本函数的深度随机量子电路中梯度方差的指数式消失。
- [Noise-induced barren plateaus in variational quantum algorithms](https://www.nature.com/articles/s41467-021-27045-6) — Samson Wang, Enrico Fontana, M. Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, Patrick J. Coles (2021)
  Journal: Nature Communications; Publisher: Nature Publishing Group; Volume: 12; Pages: 1-11; DOI: [10.1038/s41467-021-27045-6](https://doi.org/10.1038/s41467-021-27045-6)
  讨论了去极化噪声等全局噪声过程如何导致变分量子算法中出现贫瘠高原，即使在浅层量子电路中也是如此，这对近期量子硬件构成了重大挑战。
