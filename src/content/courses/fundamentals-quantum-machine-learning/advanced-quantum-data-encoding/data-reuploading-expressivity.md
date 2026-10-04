---
course: "fundamentals-quantum-machine-learning"
chapter: "advanced-quantum-data-encoding"
lesson: "data-reuploading-expressivity"
sourceId: 1042
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding/data-reuploading-expressivity"
title: "数据重上传技术：提升表达能力"
description: "分析数据重上传电路设计，以增强参数化量子模型的表达能力。"
order: 3
plots: []
sourceHash: "5b24914cd2131173299dca484e8b4189f20c5ba64e49f607026b262995cb3937"
sourceCorrections: []
---

在上一节中，我们考察了量子特征映射的数学基础。经典数据 $x$ 通过编码电路 $U(x)$ 映射到量子态 $|\phi(x)\rangle = U(x)|\psi_0\rangle$。通常，这种编码只进行一次，常常在量子电路的起始阶段，之后是一个参数 (parameter)化处理单元 $W(\theta)$。尽管这种“单次上传”策略直接明了，但有时会限制您的量子模型可学习函数的复杂度。模型的表达能力，即它能表示的函数范围和丰富程度，受限于这种初始的、固定的映射与变分电路的结合。

为了构建可能更强大的量子机器学习 (machine learning)模型，使其能够捕获数据中复杂的模式，我们可以采用**数据重上传**。我们不是只编码一次数据，而是将数据编码层与参数化量子门层交错排列。

### 数据重上传电路的结构

设想将量子电路 $W(\theta)$ 的变分部分分解为更小的块或层，例如 $W_1(\theta_1), W_2(\theta_2), \dots, W_L(\theta_L)$。在数据重上传方案中，经典输入数据 $x$ 会在这些变分块的每一个（或某些）之前被*重复*编码。

一般结构如下：


$$
|\psi(x, \theta)\rangle = W_L(\theta_L) U(x) W_{L-1}(\theta_{L-1}) U(x) \dots W_1(\theta_1) U(x) |\psi_0\rangle
$$


此处：

- $|\psi_0\rangle$ 是初始态（例如 $|0\dots 0\rangle$）。
- $U(x)$ 是数据编码电路，将经典数据 $x$ 映射为作用于量子位的操作。这可以是任何标准编码，例如角度编码，但重要的一点是，它被应用了多次。
- $W_i(\theta_i)$ 是具有可训练参数 (parameter) $\theta_i$ 的参数化量子门层（例如旋转门、CNOT门）。
- $L$ 是层数或重上传步骤数。

主要思想是，后面层 $W_i$ 中的参数 $\theta_i$ 现在作用于一个量子态，该量子态已经受到*之前*参数 $\theta_{j<i}$ *和*数据编码 $U(x)$ 多次应用的影响。

> 数据重上传电路的示意图。经典数据 $x$ 通过 $U(x)$ 多次编码，并与参数化处理层 $W(\theta_i)$ 交错排列。

### 增强表达能力

这为何有用？思考最终量子态是如何构建的。$U(x)$ 的每次应用都可能以依赖于输入 $x$ 的方式旋转或修改希尔伯特空间中的态矢量。参数 (parameter)化层 $W(\theta_i)$ 随后处理该态。当 $U(x)$ 再次被应用时，它作用于这个*已处理过的*态。

这种交错排列使得在量子态幅值内创建关于输入数据 $x$ 的更复杂函数成为可能。如果 $U(x)$ 和 $W(\theta)$ 涉及非对易门（通常情况下确实如此），那么顺序很重要，重复作用会产生 $\theta$ 和 $x$ 之间复杂的依赖关系。

设想一个简单类比：在经典神经网络 (neural network)中，重复应用非线性激活函数 (activation function)使得网络能够学习复杂的非线性决策边界。数据重上传可被视为在量子领域提供了一种类似的机制。每个 $U(x)$ 层都会再次引入输入数据的影响，使得 $W(\theta_i)$ 层能够逐步建立与 $x$ 相关的更复杂关联和特征。研究表明，这类分类器原则上可以近似输入数据的任何连续函数，与经典网络的通用近似定理类似。

### 益处与考量

数据重上传的主要益处在于其增加**表达能力**的潜力。这意味着模型可能能够：

- 表示更复杂的函数或决策边界。
- 相较于尝试学习相同功能的单次上传模型，可能以更少的量子位达到良好性能。
- 创建在经典计算中可能难以复制的有效特征映射。

然而，也存在重要的设计选择和潜在挑战：

- **电路设计：** 多少个重上传层 $L$ 最优？$U(x)$ 和 $W(\theta_i)$ 应包含哪些特定门？这些选择严重依赖于具体问题和可用的量子硬件。没有放之四海而皆准的答案。
- **可训练性：** 尽管表达能力增加，训练的复杂度也可能增加。深度重上传电路可能更容易受到优化困难的影响，例如贫瘠高原（我们将在第4章详细讨论）。梯度可能消失，使得训练非常缓慢或不可能。
- **过拟合 (overfitting)：** 高度表达能力模型有时可能过拟合训练数据，无法泛化到新的、未见的数据。正则化 (regularization)技术可能变得有必要。
- **硬件资源：** 每个重上传步骤都会增加电路深度，从而增加计算时间以及当前量子设备对噪声的敏感性。设计硬件高效的 $U(x)$ 和 $W(\theta_i)$ 层非常重要（将在第7章进一步介绍）。

### 在量子模型中的作用

数据重上传不仅仅是一种编码技巧；它是构建某些类型QML模型的根本组成部分，尤其是**量子神经网络 (neural network)（QNNs）**和**变分量子分类器（VQCs）**。这种结构自然地适合分层解释，类似于经典深度学习 (deep learning)架构。当我们讨论变分量子算法（VQA，第4章）和量子神经网络（QNNs，第5章）时，将再次看到这些结构。

总之，数据重上传通过在参数 (parameter)化量子电路中重复编码经典数据，提供了一种增强量子机器学习 (machine learning)模型表达能力的强大技术。这使得模型能够学习输入数据与所需输出之间更复杂的关系，可能在具有挑战性的任务上带来更好的性能，前提是关于电路设计、可训练性和硬件限制的考量得到仔细管理。接下来的章节将更详细地讨论通过数据重上传等技术创建的这些更丰富特征映射的属性。

## 参考资料

- [Data re-uploading for a universal quantum classifier](https://iopscience.iop.org/article/10.1088/2058-9565/ab851a) — D. Pérez-Salinas, A. F. Bravo-Pérez, J. K. Ochoa, and J. K. Salinas (2020)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 5; Pages: 045003; DOI: [10.1088/2058-9565/ab851a](https://doi.org/10.1088/2058-9565/ab851a)
  介绍了用于量子分类器的数据重上传电路概念，并展示了其通用近似能力。
- [The power of data re-uploading in quantum machine learning](https://link.springer.com/article/10.1007/s42484-020-00039-4) — Maria Schuld, Ryan Sweke, Johannes J. Meyer (2021)
  Journal: Quantum Machine Intelligence; Publisher: Springer Science and Business Media LLC; Volume: 3; Pages: 36-42; DOI: [10.1007/s42484-020-00039-4](https://doi.org/10.1007/s42484-020-00039-4)
  探讨了数据重上传在增强量子模型表达能力和学习能力方面的理论方面和优势。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  一篇识别了荒漠高原问题的基础论文，这是训练深度量子电路（如数据重上传中使用的电路）的关键挑战。
- [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://doi.org/10.1002/qute.201900070) — Sukin Sim, Peter D. Johnson, and Alán Aspuru-Guzik (2019)
  Journal: Advanced Quantum Technologies; Publisher: John Wiley & Sons Inc.; Volume: 2; Pages: 1900070; DOI: [10.1002/qute.201900070](https://doi.org/10.1002/qute.201900070)
  分析了参数化量子电路的设计（包括深度和纠缠层）如何影响其表达能力和表示能力。
