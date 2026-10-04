---
course: "fundamentals-quantum-machine-learning"
chapter: "hardware-error-mitigation-qml"
lesson: "characterizing-quantum-hardware-noise"
sourceId: 1078
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml/characterizing-quantum-hardware-noise"
title: "表征量子硬件噪声"
description: "理解当前量子硬件中各类噪声，例如退相干、门错误和读出错误。"
order: 1
plots: ["plots/1078-0.json"]
sourceHash: "375dc5ae50f217e0515bdff5ed5c29a0d7fd6a34797f0f8d1999ba757d59972e"
sourceCorrections: []
---

有效运行量子机器学习 (machine learning)算法，需要面对当前量子硬件的噪声现实。与教科书中描述的理想操作不同，物理量子处理器容易受到各种缺陷影响，从而降低计算准确性。在减轻这些影响之前，我们首先需要理解并量化 (quantization)它们。这个过程被称为噪声表征。

可以将其想象成调试经典代码。你不会在不清楚问题所在的情况下尝试修复错误。同样，在NISQ设备上的量子机器学习中，表征影响量子比特和门的噪声是构建应用的基础一步。

### 主要量子噪声来源

硬件噪声并非单一的巨大问题；它以几种不同的方式出现。理解这些类别有助于查明问题并选择合适的缓解策略。

#### 退相干：量子特性的丧失

量子态以其脆弱性而闻名。它们与周围环境的作用导致其失去独特的量子属性，如叠加和纠缠，最终行为更像经典比特。这个过程被称为退相干。有两个主要的时间尺度来表征退相干：

1. **$T_1$ 时间（幅度衰减）：** 这表示量子比特从激发态 $|1\rangle$ 因能量弛豫而衰减到基态 $|0\rangle$ 所需的特征时间，通常通过向环境发射能量实现。如果你的计算时间超过 $T_1$，存储在量子态幅度中的信息很可能会被破坏。在数学上，这个过程使量子比特的状态进行非幺正演化。
2. **$T_2$ 时间（相位衰减或退相位）：** 它衡量的是叠加态（$\alpha|0\rangle + \beta|1\rangle$）中 $|0\rangle$ 和 $|1\rangle$ 分量之间的相对相位衰减所需的时间。量子比特能级的随机波动，通常由环境影响引起，导致这种相位相干性的丧失。退相位会破坏叠加，而叠加是许多量子机器学习 (machine learning)算法的资源。总退相位时间 $T_2$ 总是小于或等于弛豫时间的两倍（$T_2 \le 2T_1$）。关系式 $1/T_2 = 1/(2T_1) + 1/T_{\phi}$ 将 $T_1$、$T_2$ 和“纯”退相位时间 $T_{\phi}$ 联系起来，$T_{\phi}$ 表示无能量损失的相位随机化。

对于量子机器学习，退相干意味着代表数据或模型参数 (parameter)的量子态在电路执行期间会逐渐衰减，从而限制了可以可靠运行的算法的复杂性和深度。

#### 门错误：不完美的操作

量子门通过施加精心控制的物理作用（如微波脉冲或激光束）来实现。这些控制机制永远不会完美精确。门错误发生在实际执行的操作偏离预期理想幺正变换时。

- **系统误差：** 这些是持续存在的偏差，例如量子比特绕某个轴轻微过旋或欠旋（例如，应用 $R_X(\theta + \epsilon)$ 门而不是 $R_X(\theta)$）。
- **随机误差：** 这些源于门操作期间控制场或环境参数的随机波动，导致每次应用门时实现的操作都有所不同。

门错误通常用其**保真度**来量化 (quantization)，保真度衡量实际操作与理想操作的接近程度。当前硬件上典型的单量子比特门保真度可能是99.9%，而涉及量子比特之间作用的双量子比特门（如CNOT），通常噪声更大，保真度可能在99%左右或略低。这些错误会累积；一个包含许多门的电路将比门数较少的电路可靠性低得多。

#### 读出错误：结果误判

量子计算完成后，我们测量量子比特以获得经典结果。这个测量过程本身就是不完美的。当测量设备错误地判定量子比特的状态时，就会发生**读出错误**（也称为SPAM错误，代表状态准备和测量）。例如，一个实际处于状态 $|1\rangle$ 的量子比特可能被报告为处于状态 $|0\rangle$，反之亦然。

这些错误可以通过混淆矩阵来表征，该矩阵详细说明了 $P(\text{测量到 } i | \text{实际为 } j)$ 的概率，其中 $i, j \in \{0, 1\}$。读出错误直接影响从测量统计数据中计算出的期望值，这些期望值常用于定义变分量子算法（VQA）和量子神经网络 (neural network)（QNN）中的成本函数。

#### 串扰：不必要的作用

当操作特定量子比特时（例如，施加一个门或执行一次测量），控制信号有时会无意中影响相邻量子比特。这种不必要的影响被称为**串扰**。它会在多个量子比特之间引入相关错误，使得噪声结构比每个量子比特上的独立错误更复杂。在密集连接的量子比特阵列中，或当多个门同时应用以加速计算时，串扰变得尤其棘手。



![NISQ设备中的典型错误率（示意性）](plots/1078-0.json)



> 当前量子硬件上不同操作的示意性错误率。请注意，双量子比特门和读出通常比单量子比特门噪声大得多。实际值在不同设备之间差异很大。

### 如何测量噪声：表征协议

量子硬件提供商和研究人员使用几种标准技术来量化 (quantization)这些不同的噪声源。虽然查看实现细节超出了我们目前的范围，但理解这些方法的目的是很重要的：

- **$T_1$/$T_2$ 测量：** 特定的脉冲序列（如用于 $T_1$ 的反转恢复，以及用于 $T_2$ 的拉姆齐或哈恩回波序列）用于直接探测和测量每个量子比特的特征退相干时间。
- **随机基准测试（RB）：** 这是一种广泛使用的技术，用于估计一组门（通常是Clifford门）的*平均*保真度。它涉及运行一系列随机门，这些门在理想情况下应组合成恒等操作。通过测量随着序列长度增加，返回初始状态的概率如何衰减，可以提取出平均门错误率。交错RB将其扩展到通过在随机序列中交错特定门来估计其保真度。RB是可扩展的，并提供了一个良好的基准数据，但不能提供关于单个门错误的详细信息。
- **量子过程层析成像（QPT）：** 这旨在完全重建描述有噪声量子操作（例如，一个特定门）的数学映射（通常由过程矩阵或Kraus算符表示）。它需要准备一组输入态，应用操作，并对输出态进行量子态层析成像。虽然它提供了噪声过程的完整描述，但QPT是资源密集型的，其计算量随所涉及量子比特的数量呈指数级增长。
- **门集层析成像（GST）：** GST是一种更先进的技术，它克服了QPT和RB的一些局限。它能够自洽地表征给定集合内*所有*门操作，以及状态准备和测量（SPAM）错误，提供高度准确、详细的错误模型。然而，它的实现也很复杂，并且计算量大。
- **读出错误表征：** 这通常通过将量子比特准备到已知状态（$|0\rangle$ 和 $|1\rangle$），并多次测量它们来构建前面提到的混淆矩阵。

硬件提供商通常会报告其设备的一些这些指标（如 $T_1$、$T_2$、来自RB的平均门保真度、读出错误）。

### 用于仿真的噪声模型

一旦噪声被表征，就可以将其纳入量子计算机的经典模拟器中。这使得量子机器学习 (machine learning)研究人员可以估算他们的算法在真实硬件上的表现，而无需直接访问或为每次实验消耗宝贵的量子计算时间。

常用的噪声模型包括：

- **去极化信道：** 一个简单的模型，其中在理想门操作之后，量子比特状态有一定概率 $p$ 被完全混合态（按 $1/2$ 缩放的单位矩阵）替换，从而有效丢失所有信息。
- **幅度衰减信道：** 模拟 $T_1$ 衰减过程。
- **相位衰减信道：** 模拟 $T_2$ 退相位过程。
- **组合信道：** 模拟器通常允许组合这些基本信道，以近似真实设备中观察到的复杂噪声。

这些噪声模型通常在模拟电路中每次理想门操作后以概率方式应用。使用现实噪声模型运行量子机器学习算法模拟，对于评估性能下降和测试错误缓解技术的有效性，在实际量子处理器上部署之前非常重要。

### 为什么表征对量子机器学习 (machine learning)很重要

理解你打算使用的硬件的特定噪声特性不仅仅是学术上的，它直接影响量子机器学习的开发：

- **算法设计：** 了解主要的错误来源（例如，相干时间短与CNOT错误率高）可以影响量子机器学习模型的选择（例如，如果 $T_2$ 或CNOT保真度差，则偏好较浅的电路或较少依赖纠缠的模型）。
- **Ansatz选择：** 硬件高效的Ansatz（本章稍后讨论）是专门设计用来最小化在给定架构上受噪声影响最大的资源（门、深度）。
- **成本函数评估：** 读出错误系统地偏差了期望值的估计，而期望值是训练VQA和QNN所用成本函数的核心。表征这些错误允许在后处理期间对其进行校正。
- **梯度计算：** 噪声影响梯度的估计（例如，使用参数 (parameter)偏移规则），可能阻碍优化算法的收敛。
- **错误缓解策略：** 选择应用哪种错误缓解技术（接下来介绍）通常取决于表征期间识别出的噪声的性质和大小。

本质上，表征硬件噪声提供了必要的诊断信息，以弥合理论量子机器学习算法设计与在不完美NISQ设备上实际实现之间的差距。有了这份理解，我们现在可以研究旨在减少这些错误影响的技术。

## 参考资料

- [Quantum Computation and Quantum Information](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGoarRXCKQRCrmYKOm0OiEGs06RxA-Vgubc07J5e-28POIJuvoVTuaG4AaA-wW7dYIEmPJTSLyorCP5e_7Lzjyzvs74z_m8wTNSAvG4r2DkAFRBEFaC4md1l9mGygLsECWly_9KrcbS-HVN36_7DJOyE2En7-OicvN5YKeaKvG9k-kz1qk0bd7zjdKaRsy_4c49XMtJgYHj9f0TbMqqf-aRz10) — Michael A. Nielsen, Isaac L. Chuang (2010)
  Publisher: Cambridge University Press; Pages: Chapter 8
  一本关于量子力学和量子信息的经典教材，全面涵盖量子噪声、退相干和基本错误模型等基础概念。
- [Randomized benchmarking of quantum gates](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.77.012307) — E. Knill, D. Leibfried, R. Reichle, J. Britton, R. B. Blakestad, J. D. Jost, C. Langer, R. Ozeri, S. Seidelin, and D. J. Wineland (2008)
  Journal: Physical Review A; Publisher: American Physical Society; Volume: 77; Pages: 012307; DOI: [10.1103/PhysRevA.77.012307](https://doi.org/10.1103/PhysRevA.77.012307)
  一篇介绍随机基准测试的奠基性论文，这是一种用于评估量子门平均保真度的可扩展协议，对表征硬件性能至关重要。
- [Robust, self-consistent, closed-form characterization of quantum devices by Gate Set Tomography](https://iopscience.iop.org/article/10.1088/1367-2630/aa65f6) — Robin Blume-Kohout, John King Gamble, Erik Nielsen, Jonathan Maunz, Nick DeMarco, Kevin Fortier, Clinton Tezak, and David Wineland (2017)
  Journal: New Journal of Physics; Volume: 19; Pages: 043004; DOI: [10.1088/1367-2630/aa65f6](https://doi.org/10.1088/1367-2630/aa65f6)
  一篇详细介绍门集层析成像 (GST) 的重要出版物，这是一种同时表征量子门、态制备和测量误差的先进且高精度方法。
- [Quantum Computing in the NISQ era and beyond](https://quantum-journal.org/papers/q-2018-08-06-79/) — John Preskill (2018)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 2; Pages: 79; DOI: [10.22331/q-2018-08-06-79](https://doi.org/10.22331/q-2018-08-06-79)
  一篇极具影响力的综述文章，定义了NISQ时代，并强调了当前量子硬件中噪声和错误带来的挑战，为表征的必要性提供了背景。
