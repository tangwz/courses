---
course: "introduction-quantum-computing"
chapter: "basic-quantum-circuits"
lesson: "no-cloning-theorem"
sourceId: 8164
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-5-basic-quantum-circuits/no-cloning-theorem"
title: "不可克隆定理"
description: "为什么任意量子态都无法被复制。"
order: 2
plots: []
sourceHash: "85e2808d885cc86f9baec49710b658dc7fde5358975f85243eab2fe349d13ef5"
sourceCorrections: []
---

在传统软件开发中，复制数据是一项极其普通的操作。当你执行类似 `b = a` 的命令时，计算机会读取存储在内存地址 `a` 中的位序列，并将其副本存入为 `b` 分配的内存中。这种备份、复制和“分发”信息的能力是现代计算逻辑的基石。然而，当我们转到量子系统时，这种直觉会导致错误。

不可克隆定理为量子算法划定了严格的边界。它指出，不可能为任意未知的量子态创建一个独立且完全相同的副本。如果你有一个处于叠加态 $|\psi\rangle$ 的量子比特，不存在任何幺正操作能够将 $|\psi\rangle$ 和一个空白量子比特 $|0\rangle$ 转化为 $|\psi\rangle \otimes |\psi\rangle$。

这一限制并不是更好的硬件就能解决的技术障碍，而是量子力学线性特性的数学必然结果。对于人工智能工程师来说，这意味着“将变量复制到临时存储”等典型模式在量子比特上行不通。你无法保存量子神经网络 (neural network)某一层的状态以便稍后恢复。

### 线性论证

要理解为什么禁止克隆，我们必须观察支配幺正矩阵的线性代数。回顾第 2 章，所有量子门都是幺正算子，而幺正算子必须是线性的。

假设我们有一个“克隆幺正”矩阵，记作 $U$。我们希望这个算子能够接收处于 $|\psi\rangle$ 态的源量子比特和处于 $|0\rangle$ 态的目标量子比特，然后更新目标状态以匹配源状态。

复制基态 $|0\rangle$ 的操作如下：

$U(|0\rangle \otimes |0\rangle) = |0\rangle \otimes |0\rangle$

同样，复制基态 $|1\rangle$ 的操作如下：

$U(|1\rangle \otimes |0\rangle) = |1\rangle \otimes |1\rangle$

到目前为止，该算子对计算基态是有效的。问题出现在我们尝试复制叠加态时。让我们定义一个通用状态 $|\psi\rangle$ 为：

$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$

如果我们的算子 $U$ 是一个真正的克隆器，将其应用于 $|\psi\rangle$ 和一个空白目标，应该得到 $|\psi\rangle$ 的两个独立副本：

$\text{期望结果} = |\psi\rangle \otimes |\psi\rangle$

让我们在数学上展开这个期望结果：

$(\alpha|0\rangle + \beta|1\rangle) \otimes (\alpha|0\rangle + \beta|1\rangle) = \alpha^2|00\rangle + \alpha\beta|01\rangle + \beta\alpha|10\rangle + \beta^2|11\rangle$

然而，量子力学规定算子必须是线性的。线性意味着 $U$ 分别作用于叠加态的各个分量。我们只需将之前定义的基态操作结果相加即可：

$\text{线性结果} = U(\alpha|0\rangle \otimes |0\rangle + \beta|1\rangle \otimes |0\rangle)$

$\text{线性结果} = \alpha U(|00\rangle) + \beta U(|10\rangle)$

$\text{线性结果} = \alpha|00\rangle + \beta|11\rangle$

比较“期望结果”和“线性结果”可以看出矛盾所在。期望的克隆输出包含代表两个量子比特混合概率的交叉项（$|01\rangle$ 和 $|10\rangle$）。而物理规律所允许的实际结果仅包含纠缠分量 $|00\rangle$ 和 $|11\rangle$。

$\alpha^2|00\rangle + \alpha\beta|01\rangle + \beta\alpha|10\rangle + \beta^2|11\rangle \neq \alpha|00\rangle + \beta|11\rangle$

因为这两个表达式不相等（除非 $\alpha$ 或 $\beta$ 为 0），所以通用的克隆机不存在。

> 线性数学要求与克隆逻辑要求之间的分歧。

### 对电路设计的影响

与传统电路相比，该定理从根本上改变了我们设计量子电路的方式。

**1. 无扇出**
在传统逻辑门中，一根导线可以分成两根，同时将相同的电信号发送给一个“与”门和一个“或”门。在量子电路中，你不能拆分携带未知状态的线路。量子电路中的示意线代表特定量子比特的时间线，而不是可复制电流的流动。

**2. 破坏性测量**
由于我们无法复制状态，因此在测量之前无法对其进行备份。正如第 3 章所述，测量会使波函数塌缩。如果你为了调试电路而测量量子比特，就会破坏你原本想要检查的叠加态。这使得调试量子算法比在 Python 中检查变量值要困难得多。

**3. 纠错**
传统纠错通常依赖于冗余，即保存一个位的三个副本并检查是否有一个发生了翻转（多数投票）。由于我们无法通过克隆量子比特来产生冗余，因此量子纠错需要更复杂的方法，通过纠缠来保护信息，而不是直接复制信息。

### 复制与移动

虽然我们不能 *复制* 量子态，但我们可以 *移动* 它。不可克隆定理禁止重复，但它允许将信息从一个量子比特传输到另一个量子比特，前提是原始状态在此过程中被销毁或重置。

这种总信息的守恒与幺正矩阵的可逆性质是一致的。如果我们可以将 $|\psi\rangle$ 克隆为 $|\psi\rangle|\psi\rangle$，我们就是在凭空创造信息，这违反了幺正性原理。

在接下来的“隐形传态”和“超密集编码”部分中，你将看到我们如何应对这一限制。我们不会使用 `copy()` 函数，而是利用纠缠作为传输状态的桥梁。原始量子比特将失去其状态信息，而目标量子比特将重构该信息，严格遵守此处推导出的定律。
