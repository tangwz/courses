---
course: "introduction-quantum-computing"
chapter: "multiple-qubits-entanglement"
lesson: "bell-states"
sourceId: 8161
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-4-multiple-qubits-entanglement/bell-states"
title: "贝尔态"
description: "四种特定的最大量子纠缠态。"
order: 4
plots: ["plots/8161-0.json"]
sourceHash: "28247925f506d7015adb2aecaff343f2fe6388954286cfaba0cb8e37e05d82c9"
sourceCorrections: []
---

将单量子比特的叠加态与 CNOT 门的条件逻辑结合，会产生一种全新的状态。CNOT 门的条件逻辑通常作用于计算基态（如 $|00\rangle$ 或 $|10\rangle$），从而确定性地翻转比特。然而，如果在应用 CNOT 之前，控制量子比特处于不确定状态，就会产生量子纠缠。

这一过程产生的一系列特定状态被称为贝尔态（Bell states）。它们是纠缠最简单的例子，也是量子隐形传态和超稠密编码等许多量子协议的标准资源。贝尔态共有四个，代表了两量子比特系统中的四个最大纠缠态。

### 构建电路

要创建一个贝尔态，需要一个包含两个特定组件的电路：阿达马门 (H) 和受控非门 (CNOT)。

1. **准备**：首先将两个量子比特初始化为零态：$|00\rangle$。
2. **叠加**：对第一个量子比特（控制比特）应用阿达马门。这使其处于 $|0\rangle$ 和 $|1\rangle$ 的叠加态。
3. **纠缠**：应用 CNOT 门，将第一个量子比特作为控制位，第二个量子比特作为目标位。

下图展示了该电路的流程。

> 生成贝尔态的电路逻辑。上方线路代表控制量子比特，下方线路代表目标量子比特。

### 数学推导

让我们逐步通过数学方式分析该电路，以证明量子比特产生了纠缠。我们使用标准的张量积符号，其中 $|ab\rangle = |a\rangle \otimes |b\rangle$。

**第 1 步：初始化**
初始时，两个量子比特都处于基态。
$|\psi_0\rangle = |0\rangle \otimes |0\rangle = |00\rangle$

**第 2 步：叠加**
对第一个量子比特应用阿达马门。已知 $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$。第二个量子比特保持不变。
$|\psi_1\rangle = (H \otimes I) |00\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) \otimes |0\rangle$

展开张量积，得到：
$|\psi_1\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$

此时，系统处于 $|00\rangle$ 和 $|10\rangle$ 的等权叠加态。第一个量子比特是随机的（出现 0 或 1 的概率各占 50%），而第二个量子比特确定为 0。它们尚未产生纠缠。

**第 3 步：CNOT 操作**
现在应用 CNOT 门。第一个量子比特是控制位。

- 对于 $|00\rangle$ 项，控制位为 0，因此目标位保持为 0。结果：$|00\rangle$。
- 对于 $|10\rangle$ 项，控制位为 1，因此目标位从 0 翻转为 1。结果：$|11\rangle$。

结合这些结果，得到最终的状态向量 (vector)：
$|\text{贝尔态}\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$

### 分析结果

这种特定的状态记作 $|\Phi^+\rangle$ (Phi-plus)。观察最终等式：
$|\Phi^+\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle$

这描述了一个系统，在测量时，有 50% 的概率塌缩为 $|00\rangle$，有 50% 的概率塌缩为 $|11\rangle$。

该状态的特征是两个量子比特之间存在完全的相关性。如果测量第一个量子比特并发现它是 0，状态就会塌缩为 $|00\rangle$，这意味着第二个量子比特必然立即变为 0。反之，如果你测得 1，第二个量子比特必然为 1。系统不可能处于 $|01\rangle$ 或 $|10\rangle$ 状态。

我们可以将此状态的概率分布与随机非纠缠态进行比较。



![Phi-Plus 态的测量概率](plots/8161-0.json)



> 测量 $|\Phi^+\rangle$ 贝尔态时的概率结果。注意，像 |01> 和 |10> 这样的混合结果是不可能出现的。

无论量子比特之间的物理距离有多远，这种相关性都保持不变。一旦产生纠缠，它们就表现为一个单一的数学对象。如果不提及另一个量子比特，你就无法描述其中一个量子比特的状态。如果你尝试将 $|\Phi^+\rangle$ 分解为乘积形式 $(a|0\rangle + b|1\rangle) \otimes (c|0\rangle + d|1\rangle)$，你会发现不存在满足该等式的系数 $a, b, c, d$。这种数学上的不可分性正是纠缠的定义。

### 四个贝尔态

根据应用阿达马门和 CNOT 门之前两个量子比特的初始状态，可以生成四种不同的最大纠缠态。这些构成了贝尔基（Bell Basis）。

如果我们把输入从 $|00\rangle$ 更改为其他基态（$|01\rangle, |10\rangle, |11\rangle$），生成的贝尔态会在相位符号或比特翻转上有所不同。

1. **输入 $|00\rangle \rightarrow |\Phi^+\rangle$**：
   $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$
   *结果：* 正相关（00 或 11），相位相同。
2. **输入 $|01\rangle \rightarrow |\Psi^+\rangle$**：
   我们从 $|01\rangle$ 开始。对第一个量子比特进行阿达马变换得到 $\frac{|01\rangle + |11\rangle}{\sqrt{2}}$。CNOT 门（控制位为 1 时将目标位 1 翻转为 0）导致：
   $|\Psi^+\rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}$
   *结果：* 负相关（01 或 10）。如果量子比特 A 是 0，则量子比特 B 是 1。
3. **输入 $|10\rangle \rightarrow |\Phi^-\rangle$**：
   我们从 $|10\rangle$ 开始。阿达马门产生相位差：$\frac{|00\rangle - |10\rangle}{\sqrt{2}}$。CNOT 门导致：
   $|\Phi^-\rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}$
   *结果：* 正相关（00 或 11），但带有相位差。在标准测量下，概率看起来与 $\Phi^+$ 相同，但相位会影响后续操作中的干涉模式。
4. **输入 $|11\rangle \rightarrow |\Psi^-\rangle$**：
   我们从 $|11\rangle$ 开始。结果为：
   $|\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}$
   *结果：* 负相关且带有相位差。

在 Qiskit 等 Python 框架中，生成这些状态是一项常规练习。你只需将量子比特初始化为所需的起始状态（必要时使用 X 门将 $|0\rangle$ 翻转为 $|1\rangle$），应用 H 门，然后应用 CX (CNOT) 门。我们将在本节说明后的练习部分直接实现这一点。
