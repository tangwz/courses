---
course: "fundamentals-quantum-machine-learning"
chapter: "advanced-quantum-data-encoding"
lesson: "practical-custom-quantum-feature-maps"
sourceId: 1046
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding/practical-custom-quantum-feature-maps"
title: "动手实践：实现定制量子特征映射"
description: "使用量子计算框架实现并尝试不同的定制量子特征映射设计。"
order: 7
plots: []
sourceHash: "667bffd89bc544f634267a0f99d149124ff68efb4531cdf1991c0918300e4baf"
sourceCorrections: []
---

理论是基础，而实现量子特征映射有助于巩固认识，并为构建实际的量子机器学习 (machine learning)模型做好准备。在前面的章节中，我们考察了高阶多项式映射和数据重上传等各种编码策略的数学结构和属性。现在，我们将这些构想转化为可执行代码，使用一个量子计算库。我们将选用PennyLane，因为它能与经典机器学习工具顺利集成，但这些原理同样适用于Qiskit等其他框架。

我们的目标是创建可复用的函数或模板，它们以经典数据向量 (vector)为输入，并生成对应的量子态 $|\phi(x)\rangle$。这种实际操作经验将使您能够尝试不同的电路结构，并了解它们的具体实现细节。

### 环境配置

首先，请确保您已安装PennyLane和NumPy。我们将从必要的导入开始：

```python
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 为我们的例子设置一个默认设备
# 使用 'default.qubit' 进行模拟。之后您可能会选择其他设备。
num_qubits = 2 # 例子中的默认量子比特数量
dev = qml.device('default.qubit', wires=num_qubits)

print(f"Using PennyLane version: {qml.__version__}")
print(f"Default device set to: {dev.name} with {num_qubits} qubits.")
```

我们定义了一个默认模拟器设备 `default.qubit`。所需的量子比特数量（在PennyLane中为 `wires`）通常取决于输入数据的维度和所选的编码策略。

### 基本角度编码特征映射

最简单的编码策略是将经典数据特征直接映射到单量子比特门的旋转角度。我们来构建一个特征映射，其中每个特征 $x_i$ 控制对应量子比特上的 $RY$ 旋转。

```python
def simple_angle_encoding_map(x):
    """使用RY旋转将N个数据特征编码到N个量子比特中。"""
    if len(x) > dev.num_wires:
        raise ValueError(f"Input vector dimension ({len(x)}) exceeds number of qubits ({dev.num_wires}).")
    
    # 应用由输入特征控制的RY旋转
    for i in range(len(x)):
        qml.RY(x[i], wires=i)

# 用例：定义使用此特征映射的QNode
@qml.qnode(dev)
def simple_angle_circuit(x):
    simple_angle_encoding_map(x)
    # 通常之后会进行测量或进一步处理
    # 为了演示，我们返回态向量
    return qml.state()

# 尝试编码一个样本数据点
sample_data_point = np.array([0.5, -1.2]) # 2个量子比特的例子
encoded_state = simple_angle_circuit(sample_data_point)

print("Example usage of Simple Angle Encoding:")
print(f"Input data: {sample_data_point}")
# print(f"Encoded state vector (showing first 4 components): \n{encoded_state[:4]}") # 态向量可能很大
qml.draw_mpl(simple_angle_circuit, style='pennylane')(sample_data_point)
plt.show()
```

`simple_angle_encoding_map` 函数接收经典向量 (vector) `x` 并应用旋转。`@qml.qnode` 装饰器将描述量子电路的Python函数转变为链接到我们设备 `dev` 的可执行量子节点。这种基本编码对于复杂任务而言通常不足，因为它不会引入特征之间的纠缠或高阶关联。

### 实现ZZ特征映射

为了捕获输入特征之间的关联，我们需要纠缠门。ZZ特征映射常在二阶多项式核的背景下被提及，它在数据编码旋转后使用受控相位 ($CZ$) 或等效的基于 $CNOT$ 的门。它可以近似包含 $x_i x_j$ 等项的核函数。

我们来构建一个使用 $RX$ 编码，然后是 $CZ$ 纠缠门的版本。

```python
def zz_feature_map(x, num_layers=1):
    """实现ZZ型特征映射。
    
    参数:
        x (np.ndarray): 输入数据向量。维度必须与量子比特数量匹配。
        num_layers (int): 重复编码-纠缠块的次数。
    """
    num_qubits_map = len(x)
    if num_qubits_map != dev.num_wires:
         raise ValueError(f"Input vector dimension ({num_qubits_map}) must match number of qubits ({dev.num_wires}).")

    for _ in range(num_layers):
        # 数据编码层
        for i in range(num_qubits_map):
            qml.RX(x[i], wires=i) # 这里使用RX编码
        
        # 纠缠层 (例子：线性纠缠)
        for i in range(num_qubits_map - 1):
            qml.CZ(wires=[i, i+1])
        # 可选：在最后一个和第一个量子比特之间添加纠缠，实现循环边界
        # if num_qubits_map > 1:
        #    qml.CZ(wires=[num_qubits_map - 1, 0])

@qml.qnode(dev)
def zz_map_circuit(x, num_layers=1):
    # 通常以Hadamard门开始以创建叠加态
    for i in range(dev.num_wires):
        qml.Hadamard(wires=i)
        
    zz_feature_map(x, num_layers=num_layers)
    
    # 返回态用于分析，或添加测量用于核估计
    return qml.state()

# 用例
sample_data_point_zz = np.array([np.pi/4, np.pi/2]) # 2个量子比特的例子
num_encoding_layers = 2
encoded_state_zz = zz_map_circuit(sample_data_point_zz, num_layers=num_encoding_layers)

print("\nExample usage of ZZ Feature Map:")
print(f"Input data: {sample_data_point_zz}")
print(f"Number of layers: {num_encoding_layers}")
# print(f"Encoded state vector (showing first 4 components): \n{encoded_state_zz[:4]}")
qml.draw_mpl(zz_map_circuit, style='pennylane')(sample_data_point_zz, num_layers=num_encoding_layers)
plt.show()
```

在此例子中，`zz_feature_map` 首先应用单量子比特旋转，使用 $RX$ 门编码数据特征 $x_i$。然后，它在相邻量子比特之间应用 $CZ$ 门以引入纠缠。最初的Hadamard层创建了叠加基，这在许多特征映射构建中都很常见。重复编码和纠缠层（`num_layers > 1`）可以增加映射的复杂性和表现力。

### 实现数据重上传特征映射

如前所述，数据重上传是在参数 (parameter)化量子电路中重复编码数据。这种结构可以显著增强模型近似复杂函数的能力。电路通常在数据编码层和参数化（或固定）变分门层之间交替。

我们来构建一个简单的数据重上传电路，其中数据使用 $RZ$ 旋转编码，然后是固定的 $CX$ 纠缠和参数化的 $RY$ 旋转。

```python
def data_reuploading_map(x, weights, num_layers=1):
    """实现数据重上传特征映射。
    
    参数:
        x (np.ndarray): 输入数据向量。维度应小于或等于量子比特数量。
        weights (np.ndarray): 电路的变分参数。 
                              形状应与基于层数和量子比特数的预期相符。
        num_layers (int): 重上传层的数量。
    """
    num_qubits_map = dev.num_wires
    # 假设weights是一个扁平数组，根据层数和量子比特数进行重塑
    # 例子：每层需要 num_qubits 个 RY 参数
    expected_weights_per_layer = num_qubits_map
    if len(weights) != num_layers * expected_weights_per_layer:
        raise ValueError("Incorrect number of weights provided.")
        
    weights_reshaped = weights.reshape((num_layers, num_qubits_map))

    feature_dim = len(x)
    
    for layer in range(num_layers):
        # 数据编码层 (例子：RZ旋转，如果维度小于量子比特数，则循环使用特征)
        for i in range(num_qubits_map):
            qml.RZ(x[i % feature_dim], wires=i) # 使用模运算实现特征复用
            
        # 变分层 (例子：RY旋转)
        for i in range(num_qubits_map):
            qml.RY(weights_reshaped[layer, i], wires=i)
            
        # 纠缠层 (例子：固定CNOT梯形结构)
        for i in range(num_qubits_map - 1):
            qml.CNOT(wires=[i, i+1])
        # 可选：在最后一个和第一个量子比特之间添加纠缠
        # if num_qubits_map > 1:
        #    qml.CNOT(wires=[num_qubits_map - 1, 0])

@qml.qnode(dev)
def data_reuploading_circuit(x, weights, num_layers=1):
    # 如果编码从|0>态开始，则不需要初始Hadamard门
    data_reuploading_map(x, weights, num_layers=num_layers)
    # 返回态或期望值
    return qml.expval(qml.PauliZ(0)) # 测量例子

# 用例
num_reupload_layers = 3
num_qubits_reupload = 2 # 我们保持2个量子比特
dev.num_wires = num_qubits_reupload # 如果需要，更新设备设置

# 变分层的权重 (3层 * 2量子比特 = 6个权重)
sample_weights = np.random.uniform(0, 2 * np.pi, num_reupload_layers * num_qubits_reupload)
sample_data_point_reupload = np.array([0.2, 0.8]) # 2D数据例子

# 执行电路
output_expval = data_reuploading_circuit(sample_data_point_reupload, sample_weights, num_layers=num_reupload_layers)

print("\nExample usage of Data Re-uploading Feature Map:")
print(f"Input data: {sample_data_point_reupload}")
print(f"Sample weights (first 4): {sample_weights[:4]}...")
print(f"Number of layers: {num_reupload_layers}")
print(f"Output expectation value <Z_0>: {output_expval:.4f}")

qml.draw_mpl(data_reuploading_circuit, style='pennylane')(sample_data_point_reupload, sample_weights, num_layers=num_reupload_layers)
plt.show()

# 如果已更改，恢复默认量子比特数量
dev.num_wires = num_qubits 
```

在此实现中，`data_reuploading_map` 接收数据向量 (vector) `x` 和变分 `weights`。在定义层的循环内部，我们首先应用依赖于数据的 $RZ$ 旋转。请注意 `x[i % feature_dim]` 的使用，它通过复用特征来编码小于量子比特数量的数据向量。之后是一个可训练的 $RY$ 旋转层（由 `weights` 控制），以及一个固定的纠缠结构（$CNOT$ 门）。参数 `weights` 通常在训练更大的QML模型（如VQC）期间进行优化。

### 定制与尝试

上述例子提供了起点。真正的优势在于设计针对特定问题的特征映射，或尝试不同的变体。以下是一些定制的建议：

1. **编码门：** 将 $RX$、$RY$、$RZ$ 替换为其他单量子比特旋转或其组合。您可以将数据编码到相位门（$qml.PhaseShift$）中，或使用像 $qml.U3$ 这样的多参数 (parameter)门。
2. **纠缠策略：** 改变纠缠门的模式。除了线性的 $CZ$ 或 $CNOT$ 链，可以尝试全对全纠缠、循环模式或特定硬件的高效结构。使用不同的双量子比特门，例如 $qml.IsingXX$、$qml.IsingYY$ 或 $qml.IsingZZ$。
3. **层结构：** 改变ZZ映射或数据重上传映射中的层数。尝试编码、变分和纠缠块的不同排序。
4. **混合方法：** 组合不同策略的元素。例如，在某些量子比特上使用角度编码，而在其他量子比特上使用ZZ型作用。
5. **输入预处理：** 在编码前对 `x` 进行经典预处理。这可能包括缩放、归一化 (normalization)或像PCA这样的降维技术，这可能影响量子特征映射的有效性。

这是一个您可以调整的基本模板结构：

```python
def my_custom_feature_map(x, params=None):
    """自定义量子特征映射的模板。"""
    num_qubits_map = dev.num_wires 
    # 检查输入维度一致性
    if len(x) > num_qubits_map: 
         print(f"Warning: Input dimension {len(x)} > qubits {num_qubits_map}. Using first {num_qubits_map} features.")
         x_eff = x[:num_qubits_map]
    else:
         x_eff = x
         # 可选：如果len(x) < num_qubits_map，则用零填充x或复用特征
         
    # --- 自定义电路定义开始 ---
    
    # 例子：第1层 - 基于数据的初始旋转
    for i in range(len(x_eff)):
        qml.RX(x_eff[i] * np.pi, wires=i) # 缩放输入以适应旋转范围
        
    # 例子：第2层 - 固定纠缠
    if num_qubits_map > 1:
        qml.broadcast(qml.CNOT, wires=range(num_qubits_map), pattern="ring")

    # 例子：第3层 - 依赖于数据的受控旋转（如果可训练则需要参数）
    # if params is not None: ...
    for i in range(num_qubits_map - 1):
         # 例子：基于特征乘积的受控旋转
         angle = x_eff[i] * x_eff[i+1] 
         qml.CRZ(angle, wires=[i, i+1])

    # --- 自定义电路定义结束 ---

@qml.qnode(dev)
def custom_circuit(x, params=None):
    my_custom_feature_map(x, params)
    # 返回态或测量可观量
    return qml.probs(wires=range(dev.num_wires))

# 后续将有使用示例...
```

### 分析特征映射输出：量子核

分析特征映射效果的一种方法是计算量子核矩阵。两个数据点 $x_i$ 和 $x_j$ 之间的核元素由其对应量子态的平方重叠给出：$K(x_i, x_j) = |\langle \phi(x_j) | \phi(x_i) \rangle|^2$。此值反映了量子特征空间中数据点之间的相似度。

我们可以使用已实现的特征映射来计算此值。PennyLane提供了计算态重叠或保真度的便捷方法。

```python
# 以ZZ映射电路为例
@qml.qnode(dev)
def zz_map_state_vector(x, num_layers=1):
    # 准备初始态 |0...0>
    # 应用Hadamard层
    for i in range(dev.num_wires):
        qml.Hadamard(wires=i)
    # 应用ZZ特征映射
    zz_feature_map(x, num_layers=num_layers)
    return qml.state()

# 定义两个样本数据点（必须与量子比特数量匹配）
data_point_1 = np.array([0.1, 0.2])
data_point_2 = np.array([0.3, 0.4])
num_layers_kernel = 1

# 生成态向量
state_1 = zz_map_state_vector(data_point_1, num_layers=num_layers_kernel)
state_2 = zz_map_state_vector(data_point_2, num_layers=num_layers_kernel)

# 计算重叠 |<state_2 | state_1>|^2
# 注意：态向量可以是复数，使用共轭转置
overlap = np.abs(np.vdot(state_2, state_1))**2 

print(f"\nQuantum Kernel Element Example (ZZ Map, {num_layers_kernel} layer):")
print(f"Data point 1: {data_point_1}")
print(f"Data point 2: {data_point_2}")
print(f"Kernel value K(x1, x2) = |<phi(x2)|phi(x1)>|^2 = {overlap:.6f}")

# 使用PennyLane的qml.kernels模块（对于完整矩阵更高效）
kernel_func = qml.kernels.EmbeddingKernel(zz_feature_map, dev) # 传递映射*函数*
kernel_value = kernel_func(data_point_1, data_point_2, num_layers=num_layers_kernel)
# 注意：EmbeddingKernel 使用保真度，如果态是纯态，它等于我们上面计算值的平方根。
# 请查看qml.kernels使用的精确定义文档
# 对于纯态|psi>、|phi>，保真度 F = |<psi|phi>|。核通常使用 F^2。
print(f"Kernel value using qml.kernels (may differ definitionally): {kernel_value:.6f}") 
```

计算和分析数据集的核矩阵可以展现特征映射如何改变数据几何结构，有可能使线性不可分的数据在高维希尔伯特空间中变得可分。这对量子核方法（如QSVM）来说是根本的，我们将在下一章进行了解。

### 集成与后续步骤

这里创建的特征映射实现是构建模块。它们可以直接集成到：

1. **量子核估计：** 使用态重叠计算（或专用核函数）来计算QSVM等算法的核矩阵。
2. **变分量子分类器（VQCs）：** 将特征映射与随后的参数 (parameter)化变分电路结合。特征映射准备输入态，而变分电路学习对其进行分类。数据重上传电路通常将这两个方面结合。

本次实践旨在使您具备实现各种量子特征映射的能力。尝试不同的结构、编码门和纠缠模式对开发有效的QML模型具有重要意义。请记住，特征映射的选择可以显著影响性能，而了解如何构建和修改它们是QML开发中的一项宝贵技能。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (2019)
  Journal: Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  介绍量子增强特征空间，包括一种ZZ特征映射，并展示其在基于核的分类中的应用。
- [Data Re-uploading for a Universal Quantum Classifier](https://doi.org/10.22331/q-2020-05-18-261) — Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, José I. Latorre (2020)
  Journal: Quantum; Volume: 4; Pages: 226; DOI: [10.22331/q-2020-05-18-261](https://doi.org/10.22331/q-2020-05-18-261)
  提出了数据重上传技术，展示了单个量子比特如何对复杂模式进行分类，并讨论了其普遍性。
- [Quantum machine learning in feature Hilbert spaces](https://doi.org/10.1103/PhysRevLett.122.040504) — Maria Schuld, Nathan Killoran (2019)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 122; Pages: 040504; DOI: [10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)
  一篇基础论文，通过将经典数据映射到量子特征希尔伯特空间，建立了量子核方法的理论框架。
- [Quantum Feature Maps](https://pennylane.ai/qml/glossary/quantum_feature_map.html) — Xanadu Quantum Technologies (2024)
  PennyLane官方文档，涵盖各种量子特征映射及其使用该库的实现。
