---
course: "introduction-quantum-computing"
chapter: "from-bits-to-qubits"
lesson: "python-environment-setup-practice"
sourceId: 8145
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-1-from-bits-to-qubits/python-environment-setup-practice"
title: "Python 环境搭建实践"
description: "安装和配置 Python 量子库的实操指南。"
order: 5
plots: []
sourceHash: "d7eb36c844ab4c79de95aa6106b9eb03831d2435071bd471efff200aef6f3fbe"
sourceCorrections: []
---

理论为理解量子力学提供了蓝图，但实现则需要软件环境。为了构建量子系统，我们必须将数学模型、向量 (vector)、矩阵和算子转化为可执行的代码。

在本节中，我们将建立一个能够定义量子比特并模拟其行为的本地开发环境。虽然存在真实的量子硬件，但访问通常需要排队或费用昂贵。因此，标准的工作流程是在 Python 中设计电路，并在模拟量子概率的经典模拟器上运行它们。

下图展示了我们将要构建的软件栈。

> 在经典计算机上模拟量子力学所需的工具层级。

### 验证 Python 安装

量子开发工具包高度依赖较新版本的 Python。在安装特定库之前，请确认您的系统运行的是 Python 3.8 或更高版本。这可以确保与类型提示和最新的矩阵计算优化兼容。

打开终端或命令行提示符，执行以下命令：

```bash
python --version
```

如果版本低于 3.8，请更新您的 Python 安装。为了与科学计算包获得最佳兼容性，我们还建议在 Windows 上使用基于 Unix 的终端（Linux 或 macOS）或 PowerShell。

### 创建隔离环境

依赖管理在科学计算中非常关键。量子库更新频繁，经常会引入不兼容的更改。为了防止与其他项目产生冲突，我们创建一个虚拟环境。它作为一个独立的目录，包含特定的 Python 版本和一组库。

进入您想要存放项目的文件夹，运行以下命令。

macOS 和 Linux 用户：

```bash
python -m venv quantum_lab
source quantum_lab/bin/activate
```

Windows 用户：

```bash
python -m venv quantum_lab
.\quantum_lab\Scripts\activate
```

激活后，您的终端提示符应显示 `(quantum_lab)`，表示随后的任何安装都将限制在此环境中。

### 安装核心库

我们需要三类特定的库来完成本课程。

1. **线性代数引擎：** 我们需要一个库来处理复数和矩阵乘法。正如在引言中所述，量子比特状态是一个向量 (vector)。
2. **量子 SDK：** 该库提供了“量子比特”、“门”和“电路”的抽象。
3. **可视化：** 为了渲染布洛赫球和电路图，我们需要绘图工具。

我们将使用 **NumPy** 进行数学运算，并使用 **Qiskit** 作为我们的主要量子 SDK。Qiskit 在业内应用广泛，为学习代码背后的物理原理提供了很好的支持。

运行以下安装命令：

```bash
pip install numpy matplotlib qiskit qiskit-aer pylatexenc
```

- `numpy`：处理向量数学。
- `qiskit`：编写量子电路的框架。
- `qiskit-aer`：高性能模拟器后端。
- `pylatexenc`：帮助渲染高质量电路图的工具。

### 验证工具链

安装完成后，验证这些库是否能正常协同工作非常重要。我们将编写一个简单的 Python 脚本，初始化单个量子比特并检查其属性。这可以确认我们的环境能够处理狄拉克符号部分定义的数学对象。

创建一个名为 `verify_setup.py` 的文件并输入以下代码：

```python
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

# 1. 使用 NumPy 定义一个经典向量
# 这代表状态 |0>
vector_zero = np.array([1, 0])
print(f"|0> 的 NumPy 向量: {vector_zero}")

# 2. 初始化一个包含 1 个量子比特的量子电路
q = QuantumRegister(1, 'q')
circuit = QuantumCircuit(q)

# 3. 查看量子比特的状态
# 在模拟器中，我们可以查看向量而不使其坍缩
state = Statevector.from_instruction(circuit)
print(f"量子态向量: {state.data}")

print("\n环境搭建成功。")
```

在终端运行该脚本：

```bash
python verify_setup.py
```

如果输出显示了 NumPy 向量 (vector)和量子态向量（应为复数，例如 `1.+0.j` 和 `0.+0.j`），则说明您的环境配置正确。

### 可视化量子比特

使用 Python 的一个主要优势是能够将抽象定义可视化。在整个课程中，我们将把状态向量 (vector)映射到布洛赫球上。让我们测试一下环境的可视化功能。

创建一个名为 `test_plot.py` 的文件：

```python
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# 创建一个简单的电路
qc = QuantumCircuit(1)

# 获取状态向量表示
state = Statevector.from_instruction(qc)

# 在布洛赫球上绘制状态
plot_bloch_multivector(state)

# 显示图形
plt.show()
```

运行此脚本应打开一个窗口，显示一个带有一个指向北极（即 $|0\rangle$ 状态）箭头的球体。这种视觉确认确保了 `matplotlib` 已正确与量子 SDK 集成。

环境准备就绪后，我们现在可以从静态定义转向动态操作。在下一章中，我们将利用此设置通过 NumPy 执行矩阵运算，在构建第一个功能性量子门之前夯实基础。
