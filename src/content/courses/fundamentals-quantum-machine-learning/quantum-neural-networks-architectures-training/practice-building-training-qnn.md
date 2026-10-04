---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-neural-networks-architectures-training"
lesson: "practice-building-training-qnn"
sourceId: 1070
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-5-quantum-neural-networks-architectures-training/practice-building-training-qnn"
title: "实践：构建与训练一个简单QNN"
description: "动手实践，构建一个基础量子神经网络模型，并尝试在一个简单数据集上对其进行训练。"
order: 8
plots: ["plots/1070-0.json"]
sourceHash: "873d1c852317f9102dc96521ecb61501b62ccbccb045b8069d27373f38390e20"
sourceCorrections: []
---

好的，我们来将理论付诸实践。在了解了各种量子神经网络 (neural network)（QNN）架构及其训练中的问题之后，本节提供了一个动手实践，逐步呈现如何构建和训练一个简单QNN。我们将运用前面讨论的参数 (parameter)化量子电路（PQC）、数据编码、测量方法和优化技术等思想。我们的目标不一定是达到顶尖性能，而是加深对构建和训练这些模型所涉及的基本组成部分和工作流程的认识。

我们将构建一个基本的变分量子分类器（VQC），它是一种常用于监督学习 (supervised learning)任务的QNN。我们将使用Scikit-learn这样的标准机器学习 (machine learning)库生成数据，并使用PennyLane这样的量子计算框架实现量子部分，这突出了许多实际QML实现中的混合特性。

### 问题设置：简单二分类

为了简化操作并侧重于QNN的机制，我们来处理一个简单的二分类问题。我们将使用Scikit-learn的`make_moons`函数生成一个合成数据集，该函数会创建两个交错的半圆。这个数据集是非线性可分的，对我们的简单分类器来说是一个适当的挑战。

```python
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成合成数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 将特征缩放到适合编码的范围（例如，[0, pi]）
# 这对于角度编码通常很重要。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 为了方便某些成本函数，将标签从 {0, 1} 转换为 {-1, 1}
y_shifted = y * 2 - 1

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_shifted, test_size=0.3, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Data shape: {X_train.shape}") # 对于make_moons，形状应为 (n_samples, 2)

# 可选：可视化数据
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_shifted, cmap='viridis', edgecolors='k')
plt.title("合成月亮数据集 (已缩放)")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.show()
```

### 量子电路设计

现在，我们来设计QNN的核心：参数 (parameter)化量子电路（PQC）。我们需要一种方法来编码经典输入数据 ($x$)，并施加由权重 (weight) ($\theta$) 参数化的可训练量子门。

1. **数据编码：** 我们将使用角度编码，将输入数据 $x = (x_1, x_2)$ 的两个特征映射到量子位的旋转角度上。由于我们有两个特征，我们将使用两个量子位。例如，我们可以使用 `qml.AngleEmbedding`。
2. **PQC 变分态（变分层）：** 编码之后，我们将添加可训练的门层。一种常见做法是交替使用单量子位旋转门和纠缠门（如CNOT）。旋转角度是我们的可训练参数 $\theta$。我们将使用一个包含几层的简单结构。QNN的表达能力很大程度上取决于这种变分态设计。
3. **测量：** 为了获得预测，我们需要测量量子态。对于二分类，测量其中一个量子位（例如第一个量子位 $\langle Z_0 \rangle$）的泡利Z算符的期望值是一种常见选择。期望值范围从-1到1，这与我们转换后的标签{-1, 1}保持一致。

我们来使用PennyLane定义量子设备和电路。

```python
# 定义PQC的量子比特数和层数
n_qubits = 2
n_layers = 3 # 变分层数

# 使用默认模拟器
dev = qml.device("default.qubit", wires=n_qubits)

# 定义PQC结构（变分态）
def pqc_ansatz(params):
    """一个变分门层。"""
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

# 定义完整的量子节点（电路）
@qml.qnode(dev, interface="autograd")
def quantum_circuit(params, x):
    """完整的QNN电路：编码 -> 变分态 -> 测量"""
    # 编码输入特征 x
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y') # 示例：Y旋转
    # 应用变分层
    pqc_ansatz(params)
    # 测量第一个量子位上泡利Z的期望值
    return qml.expval(qml.PauliZ(0))

# 初始化PQC层的参数
# 形状需符合qml.StronglyEntanglingLayers的要求：(L, N, 3)
# L = 层数, N = 量子比特数
param_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
initial_params = np.random.uniform(low=0, high=2 * np.pi, size=param_shape)

print(f"参数形状: {initial_params.shape}")

# 示例：绘制一层电路（需要matplotlib）
# drawer = qml.draw(quantum_circuit)
# example_params = np.random.uniform(0, 2 * np.pi, size=param_shape)
# example_x = X_train[0]
# print(drawer(example_params, example_x))

# 示例：使用初始参数和一个数据点测试电路
output = quantum_circuit(initial_params, X_train[0])
print(f"第一个数据点的初始电路输出: {output}")
```

> QNN的简化流程：输入数据 `x` 被编码，通过带有参数 `θ` 的变分层处理，并测量以生成预测 `ŷ`。

### 成本函数与优化

为了训练QNN，我们需要一个成本函数来衡量模型表现如何。由于我们的输出 $\langle Z_0 \rangle$ 介于-1和1之间，且我们的标签是{-1, 1}，我们可以使用均方误差（MSE）。


$$
\text{成本}(\theta) = \frac{1}{M} \sum_{i=1}^{M} (y_i - \hat{y}_i(\theta, x_i))^2
$$


M是训练样本的数量，$y_i$是样本$i$的真实标签，而模型的预测是$\hat{y}_i(\theta, x_i) = \text{quantum\_circuit}(\theta, x_i)$。

我们将使用PennyLane提供的优化器（如Adam）来调整参数 (parameter) $\theta$，以最小化这个成本函数。PennyLane与`autograd`（或其他框架，如TensorFlow、PyTorch）的集成，可以实现自动微分，通常在后台使用参数偏移规则计算量子梯度。

```python
# 定义成本函数（均方误差）
def cost_function(params, X, y):
    """计算MSE成本。"""
    predictions = np.array([quantum_circuit(params, x) for x in X])
    return np.mean((y - predictions) ** 2)

# 定义准确率指标
def accuracy(params, X, y):
    """计算分类准确率。"""
    predictions = np.array([quantum_circuit(params, x) for x in X])
    # 将预测值（-1到1）和标签（-1, 1）转换为二进制 {0, 1} 进行比较
    predicted_labels = np.sign(predictions) # 映射到 {-1, 1}
    # 如果需要，转换标签 {-1, 1}，或直接比较
    acc = np.mean(predicted_labels == y)
    return acc

# 选择优化器
opt = qml.AdamOptimizer(stepsize=0.05)

# 训练循环
batch_size = 10
num_epochs = 15
params = initial_params # 从随机参数开始

cost_history = []
accuracy_history_train = []
accuracy_history_test = []

print("开始训练...")
for epoch in range(num_epochs):
    # 打乱训练数据
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_costs = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # 梯度下降步骤
        params, cost_val = opt.step_and_cost(lambda p: cost_function(p, X_batch, y_batch), params)
        epoch_costs.append(cost_val)

    avg_epoch_cost = np.mean(epoch_costs)
    cost_history.append(avg_epoch_cost)

    # 计算训练集和测试集的准确率
    train_acc = accuracy(params, X_train, y_train)
    test_acc = accuracy(params, X_test, y_test)
    accuracy_history_train.append(train_acc)
    accuracy_history_test.append(test_acc)

    print(f"周期 {epoch+1}/{num_epochs} - 成本: {avg_epoch_cost:.4f} - 训练准确率: {train_acc:.4f} - 测试准确率: {test_acc:.4f}")

print("训练完成。")
```

### 结果可视化

我们来绘制训练周期中的成本函数和准确率，以查看QNN是如何学习的。

```python
# 绘制结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制成本历史
ax1.plot(range(num_epochs), cost_history, marker='o', linestyle='-', color='#1c7ed6')
ax1.set_title("成本函数历史")
ax1.set_xlabel("周期")
ax1.set_ylabel("均方误差成本")
ax1.grid(True, linestyle='--', alpha=0.6)

# 绘制准确率历史
ax2.plot(range(num_epochs), accuracy_history_train, marker='s', linestyle='-', color='#40c057', label='训练准确率')
ax2.plot(range(num_epochs), accuracy_history_test, marker='^', linestyle='--', color='#fd7e14', label='测试准确率')
ax2.set_title("准确率历史")
ax2.set_xlabel("周期")
ax2.set_ylabel("准确率")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(0, 1.05) # 准确率介于0和1之间

plt.tight_layout()
plt.show()
```



![QNN训练进程](plots/1070-0.json)



> 训练曲线显示了简单QNN的成本随周期减少，准确率随周期增加。注意：实际值取决于随机初始化和超参数 (parameter) (hyperparameter)。

### 讨论与后续步骤

这个动手实践例子展示了构建一个简单QNN的端到端过程：问题定义、数据编码、PQC变分态设计、成本函数选择以及优化循环的运行。结果通常表明，即使是简单的QNN也能学会对简单的非线性数据进行分类。

然而，这只是一个起点。请考虑以下几点和潜在的扩展：

1. **超参数 (parameter) (hyperparameter)调整：** 试着调整层数（`n_layers`）、学习率（`stepsize`）、批量大小和周期数。这些如何影响收敛和最终准确率？
2. **变分态选择：** 用不同的PQC结构替换`qml.StronglyEntanglingLayers`。尝试更简单或更复杂的变分态。这种选择如何影响性能和可训练性（例如，荒漠高原的风险）？请回顾第4章中PQC设计方法。
3. **数据编码：** 尝试不同的编码方式（例如，幅度编码、角度编码中不同的旋转轴）。模型对编码策略的敏感程度如何？（见第2章）。
4. **优化器：** 试用不同的优化器，如`qml.GradientDescentOptimizer`、`qml.AdagradOptimizer`，甚至量子自然梯度（如果已涵盖且适用）。
5. **成本函数：** 对于二分类，尝试其他方法，如二元交叉熵（需要调整输出映射，也许在期望值上经典地使用sigmoid函数）。
6. **数据集复杂性：** 将此框架应用于更复杂的数据集。性能如何扩展？何时会出现荒漠高原或表达能力有限等制约？
7. **过拟合 (overfitting)：** 关注训练准确率和测试准确率之间的差距。如果发生过拟合（训练准确率高，测试准确率低），请考虑适用于QNN或更简单模型（更少层/参数）的正则化 (regularization)技术，如本章前面所讨论的。
8. **硬件执行：** 如果条件允许，调整代码使其在真实量子设备或带噪声的模拟器上运行。研究噪声的影响并应用纠错技术（第7章涵盖）。

本次练习奠定了基础。构建更复杂的QNN需要仔细考虑这些因素，平衡电路的表达能力、可训练性以及对噪声的抵抗力，特别是在面向近期量子硬件时。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (2019)
  Journal: Nature; Publisher: Springer Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  这篇基础论文介绍了量子特征映射的概念，这对于量子机器学习中的数据编码策略至关重要，与本节使用的角度编码方法直接相关。
- [Barren plateaus in quantum neural networks](https://doi.org/10.1038/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Publisher: Nature Publishing Group; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  这篇有影响力的论文识别并分析了“贫瘠高原”现象，这是训练深度参数化量子电路的一个重大挑战，正如本节后续讨论中所述。
- [Variational Quantum Algorithms](https://doi.org/10.1038/s42254-021-00348-9) — M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Publisher: Springer Nature; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  一篇关于变分量子算法（VQA）的综述文章，为本节构建的变分量子分类器（VQC）提供了更广阔的视野和对其原理与挑战的理解。
