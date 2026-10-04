---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-kernel-methods-theory-implementation"
lesson: "practical-comparing-quantum-kernels"
sourceId: 1054
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation/practical-comparing-quantum-kernels"
title: "动手实践：比较量子核函数"
description: "实现并比较不同量子核函数在基准数据集上的性能。"
order: 8
plots: []
sourceHash: "b27236e8628ad04b55a6716cf51a8a8a17dc1913b3c8352e6db03e9370ff3f47"
sourceCorrections: []
---

好的，现在我们将理论付诸实践。在前面的章节中，我们已经确立了量子核函数的数学依据，$k(x, x') = |\langle\phi(x)|\phi(x')\rangle|^N$，其中 $|\phi(x)\rangle$ 是编码数据点 $x$ 的量子态，通常由作用于 $|0\rangle^{\otimes n}$ 的参数 (parameter)化量子电路 $U_{\phi(x)}$ 生成。我们也讨论了如何估计这些核函数值以及核函数集中等潜在问题。现在，我们将实现不同的量子核函数，将它们与经典支持向量 (vector)机（SVM）结合，并在一个分类任务上比较它们的性能。这个练习将帮助您巩固对不同特征映射如何影响最终核函数和模型性能的理解。

我们将使用 Python，借助 Qiskit 等库进行量子电路构建和模拟，NumPy 用于数值运算，scikit-learn 用于经典 SVM 实现和数据集生成。

### 环境设置

首先，请确保您已安装必要的库。通常可以使用 pip 进行安装：

```bash
pip install qiskit qiskit-machine-learning numpy scikit-learn matplotlib plotly
```

我们来导入所需的模块并设置一个简单的数据集。我们将使用 scikit-learn 的 `make_moons` 数据集，它是非线性可分的，通常是核函数方法的一个良好测试用例。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel

# 设置随机种子以保证结果可复现
seed = 12345
algorithm_globals.random_seed = seed

# 生成数据集
X, y = make_moons(n_samples=100, noise=0.2, random_state=seed)

# 将特征缩放到 [0, 1] 范围 - 通常对角度编码有利
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=seed
)

# 定义特征和量子比特的数量
num_features = X_train.shape[1]
num_qubits = num_features # 为简化起见，这里每个特征使用一个量子比特

# 设置 Qiskit QuantumInstance 用于模拟
backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

print(f"训练样本数量: {len(X_train)}")
print(f"测试样本数量: {len(X_test)}")
print(f"特征/量子比特数量: {num_qubits}")
```

### 核函数 1：简单 Z 特征映射

我们的第一个量子核函数将使用 Qiskit 的 `ZFeatureMap`。这个映射使用围绕 Z 轴的单量子比特旋转来编码数据。对于 $n$ 个特征 $x = (x_1, ..., x_n)$，它在每个量子比特 $i$ 上应用 $H^{\otimes n}$，然后是 $P(2x_i)$ 门（相位门）。它本身不会在量子比特之间产生纠缠。

电路结构如下：
$|\phi(x)\rangle = U_{ZFeatureMap}(x) |0\rangle^{\otimes n}$
其中 $U_{ZFeatureMap}(x) = \prod_{i=1}^n P(2x_i)_i H_i$.

我们来定义并计算这个特征映射的核矩阵。

```python
# 定义 Z 特征映射
z_feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
# z_feature_map.decompose().draw('mpl', style='iqx') # 取消注释以可视化

# 实例化 QuantumKernel 类
z_kernel = QuantumKernel(feature_map=z_feature_map, quantum_instance=quantum_instance)

# 计算核矩阵（训练和测试）
# 训练数据核矩阵 (X_train 与 X_train)
print("正在计算 Z 核矩阵（训练）...")
kernel_matrix_train_z = z_kernel.evaluate(x_vec=X_train)

# 测试数据核矩阵 (X_test 与 X_train)
print("正在计算 Z 核矩阵（测试）...")
kernel_matrix_test_z = z_kernel.evaluate(x_vec=X_test, y_vec=X_train)

print("Z 核矩阵计算完成。")
```

> 训练核矩阵 `kernel_matrix_train_z` 将是一个大小为 `(n_train_samples, n_train_samples)` 的方阵，其中元素 $(i, j)$ 是 $k(x_i^{train}, x_j^{train})$。测试核矩阵 `kernel_matrix_test_z` 的大小将为 `(n_test_samples, n_train_samples)`，其中元素 $(i, j)$ 是 $k(x_i^{test}, x_j^{train})$。

### 核函数 2：纠缠 ZZ 特征映射

接下来，我们将使用 `ZZFeatureMap`。这个特征映射在初始数据编码旋转之后包含纠缠门（具体来说，是通过受控相位门实现的 $ZZ$ 相互作用）。这使得特征映射能够捕获特征之间的相关性，并将数据映射到更复杂的希尔伯特空间结构中。

该电路包含多层哈达玛门、单量子比特相位旋转 $P(2x_i)$，以及针对对 $(i, j)$ 的双量子比特受控相位旋转 $P(2(\pi - x_i)(\pi - x_j))$。
$|\phi(x)\rangle = U_{ZZFeatureMap}(x) |0\rangle^{\otimes n}$.

我们来定义并计算这个映射的核函数。我们将使用 `reps=2` 以使电路稍深一些。

```python
# 定义 ZZ 特征映射
zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
# zz_feature_map.decompose().draw('mpl', style='iqx') # 取消注释以可视化

# 实例化 QuantumKernel 类
zz_kernel = QuantumKernel(feature_map=zz_feature_map, quantum_instance=quantum_instance)

# 计算核矩阵
print("正在计算 ZZ 核矩阵（训练）...")
kernel_matrix_train_zz = zz_kernel.evaluate(x_vec=X_train)

print("正在计算 ZZ 核矩阵（测试）...")
kernel_matrix_test_zz = zz_kernel.evaluate(x_vec=X_test, y_vec=X_train)

print("ZZ 核矩阵计算完成。")
```

### 核函数 3：经典 RBF 核函数（基准）

为了提供量子核函数性能的参考，我们来计算一个标准的经典核函数，即径向基函数（RBF）核函数，其定义为 $k(x, x') = \exp(-\gamma \|x - x'\|^2)$。我们可以为此使用 scikit-learn 的内置函数。

```python
from sklearn.metrics.pairwise import rbf_kernel

# 计算 RBF 核矩阵
gamma_val = 'scale' # 常见启发式方法
print("正在计算 RBF 核矩阵（训练和测试）...")
kernel_matrix_train_rbf = rbf_kernel(X_train, X_train, gamma=gamma_val)
kernel_matrix_test_rbf = rbf_kernel(X_test, X_train, gamma=gamma_val)

print("RBF 核矩阵计算完成。")
```

### 训练和评估支持向量 (vector)机

现在我们有了三组预计算的核矩阵（Z 特征映射、ZZ 特征映射、RBF）。我们可以使用这些核函数训练独立的 SVM 分类器。Scikit-learn 的 `SVC` 支持预计算的核函数。

```python
# 使用 Z 特征映射核函数训练 SVM
svm_z = SVC(kernel='precomputed', C=1.0, random_state=seed)
print("正在使用 Z 核函数训练 SVM...")
svm_z.fit(kernel_matrix_train_z, y_train)
print("训练完成。")

# 使用 ZZ 特征映射核函数训练 SVM
svm_zz = SVC(kernel='precomputed', C=1.0, random_state=seed)
print("正在使用 ZZ 核函数训练 SVM...")
svm_zz.fit(kernel_matrix_train_zz, y_train)
print("训练完成。")

# 使用 RBF 核函数训练 SVM
svm_rbf = SVC(kernel='precomputed', C=1.0, random_state=seed)
print("正在使用 RBF 核函数训练 SVM...")
svm_rbf.fit(kernel_matrix_train_rbf, y_train)
print("训练完成。")
```

模型训练完成后，我们来使用相应的测试核矩阵评估它们在测试集上的性能。

```python
# 评估 Z 核函数 SVM
print("正在评估 Z 核函数 SVM...")
score_z = svm_z.score(kernel_matrix_test_z, y_test)
print(f"准确率（Z 核函数）: {score_z:.4f}")

# 评估 ZZ 核函数 SVM
print("正在评估 ZZ 核函数 SVM...")
score_zz = svm_zz.score(kernel_matrix_test_zz, y_test)
print(f"准确率（ZZ 核函数）: {score_zz:.4f}")

# 评估 RBF 核函数 SVM
print("正在评估 RBF 核函数 SVM...")
score_rbf = svm_rbf.score(kernel_matrix_test_rbf, y_test)
print(f"准确率（RBF 核函数）: {score_rbf:.4f}")
```

### 可视化决策边界

对于像 `make_moons` 这样的二维数据集，可视化决策边界能够提供有价值的视角，了解每个核函数如何分离数据。我们创建一个网格，计算每个网格点与训练数据之间的核函数，然后使用训练好的 SVM 来预测每个网格点的类别。

```python
def plot_decision_boundary(X, y, svm_model, kernel_matrix_func, title):
    """
    绘制预计算核函数 SVM 的决策边界。
    kernel_matrix_func(grid_points, X_train) -> 核矩阵
    """
    h = .02  # 网格中的步长
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 在网格上进行预测
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # 重要：像训练数据一样缩放网格点
    grid_points_scaled = scaler.transform(grid_points) # 使用相同的缩放器

    # 计算网格点与训练数据之间的核矩阵
    kernel_grid = kernel_matrix_func(grid_points_scaled, X_train)

    Z = svm_model.predict(kernel_grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # 绘制训练点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    # 稍有不同地绘制测试点
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k', marker='s', alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(f'{title} - Accuracy: {svm_model.score(kernel_matrix_func(X_test, X_train), y_test):.4f}')
    plt.show()

# 创建用于绘图的核函数计算函数
def kernel_z_eval(X1, X2):
    return z_kernel.evaluate(x_vec=X1, y_vec=X2)

def kernel_zz_eval(X1, X2):
    return zz_kernel.evaluate(x_vec=X1, y_vec=X2)

def kernel_rbf_eval(X1, X2):
    return rbf_kernel(X1, X2, gamma=gamma_val)


# 绘制决策边界（使用原始未缩放数据确定绘图范围）
plot_decision_boundary(X_scaled, y, svm_z, kernel_z_eval, "使用 Z 量子核函数的 SVM")
plot_decision_boundary(X_scaled, y, svm_zz, kernel_zz_eval, "使用 ZZ 量子核函数的 SVM")
plot_decision_boundary(X_scaled, y, svm_rbf, kernel_rbf_eval, "使用经典 RBF 核函数的 SVM")
```

### 讨论

运行代码后，您应该会观察到每个核函数具有不同的准确率分数和决策边界。

- **性能比较：** 纠缠 `ZZFeatureMap` 核函数是否胜过简单的 `ZFeatureMap` 核函数？在这个特定数据集上，是否有任何量子核函数胜过经典 RBF 核函数？结果可能因数据集、特征映射设计、量子比特数量和超参数 (parameter) (hyperparameter)（`reps`、`entanglement`、SVM 的 `C` 参数）而异。有时，简单的映射表现良好；另一些时候，纠缠会带来益处。像 RBF 这样的经典核函数通常经过高度优化，性能出色。
- **特征映射选择：** 这个实验表明，特征映射 $U_{\phi(x)}$ 的选择对量子核函数方法的性能非常重要。它定义了发生分离的特征空间的几何结构。设计有效的特征映射是一个活跃的研究方向。
- **计算成本：** 请注意，计算量子核矩阵涉及为训练矩阵运行 $O(N_{train}^2)$ 次量子电路，以及为测试矩阵运行 $O(N_{test} \times N_{train})$ 次（使用基本的成对方法）。尽管模拟器对小型示例有用，但在真实硬件上执行此操作需要大量的量子资源，并且容易受到噪声影响，因此需要第 7 章中讨论的错误缓解技术。通过 SWAP 测试或反演测试等方法估计核函数项通常在硬件上更受青睐，但会引入统计不确定性。
- **核函数集中：** 尽管只有 2 个特征/量子比特时不太可能出现严重情况，但请记住，随着量子比特数量的增加，可能会出现核函数集中现象（核矩阵元素变得非常相似），这会阻碍训练能力。特征映射的选择会影响这种现象。

这个实践练习展示了应用量子核函数方法的工作流程：选择特征映射，使用量子后端（这里是模拟器）计算核矩阵，并将其与像 SVM 这样的经典核函数机器结合。通过比较不同的量子核函数和经典基准，您将获得对其行为的实践理解以及特征映射设计在量子机器学习 (machine learning)中的重要性。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (2019)
  Journal: Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  一篇基础论文，介绍了通过将经典数据映射到量子特征空间进行分类的量子核方法，与理论基础高度相关。
- [Qiskit Machine Learning Documentation](https://qiskit.org/documentation/machine-learning/) — Qiskit Development Team (2024)
  Qiskit机器学习库的官方文档，提供了关于`QuantumKernel`、`ZFeatureMap`和`ZZFeatureMap`等在实现中的实用信息。
- [Supervised Learning with Quantum Computers](https://link.springer.com/book/10.1007/978-3-319-96424-9) — Maria Schuld, Francesco Petruccione (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)
  一本内容全面的书籍，提供了量子机器学习的广泛介绍，其中包含关于量子核方法及其理论基础的相关章节。
