---
course: "bayesian-machine-learning"
chapter: "gaussian-processes"
lesson: "practice-gaussian-process-modeling"
sourceId: 3607
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-4-gaussian-processes/practice-gaussian-process-modeling"
title: "动手实践：高斯过程建模"
description: "实现高斯过程回归和分类，包括核函数选择和超参数调整。"
order: 8
plots: ["plots/3607-0.json", "plots/3607-1.json"]
sourceHash: "27d2191dd646a28762784c4f9dd65d9dba6773e4273fd99417dfe86709b7f2fe"
sourceCorrections: []
---

此处提供高斯过程模型在回归和分类中的实际实现。其中运用了高斯过程的定义和核函数的作用。我们将使用 Python 和流行的 `scikit-learn` 库，它为标准高斯过程模型提供了便捷的实现。虽然 `GPy` 或 `GPflow` 等其他库提供更多高级功能和灵活性，但 `scikit-learn` 非常适合开始实际操作。

请确保您已安装 `scikit-learn`、`numpy` 和 `matplotlib`（或 `plotly` 等其他绘图库）。

```bash
pip install scikit-learn numpy matplotlib plotly
```

### 高斯过程回归

让我们从一个常用场景开始：非线性回归。我们希望对带噪声的数据点进行函数拟合，不仅得到点预测，还能获得这些预测的不确定性量度。

#### 1. 生成样本数据

首先，我们需要一些数据。我们将基于一个已知的非线性函数生成一个合成数据集，并加入一些高斯噪声。这使我们能够直观地将高斯过程拟合结果与真实值进行比较。

```python
import numpy as np

# 定义真实函数
def true_function(X):
    return np.sin(X * 1.5 * np.pi).ravel()

# 生成训练数据
rng = np.random.RandomState(1)
X_train = rng.rand(30) * 2 - 1 # -1 到 1 之间的 30 个点
y_train = true_function(X_train) + rng.randn(30) * 0.2 # 添加噪声

# 生成用于预测的测试点
X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)

# 为 scikit-learn 重塑训练数据
X_train = X_train.reshape(-1, 1)
```

#### 2. 定义核函数和模型

核函数的选择很重要，因为它编码了我们对函数的事先假设（例如，平滑性、周期性）。一个常见的默认选择是径向基函数（RBF）核，也称为平方指数核。它假设函数是平滑的。我们通常将其与 `WhiteKernel` 结合使用，以处理观测数据中的噪声。

RBF 核有一个长度尺度参数 (parameter) $l$，它控制着平滑度（相关性随距离衰减的速度），而 `WhiteKernel` 有一个噪声水平参数 $\sigma_n^2$。


$$
k_{\text{RBF}}(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{(x_i - x_j)^2}{2l^2}\right)
$$


$$
k_{\text{White}}(x_i, x_j) = \sigma_n^2 \delta_{ij}
$$


在 `scikit-learn` 中，我们使用加法组合核函数。我们还添加一个 `ConstantKernel`（乘以 RBF）来控制整体方差（$\sigma_f^2$）。

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# 定义核函数：RBF + WhiteKernel
# C() 控制幅度（方差）
# RBF() 提供平滑性假设（length_scale）
# WhiteKernel() 处理噪声
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e1))

# 实例化高斯过程模型
gp_regressor = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10, # 多次运行优化器
    random_state=42
)
```

我们为超参数 (hyperparameter)设置了边界，以引导优化过程。`n_restarts_optimizer` 有助于在拟合超参数时避免局部最优。

#### 3. 拟合模型

拟合高斯过程涉及通过最大化训练数据的对数边缘似然来优化核超参数（如 RBF 长度尺度、方差和噪声水平）。

```python
# 将高斯过程拟合到训练数据
gp_regressor.fit(X_train, y_train)

# 打印优化后的核参数
print(f"Optimized Kernel: {gp_regressor.kernel_}")
# Example Output: Optimized Kernel: 1**2 * RBF(length_scale=0.406) + WhiteKernel(noise_level=0.0416)
```

拟合过程找到了最能根据核定义的模型结构解释观测数据的超参数。

#### 4. 进行预测

现在我们可以使用拟合好的模型对测试点进行预测。重要的是，高斯过程同时提供均值预测和标准差（代表不确定性）。

```python
# 对测试点进行预测
y_pred_mean, y_pred_std = gp_regressor.predict(X_test, return_std=True)
```

#### 5. 可视化结果

让我们可视化高斯过程拟合结果，包括训练数据、真实函数（如果已知）、高斯过程均值预测以及置信区间（通常为均值 ± 1.96 \* 标准差，用于 95% 置信区间）。



![高斯过程回归](plots/3607-0.json)



> 对带噪声正弦数据进行高斯过程回归拟合。蓝线是高斯过程均值预测，红点是训练数据，灰色虚线是真实的底层函数，蓝色阴影区域代表 95% 置信区间。

请注意，在训练数据点附近，置信区间（不确定性）较小，而在数据稀疏的区域则变大。这是高斯过程的一个基本属性和优势：它们自然地量化 (quantization)自身的预测不确定性。

### 高斯过程分类

现在，我们来处理一个二元分类问题。这里的挑战在于输出是离散的（例如 0 或 1），而不是连续的，因此似然函数是非高斯的（通常是伯努利分布）。这阻碍了精确的贝叶斯推断。`scikit-learn` 的 `GaussianProcessClassifier` 内部使用拉普拉斯近似来处理这个问题。

#### 1. 生成样本数据

我们将创建一个简单的一维数据集，其中类别非线性地依赖于输入特征。

```python
# 生成一维分类数据
rng = np.random.RandomState(3)
X_class = rng.rand(80, 1) * 4 - 2 # -2 到 2 之间的 80 个点
# 类别概率非线性地依赖于 X
p_class1 = 1 / (1 + np.exp(-np.sin(X_class.ravel() * 2)))
y_class = (rng.rand(80) < p_class1).astype(int) # 根据概率分配类别 0 或 1

# 用于可视化的测试点
X_test_class = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
```

#### 2. 定义核函数和模型

与回归类似，我们需要一个核函数。RBF 核函数通常也是分类任务的一个合理起点，它假设决定类别概率的底层潜在函数具有一定的平滑性。

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 定义核函数（RBF 通常效果不错）
kernel_class = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1))

# 实例化高斯过程分类器
# 隐式使用拉普拉斯近似
gp_classifier = GaussianProcessClassifier(
    kernel=kernel_class,
    n_restarts_optimizer=10,
    random_state=42
)
```

#### 3. 拟合模型

拟合过程涉及根据近似边缘似然（由拉普拉斯近似推导而来）优化超参数 (parameter) (hyperparameter)。

```python
# 拟合高斯过程分类器
gp_classifier.fit(X_class, y_class)

# 打印优化后的核函数
print(f"Optimized Kernel (Classification): {gp_classifier.kernel_}")
# Example Output: Optimized Kernel (Classification): 2.91**2 * RBF(length_scale=1.04)
```

#### 4. 进行预测（概率）

对于分类，我们通常关注属于每个类别的概率。

```python
# 对测试点预测类别概率
y_prob_class = gp_classifier.predict_proba(X_test_class)
```

#### 5. 可视化结果

我们可以绘制训练数据点和属于类别 1 的预测概率。



![高斯过程分类](plots/3607-1.json)



> 高斯过程分类结果。圆点代表训练数据点（红色圆圈表示类别 0，青色叉号表示类别 1）。蓝线显示属于类别 1 的预测概率。

高斯过程分类器提供类别间平滑的概率转换。概率接近 0.5 的区域表示分类的不确定性较高。

### 实践考量

- **核函数选择：** 选择正确的核函数（或核函数组合）很重要。它编码了先验知识。如果您期望周期性，请使用 `ExpSineSquared` 核函数。如果您期望线性趋势，请添加 `DotProduct` 核函数。组合核函数（`+` 表示加法，`*` 表示乘法）可以构建复杂的先验。超参数 (parameter) (hyperparameter)优化有助于调整所选的核函数结构，但初始结构选择依赖于领域知识或尝试。
- **可扩展性：** 标准高斯过程实现的计算复杂度在训练时为 $O(N^3)$，预测时为 $O(N^2)$，其中 $N$ 是训练点的数量。这对于大型数据集（N 大于数千）来说是不可行的。理论中讨论的近似方法（如使用诱导点的稀疏高斯过程）对于更大规模的数据是必要的。`GPflow`（基于 TensorFlow）或 `GPyTorch`（基于 PyTorch）等库是专门设计来处理这些可扩展近似的。
- **超参数优化：** 尽管 `scikit-learn` 自动化了这一点，但理解它最大化（对数）边缘似然很重要。`n_restarts_optimizer` 参数很有用，因为优化过程可能存在多个局部最优。
- **结果解读：** 高斯过程的主要优势在于其灵活的非线性建模，以及系统化的不确定性量化 (quantization)。始终检查回归中的方差或标准差，或分类中的概率，以理解模型的置信度。

本次实践演示了如何使用 `scikit-learn` 实现高斯过程回归和分类。您已经了解了如何定义核函数、拟合模型、进行带不确定性的预测以及可视化结果。请记住，尽管这些示例使用了简单的核函数和数据集，但高斯过程的强大之处在于其应对更复杂问题的灵活性，尤其是在结合了周全的核函数设计以及对于大型数据集采用适当的近似技术时。

## 参考资料

- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) — Carl Edward Rasmussen and Christopher K. I. Williams (2006)
  Publisher: The MIT Press
  这本权威教材为高斯过程提供了全面的理论和实践基础，涵盖核函数、推断和应用。
- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Kevin P. Murphy (2022)
  Publisher: The MIT Press
  一本关于概率机器学习的综合教材，其中包含高斯过程的详细章节，在高斯过程的贝叶斯方法中提供了更广阔的背景。
- [1.7. Gaussian Processes - scikit-learn 1.4.2 documentation](https://scikit-learn.org/stable/modules/gaussian_process.html) — scikit-learn developers (2024)
  scikit-learn 库中高斯过程回归器和分类器实现与使用的官方文档。
