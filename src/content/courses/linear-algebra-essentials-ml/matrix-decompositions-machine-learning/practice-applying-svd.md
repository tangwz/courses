---
course: "linear-algebra-essentials-ml"
chapter: "matrix-decompositions-machine-learning"
lesson: "practice-applying-svd"
sourceId: 1542
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/practice-applying-svd"
title: "SVD实践应用"
description: "使用Python将SVD应用于数据集（例如图像数据），以进行降维或压缩。"
order: 10
plots: ["plots/1542-0.json"]
sourceHash: "dc8c6f8c5965e762970ed9da86ad1dc50e906c036fa7750ade3b881f8e9d7f84"
sourceCorrections: []
---

奇异值分解 (SVD) 将矩阵 $A$ 分解为 $A = U\Sigma V^T$ 的乘积。这种分解在理解矩阵性质、降维甚至数据压缩方面都非常有用。实践中，将使用Python的NumPy库执行SVD，并观察它在实际数据上的表现。

### 环境设置

首先，请确保您已安装NumPy。如果您在标准的科学计算Python环境中工作，您很可能已经安装了它。我们将主要使用NumPy中的`linalg`子模块。让我们导入它：

```python
import numpy as np
# 如果后续需要图像可视化，我们可能会使用matplotlib，但目前只用NumPy。
# import matplotlib.pyplot as plt
# from scipy import misc # 加载图像的例子

print(f"使用的NumPy版本: {np.__version__}")
```

### 对简单矩阵进行SVD计算

让我们从一个基本矩阵开始，计算其SVD。

```python
# 定义一个简单的2x3矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("原始矩阵A:")
print(A)

# 执行SVD
# 'full_matrices=False' 计算“薄”SVD，这通常更高效。
# U的形状将是(m, k)，s的形状将是(k,)，Vh的形状将是(k, n)
# 其中 k = min(m, n)
U, s, Vh = np.linalg.svd(A, full_matrices=False)

print("\nU矩阵（左奇异向量）:")
print(U)
print(f"U的形状: {U.shape}")

print("\n奇异值（Sigma的对角线元素）:")
print(s)
print(f"s的形状: {s.shape}")

print("\nVh矩阵（V的转置，右奇异向量的转置）:")
print(Vh)
print(f"Vh的形状: {Vh.shape}")
```

注意到 `np.linalg.svd` 将奇异值 $s$ 作为一维数组返回。为了重构原始矩阵 $A$，我们需要从这些奇异值构建对角矩阵 $\Sigma$。矩阵乘法的维度必须正确对齐 (alignment)：$U$ 是 $(m \times k)$，$\Sigma$ 必须是 $(k \times k)$，而 $V^T$（也就是`Vh`）是 $(k \times n)$，其中 $k = \min(m, n)$。

### 矩阵重构

让我们从 $U$、$s$ 和 $Vh$ 重构 $A$，以验证分解的正确性。我们需要创建 $\Sigma$ 矩阵。由于 `s` 只包含对角线元素，我们首先创建一个正确形状（$k \times k$）的零矩阵，然后填充对角线。

```python
# 从奇异值s创建Sigma矩阵
# k是奇异值的数量，即min(m, n)
k = len(s)
Sigma = np.zeros((k, k))
Sigma[:k, :k] = np.diag(s)

print("\n构建的Sigma矩阵:")
print(Sigma)
print(f"Sigma的形状: {Sigma.shape}")

# 使用 U @ Sigma @ Vh 重构A
# '@' 运算符执行矩阵乘法
A_reconstructed = U @ Sigma @ Vh

print("\n重构的矩阵A:")
print(A_reconstructed)

# 检查重构矩阵是否接近原始矩阵
# np.allclose 在给定容差范围内检查元素的相等性
print(f"\n重构成功？ {np.allclose(A, A_reconstructed)}")
```

正如您所看到的，重构的矩阵与原始矩阵 $A$ 非常接近，确认了 $A = U \Sigma V^T$。这些微小差异是由于浮点运算引起的。

### SVD用于矩阵近似（降维）

SVD最重要的应用之一是创建矩阵的低秩近似。这通过只保留最大的 $k$ 个奇异值及其对应的奇异向量 (vector)来实现。这个思想是，最大的奇异值捕获了矩阵所表示数据中最“重要”的信息或方差。

让我们在一个稍微大一点的矩阵上尝试这个方法。

```python
# 创建一个稍微大一点的矩阵，可能包含一些依赖关系
B = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],
              [5, 6, 7, 8],
              [9, 8, 7, 6]])

print("\n原始矩阵B:")
print(B)

# 对B执行SVD
U_b, s_b, Vh_b = np.linalg.svd(B)

print("\nB的奇异值:")
print(s_b)

# 选择要保留的奇异值数量（用于近似的秩）
rank_k = 2 # 让我们尝试一个秩为2的近似

print(f"\n使用秩 k={rank_k} 近似B")

# 构建截断的Sigma_k矩阵
Sigma_k = np.zeros_like(B, dtype=float) # 从B形状的零矩阵开始
Sigma_k[:rank_k, :rank_k] = np.diag(s_b[:rank_k]) # 用k个奇异值填充左上角

print("\n截断的Sigma矩阵 (Sigma_k):")
print(Sigma_k)

# 重构秩为k的近似
B_approx = U_b @ Sigma_k @ Vh_b

print("\n秩为k的近似矩阵B_approx:")
print(B_approx)

# 计算差异（近似误差）
error = np.linalg.norm(B - B_approx, 'fro') # 差值的Frobenius范数
print(f"\n近似误差（Frobenius范数）: {error:.4f}")

# 让我们尝试秩为3
rank_k_3 = 3
Sigma_k_3 = np.zeros_like(B, dtype=float)
Sigma_k_3[:rank_k_3, :rank_k_3] = np.diag(s_b[:rank_k_3])
B_approx_3 = U_b @ Sigma_k_3 @ Vh_b
error_3 = np.linalg.norm(B - B_approx_3, 'fro')
print(f"\n秩 k=3 的近似误差: {error_3:.4f}")
```

注意到奇异值如何减小。前两个奇异值远大于其他值，这表明秩为2的近似可能捕获了矩阵结构的很大一部分。随着我们增加秩 $k$，近似误差会减小。通过选择 $k < \text{rank}(B)$，我们有效地减少了表示 $B$ 中核心信息所需的维度。

### 奇异值衰减的可视化：图像压缩的应用

图像压缩是一个演示SVD近似能力的经典例子。灰度图像可以表示为一个矩阵，其中每个元素都是一个像素强度。彩色图像通常由三个这样的矩阵（红、绿、蓝）表示。

让我们考虑一个灰度图像矩阵。通过对这个矩阵执行SVD，我们常常发现奇异值快速减小。这意味着我们有可能只使用原始奇异值（和对应向量 (vector)）的一小部分，就能重构出图像在视觉上可接受的版本。

虽然我们不会在这里直接加载和显示图像，但让我们模拟这个过程，并可视化一个代表性矩阵的奇异值衰减情况。想象这代表着一个简化图像。

```python
# 生成一个矩阵（例如，模拟图像数据）
# 让我们使用一个结构可能由少量分量捕获的矩阵
np.random.seed(42)
m, n = 50, 60
base = np.random.rand(m, 2) @ np.random.rand(2, n) # 低秩基础
noise = np.random.rand(m, n) * 0.1 # 添加一些噪声
image_matrix = base + noise

# 执行SVD
U_img, s_img, Vh_img = np.linalg.svd(image_matrix, full_matrices=False)

# 绘制奇异值
num_singular_values = len(s_img)
singular_value_indices = np.arange(1, num_singular_values + 1)

# 为奇异值创建Plotly图表定义
plotly_singular_values_chart = {
    "data": [{
        "type": "line",
        "x": singular_value_indices.tolist(),
        "y": s_img.tolist(),
        "mode": "lines+markers",
        "marker": {"color": "#228be6"},
        "line": {"color": "#228be6"}
    }],
    "layout": {
        "title": "奇异值衰减",
        "xaxis": {"title": "索引 (k)"},
        "yaxis": {"title": "奇异值大小", "type": "log"}, # 对数刻度通常有助于可视化
        "height": 350,
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40}
    }
}

# 显示图表（在真实的网页环境中，此JSON将被渲染）
import json
print("\n奇异值的Plotly图表JSON:")
print(json.dumps(plotly_singular_values_chart))
```



![奇异值衰减](plots/1542-0.json)



> 生成的矩阵的奇异值，绘制在对数刻度上。注意到前几个值之后出现急剧下降。

该图清楚地表明奇异值快速减小。前几个值显著大于其余部分。对于图像压缩，您会这样做：

1. 对图像矩阵执行SVD。
2. 选择要保留的奇异值数量 $k$（例如10、20、50）。
3. 保留 $U$ 的前 $k$ 列、前 $k$ 个奇异值（作为对角矩阵 $\Sigma_k$）以及 $V^T$ 的前 $k$ 行。
4. 将近似图像重构为 $A_k = U_k \Sigma_k V^T_k$。
5. 存储 $U_k$、这 $k$ 个奇异值和 $V^T_k$，而不是完整的图像矩阵 $A$。如果 $k$ 远小于原始维度，这会显著节省存储空间。

尝试不同的 $k$ 值会展现出一种权衡：较小的 $k$ 提供更高的压缩比，但图像重构的准确性较低；而较大的 $k$ 则会提高视觉质量，但压缩比会降低。

这个实践练习演示了如何使用NumPy计算SVD，以及截断SVD如何用于创建矩阵的低秩近似，这是一种在机器学习 (machine learning)中用于降维和数据压缩应用的基本方法。

## 参考资料

- [Introduction to Applied Linear Algebra - Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — Stephen Boyd and Lieven Vandenberghe (2018)
  Publisher: Cambridge University Press
  一本专注于应用线性代数的入门教材，涵盖了奇异值分解在数据分析、矩阵近似和降维中的应用，对机器学习尤为相关。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习领域的权威资源，解释了奇异值分解和主成分分析等相关技术如何在机器学习中用于降维和特征提取。
- [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) — NumPy Developers (2023)
  NumPy 计算奇异值分解函数的官方文档，详细说明了其参数、返回值和使用示例，用于实际实现。
