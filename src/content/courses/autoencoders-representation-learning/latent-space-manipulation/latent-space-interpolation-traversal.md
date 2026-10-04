---
course: "autoencoders-representation-learning"
chapter: "latent-space-manipulation"
lesson: "latent-space-interpolation-traversal"
sourceId: 2903
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-6-latent-space-manipulation/latent-space-interpolation-traversal"
title: "潜在空间中的插值与遍历"
description: "在潜在空间中进行插值和属性向量修改。"
order: 5
plots: []
sourceHash: "e91f458f608414b5f27b492f959e39f5883414aff1c53b42333a024c171f3a9b"
sourceCorrections: []
---

既然我们已经了解了如何使用t-SNE和UMAP等方法（如“潜在空间可视化”中讨论的）可视化潜在空间的整体结构，我们就可以转向更主动的考察方法。我们不再仅仅观察编码点的静态排列，而是可以通过系统地生成新的潜在向量 (vector)并进行解码来细致观察该空间。这让我们能理解自编码器如何组织信息，以及潜在表示$z$的变化如何转化为重建输出$\hat{x}$的变化。实现这一点的两个基本方法是插值和遍历。

### 潜在空间插值

插值是指在潜在空间中在两点之间创建平滑路径，并观察解码器生成的相应输出。假设你有两个输入数据点，$x_a$和$x_b$。训练自编码器后，你可以获得它们各自的潜在表示，$z_a = \text{Encoder}(x_a)$和$z_b = \text{Encoder}(x_b)$。

核心思路是生成位于$z_a$和$z_b$连接路径上的中间潜在向量 (vector)$z_{interp}$。最简单的方法是**线性插值**。对于在0到1之间变化的标量$\alpha$，我们将插值向量定义为：


$$
z_{interp}(\alpha) = (1 - \alpha) z_a + \alpha z_b
$$


随着$\alpha$从0扫描到1，$z_{interp}(\alpha)$从$z_a$线性移动到$z_b$。通过将这些中间$z_{interp}(\alpha)$向量中的每一个输入解码器，我们得到一系列输出$\hat{x}_{interp}(\alpha) = \text{Decoder}(z_{interp}(\alpha))$。

**为何插值？**

观察序列$\hat{x}_{interp}(\alpha)$可以提供有益的了解：

1. **连续性：** 如果生成的输出从$\hat{x}_a \approx x_a$平滑且真实地过渡到$\hat{x}_b \approx x_b$，这表明潜在空间组织良好，并有意义地捕获了$x_a$和$x_b$之间的变化。解码器已学会将附近的潜在点映射到语义相似的输出。
2. **数据流形：** 生成输出的路径可以近似于所选两点之间数据的潜在流形。
3. **生成能力：** 对于像VAE这样的生成模型，平滑的插值表明良好的生成能力，模型可以合成合理的中间样本。

**实施步骤：**

1. 选择两个输入数据点，$x_a$和$x_b$。
2. 使用训练好的编码器获得它们的潜在表示$z_a$和$z_b$。
3. 选择中间步骤的数量，`num_steps`。
4. 生成一系列$\alpha$值，通常使用`np.linspace(0, 1, num_steps)`。
5. 对于每个$\alpha$，使用线性插值公式计算$z_{interp}(\alpha)$。
6. 使用训练好的解码器解码每个$z_{interp}(\alpha)$以获得$\hat{x}_{interp}(\alpha)$。
7. 可视化生成输出的序列（例如，作为一系列图像或图表）。

这是一个使用类似 Keras API 的 Python 代码片段：

```python
import numpy as np
# 假设编码器和解码器模型已训练并加载
# 假设 xa_batch 和 xb_batch 包含单个数据点（带批次维度）

# 1 & 2: 编码输入
za = encoder.predict(xa_batch)
zb = encoder.predict(xb_batch)

num_steps = 10 # 插值步数
interpolated_latents = []
interpolated_outputs = []

# 3 & 4: 生成 alpha 值
alphas = np.linspace(0, 1, num_steps)

# 5: 计算插值的潜在向量
for alpha in alphas:
    z_interp = (1.0 - alpha) * za + alpha * zb
    interpolated_latents.append(z_interp)

# 将潜在向量堆叠成批次以高效解码
latent_batch = np.vstack(interpolated_latents)

# 6: 解码插值的潜在向量
interpolated_outputs = decoder.predict(latent_batch)

# 7: 可视化 interpolated_outputs（例如，按顺序显示图像）
# ... 可视化代码取决于数据类型 ...
```

**球面线性插值 (SLERP)**

虽然线性插值（LERP）直接简单，但有时\*\*球面线性插值（SLERP）\*\*更受青睐，特别是在处理方向比幅度更重要的潜在空间时，或者在使用带有超球面先验的VAE时。SLERP在超球面上沿着$z_a$和$z_b$之间的大圆弧保持恒定速度。

两个向量$v_1$和$v_2$之间，插值因子$t \in [0, 1]$的SLERP公式是：


$$
\text{SLERP}(v_1, v_2; t) = \frac{\sin((1-t)\Omega)}{\sin \Omega} v_1 + \frac{\sin(t\Omega)}{\sin \Omega} v_2
$$


其中$\Omega$是$v_1$和$v_2$之间的角度，计算公式为$\Omega = \arccos(\frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|})$。如果不希望幅度变化，通常在应用SLERP之前对$z_a$和$z_b$进行归一化 (normalization)。在某些情况下，SLERP可以带来更自然的过渡，特别是对于涉及旋转或周期性特征的生成任务。

### 潜在空间遍历

遍历与插值不同。遍历不是在两个特定点之间移动，而是在潜在空间中沿着特定*方向*移动，从一个单点开始。这对于理解各个潜在维度的作用尤其有帮助，特别是如果你致力于获得或正在分析解耦表示。

**轴对齐 (alignment)遍历**

最常见的形式是轴对齐遍历。在这里，我们一次修改一个潜在维度，同时保持其他维度不变。

1. 从输入$x_{start}$开始，并对其进行编码以获得$z_{start} = \text{Encoder}(x_{start})$。
2. 选择一个潜在维度$i$进行研究。
3. 定义一个变化范围，例如从$-\delta_{max}$到$+\delta_{max}$。
4. 生成一系列修改后的潜在向量 (vector)$z_{traversed}(\delta)$，其中只有第$i$个维度发生变化：
   
   $$
   z_{traversed}(\delta) = z_{start} + \delta \cdot e_i
   $$
   
   其中$e_i$是一个基向量，在第$i$个位置为1，其他位置为0。$\delta$在所选范围内变化。
5. 使用解码器解码每个$z_{traversed}(\delta)$：$\hat{x}_{traversed}(\delta) = \text{Decoder}(z_{traversed}(\delta))$。
6. 可视化输出序列$\hat{x}_{traversed}(\delta)$。

**解读与解耦**

如果自编码器学习到了某种程度的解耦表示（正如$\beta$-VAE等技术所追求的，在“促进解耦的方法”中已讨论），理想情况下，遍历单个维度$i$应该导致输出$\hat{x}$中单一、特定、可解释的变化因素发生变化。例如，在一个人脸数据集中，一个维度可能控制笑容强度，另一个控制头部姿态旋转，还有一个控制光照方向。观察生成的序列$\hat{x}_{traversed}(\delta)$使你能够定性评估该维度的解耦程度。如果改变维度$i$同时改变多个属性，则表示在该轴上是纠缠的。

**代码片段：**

```python
import numpy as np
# 假设编码器和解码器模型已训练并加载
# 假设 x_start_batch 包含单个起始数据点

# 1: 编码起始点
z_start = encoder.predict(x_start_batch)

dimension_index = 5  # 要遍历的潜在维度索引
num_steps = 11       # 遍历步数（包括中心点）
traversal_range = 3.0 # 维度变化范围（+/-）

traversed_latents = []
traversed_outputs = []

# 3 & 4: 生成 delta 值并修改潜在向量
deltas = np.linspace(-traversal_range, traversal_range, num_steps)

for delta in deltas:
    z_traversed = z_start.copy() # 从原始潜在向量开始
    z_traversed[0, dimension_index] += delta # 修改特定维度
    traversed_latents.append(z_traversed)

# 将潜在向量堆叠成批次
latent_batch = np.vstack(traversed_latents)

# 5: 解码遍历的潜在向量
traversed_outputs = decoder.predict(latent_batch)

# 6: 可视化 traversed_outputs
# ... 可视化代码 ...
```

**轴对齐遍历**

尽管轴对齐遍历很常见，你也可以在潜在空间中沿着任意方向$v$遍历：$z_{traversed}(\delta) = z_{start} + \delta \cdot v$。这些方向可以通过应用于潜在向量集合的主成分分析（PCA）等方法找到，或者它们可能对应于表示特定语义属性的向量（这与潜在空间算术的主题密切相关，将在后续讨论）。

> 通过潜在空间修改生成输出的流程图。输入被编码为潜在向量，这些向量被系统地修改（插值或遍历），结果被解码回原始数据空间。

### 实际考量

- **模型架构：** 插值和遍历的质量在很大程度上取决于自编码器类型。VAEs由于其概率性质和施加先验（通常是高斯分布）的KL散度正则化 (regularization)项，与基本的确定性自编码器相比，倾向于生成更平滑、更连续的潜在空间，适合进行生成。像$\beta$-VAE这样的架构明确旨在改进解耦，使轴对齐 (alignment)遍历更有意义。
- **潜在空间结构：** 学习到的潜在空间可能并非完全均匀或连续。你可能会遇到“空洞”，其中解码的点毫无意义，或者过渡突然的区域。这反映了模型对数据分布的理解（或缺乏理解）。
- **选择点/方向：** 起始点（$x_a, x_b, x_{start}$）和遍历方向的选择显著影响结果。通常需要通过实验来寻找有趣或有启发性的路径和轴。

插值和遍历是强大的工具，可用于交互式地考察你的自编码器所学到的内容。它们将抽象的潜在空间转化为数据空间中可观察到的变化，提供了关于表示的结构、连续性和解耦潜力的直观理解。这些方法为更复杂的修改铺平了道路，例如使用潜在空间算术进行语义属性编辑，我们将在接下来进行讨论。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器，这是一种生成模型，用于学习连续且可解释的潜在表示，对于平滑插值和有意义的遍历至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的教科书，提供了自编码器、变分自编码器以及表示学习和潜在空间理论基础的全面内容。
- [CS236: Deep Generative Models - Variational Autoencoders (VAEs)](https://deepgenerativemodels.github.io/variational_autoencoder/) — Stefano Ermon, Aditya Grover (2023)
  Journal: Stanford University; Publisher: Stanford University
  斯坦福大学CS236课程的讲义，涵盖变分自编码器和潜在空间操作的实际应用。
