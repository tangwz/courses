---
course: "vae-representation-learning"
chapter: "disentangled-representation-learning-vaes"
lesson: "practice-disentangled-vaes"
sourceId: 6371
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes/practice-disentangled-vaes"
title: "动手实践：训练与评估解耦变分自编码器"
description: "训练变分自编码器（例如 Beta-VAE）以实现解耦，并使用标准指标评估结果。"
order: 8
plots: ["plots/6371-0.json"]
sourceHash: "17a55183610e5a986734dab11831c00ec0032e798def1195be7cbf3bc73dfda3"
sourceCorrections: []
---

解耦表示学习的实践应用包括训练旨在实现更好解耦的 VAE 变体。具体来说，训练 $\beta$-VAE 并使用互信息间隙（MIG）和独立属性可预测性（SAP）等指标来评估其效果。

这里的目的不是提供一个完整的、可直接复制粘贴的代码库，而是概述必要的步骤和注意事项，让您能够试验并加深您的理解。我们假定您能够熟练地在 PyTorch 或 TensorFlow 等框架中实现一个标准 VAE。

### 准备工作：数据集与库

对于解耦实验，具有已知真实变异因素的合成数据集非常有用。**dSprites 数据集**是一个流行的选择。它由 6 个独立的潜在因素生成的 2D 形状（正方形、椭圆形、心形）组成：颜色（始终为白色）、形状、大小、方向、X 轴位置和 Y 轴位置。能够使用这些真实因素使我们能够定量测量模型解耦它们的程度。

您将需要标准的深度学习 (deep learning)工具包：

- 一个深度学习框架（PyTorch 或 TensorFlow）。
- 用于数值运算的 NumPy。
- Scikit-learn 用于可能的辅助函数，特别是指标计算（例如，`mutual_info_regression` 或用于训练简单分类器）。
- Matplotlib 或 Seaborn 用于可视化。

### 实现和训练 $\beta$-VAE

回想我们讨论过的，即 $\beta$-VAE 通过修改标准 VAE 目标函数，在 KL 散度项中引入系数 $\beta$：


$$
\mathcal{L}_{\beta-VAE} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \beta \cdot D_{KL}(q_{\phi}(z|x) || p(z))
$$


当 $\beta > 1$ 时，KL 散度会受到更强的限制，促使近似后验分布 $q_{\phi}(z|x)$ 更接近先验分布 $p(z)$ (通常是各向同性高斯分布 $\mathcal{N}(0, I)$)。这种压力可以促使模型学习到更解耦的表示。

**1. 模型架构：**
您的 VAE 架构可以是处理 dSprites 这类图像数据的标准卷积设置。

- **编码器**：由若干卷积层和全连接层组成，输出潜在分布 $q_{\phi}(z|x)$ 的均值 $\mu_z$ 和对数方差 $\log \sigma_z^2$。
- **解码器**：由全连接层和反卷积（或上采样 + 卷积）层组成，从潜在样本 $z$ 重建输入图像。

**2. $\beta$-VAE 损失函数 (loss function)：**
与标准 VAE 相比，实现上的改动很小。假设 `reconstruction_loss` 是您的负对数似然项（例如，二元交叉熵或均方误差），而 `kl_divergence` 是 KL 项，那么您的组合损失计算如下：

```python
# beta-VAE 损失的伪代码
# mu, log_var 是编码器的输出
# x_reconstructed 是解码器的输出
# x_original 是输入图像
# beta 是超参数

reconstruction_loss = reconstruction_criterion(x_reconstructed, x_original)
# N(mu, sigma^2) 与 N(0, I) 之间的 KL 散度
# 0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp())
kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1) 
kl_divergence = torch.mean(kl_divergence) # 对批次求平均

total_loss = reconstruction_loss + beta * kl_divergence 
# 反向传播 total_loss
```

**3. 训练要点：**

- 训练您的 $\beta$-VAE 足够多的周期。
- 试验不同的 $\beta$ 值。常见的尝试值可以是 $\beta=1$（标准 VAE）、$\beta=2, 4, 8, 16$。
- 同时监控重建损失和 KL 散度。您可能会观察到，随着 $\beta$ 值的增加，KL 散度项会变小（如果使用标准高斯先验，则更接近每个维度的目标值 0），但重建质量可能会下降。这是经典的权衡。

### 评估解耦

一旦您的模型训练完成，您需要评估其学习到的表示的解耦程度。

**1. 定性评估：潜在空间遍历**
一种简单而能说明问题的定性评估解耦的方法是执行潜在空间遍历。

- 取一个输入图像并将其编码以获得其潜在均值 $\mu_z$。
- 选择一个潜在维度 $z_i$。
- 通过解码 $\mu_z$ 的变体来生成新图像，其中只有 $z_i$ 在一定范围（例如，如果 $p(z) = \mathcal{N}(0,I)$，则从 -3 到 3 标准差）内变化，而其他潜在维度则固定为 $\mu_z$ 中的值。
- 如果第 $i$ 个潜在维度解耦良好，则仅改变 $z_i$ 应该导致生成图像中单个可解释的变异因素发生变化（例如，仅大小变化，或仅 x 轴位置变化）。

您可以创建图像网格，其中每行（或每列）对应于遍历一个不同的潜在维度。这种视觉检查可以很能说明问题。

**2. 定量指标**
为了更严谨的评估，我们使用定量指标。这些通常需要数据集中的真实因素标签。

**互信息间隙（MIG）**
MIG 旨在测量每个真实因素被单个潜在维度捕获的程度。
对于每个真实因素 $y_k$：

1. 编码一批数据以获得潜在均值 $Z$。
2. 对于每个潜在维度 $z_j$，计算经验互信息 $I(z_j; y_k)$。这通常涉及将 $z_j$ 离散化为多个区间。
3. 找到与 $y_k$ 具有最高互信息的潜在维度 $z_{j^*}$：$\max_j I(z_j; y_k)$。
4. 找到与 $y_k$ 具有第二高互信息的潜在维度 $z_{j^{**}}$：$\max_{j \neq j^*} I(z_j; y_k)$。
5. 因素 $y_k$ 的间隙是 $\frac{I(z_{j^*}; y_k) - I(z_{j^{**}}; y_k)}{H(y_k)}$，其中 $H(y_k)$ 是真实因素 $y_k$ 的熵。
   最终的 MIG 分数是所有真实因素的这些间隙的平均值。

更高的 MIG 分数表示更好的解耦效果，因为它意味着每个因素主要由一个潜在维度表示，与次优信息维度之间存在明显的“间隙”。

```python
# 计算 MIG 的伪代码（简化版）
# latents: (样本数, 潜在维度数) - 来自 VAE 的编码均值
# factors: (样本数, 因素数) - 真实因素值
# 潜在维度离散化的 bin 数量 = 20

def calculate_mig(latents, factors):
    num_latents = latents.shape[1]
    num_factors = factors.shape[1]
    mig_scores_per_factor = []

    for k in range(num_factors): # 对于每个真实因素 y_k
        y_k = factors[:, k]
        # 估计 H(y_k) - 如果是连续值可能需要离散化，或者使用已知值
        # 对于 dSprites，因素是离散的，因此可以直接计算 H(y_k)
        h_y_k = calculate_entropy(y_k) 

        mutual_informations = []
        for j in range(num_latents): # 对于每个潜在维度 z_j
            z_j_discretized = discretize_latent(latents[:, j], n_bins_for_latent_discretization)
            # 使用 sklearn.metrics.mutual_info_score 或类似函数
            mi_zj_yk = compute_mutual_information(z_j_discretized, y_k) 
            mutual_informations.append(mi_zj_yk)

        sorted_mi = sorted(mutual_informations, reverse=True)
        if len(sorted_mi) < 2: continue # 潜在维度不足以计算间隙

        gap_k = (sorted_mi[0] - sorted_mi[1]) / h_y_k if h_y_k > 0 else 0
        mig_scores_per_factor.append(gap_k)

    return sum(mig_scores_per_factor) / len(mig_scores_per_factor) if mig_scores_per_factor else 0

# 辅助函数，例如 discretize_latent, calculate_entropy, compute_mutual_information
# 需要实现。对于 dSprites，因素是离散的，简化了熵的计算。
# sklearn.feature_selection.mutual_info_regression (如果 y_k 是连续的) 
# 或 sklearn.metrics.mutual_info_score (如果 y_k 是离散的，在 z_j 离散化后) 可以使用。
```

**独立属性可预测性（SAP）**
SAP 通过评估每个潜在维度预测单个真实因素的效果来衡量解耦。
对于每个真实因素 $y_k$：

1. 训练一个简单的线性分类器（例如，逻辑回归）仅使用潜在维度 $z_j$ 来预测 $y_k$。计算其准确率（如果 $y_k$ 是连续的，则计算 R-平方）。对所有 $j$ 都这样做。
2. 找到对 $y_k$ 预测能力最强的潜在维度 $z_{j^*}$。
3. 找到对 $y_k$ 预测能力第二强的潜在维度 $z_{j^{**}}$。
4. 因素 $y_k$ 的 SAP 分数是使用 $z_{j^*}$ 和 $z_{j^{**}}$ 时的预测分数（例如，准确率）之差。
   最终的 SAP 分数是所有真实因素的这些分数差异的平均值。

更高的 SAP 分数表示单个潜在维度能够预测单个变异因素。

```python
# 计算 SAP 的伪代码（简化版）
# latents: (样本数, 潜在维度数) - 来自 VAE 的编码均值
# factors: (样本数, 因素数) - 真实因素值
# (假定因素是离散的，以便计算分类准确率)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def calculate_sap(latents, factors, test_size=0.2, random_state=42):
    num_latents = latents.shape[1]
    num_factors = factors.shape[1]
    sap_scores_per_factor = []

    # 拆分数据用于训练分类器
    # 注意：更完整的评估会使用交叉验证。
    latents_train, latents_test, factors_train, factors_test = train_test_split(
        latents, factors, test_size=test_size, random_state=random_state
    )

    # 缩放潜在维度（可选，但对线性模型是良好实践）
    scaler = StandardScaler()
    latents_train_scaled = scaler.fit_transform(latents_train)
    latents_test_scaled = scaler.transform(latents_test)

    for k in range(num_factors): # 对于每个真实因素 y_k
        y_k_train = factors_train[:, k]
        y_k_test = factors_test[:, k]

        prediction_scores = []
        for j in range(num_latents): # 对于每个潜在维度 z_j
            z_j_train = latents_train_scaled[:, j].reshape(-1, 1)
            z_j_test = latents_test_scaled[:, j].reshape(-1, 1)

            # 训练一个简单的分类器（例如，逻辑回归）
            # 处理 y_k 在训练/测试集中只有一个类别的情况
            try:
                if len(np.unique(y_k_train)) < 2: 
                    score = 0.0 # 或根据需要处理
                else:
                    model = LogisticRegression(solver='liblinear', multi_class='auto', C=0.1) # 保持模型简单
                    model.fit(z_j_train, y_k_train)
                    score = model.score(z_j_test, y_k_test)
                prediction_scores.append(score)
            except ValueError: # 例如，如果 y_k_train 只有一个类别
                prediction_scores.append(0.0)

        if not prediction_scores: continue

        sorted_scores = sorted(prediction_scores, reverse=True)
        if len(sorted_scores) < 2: continue # 潜在维度不足

        # 前两名分数之差
        sap_k = sorted_scores[0] - sorted_scores[1]
        sap_scores_per_factor.append(sap_k)

    return sum(sap_scores_per_factor) / len(sap_scores_per_factor) if sap_scores_per_factor else 0
```

> **关于指标实现的说明**：上述伪代码简化了某些方面。实际实现需要仔细处理数据拆分（用于指标分类器的训练/验证/测试）、对探测分类器进行超参数 (parameter) (hyperparameter)调优（尽管通常偏好使用简单模型来测试潜在维度固有的可预测性），以及可能对多次运行的结果取平均。对于 dSprites，因素值是离散的，这简化了操作。

### 整合所有内容：一个示例工作流程

1. **准备数据**：加载 dSprites 数据集。分离图像及其真实因素标签。
2. **训练模型**：
   - 训练一个标准 VAE ($\beta=1$) 作为基准。
   - 训练几个 $\beta$ 值逐渐增加的 $\beta$-VAE（例如，$\beta \in \{2, 4, 8, 16, 32\}$）。
   - 对于每个模型，保存学习到的编码器。
3. **评估模型**：
   - 对于每个训练好的编码器：
     - 将 dSprites 图像的保留测试集通过编码器，以获得其潜在表示（例如，均值 $\mu_z$）。
     - 对一些示例图像进行潜在空间遍历，以定性评估解耦。
     - 使用潜在表示和测试集的真实因素计算 MIG 分数。
     - 计算 SAP 分数。
     - 此外，记录该模型在测试集上的重建误差（例如，MSE 或 BCE）。
4. **分析结果**：
   - 绘制解耦分数（MIG、SAP）与 $\beta$ 值之间的关系图。
   - 绘制重建误差与 $\beta$ 值之间的关系图。
   - 您应该观察到一个趋势：更高的 $\beta$ 值通常会带来更好的解耦分数（达到一定程度），但代价是更高的重建误差。这显示了其中的权衡。



![解耦程度与 Beta-VAE 中的 Beta 值关系](plots/6371-0.json)



> 结果显示互信息间隙（MIG）和独立属性可预测性（SAP）分数随 $\beta$ 值的增加而提高，同时重建损失也倾向于增加。这显示了 $\beta$-VAE 中常见的权衡。请注意，使用了对数刻度以更好地显示不同数量级的数据。

### 更多实践

本次动手练习提供了一个起点。您可以通过以下方式进行扩展：

- **实现其他指标**：尝试 DCI（解耦性、完整性、信息性）或 Factor-VAE 指标，它们更复杂但提供了不同的视角。
- **训练其他模型**：实现和评估 FactorVAE 或 TCVAE（总相关性 VAE），它们以不同方式修改 VAE 目标以促进解耦，正如本章讨论的那样。比较它们的性能与 $\beta$-VAE。
- **使用不同数据集**：尝试其他解耦数据集，如 3D Shapes、MPI3D，甚至尝试将这些技术应用于更复杂的数据集（尽管没有真实因素，指标计算可能更难）。
- **考察局限性**：观察指标对超参数 (parameter) (hyperparameter)（MIG 的分箱数量、SAP 的分类器复杂度）的敏感程度。思考当前指标的局限性以及在定义和实现“真正”解耦方面持续存在的研究挑战。

通过积极使用这些模型和指标，您将对解耦表示学习领域的挑战和成果形成更强的直觉。请记住，这是一个活跃的研究领域，完美的解耦，特别是在没有监督的复杂数据集上，仍然是一个未解决的问题。
