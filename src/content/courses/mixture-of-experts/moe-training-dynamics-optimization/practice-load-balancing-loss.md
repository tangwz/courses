---
course: "mixture-of-experts"
chapter: "moe-training-dynamics-optimization"
lesson: "practice-load-balancing-loss"
sourceId: 4697
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/practice-load-balancing-loss"
title: "实践操作：实现和调整负载均衡损失"
description: "MoE训练中辅助损失函数的代码实现与调整。"
order: 7
plots: ["plots/4697-0.json"]
sourceHash: "4d3ff26b585d74b96f2498522da3ebc975c28498c439726cd23a065b739701dc"
sourceCorrections: []
---

实现和调整旨在促进MoE层中专家之间负载均衡的辅助损失函数 (loss function)在此展示。确保令牌在专家之间相对均匀分布对训练稳定性和计算效率很重要。如果没有明确的机制，路由器可能会压倒性地偏向一小部分专家，导致其他专家利用率不足和潜在的训练崩溃。

我们将侧重于将辅助损失 $L_{aux}$ 整合到总损失函数中：
$L_{total} = L_{task} + \alpha L_{aux}$
其中 $L_{task}$ 是主要目标（例如，用于分类的交叉熵），$\alpha$ 是一个控制平衡激励强度的超参数 (parameter) (hyperparameter)。

### 实现常见负载均衡损失

我们来看一个大型模型中的单个MoE层。对于一批 $T$ 个令牌和 $N$ 个专家，路由器为每个令牌-专家对输出逻辑值。这些逻辑值通常通过softmax函数得到门控概率 $G_{i,j}$，表示路由器分配给令牌 $i$ 由专家 $j$ 处理的概率。在top-k门控场景中（通常 k=1 或 k=2），这些概率决定每个令牌被路由到哪个或哪些专家。

为了计算 $L_{aux}$，我们需要从门控网络的输出中，在进行top-k选择*之前*，得出两个主要量：

1. **分配给专家$j$的令牌分数 ($f_j$)：** 这衡量了在top-k路由决策后，当前批次中分配给专家$j$的令牌比例。如果 $T_j$ 是路由到专家$j$的令牌数量，那么 $f_j = T_j / T$。
2. **专家$j$的平均路由器概率 ($P_j$)：** 这是批次中所有令牌分配给专家$j$的平均概率，计算时*使用top-k选择之前的概率*。
   $P_j = \frac{1}{T} \sum_{i=1}^T G_{i,j}$

#### 负载均衡损失（基于路由器概率）

一种常见方法，受MoE原始论文启发并在Switch Transformers等系统中使用，通过最小化专家工作量的方差来促进平衡，该方差使用路由器概率 $P_j$ 近似。损失通常被表述为向量 (vector) $f$ 和 $P$ 的点积：

$L_{aux} = N \cdot \sum_{j=1}^N f_j P_j$

这里，$N$是专家数量。最小化此损失会促使路由器将相似的概率（$P_j$）分配给接收相似比例令牌（$f_j$）的专家。直观地讲，如果一个专家获得许多令牌（$f_j$高），我们希望分配给它的平均概率（$P_j$）在整个批次中相对较低，这表明路由器没有普遍偏向它。反之，如果$f_j$低，我们希望$P_j$更高以鼓励其使用。乘以$N$可以适当地调整损失的比例。

#### 变异系数平方损失（CV损失）

另一种有效的损失旨在最小化每个专家令牌分布的平方变异系数（CV）。设 $T_j$ 是路由到专家 $j$ 的令牌数量。CV平方损失为：

$L_{aux} = \frac{\text{方差}(T_1, T_2, ..., T_N)}{(\text{均值}(T_1, T_2, ..., T_N))^2}$

由于每个专家的平均令牌数量为 $T/N$，这简化为：

$L_{aux} = \frac{N \sum_{j=1}^N T_j^2}{T^2} - 1$

这种损失直接惩罚分配给每个专家令牌数量 $T_j$ 的不平衡。接近零的值表示完美平衡（对于所有 $j$， $T_j = T/N$）。

### 实现草图（PyTorch风格伪代码）

我们来概述一下如何在模型的正向传播或训练步骤中计算这些值。假设 `router_logits` 是门控网络对一批令牌的输出，形状为 `(T, N)`。

```python
import torch
import torch.nn.functional as F

def compute_load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int = 1) -> torch.Tensor:
    """
    计算常见的负载均衡辅助损失。

    参数：
        router_logits：来自门控网络的原始逻辑值（形状：[T, N]）。
        num_experts：专家总数（N）。
        top_k：每个令牌路由到的专家数量。

    返回：
        标量辅助损失值。
    """
    T, N = router_logits.shape # T = 令牌数量, N = 专家数量

    # 获取路由器概率（对每个令牌的专家进行softmax）
    router_probs = F.softmax(router_logits, dim=-1) # 形状: [T, N]

    # --- 计算 f_j: 分配给专家 j 的令牌分数 ---
    # 获取每个令牌的 top-k 专家索引和门控值
    # gates: 所选专家的概率
    # indices: 所选专家的索引
    gates, indices = torch.topk(router_probs, top_k, dim=-1) # gates 形状: [T, k], indices 形状: [T, k]

    # 创建一个掩码，指示每个令牌被路由到哪个专家（k=1的简化）
    # 对于 k > 1，这需要根据分配方式进行调整。
    # 为简单起见，这里假设 k=1。
    if top_k == 1:
        # 创建一个独热张量，指示每个令牌选择的专家
        # 使用 scatter_add_ 来统计每个专家的令牌数量
        tokens_per_expert = torch.zeros(N, device=router_logits.device, dtype=torch.float32)
        # 如果可用/适用，可使用 index_add_ 以获得更好的性能/清晰度
        # indices.squeeze(1) 移除 k=1 维度
        tokens_per_expert.index_add_(0, indices.squeeze(1), torch.ones(T, device=router_logits.device)) # 形状: [N]
        # 注意：如果需要，确保梯度正确流动，可能使用 scatter_add 或类似的可微分操作
        # 对于损失计算，通常只需要直接计数，梯度通过 P_j 流动。

        # 计算分数 f_j
        f_j = tokens_per_expert / T # 形状: [N]

    else:
        # 处理 k > 1 需要对 f_j 有更明确的定义。
        # 通常，f_j 仍然表示专家 j 所填充的*槽位*的比例，
        # 考虑到每个令牌使用 k 个槽位。或者它可能计算独有的令牌。
        # 为清晰起见，我们将继续使用 k=1 简化损失公式。
        # 一种常见方法可能涉及根据分配计算负载。
        # 对于本例，我们引发错误或实现一个特定的 k>1 策略。
        raise NotImplementedError("k > 1 的负载均衡需要为 f_j 进行专门实现")

    # --- 计算 P_j: 专家 j 的平均路由器概率 ---
    P_j = router_probs.mean(dim=0) # 形状: [N]

    # --- 计算辅助损失 ---
    # L_aux = N * sum(f_j * P_j)
    # 如果梯度只应通过 P_j 流动，请确保 f_j 是分离的
    # 根据实现方式，通过 f_j 的梯度可能也是期望的或有问题的
    loss = num_experts * torch.sum(f_j * P_j)

    # --- 替代方案：CV 平方损失 ---
    # 需要上面计算的 T_j = tokens_per_expert
    # mean_tokens = T / N
    # variance = torch.sum((tokens_per_expert - mean_tokens)**2) / N
    # cv_squared_loss = variance / (mean_tokens**2)
    # loss = cv_squared_loss # 或使用涉及 sum(T_j^2) 的简化公式

    return loss

# --- 在训练循环中 ---
# model_output, router_logits = model(input_data) # 假设模型返回逻辑值
# task_loss = compute_task_loss(model_output, labels)
# aux_loss = compute_load_balancing_loss(router_logits, model.num_experts, model.top_k)
# alpha = 0.01 # 系数示例
# total_loss = task_loss + alpha * aux_loss
# total_loss.backward()
# optimizer.step()
```

\*\*注意：\*\*计算 $f_j$ 和处理梯度的具体实现，特别是对于 $k > 1$ 或使用专家容量限制时，在DeepSpeed、Tutel等框架或自定义实现中可能会有所不同。目标不变：得出负载度量（$f_j$或$T_j$）和平均分配概率（$P_j$）以计算$L_{aux}$。

### 调整平衡系数（$\alpha$）

$\alpha$的选择很重要，通常需要经验性调整。

- **起始点：** 值通常较小，常在 $10^{-2}$ 到 $10^{-3}$ 的范围。过大的 $\alpha$ 可能会主导任务损失，以牺牲学习质量为代价强制实现完美平衡。过小的值可能不足以抵消路由器崩溃或严重不平衡。
- **监控指标：** 训练期间，监控重要指标：
  - **专家利用率：** 跟踪每个专家在每批次或多个步骤中处理的令牌数量或比例。可以将其可视化为随时间变化的直方图或折线图。理想情况下，专家应具有大致相似的利用率，尽管完美一致并非总是必要或最佳的。
  - **$L_{aux}$ 的大小：** 观察辅助损失本身的值。它应随着训练的进行和平衡的改善而普遍下降。
  - **任务损失和验证性能：** 确保增加 $\alpha$ 不会过度损害模型学习主要任务的能力。跟踪 $L_{task}$ 和相关的验证指标（准确率、困惑度等）。
- **权衡分析：** 存在固有的权衡。增加 $\alpha$ 通常会改善平衡（减少每个专家令牌的方差），但如果路由器被迫偏离任务本身的最佳路由，可能会轻微降低任务性能。找到一个 $\alpha$，它在提供可接受的平衡的同时，不会显著影响验证指标。

#### 可视化示例

可视化专家利用率有助于诊断不平衡。您可以绘制在不同$\alpha$值下，每个专家令牌的标准差随训练步骤的变化情况。



![α 对负载均衡的影响（每个专家令牌标准差）](plots/4697-0.json)



> 训练期间路由到每个专家令牌数量的标准差，针对不同平衡系数$\alpha$值绘制。较高的$\alpha$通常会导致较低的标准差，这表示更好的负载平衡，但需要监控任务性能。

- **实验：** 尝试几个不同的 $\alpha$ 值（例如，0.001、0.01、0.05），并比较由此产生的利用模式和验证性能曲线。选择提供最佳折衷的值。
- **$\alpha$ 调度：** 一些实践者发现调度 $\alpha$ 很有用，可能在训练早期开始时设置较高值以建立平衡，然后后期降低它以允许更精细的任务专门化。然而，一个恒定且经过良好调整的 $\alpha$ 通常就足够了。

### 最终思考

在MoE训练中，实现和调整负载均衡损失是常见做法。虽然具体公式可能略有不同，但根据利用率不平衡（$f_j$, $T_j$）和路由器置信度（$P_j$）添加惩罚的原则是普遍的。仔细监控和调整 $\alpha$ 系数对于从稀疏专家模型获得稳定训练和最佳性能是必需的。请记住，这些损失与专家容量和路由器设计等其他因素相互作用，因此需要采用整体方法来优化MoE训练。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合（MoE）层和负载平衡辅助损失的概念，这对所讨论的损失至关重要。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0998.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 23; Pages: 1-39
  描述了Switch Transformer架构，并详细阐述了用于确保大规模MoE模型中专家均匀利用的 N ⋅ ∑ f_j P_j 负载平衡损失。
- [Mixtral 8x7B (MoE) Model Documentation](https://huggingface.co/docs/transformers/en/model_guides/mixtral) — Hugging Face (2023)
  Publisher: Hugging Face
  提供了实践背景和示例，说明了负载平衡损失（特别是基于路由概率和专家负载的损失）如何在Transformers库中的一个知名开源MoE模型中实现和使用。
