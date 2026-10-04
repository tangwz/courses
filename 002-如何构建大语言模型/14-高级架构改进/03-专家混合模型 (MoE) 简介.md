# 专家混合模型 (MoE) 简介

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications/introduction-mixture-of-experts-moe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

扩展大型语言模型通常需要在模型容量与计算成本之间做出权衡。标准Transformer模型随着规模增大，参数 (parameter)密度也越高。在一次前向传播中，每个参数都参与了对每个输入token的计算。尽管增加参数通常能提升性能（正如缩放法则所示），但计算需求（以每秒浮点运算次数，即FLOPs衡量）也会成比例增长。当目标是达到万亿级参数的模型时，这构成了一个显著的障碍。

专家混合模型 (MoE) 提供了一种基于条件计算的不同扩展方法。不同于每个Transformer块中单一的前馈网络（FFN）层，MoE层将其替换为多个并行的FFN“专家”网络。对于每个输入token，一个路由机制（通常称为“门控网络”）会动态选择这些专家中的一个小子集（例如，前1个或前2个）来处理该token。

> MoE层的简化视图。输入token由路由器引导至可用专家网络中的一个小子集。只有被选中的专家对该token执行计算。

主要思想是**稀疏性**。尽管MoE层中的参数总数（路由器和所有专家的参数总和）可以显著多于标准FFN层，但用于处理单个token的*实际使用*参数数量保持相对较小，这由激活的专家数量决定。

让我们考虑一个典型的Transformer块中的FFN层。如果隐藏维度是 $d_{model}$ 且FFN内部维度是 $d_{ff}$，则计算量大约为 $2 \times d_{model} \times d_{ff}$ FLOPs。

在一个有 $N$ 个专家的MoE层中，每个专家都与原始FFN具有相同维度，并且路由器选择前 $k$ 个专家（其中 $k$ 远小于 $N$，通常 $k=1$ 或 $k=2$），每个token的计算成本大致为：


$$
\text{MoE层FLOPs} \approx \text{路由器FLOPs} + k \times (2 \times d_{model} \times d_{ff})
$$


路由器的计算量与专家相比通常可以忽略不计。如果 $k=1$，每个token的FLOPs与原始的密集型FFN层相似，尽管总参数数量大约增加了 $N$ 倍。这使得构建具有潜在万亿级参数的模型成为可能，同时在训练和推理 (inference)期间保持可控的计算预算。

这里是PyTorch中的结构示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """一个简单的前馈网络专家"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU() # 或者GeLU、SwiGLU等

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
"""专家混合层"""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家池
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

        # 门控网络（学习将token路由到专家）
        # 输入：token表示 (d_model),
        # 输出：每个专家的分数 (num_experts)
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x 的形状：(序列长度, 批次大小, d_model)
        # 或者 (批次大小, 序列长度, d_model)
        # 为简化起见，此处假设形状为 (批次大小, 序列长度, d_model)
        batch_size, seq_len, d_model = x.shape
        # 重塑为 (批次*序列长度, d_model)
        x_reshaped = x.view(-1, d_model)

        # 1. 从门控网络获取路由权重
        # logits 形状：(批次*序列长度, 专家数量)
        router_logits = self.gating_network(x_reshaped)
        # 对专家进行Softmax
        routing_weights = F.softmax(router_logits, dim=1)

        # 2. 选择Top-k专家并获取其权重
        # top_k_weights, top_k_indices 形状：
        # (批次*序列长度, top_k)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=1
        )

        # 归一化Top-k权重
        top_k_weights_norm = (
            top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        )

        # 3. 计算专家输出（简化 -
        #    实际实现为提高效率更为复杂）
        # 初始化最终输出张量
        final_output = torch.zeros_like(x_reshaped)

        # 这是一个简化的循环，
        # 高效实现使用scatter/gather操作
        for i in range(batch_size * seq_len):
            token_input = x_reshaped[i]
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                expert_weight = top_k_weights_norm[i, k]

                # 计算所选专家的输出
                expert_output = self.experts[expert_idx](token_input)

                # 累加加权输出
                final_output[i] += expert_weight * expert_output

        # 重塑回原形状
        return final_output.view(
            batch_size, seq_len, d_model
        ) # 重塑回原形状
```

*注：上述`forward`方法为清晰起见使用了简单的循环。实际实现依赖于优化操作，以高效地处理跨批次和设备的稀疏路由，避免对token进行显式循环。*

MoE方法的主要益处是这种参数数量与计算量的解耦。它使得模型能够显著增加其容量，可能带来更佳的性能和知识表示，而无需按比例增加处理每个token所需的FLOPs。此外，专家可能能够专注于处理不同类型的输入或语言现象，尽管验证和控制这种专业化仍然是活跃的研究方向。

然而，MoE带来了一系列自身挑战，特别是在训练动态和实现复杂性方面。确保路由器有效地将token分发给专家（负载均衡）以及管理分布式环境中的通信开销，是我们将在后续章节中讨论的重要考量。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合（MoE）层，这是将MoE应用于大规模神经网络的基础工作，包括门控机制和负载均衡。
- [Switch Transformers: Scaling to Trillion Parameter Models with Constant Cost](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  展示了MoE层如何使Transformer模型扩展到万亿参数，同时保持每令牌计算成本不变，引入了高效的Switch机制。
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) — Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, Claire Cui (2022)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2112.06905](https://doi.org/10.48550/arXiv.2112.06905)
  介绍了大规模MoE语言模型GLaM，侧重于超稀疏MoE架构的高效训练和推理策略。

---

[上一节](02-Transformer%E7%9A%84%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md) · [下一节](04-MoE%20%E4%B8%AD%E7%9A%84%E8%B7%AF%E7%94%B1%E6%9C%BA%E5%88%B6.md)
