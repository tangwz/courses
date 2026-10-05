# DiT 的实现考量

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/dit-implementation-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实现 Diffusion Transformer (DiT) 需要留意多个实用细节。尽管用 Transformer 块替换 U-Net 骨干网络，在建模长距离依赖性方面可能带来优势，但这也引出计算成本、数据处理和训练稳定性方面的特定难题与考量。构建或训练 DiT 模型时需要处理的重要方面将被检查。

### 计算成本与可扩展性

Transformer 的主要组成是自注意力 (self-attention)机制 (attention mechanism)。标准自注意力机制的计算和内存复杂度为 $O(N^2)$，其中 $N$ 是序列长度。在 DiT 处理图像数据的情况下，$N$ 对应于图像被划分的补丁数量。对于分辨率为 $H \times W$、补丁大小为 $P \times P$ 的图像，补丁数量为 $N = (H \times W) / P^2$。

这种二次方扩展意味着将图像分辨率加倍（像素数量增加四倍）或将补丁大小减半（补丁数量增加四倍），都会使注意力计算的成本增加约 16 倍。这显著影响训练时间和 GPU 内存需求，特别是对于高分辨率图像生成。尽管像 FlashAttention 这样的技术可以优化注意力实现，但它们并未改变基本的二次方复杂度。这与基于 CNN 的 U-Net 形成对比，在 U-Net 中，卷积操作通常与像素数量呈线性关系，即 $O(N_{pixels})$。因此，选择补丁大小和管理序列长度是在设计和训练 DiT 时要点考量。

### 块嵌入 (embedding)策略

Transformer 处理的是令牌序列。要将它们应用于图像，输入图像 $x_t$ 在时间步 $t$ 必须转换为这样的序列：

1. **图像分块**: 图像张量（例如，形状为 `[批次, 通道, 高度, 宽度]`）被划分为不重叠的块网格。对于大小为 $256 \times 256$、块大小为 $16 \times 16$ 的图像，您将得到 $(256/16) \times (256/16) = 16 \times 16 = 256$ 个块。
2. **线性投影**: 每个块（例如，形状为 `[通道, P, P]`）被展平并线性投影到一个维度为 $D$（Transformer 的隐藏维度）的嵌入向量 (vector)。这会产生一个包含 $N$ 个嵌入向量的序列，通常形状为 `[批次, N, D]`。
3. **位置嵌入**: 由于自注意力 (self-attention)机制 (attention mechanism)是置换不变的，模型需要关于每个块原始空间位置的信息。位置嵌入被添加到块嵌入中。这些可以在训练期间学习，也可以是固定的（例如，二维正弦嵌入）。应用于序列顺序的标准一维位置嵌入在许多实现中都很常见。

块大小 $P$ 的选择影响很大。

- **较小的 $P$**: 导致序列 $N$ 更长，二次方地增加计算成本。但是，它允许模型通过初始投影潜在地捕获每个块内更精细的细节。
- **较大的 $P$**: 减少 $N$ 和计算成本，但在 Transformer 层处理它们之前，可能会将块内的重要局部细节平均化。

### Transformer 块设计与条件作用

标准 DiT 架构使用一系列 Transformer 块。每个块通常包含层归一化 (normalization) (LN)、多头自注意力 (self-attention) (MHSA) 和一个 MLP（通常是两个带有 GeLU 等激活函数 (activation function)的线性层）。DiT 中的一个重要创新是如何使用自适应层归一化，特别是 `adaLN-Zero`，来融入时间步 $t$ 和条件信息 $c$（例如类别标签）。

`adaLN-Zero` 不是简单地将时间步和条件嵌入 (embedding)添加到序列中，而是调制 Transformer 子块的输出。对于隐藏状态 $h$，`adaLN-Zero` 操作是：


$$
\text{adaLN-Zero}(h, \gamma, \beta, \alpha) = \alpha \cdot \text{LayerNorm}(h) + \beta
$$


这里，$\gamma$（LayerNorm 内部用于缩放），$\beta$（偏移），和 $\alpha$（输出缩放）由一个小型 MLP 动态计算，该 MLP 以时间步 $t$ 和条件 $c$ 的嵌入作为输入。这些嵌入通常首先进行处理：

1. 时间步 $t$ 使用正弦特征后接一个 MLP 转换为嵌入。
2. 条件 $c$（例如，一个类别索引）被映射到一个学习到的嵌入向量 (vector)。
3. 这两个嵌入相加：$emb = \text{MLP}(\text{sinusoidal}(t)) + \text{Embedding}(c)$。
4. 最终的 MLP 预测参数 (parameter)：$(\gamma, \beta, \alpha) = \text{MLP}_{\text{adaLN}}(emb)$。

这些自适应参数应用于特定点，通常在每个 Transformer 块内的 MHSA 和 MLP 层之前，有时也用于调制残差连接。“Zero”部分指的是将生成 $\alpha$ 和 $\beta$（并影响 $\gamma$）的最终 MLP 层初始化为输出零。这意味着这些自适应层在初始阶段充当恒等函数，有助于训练稳定性，尤其是在早期。

### 训练稳定性与优化

训练 DiT 等大型 Transformer 模型需要仔细优化：

- **优化器**: AdamW 是一个常见选择，通常使用 $\beta_1=0.9$，$\beta_2=0.999$。
- **学习率**: 适当的学习率调度（例如，带预热的余弦衰减）很重要。峰值学习率可能在 $1e^{-4}$ 到 $5e^{-4}$ 的范围，具体取决于模型大小和批次大小。
- **权重 (weight)衰减**: 应用于稳定训练并改进泛化能力。
- **混合精度训练**: 使用 FP16（16 位浮点）或 BF16（脑浮点）对于管理内存使用和加速现代 GPU（如 NVIDIA Tensor Cores）上的训练几乎是不可或缺的。这需要梯度缩放以防止低精度格式的数值下溢/溢出问题。PyTorch 的 `torch.cuda.amp` 或 TensorFlow 的混合精度 API 大致上能自动处理此问题。
- **梯度裁剪**: 有时需要防止梯度爆炸，尤其是在训练早期或批次较大时。通常按全局范数裁剪。
- **EMA（指数移动平均）**: 相比直接使用优化器中的原始权重，维护模型权重的 EMA 通常能使最终评估的模型表现更好。EMA 权重通常在推理 (inference)/采样期间使用。

### 模型大小选择与扩展

最初的 DiT 论文显示，这些模型表现出可预测的扩展特性。性能，以 FID (Fréchet Inception Distance) 等指标衡量，通常随模型大小（参数 (parameter)数量、深度、宽度）和计算预算的增加而提高。常见配置包括：

- **DiT-S (小型)**: 块较少，隐藏维度 $D$ 较小。
- **DiT-B (基础型)**: 中等大小。
- **DiT-L (大型)**: 块更多， $D$ 较大。
- **DiT-XL (超大型)**: 更大，需要大量计算。

下表概括了扩展的权衡（数值为示例）：

| 模型 | 参数量（百万） | 相对计算量 | 潜在 FID（越低越好） |
| --- | --- | --- | --- |
| DiT-S | ~30 | 1x | 中等 |
| DiT-B | ~100 | 3-4x | 良好 |
| DiT-L | ~400 | 10-15x | 很好 |
| DiT-XL | ~600+ | 20-25x | 目前最佳 |

这种扩展行为使得研究人员可以估算通过投入更多计算资源可获得的性能提升。



[交互图表：DiT 扩展趋势示例](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/dit-implementation-considerations#plot-qy43fg)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "DiT-S",
        "DiT-B",
        "DiT-L",
        "DiT-XL"
      ],
      "y": [
        30,
        100,
        400,
        650
      ],
      "type": "bar",
      "name": "参数量（百万）",
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        "DiT-S",
        "DiT-B",
        "DiT-L",
        "DiT-XL"
      ],
      "y": [
        12,
        8,
        4,
        2.5
      ],
      "type": "scatter",
      "name": "示例 FID",
      "yaxis": "y2",
      "mode": "lines+markers",
      "line": {
        "color": "#f76707"
      },
      "marker": {
        "color": "#f76707"
      }
    }
  ],
  "layout": {
    "title": "DiT 扩展趋势示例",
    "xaxis": {
      "title": "模型大小"
    },
    "yaxis": {
      "title": "参数量（百万）",
      "color": "#4263eb"
    },
    "yaxis2": {
      "title": "示例 FID 分数（越低越好）",
      "overlaying": "y",
      "side": "right",
      "color": "#f76707",
      "range": [
        0,
        15
      ]
    },
    "legend": {
      "x": 0.1,
      "y": -0.3,
      "orientation": "h"
    },
    "autosize": true,
    "margin": {
      "l": 50,
      "r": 50,
      "b": 100,
      "t": 50
    }
  }
}
```

</details>



> Diffusion Transformer 模型大小、大致参数数量与潜在 FID 分数改进（分数越低表明图像质量越好）之间的关系。计算需求随模型大小显著增长。

### 实现实用建议

- **借鉴现有实现**: 从头开始构建 DiT 是复杂的。仔细查看受认可的开源实现（例如原始 DiT 仓库或 `diffusers` 等框架中的实现），以理解实用选择。
- **验证形状**: 在分块、嵌入 (embedding)、注意力以及反分块阶段，仔细追踪张量形状。形状不匹配是常见的错误源。
- **从小规模开始**: 在扩展之前，先用较小的模型（如 DiT-S）和较低分辨率进行实验。这能加快调试周期。
- **监控训练**: 使用 TensorBoard 或 Weights & Biases 等工具，在整个训练过程中追踪损失曲线、梯度范数和生成的图像样本。这有助于诊断发散或收敛缓慢等问题。
- **硬件**: 对硬件需求要有切实际的认知。有效训练更大的 DiT 通常需要多块高显存 (VRAM) GPU（例如 A100、H100）和分布式训练设置（例如使用 PyTorch 的 DistributedDataParallel）。

通过仔细考量这些计算、架构和优化方面的要点，您可以成功实现和训练 Diffusion Transformer 模型，用于高质量图像生成任务。

## 参考资料

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — William Peebles, Saining Xie (2023)
  Journal: Proceedings of the 40th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2212.09748](https://doi.org/10.48550/arXiv.2212.09748)
  介绍了扩散Transformer (DiT) 架构、其自适应层归一化 (adaLN-Zero) 条件化机制，并展示了扩散模型中可预测的缩放特性。
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  提出了视觉Transformer (ViT) 以及将图像转换为补丁序列以供Transformer处理的方法，这是DiT输入处理的基础。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  描述了一种优化的自注意力算法，显著提升了Transformer的计算速度和内存效率，对训练大型DiT模型有益。
- [Hugging Face Diffusers Library](https://huggingface.co/docs/diffusers/index) — Hugging Face (2024)
  一个广泛使用的开源库的官方文档，该库提供预训练扩散模型和工具，用于实现和训练自定义扩散模型，包括DiT。

---

[上一节](05-U-Net%E4%B8%8ETransformer%E5%9C%A8%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E6%AF%94%E8%BE%83.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%20DiT%20%E6%A8%A1%E5%9D%97.md)
