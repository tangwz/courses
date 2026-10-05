# 专家卸载到 CPU 或 NVMe

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading)

[返回章节目录](README.md) · [返回课程目录](../README.md)

专家混合模型（MoE）庞大的参数 (parameter)量是其最大的优势，也是部署时最主要的挑战。像 Mixtral 8x7B 这样的模型，每个 MoE 层包含八个不同专家的权重 (weight)，导致总参数大小远超绝大多数大型且昂贵的 GPU 加速器的容量。然而，在任何给定的前向传播过程中，每个 token 仅激活这些专家中的一到两个。这种计算稀疏性是管理推理 (inference)性能的主要方面。专家卸载运用这一特性，通过将大部分非活跃专家参数存储在一个更大、更经济的内存池中，例如系统内存或 NVMe 存储设备。

基本原则是将 GPU 显存（VRAM）视为专家权重的高速、容量有限的缓存。我们不是将整个模型加载到 GPU 上，而是只加载非专家层（模型的骨干部分）和门控网络。专家权重本身存储在“芯片外部”，并根据需要动态移入显存。

### 内存与延迟的权衡

卸载并非没有代价。它解决了显存 (VRAM)容量问题，但代价是引入了数据传输延迟。通过 PCIe 总线将数 GB 的专家权重 (weight)从 CPU 内存或 NVMe 驱动器移动到 GPU 需要时间，此操作的速度比访问 GPU 高带宽内存 (HBM) 中已有的数据慢几个数量级。

因此，卸载系统的性能受到这种权衡的制约。目标是设计一个能最大限度减少这些数据传输并尽可能隐藏其延迟的系统。

> 卸载专家前向传播的数据流。当所需专家不在 GPU 缓存中时，其权重必须通过 PCIe 总线从系统内存传输，这会引入延迟。

### 卸载策略

#### 卸载到 CPU 内存

这是最常见且平衡的方式。系统内存远大于 GPU 显存 (VRAM)且成本更低，同时仍提供合理的传输速度。

工作流程如下：

1. **初始化**：模型的共享组件（注意力层、嵌入 (embedding)层、归一化 (normalization)层）和所有门控网络加载到 GPU 显存。所有专家权重 (weight)加载到 CPU 内存。
2. **门控**：对于输入的 token 批次，GPU 上的门控网络运行并确定每个 token 需要哪些专家。
3. **传输**：系统识别整个批次所需的唯一专家集合。对于 GPU 上尚不存在的每个所需专家，会启动异步内存复制操作，将权重从 CPU 内存传输到 GPU 显存中的分配缓冲区。
4. **计算**：一旦专家权重位于 GPU 上，相应的 token 就会被分派给它们进行计算。
5. **逐出**：计算完成后，如果内存允许，专家权重可以保留在 GPU 上；或者被逐出，为下一个所需专家腾出空间。

使用异步复制操作（例如 PyTorch 中的 `torch.Tensor.to('cuda', non_blocking=True)`）很重要，因为它允许 CPU 在 GPU 忙于其他工作时准备下一次传输，部分隐藏了传输延迟。

#### 卸载到 NVMe 存储

对于甚至可能不适合系统内存的超大型模型，或在内存有限的硬件上，卸载可以扩展到高速 NVMe 固态硬盘。这提供了对数 TB 存储的访问，但会带来严重的延迟开销。

数据路径变得更长：`NVMe -> CPU 内存 -> GPU 显存`。尽管像 GPUDirect Storage 这样的技术可以创建从 NVMe 到 GPU 更直接的路径，但它们会增加系统复杂性。该策略通常保留用于离线、注重吞吐量 (throughput)的批处理作业，其中每 token 延迟不是首要考虑事项。

### 通过缓存提高性能

通过在 GPU 显存 (VRAM)中为专家权重 (weight)实现缓存，可以大幅减少卸载的延迟开销。一个简单的最近最少使用 (LRU) 缓存策略非常有效。GPU 分配其一部分显存来存储少量专家。

当需要一个专家时：

- **缓存命中**：如果专家已在 GPU 缓存中，它会立即被使用，不会发生数据传输。
- **缓存未命中**：如果专家不在缓存中，系统必须从卸载位置（CPU 内存）获取它。如果缓存已满，最近最少使用的专家会从 GPU 中逐出，为传入的专家腾出空间。

此缓存的大小是一个重要的超参数 (parameter) (hyperparameter)。更大的缓存会增加缓存命中的概率，从而降低平均延迟，但它也会消耗更多宝贵的 GPU 显存，而这些显存原本可以用于处理更大的批次。



[交互图表：专家缓存大小对推理延迟的影响](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading#plot-1nz5ofr)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "专家缓存大小对推理延迟的影响"
    },
    "xaxis": {
      "title": {
        "text": "GPU 显存中缓存的专家数量（总共 64 个）"
      }
    },
    "yaxis": {
      "title": {
        "text": "每个 Token 的平均延迟 (ms)"
      },
      "range": [
        5,
        35
      ]
    },
    "font": {
      "family": "sans-serif"
    },
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "#ffffff"
  },
  "data": [
    {
      "x": [
        0,
        4,
        8,
        16,
        32
      ],
      "y": [
        32.5,
        18.1,
        11.2,
        7.4,
        5.8
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#ff8787",
          "#ffd43b",
          "#69db7c",
          "#4dabf7",
          "#9775fa"
        ],
        "line": {
          "width": 1,
          "color": "#495057"
        }
      }
    }
  ]
}
```

</details>



> GPU 专家缓存对推理 (inference)延迟的影响。即使是仅存储总专家一小部分的缓存，也能通过避免 PCIe 总线上的频繁数据传输来显著降低平均延迟。

### 一个实现示例

为具体说明这一点，可以考虑一个简化的 Python `OffloadedMoE` 层。这个示例展示了检查缓存和在未命中时加载专家的核心逻辑。

```python
import torch

class OffloadedMoE(torch.nn.Module):
    def __init__(self, experts, cache_size=4):
        super().__init__()
        # 专家最初在 CPU 上
        self.experts_cpu = torch.nn.ModuleList(experts)
        self.num_experts = len(experts)

        # GPU 缓存管理
        self.cache_size = cache_size
        self.expert_cache_gpu = {} # 将 expert_id 映射到 GPU 张量
        self.expert_cache_lru = [] # 按使用顺序存储 expert_id

    def _load_expert_to_gpu(self, expert_id):
        # 如果缓存已满则逐出
        if len(self.expert_cache_lru) >= self.cache_size:
            evict_id = self.expert_cache_lru.pop(0)
            del self.expert_cache_gpu[evict_id]

        # 将专家从 CPU 加载到 GPU
        expert = self.experts_cpu[expert_id]
        self.expert_cache_gpu[expert_id] = expert.to('cuda', non_blocking=True)
        self.expert_cache_lru.append(expert_id)

    def forward(self, x, gating_output):
        # gating_output 包含路由器决策，例如专家索引
        required_ids = torch.unique(gating_output.top_k_indices).tolist()

        # 确保所有所需专家已加载
        for expert_id in required_ids:
            if expert_id not in self.expert_cache_gpu:
                self._load_expert_to_gpu(expert_id)
            else:
                # 移动到 LRU 列表末尾以标记为最近使用
                self.expert_cache_lru.remove(expert_id)
                self.expert_cache_lru.append(expert_id)

        # ... 将 token 分派给 GPU 上相应专家的逻辑 ...
        # final_output = perform_expert_computation(x, self.expert_cache_gpu)
        # return final_output
```

这个示例省略了复杂的 token 分派逻辑，但展示了缓存机制。像 `deepspeed-mii` 或 Hugging Face 的 `accelerate` 这样的生产系统提供了此模式的优化实现。通过将智能缓存与异步传输结合，专家卸载使得在普通硬件上运行大型 MoE 模型成为可能，从而普及了其使用。

## 参考资料

- [Big Model Inference](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Accelerate 的官方文档，解释了将大型模型参数卸载到 CPU 和磁盘以在内存受限硬件上进行推理的实用方法。
- [GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage) — NVIDIA (2024)
  Publisher: NVIDIA
  NVIDIA 关于 GPUDirect Storage 的技术概述，解释了它如何实现 NVMe 存储和 GPU 内存之间的直接数据路径，提高卸载的数据传输速度。

---

[上一节](01-%E6%8E%A8%E7%90%86%E9%9D%A2%E4%B8%B4%E7%9A%84%E5%9B%B0%E9%9A%BE%EF%BC%9A%E5%86%85%E5%AD%98%E4%B8%8E%E5%BB%B6%E8%BF%9F.md) · [下一节](03-%E7%A8%80%E7%96%8F%E6%BF%80%E6%B4%BB%E7%9A%84%E6%89%B9%E5%A4%84%E7%90%86%E7%AD%96%E7%95%A5.md)
