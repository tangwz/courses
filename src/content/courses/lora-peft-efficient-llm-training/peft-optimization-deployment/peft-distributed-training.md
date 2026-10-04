---
course: "lora-peft-efficient-llm-training"
chapter: "peft-optimization-deployment"
lesson: "peft-distributed-training"
sourceId: 4167
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-distributed-training"
title: "PEFT 的分布式训练方法"
description: "调整分布式训练框架（DDP、DeepSpeed ZeRO），以实现高效的PEFT微调。"
order: 6
plots: ["plots/4167-0.json"]
sourceHash: "954b1a8be39c14b61ee3be08f9ebbd45887b27033a9d6d2d255a12b29dbf97d2"
sourceCorrections: []
---

LoRA等参数 (parameter)高效微调 (fine-tuning)（PEFT）方法相较于完全微调，能显著减少可训练参数的数量，但扩大PEFT训练规模，通常仍需要采用分布式计算方法。这主要有两方面原因：基础大型语言模型（LLM）本身仍然非常庞大，其正向和反向传播 (backpropagation)需要大量内存和计算资源，并且在大规模数据集上高效训练得益于并行处理，可以缩短实际运行时间。为PEFT调整标准分布式训练框架，需要了解PEFT如何与数据并行和模型并行技术配合使用。

### PEFT 的数据并行

最常用的分布式方法是数据分布式并行（DDP）。在标准DDP中（例如，使用PyTorch的`DistributedDataParallel`），模型会在多个设备（GPU）上进行复制。每个设备处理不同的mini-batch数据，本地计算梯度，然后梯度会在所有设备之间同步（通常通过AllReduce操作），之后优化器在每个副本上更新模型权重 (weight)。

当将DDP应用于PEFT时：

1. **基础模型复制：** 庞大且冻结的基础模型仍在每个设备上复制。这意味着与存储基础模型权重相关的主要内存瓶颈依然存在。
2. **PEFT 参数 (parameter)复制：** 小量可训练的PEFT参数（例如LoRA矩阵 $A$ 和 $B$、适配器权重或提示嵌入 (embedding)）也会被复制。
3. **梯度同步减少：** 这是PEFT在DDP中一个显著的优点。由于只需要同步PEFT参数的梯度，AllReduce步骤中的通信开销相对于完全微调 (fine-tuning)中同步整个模型的梯度会大幅降低。如果一个模型有 $N$ 个总参数和 $P$ 个PEFT参数，且 $P \ll N$，那么梯度通信量将按比例减少。
4. **实施：** 将PEFT与标准DDP框架结合通常很直接。Hugging Face的`PEFT`等库会自动将只有适配器参数标记 (token)为可训练。当使用`DistributedDataParallel`包装模型时，它能正确识别这些可训练参数，并且只同步它们的梯度。

```python
# 示例伪代码：PyTorch DDP 结合 PEFT
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig # 假设使用了PEFT库

# 初始化分布式环境（例如，使用torchrun）
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
base_model.to(local_rank)

# 配置并应用PEFT（例如，LoRA）
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters() # 验证只有PEFT参数是可训练的

# 使用DDP包装模型
# DDP将自动处理可训练（PEFT）参数的梯度同步
ddp_model = DDP(model, device_ids=[local_rank])

# 优化器仅针对可训练参数
optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4) # AdamW会隐式过滤requires_grad=True的参数

# --- 训练循环 ---
# for batch in dataloader:
#     outputs = ddp_model(**batch)
#     loss = outputs.loss
#     loss.backward() # 只计算PEFT参数的梯度
#     optimizer.step() # 梯度同步后更新PEFT参数
#     optimizer.zero_grad()
# --- 训练循环结束 ---

dist.destroy_process_group()
```

即使梯度通信量减少，大型基础模型的正向和反向传播 (backpropagation)计算成本在DDP中每步训练时间中仍是主导因素。

### DeepSpeed ZeRO 提升内存效率

尽管DDP通过分配数据批次有助于扩展计算能力，但它不减少每个GPU存储模型权重 (weight)、梯度和优化器状态所需的内存占用。DeepSpeed的ZeRO（零冗余优化器）提供将这些组件在数据并行工作器之间分区的技术，显著降低了每个GPU的内存需求。

ZeRO阶段与PEFT的配合：

1. **ZeRO Stage 1 (优化器状态分区)：** 在设备间分区优化器状态（例如AdamW中的动量）。这对PEFT非常有利，因为即使是少量参数 (parameter)的优化器状态也能占用大量内存（例如，AdamW的FP32状态每个参数使用12字节）。ZeRO-1有效降低了这种内存开销。
2. **ZeRO Stage 2 (梯度 + 优化器状态分区)：** 额外地在设备间分区梯度。对于PEFT来说，梯度本身就较小，因此分区它们所带来的额外内存节省可能不如完全微调 (fine-tuning)那样显著。不过，它仍然有助于整体内存减少，并且与DDP风格的梯度计算很好地配合。
3. **ZeRO Stage 3 (参数 + 梯度 + 优化器状态分区)：** 分区模型参数本身。这是内存节省最激进的阶段。
   - **PEFT 的适用性：** ZeRO-3在PEFT中的主要使用场景出现在当*基础模型*即使在推理 (inference)模式下也太大以至于无法放入单个GPU内存中时。ZeRO-3分区所有参数，包括冻结的基础模型权重。在正向和反向传播 (backpropagation)过程中，每个设备会动态地从其他设备收集必要的参数。
   - **开销：** ZeRO-3引入了显著的参数收集通信开销，这有时会抵消数据并行带来的计算加速，特别是在较慢的网络互连下。
   - **PEFT 特定参数：** ZeRO-3也会分区可训练的PEFT参数。虽然这能节省少量内存，但其在PEFT环境中的主要优点是能够使用原本无法容纳的庞大基础模型。

**为PEFT选择ZeRO阶段：**

- 如果基础模型可以轻松地适应每个GPU：ZeRO-1或ZeRO-2通常是最佳选择。它们减少了优化器状态（以及ZeRO-2的梯度）的内存，同时没有ZeRO-3的参数收集通信开销。这允许比普通DDP更大的批次大小或适应稍微更大的基础模型。
- 如果基础模型无法适应单个GPU：ZeRO-3变得必要，用于分区基础模型参数。PEFT参数会随之一起分区。请注意增加的通信成本。

将PEFT与DeepSpeed结合通常涉及配置DeepSpeed JSON文件，并使用`deepspeed.initialize`函数初始化模型、优化器和数据加载器。请确保传递给DeepSpeed的优化器配置为只优化可训练的PEFT参数。

```json
// 示例DeepSpeed配置文件片段 (ds_config.json) 用于PEFT的ZeRO Stage 2
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 100
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
        "device": "cpu", // 可选：将优化器状态卸载到CPU内存
        "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true // 或 bf16: { "enabled": true }
  }
}
```

```python
# 示例伪代码：DeepSpeed 结合 PEFT
import deepspeed
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# --- 设置PEFT模型（如DDP示例中所示）---
# model = get_peft_model(base_model, peft_config)

# 如果需要，可以显式过滤优化器参数，
# 尽管如果模型设置正确，DeepSpeed通常会处理此问题。
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if p.requires_grad], 'weight_decay': 0.01}
# ]
# optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4) # 或使用DeepSpeed优化器配置

# 初始化DeepSpeed
# model.parameters()调用应只返回可训练的PEFT参数
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    # model_parameters=model.parameters(), # 或显式传递参数
    # optimizer=optimizer, # 可以传递预配置的优化器
    config_params='ds_config.json' # DeepSpeed配置文件的路径
)

# --- 使用model_engine的训练循环 ---
# for batch in dataloader:
#     loss = model_engine(**batch).loss
#     model_engine.backward(loss)
#     model_engine.step() # 处理优化器步骤、梯度裁剪、调度器
# --- 训练循环结束 ---
```

### 内存使用对比

分布式方法的选择显著影响每个GPU的内存使用。与完全微调 (fine-tuning)相比，PEFT本身就减少了梯度和优化器状态所需的内存。ZeRO进一步优化了这一点。



![每个GPU的预估内存使用（示意图）](plots/4167-0.json)



> 这是大型模型微调场景下，每个GPU的预估内存组成。完全微调DDP需要存储完整的权重 (weight)、梯度和优化器状态。PEFT DDP大幅减少了梯度和优化器状态的内存。ZeRO-2进一步减少了优化器状态的内存（可能进行卸载）。ZeRO-3分区所有组件，包括基础模型权重，提供最低的每个GPU占用空间，但可能带来更高的通信开销。激活内存很大程度上取决于批次大小和序列长度，这里为了比较假设为常数。

### 选择恰当的方法

选择合适的PEFT分布式方法涉及考虑：

1. **基础模型大小：** 如果基础模型能适应单个GPU，DDP或ZeRO-1/2是高效的选择。如果不能，则需要ZeRO-3。
2. **GPU内存：** 内存有限的情况下更倾向于选择ZeRO阶段，可能伴随CPU卸载优化器状态（ZeRO-1/2）或参数 (parameter)（ZeRO-3，对于PEFT来说不太常见，除非基础模型非常庞大）。
3. **PEFT参数数量：** 尽管相对于基础模型而言较少，但大量的适配器参数（例如多适配器配置）可能仍然受益于ZeRO-1/2优化器状态分区。
4. **网络互连：** 高带宽、低延迟的互连（如NVLink或Infiniband）可以减轻梯度同步（DDP、ZeRO-1/2）和参数收集（ZeRO-3）的通信开销。较慢的网络（如以太网）会使得ZeRO-3的吸引力较低，除非为了适应模型而绝对必要。
5. **训练速度与内存：** 通常存在权衡。如果内存允许，DDP通常计算速度最快。ZeRO阶段节省内存，但可能引入通信瓶颈，可能减慢训练步骤。

通过仔细考虑这些因素并善用PyTorch DDP和DeepSpeed ZeRO等框架，你可以有效地扩展PEFT微调 (fine-tuning)，以处理大型模型、大型数据集和复杂的多适配器场景，同时高效管理硬件资源。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR 2022); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种对文中讨论的参数高效微调策略至关重要的技术。
- [torch.nn.parallel.DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Documentation Team (2024)
  Publisher: PyTorch
  PyTorch官方的DistributedDataParallel文档，这是讨论的数据并行策略的核心。
- [Parameter-Efficient Fine-tuning (PEFT) with 🤗PEFT](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了PEFT方法实际实现的细节。
