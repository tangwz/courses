# 配置 Megatron-LM 中的张量和流水线并行

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/configuring-megatron-lm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解张量并行（TP）和流水线并行（PP）的理论很重要，但将它们付诸实践需要专门的工具。Megatron-LM 是 NVIDIA 开发的一个著名库，专门设计用于实现这些复杂的并行策略，尤其适合训练大型 Transformer 模型。它提供并行层的优化实现，并管理所需的复杂通信模式。Megatron-LM 以使用 TP 和 PP 的配置步骤进行展示。

Megatron-LM 中的配置通常通过命令行参数 (parameter)传递给训练脚本（如 `pretrain_gpt.py` 或类似脚本）来处理。我们来看具体如何设置张量并行和流水线并行。

### 张量并行 (TP) 配置

张量并行，有时也称为层内模型并行，涉及将单个大型层（如 MLP 块或注意力机制 (attention mechanism)中的权重 (weight)矩阵）的执行拆分到多个 GPU 上。Megatron-LM 提供专门的层实现（例如 `ColumnParallelLinear`、`RowParallelLinear`），它们自动处理此分区和必要的通信（如 AllReduce 或 AllGather）。

启用和控制 TP 的主要参数 (parameter)是 `--tensor-model-parallel-size`（或根据 Megatron-LM 的特定版本或分支而异的类似变体）。这个值，我们称之为 $TP_{size}$，指定每个张量并行层将被拆分到多少个 GPU 上。

例如，如果设置 `--tensor-model-parallel-size 4`，一个大型线性层的权重矩阵将被划分为 4 个列或行段，每个段驻留在张量并行组内的不同 GPU 上。在前向和反向传播 (backpropagation)过程中，Megatron-LM 协调这 4 个 GPU 之间所需的数据交换。

```python
# 在 Megatron-LM 环境中的示例
# (实际使用涉及 Megatron 的模型定义工具)

# 假设 tp_size = 2
# GPU 0 存放 W_A，GPU 1 存放 W_B，并且 W = [W_A W_B]
tp_group = get_tensor_model_parallel_group()
# 获取 TP 通信组的函数

# ColumnParallelLinear: 拆分权重列，
# 输入被广播，输出被收集
# 输入 X -> [X W_A, X W_B] -> AllGather -> 输出 Y
linear_column_parallel = ColumnParallelLinear(
    input_size,
    output_size,
    tp_group=tp_group,
)

# RowParallelLinear: 拆分权重行，
# 输入被拆分，输出被归约
# 输入 X -> [X_A, X_B] -> [X_A W_A, X_B W_B] -> AllReduce -> 输出 Y
linear_row_parallel = RowParallelLinear(
    input_size,
    output_size,
    tp_group=tp_group,
)

# Transformer 块内的使用示例 (简化版)
# mlp_output = linear_row_parallel(dropout(
#     linear_column_parallel(layer_input)))
```

**TP 的重要考虑因素：**

1. **可除性：** $TP_{size}$ 必须能整除模型的隐藏维度（`--hidden-size`）和注意力头的数量（`--num-attention-heads`）。Megatron-LM 的并行层依赖此进行分区。
2. **通信：** TP 会在张量并行组内引入通信开销，通常是 AllReduce 操作。效率在很大程度上取决于这些 GPU 之间的互连速度（例如 NVLink）。
3. **内存：** TP 减少了组内每个 GPU 上权重、梯度和优化器状态所需的内存，但由于通信需求，增加了激活内存使用量。

### 流水线并行 (PP) 配置

流水线并行涉及将模型的层按顺序划分到不同的 GPU 上，形成一个流水线。每个 GPU（或结合 TP/DP 时的 GPU 组）构成流水线中的一个“阶段”，负责只执行模型层的一个子集。

配置 PP 的主要参数 (parameter)是 `--pipeline-model-parallel-size`（或类似参数）。这个值，$PP_{size}$，决定流水线的阶段数量。例如，`--pipeline-model-parallel-size 4` 创建一个 4 阶段的流水线。

为了保持流水线阶段的利用率并最小化空闲时间（“气泡”），Megatron-LM 采用微批处理。整体训练批次被拆分为更小的微批，这些微批并发地流经流水线阶段。微批数量是一个重要的调整参数，通常由 `--num-microbatches` 控制，或根据全局批次大小、微批次大小和数据并行度计算得出。常见的调度方法是 1F1B（每个微批一次前向传播，一次反向传播 (backpropagation)），这有助于平衡计算和内存使用。

> 微批（MB1、MB2、MB3）通过 3 阶段流水线（$PP_{size}=3$）的前向传播（Fwd）流程。激活值（Actvs）在阶段之间传递。反向传播遵循类似的反向流程。

**PP 的重要考虑因素：**

1. **层数：** 变压器 (Transformer)层的总数（`--num-layers`）理想情况下应能被 $PP_{size}$ 整除，以实现负载均衡，尽管 Megatron-LM 可以处理不均匀的分布。
2. **通信：** PP 涉及相邻阶段之间的点对点通信，以向前传递激活值和向后传递梯度。
3. **流水线气泡：** PP 的效率取决于最小化流水线气泡（处理批次开始和结束时的空闲时间）。增加微批数量有帮助，但也会增加激活内存。
4. **内存平衡：** 内存使用在不同阶段可能无法完美平衡，因为第一和最后阶段通常包含嵌入 (embedding)层和输出层，这些层的大小可能与中间变压器块不同。

### 结合张量并行与流水线并行

Megatron-LM 擅长结合不同的并行策略。通常，TP 和 PP 会一起使用。在这种设置中，模型并行涉及的 GPU 总数是 $TP_{size} \times PP_{size}$。每个流水线阶段本身可能包含多个 GPU 通过张量并行一起工作。数据并行（DP）可以在此之上叠加，复制这种 TP/PP 结构。

训练中使用的 GPU 总数将是 $N_{gpus} = DP_{size} \times TP_{size} \times PP_{size}$。

> 8 个 GPU 的配置示例：$PP_{size}=4$ 和 $TP_{size}=2$。每个流水线阶段使用 2 个 GPU 进行张量并行。通信发生在阶段之间（PP）的点对点方式，以及每个 TP 组内部（TP）的集体操作。

### 配置示例

这里有一个简化的示例，说明在使用 TP 和 PP 配置 Megatron-LM 进行 GPT 风格模型训练时，命令行参数 (parameter)可能是什么样子（假设总共有 8 个 GPU，其中 $TP_{size}=2$ 和 $PP_{size}=4$，为简单起见，数据并行 $DP_{size}=1$）：

```bash
# 使用 Megatron-LM 参数的训练命令示例
# (实际脚本名称和其他参数可能有所不同)

python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --seq-length 2048 \
       \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       # 假设 DP 大小为 1，微批数量 = 全局批大小 / (微批大小 * dp_大小)
       # 需要根据 DP 大小仔细计算。这里假设 DP=1。
       # 每个流水线的有效批次 = 微批大小 * 微批数量
       # 全局批大小 = 微批大小 * 微批数量 * dp_大小
       # 微批数量 = 全局批大小 / (微批大小 * dp_大小) = 128 / (4 * 1) = 32
       --num-microbatches 32 \
       \
       --optimizer adam \
       --learning-rate 1.0e-4 \
       --weight-decay 0.01 \
       --clip-grad 1.0 \
       \
       --train-data-path <path_to_train_data> \
       --valid-data-path <path_to_valid_data> \
       --tokenizer-type SentencePieceTokenizer \
       --tokenizer-model <path_to_tokenizer> \
       \
       --distributed-backend nccl \
       --save <path_to_save_checkpoints> \
       --load <path_to_load_checkpoints> \
       --log-interval 10 \
       --save-interval 1000 \
       --eval-interval 100 \
       --num-workers 2
```

在此示例中：

- `--tensor-model-parallel-size 2`：将线性层和注意力层等拆分到 2 个 GPU 上。
- `--pipeline-model-parallel-size 4`：将 24 个层拆分到 4 个顺序阶段（每个阶段大约 6 层）。
- `--num-microbatches 32`：用于高效地为 4 阶段流水线提供数据，根据全局批次大小、微批次大小和数据并行度计算得出。

正确配置这些参数对于平衡计算、内存使用和可用硬件上的通信开销非常重要。Megatron-LM 提供底层的机制，但确定最佳的 $TP_{size}$、$PP_{size}$ 和微批数量通常需要根据具体的模型架构、硬件配置（GPU 类型、互连）和所需的批次大小进行实验。

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  介绍Megatron-LM及其在大语言模型中张量和流水线并行实现的基础论文。
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — NVIDIA (2024)
  Journal: GitHub repository; Publisher: NVIDIA
  官方GitHub仓库，提供Megatron-LM的源代码、最新配置选项和实用示例。

---

[上一节](04-Megatron-LM%20%E4%BB%8B%E7%BB%8D.md) · [下一节](06-%E7%BB%93%E5%90%88%E6%A1%86%E6%9E%B6%E4%B8%8E%E7%AD%96%E7%95%A5.md)
