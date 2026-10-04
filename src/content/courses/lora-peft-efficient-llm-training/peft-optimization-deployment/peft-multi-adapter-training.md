---
course: "lora-peft-efficient-llm-training"
chapter: "peft-optimization-deployment"
lesson: "peft-multi-adapter-training"
sourceId: 4160
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-multi-adapter-training"
title: "多适配器/多任务训练方法"
description: "了解同时训练多个PEFT适配器以处理不同任务或数据集的策略。"
order: 3
plots: []
sourceHash: "6670569a78a103a90e234ad3e9185c8dd1cdabca46a5f3a5f3ce2945cb71073e"
sourceCorrections: []
---

参数 (parameter)高效微调 (fine-tuning)方法显著减少了可训练参数的数量，为模型针对多个下游任务进行微调提供了可能性。一个重要优势是能够为一个单一的冻结基础模型训练和管理多个适配器，使其能够在不产生存储多个完整模型副本的高昂成本的情况下，专注于各种任务或专业方面。本节详细介绍同时或顺序管理和训练这些多适配器的策略。

### 多适配器训练的原因

在相同的基础模型上训练多个适配器有以下几个目的：

1. **多任务学习：** 为不同任务（例如，摘要、翻译、情感分析）训练不同的适配器，利用基础LLM的通用能力，同时在轻量级适配器内专精任务特定知识。
2. **专业化：** 使用相同的基础模型，开发在特定专业方面（例如，医疗、法律、金融）进行微调 (fine-tuning)的适配器。
3. **个性化：** 创建用户特定的适配器，根据个人偏好或数据调整模型行为，同时在用户之间共享核心模型。
4. **增量学习：** 通过训练新适配器来引入新功能或更新现有功能，而不影响之前训练过的适配器，也无需从头开始重新训练。
5. **效率：** 只存储每个任务的小适配器权重 (weight)，而不是整个模型的多个副本，大幅节省存储空间。在推理 (inference)过程中，可以在需要时加载相应的适配器。

### 训练多个适配器的策略

为一个基础模型训练多个适配器主要有以下几种方法：

#### 1. 顺序训练

这是最直接的方法。您一次为一个特定任务训练一个适配器。

- **过程：**

  1. 加载预训练 (pre-training)的基础模型（通常是量化 (quantization)的，例如，使用QLoRA的前提条件）。
  2. 初始化或加载*第一个*适配器（例如，`adapter_task_A`）的权重 (weight)。
  3. 配置PEFT设置（如LoRA配置）以仅针对`adapter_task_A`的权重。
  4. 在任务A的数据上训练模型。只更新`adapter_task_A`的权重。
  5. 保存训练好的`adapter_task_A`权重。
  6. 对`adapter_task_B`重复步骤2-5，使用任务B的数据，确保在此阶段只有`adapter_task_B`的权重是可训练的。对所有适配器继续此过程。
- **优点：** 实现和管理简单。每次训练运行都是独立的，简化了调试。
- **缺点：** 如果需要很多适配器，可能会很耗时。未能利用同时处理不同任务数据所带来的潜在计算效率。

#### 2. 交错或混合批次训练

这种方法通过构建包含不同任务数据样本的批次，在同一个训练循环中训练多个适配器。

- **过程：**

  1. 加载预训练的基础模型。
  2. 初始化或加载*所有*适配器配置及其权重（例如，`adapter_task_A`、`adapter_task_B`）。
  3. 修改数据集加载和批次整理过程：批次中的每个样本必须与其目标适配器关联（例如，通过任务ID或适配器名称）。批次可以包含针对不同适配器的样本。
  4. 修改模型的正向传播：核心基础模型计算执行一次。然后，根据与每个样本（或样本组）关联的任务ID，应用相应的适配器权重（例如，LoRA矩阵A和B）。
  5. 修改反向传播 (backpropagation)和优化器步骤：梯度必须*只*针对每个特定样本在正向传播中激活的适配器权重进行计算。该批次中非活动适配器的梯度应被置零或被优化器忽略。
  6. 管理优化器状态：可以为每个适配器使用单独的优化器实例，或者在一个优化器中仔细管理状态，以确保更新正确应用于各自的适配器参数 (parameter)。
- **正向传播逻辑（伪代码）：**

```python
# 假设批次包含带有'adapter_name'标签的样本
base_output = base_model(input_ids, attention_mask)
final_output = {} # 存储每个适配器的输出

for adapter_name in unique_adapter_names_in_batch:
    # 选择此适配器的样本
    adapter_mask = [sample['adapter_name'] == adapter_name for sample in batch]
    adapter_input_indices = ... # 根据adapter_mask获取索引

    # 应用特定适配器
    # 注意：这要求模型架构支持动态适配器选择
    adapter_output = get_adapter_layer(adapter_name)(base_output[adapter_input_indices])

    # 组合或存储适配器特定输出
    final_output[adapter_name] = adapter_output

# 损失计算基于final_output和每个任务的标签
```

- **优点：** 潜在地更具计算效率，尤其是在基础模型计算占主导地位的情况下。允许在单次运行中动态平衡任务训练。
- **缺点：** 实现起来复杂很多。需要仔细处理批处理、正向/反向传播和优化器状态。调试可能具有挑战性。

Hugging Face的`peft`等库提供了简化混合批次训练的抽象。使用`PeftModel.add_adapter()`和`PeftModel.set_adapter()`可以管理多个适配器配置，尽管混合批次的训练循环逻辑通常仍需要自定义实现。

> 使用共享基础模型进行多适配器训练的架构图。输入数据在经过冻结的基础模型后，会被路由到相应的适配器（例如，LoRA层）。反向传播期间的梯度只更新与特定任务样本对应的适配器的权重。

### 技术要点

- **适配器管理：** 为适配器使用清晰的命名约定。库通常提供加载、保存、激活和停用与`PeftModel`关联的特定适配器的函数。
- **优化器状态：** 对于混合批次训练，为每个适配器使用单独的优化器通常是最简洁的方法，尽管这可能会略微增加内存使用。如果使用单个优化器，请确保它正确处理与不同适配器对应的参数 (parameter)组，这可能需要自定义逻辑来屏蔽批次中非活动适配器的梯度。内存高效优化器（如8位Adam）在此处很有益。
- **批次构建：** 在创建混合批次时，决定从不同任务中采样数据的策略。选项包括均匀采样、按数据集大小比例采样，或根据任务性能动态调整采样率。这会影响学习动态。
- **梯度累积：** 此技术完全兼容且通常很有用，尤其是在混合批次的情况下。在进行单个优化器步骤之前，在多个微批次（可能包含各种任务的样本）上累积梯度。
- **内存占用：** 尽管每个适配器都很小，但与顺序训练或单适配器PEFT相比，同时将许多适配器加载到GPU内存中进行混合训练会增加整体内存需求。基础模型的QLoRA等量化 (quantization)技术变得更加重要。评估训练速度（混合批次）和内存限制之间的权衡。
- **兼容性：** 确保所使用的PEFT方法（LoRA、Adapters等）和具体实现支持多适配器场景所需的动态选择或并发应用。

有效管理多个适配器能够创建高度通用、能高效处理各种任务的模型，这代表了PEFT方法论相对于传统完整微调 (fine-tuning)的重要操作优势。顺序训练和混合批次训练之间的选择取决于适配器的数量、计算资源以及对实现复杂度的容忍度。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种基础的参数高效微调方法，构成了多适配器策略的基础。
- [PEFT (Parameter-Efficient Fine-tuning) Library Documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Hugging Face PEFT库的官方文档，提供了实现和管理适配器的实用指南，包括处理多个适配器的功能。
