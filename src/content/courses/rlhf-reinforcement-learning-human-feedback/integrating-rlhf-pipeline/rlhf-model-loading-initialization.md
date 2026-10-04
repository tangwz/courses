---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "integrating-rlhf-pipeline"
lesson: "rlhf-model-loading-initialization"
sourceId: 5173
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-5-integrating-rlhf-pipeline/rlhf-model-loading-initialization"
title: "模型加载与配置"
description: "管理基础模型、SFT 模型、奖励模型、策略模型和价值模型。"
order: 2
plots: []
sourceHash: "1e5fc97379f51e8736abbbde343faf6ca6c15e8ba7885a117eb44ca4404d7282"
sourceCorrections: []
---

RLHF 流程需要用到多种不同的模型，每个模型都有其特定用途。从各自的训练阶段（如 SFT 和奖励模型训练）正确加载这些模型，并为 PPO 阶段进行配置，是强化学习 (reinforcement learning)微调 (fine-tuning)开始前的必要步骤。常用的加载和配置方法将进行详细说明。

在 PPO 阶段（第三阶段）开始时，你通常需要使用以下模型：

1. **策略模型（行动者）：** 这是生成回复的语言模型，其参数 (parameter)由 PPO 算法主动更新。它学习生成能使奖励模型奖励信号最大的输出，同时保持与初始 SFT 行为的接近。
2. **参考模型：** 初始策略模型（通常是 SFT 模型）的冻结副本。它在 PPO 中作为 KL 散度惩罚的固定参照点，防止策略模型偏离原始 SFT 分布过远，从而避免模型性能下降或产生不佳文本。
3. **奖励模型（RM）：** 该模型在第二阶段训练，接收提示和生成的回复作为输入，并输出一个表示人类偏好（奖励）的标量分数。它在 PPO 期间仅用于推理 (inference)，以评估策略模型的输出。
4. **价值模型（评论者）：** 该模型从给定状态（提示）估计预期的未来奖励（价值函数）。它通过降低优势估计的方差来帮助稳定 PPO 训练。通常，价值模型与策略模型共享大部分层（一种行动者-评论者架构），但有一个单独的标量输出头。

### 加载方法与配置

管理这些可能很大的模型需要仔细处理检查点和计算资源。Hugging Face Transformers 等库提供了便捷的方法来加载先前阶段训练的模型。

**1. 配置策略模型和参考模型：**

策略模型（正在训练的模型）和参考模型（用于 KL 散度的冻结模型）通常从同一组权重 (weight)开始：即监督微调 (fine-tuning)（SFT）阶段（第一阶段）后保存的检查点。加载 SFT 模型为策略模型提供了一个良好的起点，因为它已经通过 SFT 数据集适应了所需的风格或特定用途。

你需要加载 SFT 模型检查点两次。一个实例成为 `policy_model`，其权重将在 PPO 训练期间更新。第二个实例成为 `ref_model`，它必须设置为评估模式（`.eval()`），并且其参数 (parameter)需要冻结以防止任何更新。许多 RLHF 库，例如 TRL (Transformer Reinforcement Learning)，通常会隐式处理参考模型，使用策略模型的初始状态作为参考。

**2. 配置价值模型（评论者）：**

价值模型的配置取决于架构选择：

- **共享骨干（行动者-评论者）：** 这对于效率来说很常见。TRL 等库提供了 `AutoModelForCausalLMWithValueHead` 等类，它们结合了语言模型（行动者）和输出标量值的价值头（评论者）。使用此类从 SFT 检查点加载时，基础语言模型权重从 SFT 模型加载，而价值头通常随机初始化，或使用可用的预训练 (pre-training)权重（尽管通常需要微调）。这个价值头随后在 PPO 期间与策略模型一同训练。
- **独立评论者：** 对于大型语言模型来说不那么常见，但可行。你可以配置一个独立的模型架构（可能更小，或基于相同的 SFT 检查点）专门用于价值估计。

价值模型在 PPO 更新步骤中计算优势估计（例如，使用广义优势估计 - GAE）方面起着重要作用。

**3. 加载奖励模型：**

奖励模型（RM）从第二阶段（RM 训练）后保存的检查点加载。它通常是一个序列分类模型（例如 Transformers 中的 `AutoModelForSequenceClassification`），具有一个表示奖励的标量输出。由于它在 PPO 期间仅用于对生成的回复进行评分，且在此阶段不再进行训练，因此应加载并立即将其设置为评估模式（`.eval()`），并禁用梯度计算以进行推理 (inference)。

### 实际实现示例

使用 Hugging Face Transformers 和 TRL 等库可以大大简化此过程。以下是一个说明性的例子：

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# 假设这些路径指向你训练过的模型检查点
sft_model_path = "./path/to/your/sft_model_checkpoint"
rm_model_path = "./path/to/your/reward_model_checkpoint"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 配置 ---
# PPO 配置包含训练的超参数
ppo_config = PPOConfig(
    batch_size=32,
    learning_rate=1.41e-5,
    # 其他 PPO 参数，如 kl_penalty, epochs 等
)

# --- 加载分词器 ---
# 通常在模型之间共享，从 SFT 模型路径加载
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
# 如有需要，确保设置填充 token（批处理中常见）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 加载模型 ---

# 1. 策略模型（行动者 + 价值头）
# 该模型既有用于生成的语言模型头，也有用于价值估计的标量头。
# 从 SFT 检查点初始化。它将由 PPO 训练。
print(f"正在从 {sft_model_path} 加载策略/价值模型")
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16, # 在兼容硬件上使用 bfloat16 以提高效率
    # low_cpu_mem_usage=True, # 对于大型模型很有用
    device_map={"": device} # 直接加载到目标设备
)
print("策略/价值模型已加载。")

# 2. 奖励模型（用于评分生成内容）
# 从 RM 检查点初始化。仅用于推理。
print(f"正在从 {rm_model_path} 加载奖励模型")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    num_labels=1, # 输出一个标量奖励分数
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True,
    device_map={"": device} # 加载到目标设备
)
reward_model.eval() # 设置为评估模式
print("奖励模型已加载并设置为评估模式。")

# 关于参考模型的说明：
# TRL 的 PPOTrainer 通常会自动处理参考模型。
# 它在初始化时创建 `policy_model` 的内部冻结副本。
# 如果构建自定义 PPO 循环，你将需要再次显式加载 SFT 模型
# 并确保其权重在整个训练过程中保持冻结。

# --- 初始化 PPOTrainer ---
# PPOTrainer 负责协调 PPO 训练循环，管理策略更新，
# 参考模型的 KL 散度计算以及价值函数训练。
print("正在初始化 PPOTrainer...")
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,   # 要训练的模型（包含价值头）
    ref_model=None,       # 如果为 None，TRL 会在内部处理
    tokenizer=tokenizer,
    # 需要提供数据集、数据整理器、优化器等
    # ... (数据集、优化器等的其他参数)
)
print("PPOTrainer 已初始化。")

# --- 准备 PPO 训练循环 ---
# 系统现在已加载并配置好所需的模型。
# 下一步涉及 PPO 训练循环：使用 `policy_model` 生成回复，
# 使用 `reward_model` 对其进行评分，并使用 `ppo_trainer` 管理的 PPO 算法更新 `policy_model`。
```

### 资源管理

加载多个大型模型会占用大量内存。请考虑以下几点：

- **数据类型：** 尽可能使用 `torch.bfloat16` 或 `torch.float16` 来减少内存占用，并可能在兼容硬件（如 NVIDIA Ampere GPU 及更新型号）上加速计算。
- **设备放置：** 在加载时使用 `.to(device)` 或 `device_map` 参数 (parameter)，仔细管理每个模型所在的设备（CPU、特定 GPU）。如果使用多个 GPU，你可以将奖励模型放置在与策略/价值/参考模型不同的 GPU 上，以平衡负载。
- **内存优化库：** 对于超出单个 GPU 内存的超大型模型，可以考虑使用 DeepSpeed（ZeRO 优化）或 PyTorch FSDP（完全分片数据并行）等工具，这些工具通常集成在 TRL 或 Accelerate 等训练库中。
- **CPU 卸载：** 一些技术允许将模型或优化器状态的部分卸载到 CPU 内存，以牺牲速度换取 GPU 内存使用的减少。

下图展示了不同阶段的模型如何在 PPO 阶段加载和使用。

> RLHF 流程中模型的来源和初始化路径。箭头表示 PPO 训练期间各阶段和组件之间的数据或权重 (weight)转移。

通过仔细管理这些不同模型的加载和配置，确保它们位于适当的设备上并正确设置（可训练与冻结），你为 PPO 微调 (fine-tuning)过程奠定了基础，这一过程旨在使模型输出与人类偏好保持一致。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  描述了从人类反馈中进行强化学习（RLHF）的流程，解释了策略模型、参考模型、奖励模型和价值模型的作用及其阶段性训练。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv; Publisher: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是RLHF强化学习微调阶段的核心算法，内容包括策略更新和KL散度等。
- [TRL Documentation](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  TRL库的官方文档，提供了设置和使用RLHF组件（如AutoModelForCausalLMWithValueHead和PPOTrainer）的实用指南。
- [Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，涵盖了与RLHF相关的通用模型加载、模型架构和检查点管理。
