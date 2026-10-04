# 端到端 RLHF 系统的代码结构

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-5-integrating-rlhf-pipeline/code-structure-e2e-rlhf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建一个完整的 RLHF 系统涉及到管理多个模型、数据集和训练阶段。一个组织良好的代码库对于重复性、可维护性和高效实验来说是必不可少的。在整合 RLHF 管线中的 SFT、奖励模型和 RL 微调 (fine-tuning)阶段时，有效地组织代码成为一项重要的工程工作。

我们来考虑一个组织端到端 RLHF 项目的实际方法。目标是创建一个模块化架构，其中组件可以独立地开发、测试和潜在地替换。

### 组织你的 RLHF 项目

一个典型的 RLHF 项目可能包含以下主要目录和文件：

```
rlhf_project/
├── configs/                 # 配置文件（YAML, JSON）
│   ├── sft_config.yaml
│   ├── rm_config.yaml
│   └── ppo_config.yaml
├── data/                    # 数据集（原始、处理过、偏好）
│   ├── sft/
│   ├── preferences/
│   └── prompts/
├── models/                  # 模型检查点及相关文件
│   ├── base_llm/            # （可选）基础模型的本地副本
│   ├── sft_model/
│   ├── reward_model/
│   ├── ppo_policy_final/
│   └── ppo_checkpoints/
├── src/                     # 源代码
│   ├── data_processing/     # 数据加载与预处理脚本/模块
│   │   ├── __init__.py
│   │   └── preference_dataset.py
│   ├── models/              # 模型定义（如果需要自定义）
│   │   ├── __init__.py
│   │   └── reward_model.py
│   ├── training/            # 训练循环与逻辑
│   │   ├── __init__.py
│   │   ├── sft_trainer.py
│   │   ├── rm_trainer.py
│   │   └── ppo_trainer.py   # 可使用如 TRL 等库
│   ├── evaluation/          # 评估脚本与指标
│   │   ├── __init__.py
│   │   └── evaluate_alignment.py
│   └── utils/               # 共享工具（日志、辅助函数）
│       ├── __init__.py
│       └── helpers.py
├── scripts/                 # 运行各阶段的可执行脚本
│   ├── run_sft.py
│   ├── run_rm.py
│   ├── run_ppo.py
│   └── run_evaluation.py
├── requirements.txt         # 项目依赖
└── README.md                # 项目文档
```

### 组件及其作用

1. **配置 (`configs/`)**：集中管理所有超参数 (parameter) (hyperparameter)、模型名称/路径、数据集路径和训练设置。使用 YAML 等格式可以便于管理不同的实验配置，而无需更改代码。例如，`ppo_config.yaml` 将包含学习率、批大小、KL 系数 ($\beta$)、PPO 训练轮数、GAE 参数 ($\lambda$，$\gamma$)，以及初始策略（SFT 模型）和奖励模型的模型路径等设置。
2. **数据处理 (`data/`, `src/data_processing/`)**：分别存储原始数据集和处理过的数据集。`src/data_processing` 模块应处理每个阶段（SFT 演示、偏好对、用于生成的提示词 (prompt)）特有的数据加载。定义清晰的数据结构或类（如 PyTorch `Dataset` 或 TensorFlow `tf.data.Dataset`）以确保一致处理。
3. **模型管理 (`models/`, `src/models/`)**：`models/` 目录存储实际的模型权重 (weight)和分词 (tokenization)器 (tokenizer)文件。`src/models/` 目录包含定义模型架构的代码，如果你正在自定义它们（例如，在基础 Transformer 之上为奖励模型添加一个特定的头部）。更常见的情况是，你会直接从 Hugging Face Transformers 等库中加载模型，但系统地管理检查点很重要。
4. **训练逻辑 (`src/training/`)**：将每个阶段（SFT、RM、PPO）的训练逻辑封装到单独的模块或类中。

   - `sft_trainer.py`：处理基础模型、SFT 数据加载以及运行监督微调 (fine-tuning)循环。
   - `rm_trainer.py`：加载 SFT 模型（或基础模型）、偏好数据，定义奖励模型架构（通常是 LLM 上的分类头），并实现偏好学习损失（例如 Bradley-Terry 损失）。
   - `ppo_trainer.py`：这通常是最复杂的部分。它协调加载 SFT 模型（作为初始策略 $\pi_{SFT}$）、奖励模型，配置 PPO 组件（策略 $\pi_{\theta}$、值函数 $V_{\phi}$、参考策略 $\pi_{ref}$ 通常固定为 $\pi_{SFT}$），生成响应，使用奖励模型评分，计算优势（使用 GAE），并执行 PPO 更新，包括 KL 惩罚项：
     
     $$
     L_{PPO} = \mathbb{E}_t [\hat{A}_t \cdot \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)] - c_1 \cdot L_{VF} + c_2 \cdot S[\pi_{\theta}] \\
     \text{此处 } r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} \\
     \text{且奖励包含 KL 惩罚: } R(s, a) = R_{RM}(s, a) - \beta \cdot \text{KL}(\pi_{\theta}(\cdot|s) || \pi_{ref}(\cdot|s))
     $$
     
     像 TRL (`trl.PPOTrainer`) 这样的库大大简化了这些复杂性，但理解其底层结构是有益的。
5. **评估 (`src/evaluation/`)**：实现函数或脚本，用于评估不同阶段的模型，特别是最终经过 PPO 微调的策略。这包括计算自动指标和准备用于人工评估的输出。
6. **工具 (`src/utils/`)**：放置通用辅助函数、日志设置、参数解析逻辑等，以避免代码重复。
7. **执行脚本 (`scripts/`)**：这些是运行管道每个部分的执行入口。它们解析参数，加载配置，实例化 `src/training` 中所需的训练器类，并启动训练或评估过程。

### 数据流和关系

下图说明了主要代码组件与数据/模型文件之间的高级关系。

> RLHF 项目的高级结构，展示了配置、数据、可执行脚本、源代码模块（训练、评估、数据处理）和模型文件之间的关系。

### 这种结构的好处

- **模块化**：每个阶段（SFT、RM、PPO）都独立封装在其各自的训练器模块和执行脚本中。这使得调试或修改某个部分变得更容易，而不会影响其他部分。
- **配置驱动**：通过更改 `configs/` 目录中的参数 (parameter)，可以轻松配置和复现实验。
- **明确分离**：代码 (`src/`)、数据 (`data/`)、模型 (`models/`)、配置 (`configs/`) 和执行脚本 (`scripts/`) 之间明确分离，提高了组织性。
- **可重用性**：工具函数 (`src/utils/`) 和数据处理逻辑 (`src/data_processing/`) 可以在不同阶段间共享。
- **可扩展性**：虽然这种结构适用于单机开发，但它提供了一个基本框架，可以通过修改训练器和执行脚本来适配分布式训练环境。

从一开始就采用这种结构化方法，将随着你的 RLHF 系统复杂性增加而节省大量精力，让你能够更专注于算法和建模方面的挑战，而不是处理杂乱的代码。

## 参考资料

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  描述了完整的InstructGPT管道，详细介绍了监督微调（SFT）、奖励模型（RM）训练和使用近端策略优化（PPO）的强化学习阶段，为RLHF系统提供了概念框架。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种广泛应用于RLHF强化学习阶段的策略梯度方法。
- [TRL - A Hugging Face Library for 🤗 Reinforcement Learning from Human Feedback](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  TRL库的官方文档，它简化了RLHF组件的实现，包括用于强化学习微调阶段的PPOTrainer。

---

[上一节](05-%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E4%B8%AD%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%90%8C%E6%AD%A5.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BF%90%E8%A1%8C%E7%AE%80%E5%8C%96%E7%9A%84RLHF%E5%BE%AA%E7%8E%AF.md)
