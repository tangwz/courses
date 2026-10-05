# 动手实践：实现RLHF的主要组成部分

来源：[原文](https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/hands-on-core-rlhf-components)

[返回章节目录](README.md) · [返回课程目录](../README.md)

提供实用指导和代码片段，以说明强化学习 (reinforcement learning)与人类反馈 (RLHF) 主要组成部分的实现：训练奖励模型和使用PPO进行策略优化。我们将使用Hugging Face的`transformers`和`trl`等库以提高效率，这假设您已安装好这些库并拥有可用的Python环境。

请记住，全面的RLHF实现需要大量的计算资源和对数据的仔细处理。我们此处侧重于理解核心部分的运作方式。

### 准备工作：环境和基础模型

首先，请确保您拥有所需的库。您通常可以通过pip安装它们：

```bash
pip install transformers datasets accelerate torch trl
```

我们假设您可以使用一个预训练 (pre-training)的基础语言模型（例如`gpt2`或经过指令微调 (fine-tuning)的类似模型）及其分词 (tokenization)器 (tokenizer)。

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLMWithValueHead
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# 占位符：加载您的基础模型和分词器
model_name = "gpt2" # 如果有指令微调模型，请替换此处
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 确保为批量处理设置了填充令牌
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 第一部分：实现奖励模型训练步骤

目标是训练一个模型 $r_\theta(x, y)$，使其对于给定提示 $x$，能为偏好的完成 ($y_w$) 分配比被拒绝的完成 ($y_l$) 更高的分数。

**1. 数据准备**

假设您有偏好数据，其结构为三元组：`(提示, 选中回应, 拒绝回应)`。您需要为奖励模型对这些数据进行适当分词 (tokenization)。奖励模型通常将拼接后的 `提示 + 回应` 作为输入。

```python
# 偏好数据示例（请替换为您的实际数据集）
preference_data = [
    {"prompt": "Explain reinforcement learning like I'm five.",
     "chosen": " RL is like teaching a dog tricks with treats! Good action, get a treat (reward). Bad action, no treat.",
     "rejected": " Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones."},
    # ... 更多对
]

# 用于奖励模型训练的格式化函数
def format_for_reward_model(examples):
    # 对提示 + 选中回应和提示 + 拒绝回应对进行分词
    tokenized_chosen = tokenizer(examples["prompt"] + examples["chosen"], truncation=True, max_length=512)
    tokenized_rejected = tokenizer(examples["prompt"] + examples["rejected"], truncation=True, max_length=512)
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

# 转换为Hugging Face数据集并映射
preference_dataset = Dataset.from_list(preference_data)
formatted_dataset = preference_dataset.map(format_for_reward_model)
```

**2. 奖励模型架构与损失**

我们使用一个标准Transformer模型（通常与策略模型使用相同的基础进行初始化），并在其顶部添加一个线性头以输出单个标量值（奖励）。训练目标采用成对排序损失。

损失函数 (loss function)旨在最大化选中回应和拒绝回应分数之间的边距：


$$
\text{损失} = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l)))]
$$


其中 $\sigma$ 是 sigmoid 函数。这会促使 $r_\theta(x, y_w) > r_\theta(x, y_l)$。

**3. 简化训练代码片段**

尽管像`trl`这样的库提供了`RewardTrainer`，但理解核心循环运作方式是很有益的。以下是一个PyTorch代码片段：

```python
# 加载用于序列分类的模型作为奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# 确保模型在正确的设备上运行（如果GPU可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model.to(device)

# 训练循环（简化版）
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)

# 假设 formatted_dataset 已加载到 DataLoader 中
# dataloader = DataLoader(formatted_dataset, batch_size=4, ...) # 设置已省略

# for batch in dataloader: # 循环的伪代码
    # --- 批量处理 ---
    # 整理选中回应和拒绝回应的输入
    # inputs_chosen = {"input_ids": batch["input_ids_chosen"], "attention_mask": batch["attention_mask_chosen"]}
    # inputs_rejected = {"input_ids": batch["input_ids_rejected"], "attention_mask": batch["attention_mask_rejected"]}
    # 将数据移动到设备
    # inputs_chosen = {k: v.to(device) for k, v in inputs_chosen.items()}
    # inputs_rejected = {k: v.to(device) for k, v in inputs_rejected.items()}

    # 前向传播以获得分数
    # rewards_chosen = reward_model(**inputs_chosen).logits # 形状: (batch_size, 1)
    # rewards_rejected = reward_model(**inputs_rejected).logits # 形状: (batch_size, 1)

    # 计算损失
    # loss = -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()

    # 反向传播
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # --- 批次结束 ---

# 训练后，保存奖励模型
# reward_model.save_pretrained("./reward_model_directory")
# tokenizer.save_pretrained("./reward_model_directory") # 也保存分词器
```

此代码片段说明了核心逻辑：获取选中和拒绝对的分数，计算成对损失，并更新奖励模型参数 (parameter)。

### 第二部分：实现策略优化步骤 (PPO)

现在，我们使用训练好的奖励模型 $r_\theta$ 来微调 (fine-tuning)策略大语言模型 (LLM) $\pi_\phi$，通过PPO进行。`trl`库极大地简化了这一复杂过程。

**1. TRL设置**

我们需要基础大语言模型（策略），参考模型（通常是初始的SFT模型，用于KL惩罚），训练好的奖励模型，以及分词 (tokenization)器 (tokenizer)。`trl`提供了`AutoModelForCausalLMWithValueHead`，它将策略模型与PPO所需的价值头打包在一起。

```python
# 加载带有价值头的PPO基础模型
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
# 加载参考模型（通常是PPO之前的同一起始点）
# 在PPO期间，此模型的权重保持冻结。
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
# 加载训练好的奖励模型（我们只需要它进行推理）
# reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model_directory") # 加载先前训练的奖励模型

# 假设奖励模型已加载并设置为评估模式
# reward_model.eval()
# reward_model.to(device) # 确保奖励模型在正确的设备上

# 配置 PPO
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-6, # 示例学习率，通常小于SFT/RM的学习率
    batch_size=64,        # 根据GPU内存调整
    mini_batch_size=4,    # 梯度累积有效进行
    gradient_accumulation_steps=4,
    ppo_epochs=4,         # 每个批次的优化周期数
    log_with="wandb",     # 可选：用于日志记录（需要wandb设置）
    # 其他参数，如kl_penalty ('kl')、target_kl等，都可以设置
)

# 初始化PPOTrainer
# 注意：我们不直接将奖励模型传递给PPOTrainer。
# 我们将在循环中手动使用它来获取奖励。
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    # dataset=your_prompt_dataset # 提供用于生成的提示
)
```

**2. PPO训练循环**

PPO的核心循环包括：

1. 从数据集中获取提示 $x$。
2. 使用当前策略 $\pi_\phi$ 生成回应 $y$。
3. 使用奖励模型 $r_\theta(x, y)$ 对 `(提示, 回应)` 对进行评分。
4. 使用提示、回应和奖励执行PPO更新步骤。

```python
# 假设 `prompt_dataloader` 提供分词后的提示批次
# generation_kwargs = { "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 64 }

# PPO 训练循环（使用TRL简化）
# for epoch in range(ppo_config.ppo_epochs):
#     for batch in prompt_dataloader: # 循环的伪代码
#         # --- 批量处理 ---
#         query_tensors = batch['input_ids'].to(device) # 获取提示张量

#         # 1. 从策略模型生成回应
#         # response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
#         # batch['response'] = tokenizer.batch_decode(response_tensors)

#         # 2. 使用奖励模型对生成的回应进行评分
#         # 拼接提示和回应以进行评分
#         # texts_to_score = [q + r for q, r in zip(batch['query'], batch['response'])]
#         # tokenized_scores = tokenizer(texts_to_score, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
#         # with torch.no_grad():
#         #    rewards = reward_model(**tokenized_scores).logits
#         # reward_tensors = [torch.tensor(reward) for reward in rewards] # 将奖励转换为PPOTrainer的张量列表

#         # 3. 执行PPO优化步骤
#         # stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
#         # ppo_trainer.log_stats(stats, batch, reward_tensors) # 记录指标
#         # --- 批次结束 ---

# 训练后保存调优的策略模型
# ppo_trainer.save_model("./policy_model_ppo_tuned")
```

`ppo_trainer.step` 函数封装了PPO更新逻辑：计算优势，根据PPO目标（奖励 - $\beta$ \* KL）计算策略和价值损失，并执行梯度更新。

**3. 监控进展**

在PPO训练期间，监控从奖励模型获得的平均奖励以及策略 $\pi_\phi$ 与参考策略 $\pi_{ref}$ 之间的KL散度是很重要的。奖励的增加表明策略正在为奖励模型进行优化，而受控的KL散度则表明它没有偏离原始模型的能力太远，这有助于防止灾难性遗忘或产生无意义的输出。



[交互图表：PPO奖励进展](https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/hands-on-core-rlhf-components#plot-90giip)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "PPO奖励进展",
    "xaxis": {
      "title": "PPO训练步数"
    },
    "yaxis": {
      "title": "平均奖励（来自奖励模型）"
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "平均奖励",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        0.5,
        1.2,
        1.8,
        2.3,
        2.6,
        2.8
      ],
      "line": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 此图显示了PPO训练步骤中平均奖励的预期趋势，随着策略适应奖励信号而增加。

### 实践考量

- \*\*计算成本：\*\*RLHF，特别是PPO步骤，计算量很大。它通常需要多个GPU和高效的实现（例如`trl`支持的DeepSpeed或FSDP集成）。
- \*\*超参数 (parameter) (hyperparameter)调优：\*\*PPO对学习率、$\beta$ (KL系数)、批量大小以及每个批次的PPO周期数等超参数很敏感。仔细调优对于训练的稳定性是必需的。
- \*\*奖励模型质量：\*\*PPO的成功在很大程度上取决于奖励模型的质量和校准。有缺陷的奖励模型可能导致策略偏离（奖励作弊）。
- \*\*稳定性：\*\*PPO训练有时会不稳定。监控奖励分布、KL散度和生成质量非常重要。经常采用诸如KL裁剪或自适应KL系数等技术。

这一实践概述为实现RLHF的核心计算步骤提供了一个起点。尽管全面应用需要更多的基础设施和数据，但理解这些组成部分能让您有效地使用和调整RLHF方法。`trl`库为在这些知识基础上发展提供了帮助。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  本文介绍了基于人类反馈的强化学习（RLHF）方法，详细阐述了奖励模型训练和使用PPO进行策略优化，这些是本节实现的核心组成部分。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种在RLHF中用于策略优化的策略梯度方法，如本节的实践实现所示。
- [TRL - Transformers Reinforcement Learning (Official Documentation)](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  提供了`trl`库的全面文档和指南，该库广泛用于实现RLHF组件（如奖励模型训练和PPO）的实践代码示例。

---

[上一节](08-RLHF%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%89%A9%E5%B1%95.md) · [下一节](../03-%E9%AB%98%E7%BA%A7%E5%AF%B9%E9%BD%90%E7%AE%97%E6%B3%95/01-%E5%AE%AA%E6%B3%95%E5%BC%8FAI%EF%BC%9A%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E8%B7%B5.md)
