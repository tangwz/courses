---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "integrating-rlhf-pipeline"
lesson: "simplified-rlhf-loop-practical"
sourceId: 5183
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-5-integrating-rlhf-pipeline/simplified-rlhf-loop-practical"
title: "动手实践：运行简化的RLHF循环"
description: "在一个小规模上执行一个基本的端到端RLHF训练过程。"
order: 7
plots: []
sourceHash: "13179c89fa66e8441894d9a2a8e6faf4e78780135f28fc98067e0ef6b2ab4c0a"
sourceCorrections: []
---

本次实践将监督微调 (fine-tuning)（SFT）、奖励模型（RM）以及使用PPO的RL微调的各个阶段整合到一个功能性的、简化的端到端循环中。此处的目的并非达成最先进的对齐 (alignment)效果，而是演示这些组成部分如何在一个训练过程中相互作用，并使用Hugging Face TRL库等工具。

我们假设您具备以下条件：

1. 一个适用于SFT的预训练 (pre-training)语言模型（例如 `gpt2`、`distilbert-base-uncased`）。为简单起见，在此最小示例中，我们甚至可能直接使用基础模型作为起始的“SFT”策略。
2. 一个与基础LM架构兼容的预训练奖励模型。该模型应接收一个`(query, response)`对（或仅接收响应，具体取决于其训练方式），并输出一个标量分数。
3. 一个已安装 `transformers`、`torch`（或 `tensorflow`）和 `trl` 的Python环境。

### 配置组成部分

首先，我们需要加载模型并配置PPO训练器。我们将使用占位符名称；请将其替换为您的实际模型路径或Hugging Face标识符。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLMWithValueHead, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. 配置 - 用于演示的最小PPO设置
ppo_config = PPOConfig(
    model_name="gpt2", # 或者您的特定SFT模型路径
    learning_rate=1.41e-5,
    batch_size=4,       # 用于演示的小批量大小
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    log_with="tensorboard", # 可选：用于日志记录
    kl_penalty="kl",      # 使用KL惩罚
    target_kl=0.1,        # 目标KL散度
    init_kl_coef=0.2,     # 初始KL系数
    adap_kl_ctrl=True,    # 使用自适应KL控制
    ppo_epochs=4,         # 每批次的优化周期数
    seed=0,
)

# 2. 加载模型和分词器
# 策略模型（Actor/Critic）：从SFT/基础模型初始化
# AutoModelForCausalLMWithValueHead 结合了LM头和价值头
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
# 参考模型（用于KL散度）：保留初始策略的副本
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
# 确保为分词器设置了pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 奖励模型（RM）：单独加载，假设它是一个文本分类风格的pipeline
# 替换！
reward_model_name = "path/to/your/reward/model" # 替换！
# 注意：这可能需要自定义pipeline或直接加载模型，具体取决于您的RM
try:
    # 简化示例，假设是兼容的情感/奖励pipeline
    reward_pipe = pipeline("text-classification", model=reward_model_name, device=policy_model.device)
    print("Reward model loaded via pipeline.")
    # 定义一个函数来获取标量分数
    def get_reward_score(texts):
        # 处理文本，如果需要，可能将其格式化为 (query, response)
        # 这高度依赖于您的奖励模型的输入格式
        # 假设RM输出一个字典列表，如 [{'label': 'POSITIVE', 'score': 0.9}]
        results = reward_pipe(texts, return_all_scores=True) # 根据您的pipeline进行调整
        # 提取所需的分数（例如，“POSITIVE”的分数或特定的分数索引）
        # 此提取逻辑高度依赖于您的RM的输出
        scores = []
        for result in results:
            # 示例：查找特定标签的分数，或假设第一个分数是奖励
            # 根据您的奖励模型结构调整此逻辑
            score = 0.0 # 默认分数
            if isinstance(result, list): # 处理变化的pipeline输出
                 for label_score in result:
                     if label_score['label'] == 'POSITIVE': # 示例标签
                          score = label_score['score']
                          break
            elif isinstance(result, dict):
                 score = result.get('score', 0.0) # 简化回退
            scores.append(torch.tensor(score, device=policy_model.device))
        return scores

except Exception as e:
    print(f"Warning: Could not load reward model pipeline '{reward_model_name}'. Using dummy rewards. Error: {e}")
    # 如果RM加载失败，则回退到虚拟奖励函数
    def get_reward_score(texts):
        # 虚拟奖励：基于长度的分数（仅用于演示）
        return [torch.tensor(len(text) / 100.0, device=policy_model.device) for text in texts]

# 3. 初始化PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    # 在此手动循环示例中，可以省略数据集
    # 如果不使用 AutoModelForCausalLMWithValueHead，则value_model需要单独设置
)

print("设置完成。已准备好运行简化的RLHF循环。")
```

### 简化的RLHF循环

现在，我们来执行RLHF循环的几个步骤。我们将手动提供查询、生成响应、获取奖励并执行PPO更新。

```python
# 定义一些示例查询
queries = [
    "Explain the concept of KL divergence in simple terms:",
    "Write a short poem about a robot learning:",
    "What are the main stages of RLHF?",
    "Suggest a name for a friendly AI assistant:",
]

# 对查询进行分词
query_tensors = [tokenizer.encode(q, return_tensors="pt").to(policy_model.device) for q in queries]

# 策略模型的生成设置
generation_kwargs = {
    "min_length": -1, # 允许提前停止
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 64, # 限制响应长度用于演示
}

# 运行几个PPO步骤（例如2步）
num_steps = 2
for step in range(num_steps):
    print(f"\n--- PPO Step {step + 1} ---")

    # 1. 策略执行：从策略模型生成响应
    response_tensors = []
    for query_tensor in query_tensors:
        # 生成响应；响应包含查询和生成的部分
        response = ppo_trainer.generate(query_tensor.squeeze(0), **generation_kwargs)
        response_tensors.append(response.squeeze())

    # 解码响应以进行奖励计算和日志记录
    decoded_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

    # 2. 奖励计算：使用RM对生成的响应进行评分
    # 如果需要，为奖励模型格式化文本（例如，组合查询+响应）
    # 此示例假设RM对包括提示在内的完整生成文本进行评分
    reward_texts = decoded_responses
    rewards = get_reward_score(reward_texts) # 张量标量列表

    # 3. PPO优化步骤
    # 为 ppo_trainer.step 准备输入
    # query_tensors 需要是 List[torch.Tensor]
    # response_tensors 需要是 List[torch.Tensor]
    # rewards 需要是 List[torch.Tensor]（每个样本的标量奖励）
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # 4. 日志记录
    print(f"Query examples: {[q[:50] + '...' for q in queries]}")
    print(f"Response examples: {[r[len(q):][:80] + '...' for q, r in zip(queries, decoded_responses)]}")
    print(f"Mean reward: {torch.mean(torch.stack(rewards)).item():.4f}")
    if 'ppo/kl' in stats:
      print(f"KL Divergence: {stats['ppo/kl']:.4f}")
    if 'ppo/loss/policy' in stats:
      print(f"Policy Loss: {stats['ppo/loss/policy']:.4f}")
    if 'ppo/loss/value' in stats:
      print(f"Value Loss: {stats['ppo/loss/value']:.4f}")

    # 可选：如果使用TensorBoard等日志记录器，则记录详细统计信息
    # ppo_trainer.log_stats(stats, queries, response_tensors, rewards)

print("\n简化的RLHF循环已完成。")
```

### 数据流图

下图描述了此简化循环单次迭代中的数据流：

> RLHF中单个PPO步骤的数据流。查询促使策略模型生成响应，这些响应随后由奖励模型评分。PPO训练器使用查询、响应和奖励分数来计算损失并更新策略模型的参数 (parameter)。

### 观察与说明

在此简化执行中，您应注意以下几点：

- **响应生成**：策略模型根据输入查询生成文本。最初，这些响应反映SFT模型（或基础模型）的行为。
- **奖励分配**：奖励模型为这些响应分配分数。如果使用真实的RM，这些分数应与期望的特点（例如，有用性、无害性）相关联。如果使用虚拟奖励，它只是一个占位符信号。
- **PPO统计数据**：`ppo_trainer.step`函数返回统计数据。请关注`mean_reward`；理想情况下，如果策略学习生成RM偏好的响应，它应该呈上升趋势。请留意`KL Divergence` (`ppo/kl`)，以确保策略不会与原始参考模型偏离过大，从而防止崩溃。策略和价值损失（`ppo/loss/policy`、`ppo/loss/value`）表示优化进度。
- **策略变化**：尽管在仅几个步骤和小批量的情况下变化不明显，但策略模型的参数 (parameter)*正在*更新。如果您使用更大的数据集运行更多步骤，您将开始看到生成响应的风格或内容发生变化，有望更好地与奖励模型中编码的偏好对齐 (alignment)。

“本次实践将RLHF过程简化为其核心循环：生成、评分、更新。它突出了策略模型、奖励模型以及由训练器管理的PPO算法之间的交互点。虽然RLHF涉及更大的规模、复杂的数据处理、仔细的超参数 (hyperparameter)调整和分布式训练，但这个动手示例为本章讨论的各组成部分之间的底层机制提供了切实的感受。在此基础上，您可以扩展实现、整合适当的数据集处理，并完善配置以进行更重要的对齐任务。”

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，一种在强化学习中广泛使用的策略梯度方法。
- [TRL: Transformer Reinforcement Learning Library](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  TRL库的官方文档，提供了使用Transformer实现RLHF的指南和API参考。
