# 持续微调（SFT/RLHF）的策略

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-30-continuous-training-model-updates/strategies-continual-fine-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型在首次部署后保持其对齐 (alignment)性是一个持续过程，而非一次性任务。用户期望会改变，新的安全问题会出现，并且期望的模型行为可能会随时间推移而漂移，或需要为特定应用进行修正。持续微调 (fine-tuning)，通过监督式方法（SFT）或强化学习 (reinforcement learning)（RLHF），提供逐步调整模型的机制。与初始微调不同，持续微调涉及将新数据或反馈整合到已运行的模型中，带来了效率、稳定性和知识保留方面的特别挑战。

### 持续监督式微调 (fine-tuning)（SFT）

持续SFT旨在根据新的监督式示例（例如，提示-完成对）更新模型遵循指令或执行特定任务的能力。这需要整合新数据的策略，同时不降低现有能力。

#### 数据来源与整合

新的SFT数据可以来自多个地方：

- **用户反馈：** 用户与模型交互时提供的明确更正或示例。
- **定向数据收集：** 识别模型表现不佳的方面（例如，特定类型的问题、安全场景），并积极整理或生成新示例。
- **合成数据：** 使用一个功能强大的模型（有时是模型本身或更大的模型）来生成新的指令-响应对，通常会进行质量筛选。

简单地在新数据上进行微调可能导致*灾难性遗忘*，即模型失去其之前学习过的任务能力。常见的缓解策略包括：

- **数据重放：** 将新的SFT示例与原始SFT数据的子集或之前微调轮次的代表性示例混合。旧数据与新数据的比例是一个重要的超参数 (parameter) (hyperparameter)。
- **权重 (weight)方案：** 在训练期间，对新数据与旧数据分配不同的重要性（损失权重），可能会对重要的安全或能力示例赋予更高的权重。
- **参数高效微调（PEFT）：** 像低秩适应（LoRA）这样的方法在这里特别有用。通过仅更新少量适配器权重，PEFT方法固有地限制了原始模型权重被修改的程度，从而减少了遗忘。仅训练适配器层在计算上也比完全微调便宜得多。

这是一个使用LoRA的简化PyTorch示例（假设`peft`等PEFT库可用），用于持续SFT步骤：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# 加载基础模型和分词器
model_name = "meta-llama/Llama-2-7b-hf" # 示例基础模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 假设存在之前微调步骤的LoRA适配器权重
# 如果开始持续微调，直接加载基础模型
# 如果继续，加载之前已适配的模型
# 为演示目的，我们假设是首次添加LoRA层
lora_config = LoraConfig(
    r=16, # 更新矩阵的秩
    lora_alpha=32, # 缩放因子
    target_modules=["q_proj", "v_proj"], # 定位特定模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 显示可训练参数显著减少

# --- 持续SFT步骤 ---
# 加载新的SFT数据批次（格式化的指令-响应对）
# new_data = load_new_sft_batch(...)
# inputs = tokenizer(
#     new_data['prompts'],
#     return_tensors='pt',
#     padding=True,
#     truncation=True
# )
# labels = tokenizer(
#     new_data['responses'],
#     return_tensors='pt',
#     padding=True,
#     truncation=True
# ).input_ids

# 假设'inputs'和'labels'是已准备好的张量
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 仅优化LoRA权重

# 简化训练步骤
model.train()
# outputs = model(
#     **inputs, labels=labels
# ) # 传入标签用于损失计算
# loss = outputs.loss
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()

# --- 在新数据上训练后 ---
# 保存更新后的LoRA适配器权重，而非整个模型
# model.save_pretrained("./updated_lora_adapters")

# 使用更新后的模型：
# updated_model = PeftModel.from_pretrained(
#     base_model, "./updated_lora_adapters"
# )
```

#### 评估

评估持续SFT涉及检查以下方面的性能：

1. **新任务/指令：** 验证在新数据所针对的特定方面是否有改进。
2. **保留集：** 测量与新数据相关的未见示例上的泛化能力。
3. **回归基准：** 运行标准基准测试（例如，GLUE的子集、特定专业测试）或重要的安全评估，以确保先前能力没有明显下降。

### 持续人类反馈强化学习 (reinforcement learning)（RLHF）

RLHF使模型与复杂的人类偏好对齐 (alignment)，这些偏好通常与有用性、诚实性和无害性相关。持续RLHF涉及根据新的偏好数据更新奖励模型（RM）和/或策略模型。

#### 更新奖励模型（RM）

RM预测人类会偏爱两个响应中的哪一个。它需要定期更新，原因如下：

- 新的偏好数据可用（例如，来自持续的标注工作）。
- 对期望行为的理解有所演变（例如，新的安全指南）。
- 策略模型漂移，可能显示RM不准确的新方面。

**数据来源：** 新的偏好对（$y_1, y_0 | x$，其中给定提示$x$时$y_1$优于$y_0$) 的收集方式与初始RM训练类似，通常侧重于*当前*策略模型生成的输出。

**训练：** RM可以通过以下方式更新：

- **完全再训练：** 使用旧数据和新偏好数据从头开始训练新的RM。计算开销大但可能更有效。
- **增量微调 (fine-tuning)：** 在新的偏好数据上继续训练现有RM，有时与旧数据混合（重放）以防止忘记之前的偏好。这更快，但有RM漂移或过度拟合近期数据的风险。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 假设'rm_model'是已加载的奖励模型（例如，基于分类头）
# 假设'rm_tokenizer'是其分词器

# 加载新的偏好数据批次：(提示, 选定响应, 被拒绝响应)对
# new_prefs = load_new_preference_batch(...)

# 为奖励模型分词输入
# chosen_inputs = rm_tokenizer(new_prefs['prompt'], new_prefs['chosen'], ...)
# rejected_inputs = rm_tokenizer(new_prefs['prompt'], new_prefs['rejected'], ...)

# 假设分词后的输入是已准备好的张量
rm_optimizer = torch.optim.AdamW(
    rm_model.parameters(), lr=1e-6
) # 使用小学习率

# 简化的RM更新步骤
rm_model.train()
# chosen_rewards = rm_model(**chosen_inputs).logits
# rejected_rewards = rm_model(**rejected_inputs).logits

# 成对铰链损失或类似损失
# loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
# loss.backward()
# rm_optimizer.step()
# rm_optimizer.zero_grad()

# 保存更新后的奖励模型状态
# torch.save(rm_model.state_dict(), "./updated_reward_model.pt")
```

#### 更新策略模型

策略模型（即LLM本身）通过RL（通常是PPO）进行微调，以最大化RM预测的奖励，同时保持与原始SFT模型的接近（由KL散度惩罚控制）。持续RLHF更新涉及：

- **使用更新后的RM：** 使用奖励模型的最新版本执行PPO更新，确保策略与最新偏好对齐。
- **频率：** RL更新可能比RM更新发生得更频繁或更不频繁，这取决于新偏好数据的产生速度和观察到的策略漂移。
- **KL惩罚参考：** KL惩罚通常将策略约束在*原始*SFT模型或近期表现良好的快照，而不一定是紧随其前的策略版本。这作为一个锚点，防止过度偏离。
- **奖励作弊：** 持续监控是否存在策略利用RM漏洞而非根据人类偏好真正改进的迹象。这可能需要更新RM或调整RL过程。

使用`trl`等库的框架可能如下所示：

```python
# 假设'ppo_trainer'已使用策略模型、
# 参考模型（SFT）、分词器和PPO配置进行初始化。
# 假设'updated_rm_model'是最新奖励模型。

# --- 持续RLHF步骤 ---
# 从数据集中采样提示
# prompts = sample_prompts(...)
# tokenized_prompts = tokenizer(prompts, ...)

# 使用当前策略模型生成响应
# responses_tensors = ppo_trainer.generate(tokenized_prompts, ...)
# responses_text = tokenizer.batch_decode(responses_tensors)

# 从更新后的奖励模型获取奖励
# rewards = get_rewards_from_rm(updated_rm_model, prompts, responses_text,
#                              tokenizer)

# 执行PPO优化步骤
# stats = ppo_trainer.step(tokenized_prompts, responses_tensors, rewards)

# 定期保存更新后的策略模型（如果使用PEFT，则保存其适配器）
# ppo_trainer.save_model("./updated_policy_model")
```

#### 评估

评估持续RLHF是复杂的。指标包括：

- **RM准确性：** 更新后的RM与保留偏好集上的人类判断相关程度如何。
- **策略性能：** 自动化评估（例如，检查有害提示的拒绝情况）和人工评估（例如，与先前版本进行A/B测试）是必要的。
- **KL散度：** 监控与参考模型的偏离程度。
- **奖励分数：** 追踪策略获得的平均奖励，但需谨慎，因为奖励可能被作弊。

> 简化工作流程，展示持续SFT和RLHF的并行循环，更新已部署的模型状态。

### 挑战与注意事项

- **灾难性遗忘：** 在SFT和RLHF中仍然是一个重要的挑战。PEFT方法、重放缓冲区和正则化 (regularization)技术是必要的，但并非总是足够。需要细致的调整和评估。
- **数据质量：** 持续微调 (fine-tuning)的有效性取决于传入数据（指令或偏好）的质量和相关性。有偏差或噪声的数据可能降低性能或引入意外行为。数据验证管道很重要。
- **计算开销：** 即使是持续微调大型模型，也需要大量的计算资源。高效的策略（PEFT、优化的训练循环、分布式设置）对于实际实施是必要的。
- **评估复杂性：** 衡量对齐 (alignment)漂移和改进需要持续的、可能昂贵的人工评估以及自动化基准测试。定义“正确”行为也可能演变，使评估复杂化。
- **同步：** 决定SFT更新、RM更新和策略RL更新的节奏和相互依赖性是复杂的。过时的RM可能误导策略训练，而策略漂移可能需要更频繁的RM或SFT更新。

实施持续微调需要成熟的MLOps基础设施，能够处理数据管道、频繁的再训练任务、可靠的版本控制、分阶段推出以及全面的监控，以确保更新能提高对齐性，同时不导致模型能力或安全性的有害退步。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），一种参数高效的微调方法，它减少了微调大型语言模型所需的可训练参数数量，有助于减轻灾难性遗忘。
- [Continual Learning in Neural Networks: A Review](https://www.sciencedirect.com/science/article/pii/S089360801930060X) — Giacomo Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, Stefan Wermter (2019)
  Journal: Neural Networks; Publisher: Elsevier; Volume: 119; Pages: 42-56; DOI: [10.1016/j.neunet.2019.03.012](https://doi.org/10.1016/j.neunet.2019.03.012)
  一篇关于神经网络持续学习技术的评论，包括减轻灾难性遗忘的策略，这与大型语言模型的持续微调相关。
- [Scaling Instruction-Finetuned Transformers](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  审视了指令微调的扩展特性，表明指令微调模型在新任务上泛化能力更强，并提升了模型对齐效果，为持续SFT提供了基础。

---

[上一节](02-%E6%8C%81%E7%BB%AD%E9%A2%84%E8%AE%AD%E7%BB%83%E7%AD%96%E7%95%A5.md) · [下一节](04-%E5%AE%89%E5%85%A8%E5%9C%B0%E6%95%B4%E5%90%88%E6%96%B0%E6%95%B0%E6%8D%AE%E6%BA%90.md)
