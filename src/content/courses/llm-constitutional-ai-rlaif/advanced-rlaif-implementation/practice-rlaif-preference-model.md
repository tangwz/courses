---
course: "llm-constitutional-ai-rlaif"
chapter: "advanced-rlaif-implementation"
lesson: "practice-rlaif-preference-model"
sourceId: 4529
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/practice-rlaif-preference-model"
title: "实践：训练基础AI偏好模型"
description: "动手实践环节，侧重于使用合成AI标签训练偏好模型。"
order: 8
plots: []
sourceHash: "c3556b546efc78dc10b6b7130efbb00afa9509d335dc63a0927a4816c392e2c2"
sourceCorrections: []
---

训练一个基础 AI 偏好模型是一项实际任务，它为后续强化学习 (reinforcement learning)阶段的奖励信号生成提供依据。目标是开发一个模型，能够预测在给定提示 $x$ 下，$y_1$ 或 $y_2$ 这两个回复中哪一个更受青睐。这种预测基于可能由另一个 AI 生成的标签。监督训练过程的主要机制会在此处详细阐述。

### 准备偏好数据集

第一步是组织数据。RLAIF 偏好建模需要一个数据集，其中每个样本包含：

1. 一个提示 $x$。
2. 为该提示生成的一对回复 $(y_1, y_2)$。
3. 一个指示哪个回复更受青睐的标签。通常，这来自一个 AI 标注器，可能由某个规则或其他准则指导。

对于本次练习，我们假设您有一个三元组 $(x, y_w, y_l)$ 的数据集，其中 $y_w$ 是根据您的 AI 标注器判定的“获胜”（更受青睐）回复，$y_l$ 是“失败”（不受青睐）回复。

一种常见表示方式是使用 Python 字典或 pandas DataFrames：

```python
# 单个数据点的示例结构
preference_data_point = {
    "prompt": "Explain the concept of quantum entanglement in simple terms.",
    "response_chosen": "Imagine two coins that are linked. If you flip one and it lands heads, you instantly know the other is tails, no matter how far apart they are. Quantum entanglement is like that for tiny particles.", # 获胜回复
    "response_rejected": "Quantum entanglement involves the superposition principle and wave function collapse correlating particle states. It's a complex Hilbert space phenomenon described by Bell's theorem.", # 失败回复
}

# 一个小型数据集可能看起来像：
dataset = [
    {"prompt": "P1", "response_chosen": "PC1", "response_rejected": "PR1"},
    {"prompt": "P2", "response_chosen": "PC2", "response_rejected": "PR2"},
    # ... 更多数据点
]
```

尽管我们在此使用合成 AI 标签，但请记住，提示的质量和多样性以及 AI 标注器的一致性会明显影响最终偏好模型的有效性。确保您的提示涵盖广泛的预期用例和可能的失败情况。

### 偏好模型架构

标准方法是使用预训练 (pre-training)语言模型作为骨干。偏好模型需要接收提示 $x$ 和两个回复 $y_1, y_2$ 来预测偏好分数。典型的架构会为每个回复单独计算一个标量分数 $r(x, y)$，然后比较这些分数。

该架构通常包含：

1. **分词 (tokenization)：** 将提示和回复（例如，`prompt + response_chosen`）连接起来并进行分词。对另一个回复重复此操作。
2. **编码：** 将分词后的序列通过 LLM 骨干（例如，GPT 变体、Llama 等）进行处理。
3. **标量头：** 提取一个表示（通常是最后一个 token 的嵌入 (embedding)或一个池化表示），并通过一个线性层（“标量头”）来为每个提示-回复对生成一个单一分数 $r(x, y)$。
4. **比较：** 通过比较 $y_1$ 和 $y_2$ 各自的分数 $r(x, y_1)$ 和 $r(x, y_2)$ 来确定它们之间的偏好。

> 偏好模型通常通过 LLM 骨干和标量头处理提示-回复对来生成分数，然后这些分数用于计算偏好损失。

### 配对偏好的损失函数 (loss function)

训练偏好模型最常用的损失函数源自 Bradley-Terry 模型，该模型认为 $y_w$ 比 $y_l$ 更受青睐的概率可以使用它们基础分数的差异来建模。我们使用逻辑函数（sigmoid，$\sigma$）将分数差异映射到概率。训练目标是最大化在数据集中观察到的偏好的对数似然。最小化负对数似然得到损失：


$$
L = - \mathbb{E}_{(x, y_w, y_l) \sim D} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]
$$


这里，$r_\theta(x, y)$ 是我们的偏好模型（参数 (parameter)为 $\theta$）对提示 $x$ 和回复 $y$ 输出的标量分数。$D$ 代表偏好数据集。此损失函数促使模型为被选择的回复 ($y_w$) 分配比被拒绝的回复 ($y_l$) 更高的分数。

### 训练循环实现概述

让我们使用类似 PyTorch 的伪代码结构来概述主要的训练循环，假设您有一个 `PreferenceModel` 类，它封装了一个预训练 (pre-training)的 transformer 和一个标量头，并且有一个提供 $(x, y_w, y_l)$ 批次数据的 `dataset`。

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer # 示例导入

# 假设 PreferenceModel 存在，接受骨干模型名称
# 假设 PreferenceDataset 已定义以处理分词和批处理
# 假设 dataset 是 PreferenceDataset 的一个实例

# 配置
model_name = "path/to/your/base_llm" # 或 Hugging Face 标识符
learning_rate = 1e-5
batch_size = 4
num_epochs = 1

# 初始化模型、分词器、优化器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 如有必要，添加填充 token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # 如有需要，调整模型嵌入层大小

preference_model = PreferenceModel(model_name)
preference_model.model.resize_token_embeddings(len(tokenizer)) # 根据添加的 token 进行调整
optimizer = AdamW(preference_model.parameters(), lr=learning_rate)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
preference_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # 批次包含分词后的提示、选择的回复、拒绝的回复
        # 示例键：'prompt_ids', 'chosen_ids', 'chosen_mask', 'rejected_ids', 'rejected_mask'

        # 确保张量在正确的设备上（例如，GPU）
        chosen_ids = batch['chosen_ids'].to(device)
        chosen_mask = batch['chosen_mask'].to(device)
        rejected_ids = batch['rejected_ids'].to(device)
        rejected_mask = batch['rejected_mask'].to(device)

        # 从偏好模型获取分数
        # 模型应在内部处理提示和回复的连接
        # 或者接收预先连接的输入
        # 假设模型返回所选和被拒绝回复的分数
        scores_chosen = preference_model(input_ids=chosen_ids, attention_mask=chosen_mask)
        scores_rejected = preference_model(input_ids=rejected_ids, attention_mask=rejected_mask)

        # 计算损失
        # 损失 = -log_sigmoid(选择的分数 - 拒绝的分数)
        loss = -F.logsigmoid(scores_chosen - scores_rejected).mean()

        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# 保存训练好的偏好模型权重
# torch.save(preference_model.state_dict(), "preference_model_basic.pth")
```

**实现要点：**

- **分词 (tokenization)：** 小心处理分词、填充和注意力掩码非常重要，尤其是要确保所选和被拒绝回复处理之间的一致性。在分词*之前*连接提示和回复，以获得最佳上下文 (context)。
- **模型正向传播：** `PreferenceModel` 需要被构建为计算标量分数 $r(x, y)$。这通常意味着将 transformer 输出的相关状态（例如，最后一个 token 的隐藏状态）通过线性头进行处理。
- **梯度累积：** 对于大型模型或 GPU 内存有限的情况，可能需要在每个优化器步骤之前对多个小批次进行梯度累积。
- **混合精度：** 使用混合精度训练（例如，`torch.cuda.amp`）可以显著提高速度并节省内存，这对于大型骨干模型尤其重要。

### 评估偏好模型

在部署偏好模型为强化学习 (reinforcement learning)生成奖励之前，请使用预留的偏好对测试集评估其性能。主要衡量标准是**准确率**：模型正确预测偏好回复（即，为 $y_w$ 分配比 $y_l$ 更高的分数）的配对百分比。


$$
\text{准确率} = \frac{1}{|D_{\text{测试}}|} \sum_{(x, y_w, y_l) \in D_{\text{测试}}} \mathbb{I}[r_\theta(x, y_w) > r_\theta(x, y_l)]
$$


其中 $\mathbb{I}[\cdot]$ 是指示函数。

高准确率（例如，>75-80%，取决于任务难度和标注器质量）表明模型已经学会了 AI 标注数据中存在的偏好模式。同时监控损失曲线以查看收敛情况并检查过拟合 (overfitting)。定性分析，即检查模型与标签不一致的情况，可以提供关于标注器潜在偏差或模型可能正在学习的不一致之处的见解。

### 专家级考虑

- **处理平局：** 基本公式假定严格偏好。如果您的 AI 标注器提供平局（相等偏好），您可能需要调整损失函数 (loss function)或数据处理（例如，省略平局，或修改损失以惩罚平局对之间较大的分数差异）。
- **基础模型选择：** 用于偏好模型的骨干 LLM 的能力很重要。一个更强的基础模型可以学习更多偏好，但需要更多计算资源。通常，偏好模型的大小与通过 RLAIF 训练的策略模型相似或略小。
- **参数 (parameter)高效微调 (fine-tuning) (PEFT)：** 像 LoRA（低秩适应）这样的技术对于训练偏好模型非常有效。通过只插入和训练少量适配器层，您可以显著降低计算成本和内存占用，与完全微调相比，同时通常能获得相似的性能。考虑到现代 LLM 骨干模型庞大的规模，这一点尤为相关。
- **校准：** 检查分数差异 $(r(x, y_w) - r(x, y_l))$ 的大小是否与偏好的置信度或强度相关。一个良好校准的模型通常有助于稳定的强化学习 (reinforcement learning)训练。

本次实践课为您提供了训练 AI 偏好模型的基本步骤。您准备了数据，定义了架构，使用适当的损失函数实现了训练循环，并考虑了评估指标。所得模型，捕捉了 AI 标注数据集中包含的偏好，现已准备好在 RLAIF PPO 训练阶段作为奖励函数提供者，这将在下一节介绍。

## 参考资料

- [Rank analysis of incomplete block designs: The method of paired comparisons](https://projecteuclid.org/journals/biometrika/volume-39/issue-3-4/Rank-analysis-of-incomplete-block-designs-the-method-of-paired-comparisons/bj/1169992518.full) — Ralph Allan Bradley and Milton E. Terry (1952)
  Journal: Biometrika; Publisher: Oxford University Press; Volume: 39; Pages: 324-345; DOI: [10.1093/biomet/39.3-4.324](https://doi.org/10.1093/biomet/39.3-4.324)
  这篇论文介绍了Bradley-Terry模型，该模型是训练AI偏好模型中使用的成对偏好损失函数的基础。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  提出低秩适应（LoRA）方法，这是一种参数高效的微调方法，对于高效训练大型语言模型和偏好模型非常重要。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法AI，这是一种利用AI反馈使模型与人类价值观对齐的方法，为在RLAIF中生成AI偏好标签提供了背景。
