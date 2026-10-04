---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "integrating-rlhf-pipeline"
lesson: "synchronizing-models-training"
sourceId: 5179
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-5-integrating-rlhf-pipeline/synchronizing-models-training"
title: "训练过程中模型的同步"
description: "（如适用）处理分布式系统中的模型状态和参数更新。"
order: 5
plots: []
sourceHash: "6c7f3f5470f744e2eab2fa18f0fe8941265f1529a91c86bb5f51eb7067241638"
sourceCorrections: []
---

整合RLHF流程的组件，特别是在近端策略优化（PPO）阶段，管理多个大型语言模型之间的交互和状态是一项重大的工程工作。PPO步骤不仅涉及正在优化的策略模型；它还依赖于一个奖励模型（RM）、一个价值模型（批评器），并且通常还有一个静态的参考模型。确保这些模型被正确加载、使用和更新（或保持静态）对稳定且高效的训练有益。

### PPO中的模型角色

在强化学习 (reinforcement learning)优化阶段，有几个模型同时运行：

1. **策略模型（Actor）：** 这是大型语言模型，其参数 (parameter)正在被PPO算法积极更新。它对提示生成回应，目标是最大化由RM提供的奖励信号，同时保持与其初始分布（通常是SFT模型）接近。
2. **价值模型（Critic）：** 该模型估算给定状态（提示或提示+部分生成）的预期回报（累积未来奖励）。它与策略模型一同训练，以通过计算优势值来帮助减少策略梯度估计的方差。它通常从策略模型或RM架构初始化，但使用不同的目标进行训练（例如，最小化与观测回报的均方误差）。其参数也会在PPO期间更新。
3. **奖励模型（RM）：** 在前一阶段根据人类偏好数据训练而成，此模型为生成的回应提供标量奖励信号。特别要指出的是，RM的参数在PPO阶段是*冻结*的。它充当一个固定的判断者，根据习得的人类偏好来评估回应质量。
4. **参考策略模型：** 这通常是SFT模型在PPO微调 (fine-tuning)开始*前*的一个副本。它的作用很重要，用于计算PPO目标函数中的KL散度惩罚项：$L_{PPO} = L_{\text{clip}} - c_{\text{vf}} L_{\text{VF}} + c_{\text{entropy}} S[\pi_{\theta}] + \beta \text{KL}[\pi_{\theta} || \pi_{\text{ref}}]$。KL项，$\beta \text{KL}[\pi_{\theta} || \pi_{\text{ref}}]$，惩罚策略$\pi_{\theta}$因为它偏离参考策略$\pi_{\text{ref}}$太远。与RM一样，参考模型的参数在整个PPO训练期间保持*冻结*，以提供一个稳定的目标分布。

### 确保正确的模型状态

保持每个模型的正确状态，特别是区分正在训练的和保持静态的模型，需要细致的实现。

- **冻结权重 (weight)：** 奖励模型和参考策略模型都必须冻结其参数 (parameter)。在PyTorch等框架中，这通常通过在将它们或策略/价值模型参数传递给优化器*前*，将其参数的`requires_grad`属性设置为`False`来实现。这确保在反向传播 (backpropagation)期间不会计算或应用梯度到这些静态模型。

  ```python
  # 示例（PyTorch）
  reward_model.eval() # 设置为评估模式
  reference_model.eval() # 设置为评估模式
  for param in reward_model.parameters():
      param.requires_grad = False
  for param in reference_model.parameters():
      param.requires_grad = False

  # 策略和价值模型保持可训练
  policy_model.train()
  value_model.train()

  # 优化器通常只包含策略和价值模型的参数
  optimizer = torch.optim.Adam(
      list(policy_model.parameters()) + list(value_model.parameters()),
      lr=learning_rate
  )
  ```
- **加载和实例化：** 工作流程必须正确加载最终的SFT模型，以用作初始策略*和*参考模型。同样地，训练好的奖励模型也必须被加载。价值模型可能会从策略模型的权重或其他策略初始化。

### 分布式训练中的同步

训练RLHF中使用的大型模型通常需要在多个GPU甚至多台机器上进行分布式计算设置。这带来了进一步的同步挑战：

- **参数 (parameter)一致性：** 所有分布式工作器必须以策略、价值、参考和奖励模型的完全相同的初始参数开始。这通常涉及在所有进程中一致地加载检查点。
- **梯度聚合：** 在PPO更新期间，不同工作器在不同数据批次上计算的梯度需要在更新策略和价值模型之前进行聚合。标准分布式训练库（如`torch.distributed`或`deepspeed`）提供诸如`AllReduce`等机制来平均所有工作器上的梯度。这确保所有工作器应用相同的更新，从而保持策略和价值模型同步。

digraph G {
rankdir=LR;
node [shape=box, style=rounded, fontname="Arial", fontsize=10];
edge [fontname="Arial", fontsize=9];

```
subgraph cluster_0 {
    label = "工作器 1";
    bgcolor="#e9ecef";
    P1 [label="策略 θ", style=filled, fillcolor="#a5d8ff"];
    V1 [label="价值 V", style=filled, fillcolor="#bac8ff"];
    RM1 [label="奖励模型（固定）", style=filled, fillcolor="#ffd8a8"];
    Ref1 [label="参考模型（固定）", style=filled, fillcolor="#b2f2bb"];
    Data1 [label="数据批次 1"];
    Grad1 [label="梯度（策略，价值）"];

    Data1 -> P1 -> Grad1;
    Data1 -> V1 -> Grad1;
    P1 -> RM1 [style=dashed, arrowhead=none, label="RM的输入"];
    P1 -> Ref1 [style=dashed, arrowhead=none, label="KL的输入"];
}

subgraph cluster_1 {
    label = "工作器 2";
    bgcolor="#e9ecef";
    P2 [label="策略 θ", style=filled, fillcolor="#a5d8ff"];
    V2 [label="价值 V", style=filled, fillcolor="#bac8ff"];
    RM2 [label="奖励模型（固定）", style=filled, fillcolor="#ffd8a8"];
    Ref2 [label="参考模型（固定）", style=filled, fillcolor="#b2f2bb"];
    Data2 [label="数据批次 2"];
    Grad2 [label="梯度（策略，价值）"];

    Data2 -> P2 -> Grad2;
    Data2 -> V2 -> Grad2;
    P2 -> RM2 [style=dashed, arrowhead=none, label="RM的输入"];
    P2 -> Ref2 [style=dashed, arrowhead=none, label="KL的输入"];
}

 subgraph cluster_2 {
    label = "工作器 N";
    bgcolor="#e9ecef";
    PN [label="策略 θ", style=filled, fillcolor="#a5d8ff"];
    VN [label="价值 V", style=filled, fillcolor="#bac8ff"];
    RMN [label="奖励模型（固定）", style=filled, fillcolor="#ffd8a8"];
    RefN [label="参考模型（固定）", style=filled, fillcolor="#b2f2bb"];
    DataN [label="数据批次 N"];
    GradN [label="梯度（策略，价值）"];

    DataN -> PN -> GradN;
    DataN -> VN -> GradN;
    PN -> RMN [style=dashed, arrowhead=none, label="RM的输入"];
    PN -> RefN [style=dashed, arrowhead=none, label="KL的输入"];
}

Sync [label="同步点\n（例如 AllReduce）", shape=diamond, style=filled, fillcolor="#ffc9c9"];
Update [label="应用全局更新\n（优化器步骤）", shape=cds, style=filled, fillcolor="#96f2d7"];

Grad1 -> Sync;
Grad2 -> Sync;
GradN -> Sync;
Sync -> Update;

Update -> P1 [label="更新 θ"];
Update -> V1 [label="更新 V"];
Update -> P2 [label="更新 θ"];
Update -> V2 [label="更新 V"];
Update -> PN [label="更新 θ"];
Update -> VN [label="更新 V"];

{rank=same; P1; P2; PN;}
{rank=same; V1; V2; VN;}
{rank=same; RM1; RM2; RMN;}
{rank=same; Ref1; Ref2; RefN;}
{rank=same; Grad1; Grad2; GradN;}
```

}
\`\`\`

```
> 图示了分布式PPO设置中的模型同步。每个工作器使用共享的固定奖励模型和参考模型局部计算梯度。策略和价值模型的梯度被聚合（例如，通过AllReduce），之后对所有工作器上的这些模型应用同步更新。
```

- **静态模型访问：** 固定的奖励模型和参考模型可能在每个工作器上复制，也可以使用参数服务器架构等方法，尽管对于仅用于推理 (inference)的模型，复制是常见做法。重点在于，在整个PPO阶段，每个工作器都必须访问这些模型的相同且不变的版本。

未能正确同步可能导致工作器之间模型行为差异、过时的梯度，最终导致不稳定或低效的训练。Hugging Face的TRL等库简化了部分细节，特别是参考模型和KL计算的处理，但理解底层的同步要求仍然很重要，尤其是在调试或定制流程时。因此，在分布式设置中细致管理模型状态和通信协议，不仅仅是实现细节，更是成功进行大规模RLHF训练的先决条件。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  解释了核心PPO算法，包括旧策略和价值函数在稳定训练中的作用。
- [Distributed Communication Package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Authors (2024)
  Publisher: PyTorch
  提供了在PyTorch中实现分布式训练的详细信息，涵盖了梯度聚合和进程同步。
- [TRL - Transformers Reinforcement Learning (Documentation)](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  提供了实现RLHF管道的实用指南和代码示例，包括PPO训练循环和模型管理。
