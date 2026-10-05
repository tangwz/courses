# CAI和RLAIF的计算成本

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-8-optimization-scalability-cai-rlaif/computational-costs-cai-rlaif)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实行宪法式AI（CAI）和基于AI反馈的强化学习 (reinforcement learning)（RLAIF）等先进对齐 (alignment)技术，相比标准监督微调 (fine-tuning)（SFT）或按每次更新计算的基础模型预训练 (pre-training)，其计算需求显著增加。了解这些成本的来源和规模，是进行对齐项目规划、预算及优化的必要条件。这里分析了CAI和RLAIF的计算要求，并找出主要瓶颈。

### 分析宪法式AI（CAI）成本

CAI的计算成本主要源于其两阶段结构：监督学习 (supervised learning)（SL）数据生成阶段和随后的微调 (fine-tuning)阶段。

1. **监督数据生成阶段（批改与修订）**：这通常是CAI中计算最密集的部分。对于初始数据集中的每个提示，该过程通常涉及：

   - **初始回应生成**：使用基础LLM（$M_{base}$）进行推理 (inference)，生成初始回应（$r_0$）。成本随序列长度和$M_{base}$大小而变化。
   - **批改生成**：使用批改模型（$M_{critique}$）（通常是另一个有能力的LLM）进行推理，根据宪法、原始提示（$p$）和初始回应（$r_0$）进行提示，以识别违规行为。成本随总输入长度和$M_{critique}$大小而变化。
   - **修订生成**：使用修订模型（$M_{revise}$）（可能与$M_{base}$或$M_{critique}$相同，但经过专门提示）进行推理，将$p$、$r_0$和批改作为输入，以生成修订后的回应（$r_1$）。成本随输入长度和$M_{revise}$大小而变化。
   - **（可选）迭代细化**：一些CAI实现会执行多轮批改和修订，从而相应地增加推理成本。

   生成SL数据集的总推理成本大约为：

   
   $$
   \text{成本}_{CAI\_推理} \approx N_{提示} \times (\text{成本}_{推理}(M_{base}) + \text{成本}_{推理}(M_{critique}) + \text{成本}_{推理}(M_{revise})) \times N_{迭代}
   $$
   

   其中$N_{提示}$是初始提示的数量，$N_{迭代}$是每个提示的批改/修订轮次。由于$M_{critique}$和$M_{revise}$本身通常是大型LLM，此阶段涉及对*每个*数据点进行多次昂贵的LLM推理调用。
2. **监督微调（SFT）阶段**：一旦（提示，修订后回应）对的数据集生成，基础LLM（$M_{base}$）将在此数据上进行微调。

   - **训练成本**：这涉及标准SFT过程。成本取决于生成数据集的大小、$M_{base}$的大小、微调持续时间（周期）以及训练超参数 (parameter) (hyperparameter)（批次大小、学习率）。尽管是标准SFT，但源自批改/修订阶段的数据集规模可能非常大，从而导致高昂的训练成本。

**CAI中的瓶颈**：

- **推理成本**：批改和修订阶段重复的LLM推理调用是一个主要瓶颈，特别是当使用大型模型进行批改/修订并执行多轮迭代时。
- **数据集大小**：生成大型、高质量的SL数据集需要推理阶段大量的预先计算投入。
- **SFT成本**：在可能庞大的生成数据集上微调大型基础模型需要大量的训练资源（GPU时间、内存）。

### 分析基于AI反馈的强化学习 (reinforcement learning)（RLAIF）成本

RLAIF用AI驱动的组件替代了人类偏好标注以及RLHF中可能有的SFT阶段，带来了其特有的成本结构。

1. **AI偏好数据生成**：与RLHF相似，RLAIF需要偏好数据，但这些数据由AI模型生成。

   - **回应对生成**：对于每个提示（$p$），使用当前策略模型（初始为$M_{base}$，后续为迭代更新后的模型）生成多个回应（例如$r_A$，$r_B$）。这需要每个提示进行$k$次推理 (inference)，其中$k$是为比较而生成的回应数量（例如成对比较时$k=2$）。
   - **AI偏好标注**：使用AI偏好标注模型（$M_{pref\_labeler}$）（通常是根据宪法提示的LLM）进行推理，根据期望标准（例如宪法定义的有益性、无害性）比较生成的回应（$r_A$，$r_B$），并输出偏好标签（例如$r_A \succ r_B$）。成本随输入长度（提示+两个回应）和$M_{pref\_labeler}$大小而变化。

   成本大致为：

   
   $$
   \text{成本}_{RLAIF\_偏好生成} \approx N_{提示} \times (k \times \text{成本}_{推理}(M_{policy}) + \text{成本}_{推理}(M_{pref\_labeler}))
   $$
   
2. **偏好模型（PM）训练**：训练一个单独的奖励模型（$M_{RM}$）来预测AI生成的偏好标签。

   - **训练成本**：取决于偏好数据集的大小、$M_{RM}$选择的架构（通常小于主LLM，但仍可能很大）以及训练超参数 (parameter) (hyperparameter)。此成本类似于RLHF中训练奖励模型的成本。
3. **强化学习（RL）微调 (fine-tuning)**：此阶段使用训练好的偏好模型（$M_{RM}$）作为奖励函数，通过PPO等RL算法微调策略LLM（$M_{policy}$）。这通常是复杂度最高且资源消耗最大的阶段。对于每个PPO步骤：

   - **策略运行**：从当前策略$M_{policy}$为一批提示生成回应（推理成本）。
   - **奖励计算**：使用$M_{RM}$计算每个生成回应的奖励（推理成本）。
   - **PPO更新**：对策略模型（$M_{policy}$）和可能的值模型（$M_{value}$）执行多次前向和反向传播 (backpropagation)，以计算策略梯度并更新策略权重 (weight)（训练成本）。PPO通常涉及每批收集数据进行多个优化周期。

   RL阶段涉及推理（策略模型和奖励模型）和训练（策略模型和值模型更新）的紧密循环，这使得它在时间和内存方面都具有高计算要求，特别是需要大量GPU内存来存储多个模型副本和激活。

**RLAIF中的瓶颈**：

- **RL训练循环**：PPO循环因结合了多个模型（策略、RM）的重复推理以及可能在大型LLM上进行的复杂梯度计算而以资源密集著称。同时存储多个模型的激活、梯度和优化器状态需要大量的GPU内存。
- **AI偏好标注推理**：与CAI的批改阶段类似，使用大型LLM作为$M_{pref\_labeler}$会产生高昂的推理成本，用于生成偏好数据集。
- **偏好模型训练**：尽管通常小于策略LLM，训练一个高质量的$M_{RM}$仍需要大量的计算资源。

### 比较概述与主要因素

CAI和RLAIF都比简单的SFT带来了显著的计算开销，主要因为额外增加了LLM推理 (inference)步骤来生成反馈（CAI中的批改/修订，RLAIF中的偏好标签），以及大规模SFT（CAI）或RL训练（RLAIF）所增加的复杂性。



[交互图表：CAI与RLAIF的相对计算成本概况](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-8-optimization-scalability-cai-rlaif/computational-costs-cai-rlaif#plot-n281v0)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "name": "推理（反馈生成）",
      "x": [
        "CAI",
        "RLAIF"
      ],
      "y": [
        60,
        45
      ],
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "type": "bar",
      "name": "SFT / PM训练",
      "x": [
        "CAI",
        "RLAIF"
      ],
      "y": [
        40,
        25
      ],
      "marker": {
        "color": "#748ffc"
      }
    },
    {
      "type": "bar",
      "name": "RL训练（PPO）",
      "x": [
        "CAI",
        "RLAIF"
      ],
      "y": [
        0,
        70
      ],
      "marker": {
        "color": "#f06595"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "CAI与RLAIF的相对计算成本概况"
    },
    "barmode": "group",
    "yaxis": {
      "title": {
        "text": "相对成本（抽象单位）"
      }
    },
    "xaxis": {
      "title": {
        "text": "对齐方法"
      }
    },
    "legend": {
      "title": {
        "text": "成本组成部分"
      }
    }
  }
}
```

</details>



> CAI和RLAIF的计算成本相对细分。CAI的成本主要由反馈生成（推理）和随后的SFT主导。RLAIF涉及反馈生成推理和PM训练，但最大的组成部分通常是基于PPO的RL训练循环。实际成本根据实施选择有很大差异。

几个因素严重影响总计算成本：

- **模型大小**：基础LLM、批改/修订模型（CAI）、偏好标注器/奖励模型（RLAIF）以及策略模型的参数 (parameter)数量直接影响推理和训练成本。更大的模型需要更多浮点运算和内存。
- **数据集规模**：初始提示数量、生成的批改/修订对（CAI）或偏好对（RLAIF）的数量直接影响各自计算阶段的规模。
- **批次大小和训练步数**：标准训练参数显著影响SFT（CAI）、PM训练（RLAIF），尤其是PPO循环（RLAIF）。更大的批次需要更多内存，而更多的训练步数会增加总计算量。
- **RL算法复杂度**：PPO本身具有影响每次迭代成本的超参数 (hyperparameter)（例如每次部署的优化周期数）。
- **硬件**：加速器（GPU/TPU）的类型和数量决定了实际运行时间和可行性。内存容量通常是一个关键限制，特别是在RL微调 (fine-tuning)期间。

有效管理这些成本需要仔细考虑模型选择、数据生成策略以及训练循环的优化方法，我们将在后续章节中探讨。规划计算预算需要根据对齐 (alignment)任务预期的具体模型和数据集大小，对这些组成部分进行实际评估。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法级AI，详细阐述了其包含AI模型批评和修订的两阶段结构，这对理解CAI的计算需求至关重要。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  近端策略优化（PPO）的基础论文，PPO是一种关键的强化学习算法，被明确提及为RLAIF计算成本中的主要瓶颈。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  提供了关于计算资源、模型大小和数据集大小如何影响大型语言模型性能及成本的基本见解，有助于理解CAI和RLAIF中的扩展因素。

---

[上一节](../07-%E9%AB%98%E7%BA%A7%E5%AF%B9%E9%BD%90%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E8%AE%BE%E8%AE%A1%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E5%A5%97%E4%BB%B6.md) · [下一节](02-%E9%AB%98%E6%95%88%E7%9A%84%E5%8F%8D%E9%A6%88%E7%94%9F%E6%88%90.md)
