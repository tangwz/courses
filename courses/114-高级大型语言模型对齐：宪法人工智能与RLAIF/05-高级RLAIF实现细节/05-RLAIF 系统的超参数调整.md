# RLAIF 系统的超参数调整

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/rlaif-hyperparameter-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

微调 (fine-tuning)强化学习 (reinforcement learning)自人工智能反馈 (RLAIF) 系统中的众多参数 (parameter)，对于实现稳定训练并最大化大型语言模型 (LLM) 的对齐 (alignment)性能来说非常重要。强化学习自人工智能反馈 (RLAIF) 系统的核心组成部分，包括AI偏好标注器、偏好数据集、偏好模型 (PM) 和近端策略优化 (PPO) 循环，构成了其基本结构。然而，它们的行为对所选超参数 (hyperparameter)高度敏感。不正确的设置可能导致训练不稳定、奖励作弊、策略崩溃，或者仅仅是未达到最佳对齐结果。

本节针对 RLAIF 中偏好建模和强化学习阶段的超参数调整提供指导。

### 偏好模型超参数 (parameter) (hyperparameter)

偏好模型的目标是准确捕捉 AI 标注器指示的偏好，学习一个函数 $P(y_1 \succ y_2 | x)$，该函数会转化为强化学习 (reinforcement learning)阶段的奖励信号。调整其训练过程是第一步。

- **学习率：** 控制梯度下降 (gradient descent)过程中的步长。过高的学习率会导致发散或不稳定，而过低则会导致收敛缓慢。对于微调 (fine-tuning)大型模型，典型值通常在 $1e^{-6}$ 到 $5e^{-5}$ 之间，但需要经验验证。考虑学习率调度（例如线性衰减、余弦退火）。
- **批处理大小：** 每个训练步骤中处理的提示-完成对的数量。较大的批处理大小提供更稳定的梯度估计，但会消耗更多内存，并且有时泛化效果不如经过适当正则化 (regularization)训练的较小批次。常见值可能在 4 到 64 之间，这很大程度上取决于 GPU 内存限制。
- **训练轮数：** 训练过程迭代整个偏好数据集的次数。训练轮数过少会导致欠拟合 (underfitting)（PM 未能很好地学习偏好），而过多则可能导致过拟合 (overfitting)（PM 记住了训练数据但未能泛化到新的比较）。监控验证损失/准确率以确定最佳停止点或训练轮数。
- **优化器：** Adam 或 AdamW 是训练 Transformer 模型的标准选择。优化器本身的参数（如 Adam 的 betas、AdamW 的权重 (weight)衰减）可能也需要调整，尽管默认值通常效果不错。
- **权重衰减：** 一种正则化方法（L2 正则化），它惩罚大权重，有助于防止过拟合。典型值在 0.0 到 0.1 之间。

调整 PM 通常涉及标准的监督学习 (supervised learning)实践：将你的 AI 标注偏好数据分成训练集和验证集，并调整超参数以最大化验证集上的准确率（或最小化交叉熵损失）。

### RLAIF 的 PPO 超参数 (parameter) (hyperparameter)

PPO 算法基于偏好模型获得的奖励信号来微调 (fine-tuning) LLM 策略。其超参数直接影响学习稳定性以及探索与利用的平衡。考虑到来自 AI 偏好模型可能存在的噪声或校准不良的奖励，在 RLAIF 中调整 PPO 尤为重要。

- **学习率（策略与价值）：** 类似于 PM，但通常需要对策略网络和价值函数网络（后者估计预期回报）分别进行调整。策略学习率通常小于 PM 学习率，常在 $1e^{-7}$ 到 $1e^{-5}$ 范围。价值函数可能容忍稍高的学习率。PPO 中的不稳定通常与过于激进的策略学习率有关。
- **PPO 截断参数 ($\epsilon$)：** 这是 PPO 的核心。它限制了策略在每次更新步骤中可以改变的程度，由比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 定义。目标函数包含一个截断项：$\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)$。较小的 $\epsilon$ 值（例如 0.1-0.2）会导致更保守的更新，增强稳定性但可能减慢学习速度。较大的值（例如 0.3-0.4）允许更快改变但有不稳定风险。
- **KL 散度惩罚系数 ($\beta$)：** PPO 通常包含一个惩罚项，该惩罚项基于当前策略 $\pi_\theta$ 与更新前策略 $\pi_{\theta_{old}}$（有时是原始 SFT 模型）之间的 Kullback-Leibler (KL) 散度。该项通常为 $-\beta \cdot \text{KL}(\pi_{\theta_{old}} || \pi_\theta)$。这作为一种软约束，防止策略过快偏离已知的良好策略，当奖励信号不完善时，这很重要。调整 $\beta$ 涉及平衡优化进程与稳定性；值可能在 0.01 到 0.2 之间。自适应 KL 惩罚也很常见。
- **每个批次的 PPO 训练轮数：** 算法迭代收集到的经验数据批次以进行策略更新的次数。值通常在 1 到 10 之间。更多的训练轮数让策略更好地适应当前批次，但如果学习率过高，可能导致在该批次上过拟合 (overfitting)和不稳定。
- **小批次大小：** 在每个 PPO 训练轮次内用于随机梯度更新的数据块大小。必须小于从环境中收集的主批次大小。影响梯度方差和计算效率。典型值在 32 到 512 之间。
- **折扣因子 ($\gamma$)：** 确定在回报计算中 ($G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$) 未来奖励相对于即时奖励的重要性。接近 1 的值（例如 0.95-0.99）鼓励长期规划，而接近 0 的值则优先考虑即时奖励。对于文本生成（剧集可能较长），接近 1 的值很常见，但可能需要根据任务细节进行调整。
- **GAE Lambda ($\lambda$)：** 广义优势估计 (GAE) 的参数，用于平衡优势函数估计中的偏差和方差。$\lambda=0$ 对应于高偏差、低方差的 TD(0) 优势，而 $\lambda=1$ 对应于低偏差、高方差的蒙特卡洛估计。常见值在 0.9 到 0.99 之间（例如 0.95）。
- **熵奖励系数：** 添加到 PPO 目标函数中的一个可选项，它通过惩罚低熵策略（即，过于确定性的策略）来鼓励策略进行探索。有助于防止过早收敛到次优策略。小的正值（例如 0.01）是典型的起始点。
- **价值函数损失系数：** 缩放价值函数损失（通常是预测回报和实际回报之间的均方误差）对总损失的贡献。通常设置为 0.5 到 1.0 左右。确保价值函数与策略一同得到充分训练。

### 调整策略与监控

考虑到训练大型 LLM 的计算成本，针对所有这些超参数 (parameter) (hyperparameter)进行穷举网格搜索通常不切实际。

- **优先：** 首先关注影响力最大的参数：PPO 学习率、$\epsilon$ 和 $\beta$ 通常是主要考虑对象。对于 PM，学习率和训练轮数通常最重要。
- **迭代调整：** 首先使用验证集调整 PM，然后固定它并调整 PPO 循环。请注意，次优的 PM 性能可能使 PPO 调整变得困难或不可能。如果 PPO 未能学习，则重新调整 PM。
- **贝叶斯优化：** 像贝叶斯优化这样的技术，对于 RLAIF 超参数调整这类计算成本高的优化任务，可以比随机搜索或网格搜索更样本高效。像 Optuna 或 Ray Tune 这样的框架支持这些方法。
- **启发式方法：** 从成功的 RLAIF 或 RLHF 研究（例如 Anthropic 的 Constitutional AI 论文、InstructGPT 论文）中报告的值开始，并根据观察到的行为进行调整。
- **监控：** 在训练运行期间仔细监控对有效调整非常重要。跟踪以下指标：
  - PM：验证准确率/损失。
  - PPO：平均奖励、奖励分布、策略更新之间的 KL 散度、策略熵、价值函数损失、梯度范数。
  - 定性：定期在固定的一组评估提示上采样模型输出，以检查对齐 (alignment)改进、重复行为或性能下降。



[交互图表：PPO 平均奖励 vs. 训练步数](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/rlaif-hyperparameter-tuning#plot-louj6y)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "name": "学习率=1e-5",
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        0.1,
        0.3,
        0.5,
        0.65,
        0.75,
        0.8,
        0.83,
        0.85,
        0.86,
        0.87,
        0.87
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#40c057"
      }
    },
    {
      "name": "学习率=5e-5（不稳定）",
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        0.1,
        0.4,
        0.7,
        0.9,
        0.6,
        0.3,
        0.5,
        0.2,
        0.4,
        0.1,
        0.3
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#fa5252"
      }
    },
    {
      "name": "学习率=5e-7（缓慢）",
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.34,
        0.37,
        0.4,
        0.42,
        0.44,
        0.45
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#f59f00"
      }
    }
  ],
  "layout": {
    "title": "PPO 平均奖励 vs. 训练步数",
    "xaxis": {
      "title": "训练步数"
    },
    "yaxis": {
      "title": "平均奖励信号",
      "range": [
        0,
        1.0
      ]
    },
    "legend": {
      "title": {
        "text": "策略学习率"
      }
    },
    "width": 600,
    "height": 400
  }
}
```

</details>



> PPO 奖励曲线显示了对策略学习率的敏感性。适当的速率会带来稳定改进，而过高的速率会导致不稳定，过低的速率则会使进展缓慢。

### 实际考量

- **实验跟踪：** 使用 Weights & Biases 或 MLflow 等工具详细记录每次实验的超参数 (parameter) (hyperparameter)、代码版本、数据集和结果指标。这对于可复现性和系统性调整是不可或缺的。
- **验证集：** 定义一组单独的提示，区别于偏好数据和 PPO 训练提示，专门用于评估模型在调整期间和调整后的定性表现和对齐 (alignment)。
- **计算预算：** 对所需的计算资源要现实。调整 RLAIF 系统，尤其是在大规模情况下，需要大量的 GPU 资源。相应地进行规划，并根据预期效果和可用资源来优先安排调整工作。
- **关联影响：** 请记住超参数之间存在关联。更改 PM 可能需要重新调整 PPO。扩展模型大小时通常需要调整学习率和批处理大小。系统地进行调整，一次更改一个或一小部分相关参数。

掌握 RLAIF 中的超参数调整是一个迭代过程，它结合了理论理解、经验实验和仔细观察。虽然没有普遍完美的设置，但一种由监控重要指标指导的有条不紊的方法将大大增加你训练出稳定且有效对齐 LLM 的机会。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://doi.org/10.48550/arXiv.1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍PPO算法的奠基性论文，详细阐述了裁剪目标、GAE及其在稳定策略优化中的实际应用。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: NeurIPS; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  一篇里程碑式的论文，展示了如何通过人类反馈强化学习（RLHF）来对齐大型语言模型，并提供了PM和PPO调优的实际考量。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法式AI，一种使用AI反馈（RLAIF）对齐LLM的方法，与课程章节主题直接相关。
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://doi.org/10.1145/3292500.3330701) — Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Publisher: Association for Computing Machinery; Pages: 2623-2631; DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)
  描述了Optuna，一个高效的超参数优化框架，支持贝叶斯优化和其他高级调优策略。

---

[上一节](04-RLAIF%20%E4%B8%AD%20PPO%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AE%9E%E6%96%BD.md) · [下一节](06-%E6%89%A9%E5%B1%95%20RLAIF%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
