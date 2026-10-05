# Rainbow DQN 集成

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/rainbow-dqn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

原始 Deep Q-Network 算法已经获得了几项重要改进，例如 Double DQN、Dueling Networks、Prioritized Experience Replay 和 Distributional RL。一个很自然的问题出现了：我们能否将这些改进结合起来，以获得更出色的表现？Hessel 等人于 2018 年提出的 Rainbow DQN 代理通过将多种互补技术整合到一个架构中，给出了肯定的回答。

Rainbow 的动机源于这样一个观察：这些增强功能通常解决基本 DQN 的不同（有时是正交的）局限性：

- **Double DQN (DDQN):** 减少 Q 值更新中的高估偏差。
- **Prioritized Experience Replay (PER):** 通过将学习集中在令人惊讶或信息量大的转换上，提升样本效率。
- **Dueling Network Architectures:** 在存在许多价值相近的动作时，通过分离状态值和动作优势，带来更好的策略评估。
- **Distributional Reinforcement Learning:** 对回报分布进行建模，而不仅仅是期望值，提供更丰富的学习信号并缓解与估计方差相关的问题。
- **Noisy Nets:** 通过在网络参数 (parameter)中注入学到的、依赖于状态的噪声来实现探索，通常替代简单的 epsilon-greedy 策略。
- **Multi-step Learning:** 使用 $n$ 步回报 ($G_{t:t+n}$) 而非单步 TD 目标 ($G_{t:t+1}$) 来计算 TD 误差，使奖励传播更快。

Rainbow DQN 通常将所有这些组件结合在一起。这种集成并非简单叠加；这些技术能够以有益的方式彼此关联。例如：

- PER 需要 TD 误差来计算优先级。在分布 Rainbow 中，“TD 误差”通常以 Kullback-Leibler (KL) 散度衡量，在当前 Q 分布预测与目标分布之间，有效地优先处理预测回报分布与目标显著不同的转换。
- 对偶网络可以调整以输出参数，用于状态值*分布*和动作优势*分布*，然后将它们结合起来形成最终的 Q 值分布。
- DDQN 在目标更新中解耦动作选择和值估计的机制，在处理 Q 值分布时仍然直接适用。
- Noisy Nets 提供独立于回放机制或值估计方法的探索。

原始的 Rainbow 论文表明，这种组合明显优于 Atari 2600 基准测试套件中的任何单个组件和基线 DQN。进行了消融研究，系统地从完整的 Rainbow 代理中移除组件，以评估每种技术在组合架构中的贡献。这些研究表明，尽管所有组件都做出了积极贡献，但分布强化学习 (reinforcement learning)和 PER 通常提供了最大的性能提升。



[交互图表：基准任务的性能比较](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/rainbow-dqn#plot-zo704q)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "DQN",
        "DQN+DDQN",
        "DQN+PER",
        "DQN+Dueling",
        "DQN+Distrib",
        "Rainbow"
      ],
      "y": [
        100,
        180,
        210,
        190,
        250,
        400
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#adb5bd",
          "#748ffc",
          "#f783ac",
          "#69db7c",
          "#ffc078",
          "#f03e3e"
        ]
      }
    }
  ],
  "layout": {
    "title": "基准任务的性能比较",
    "xaxis": {
      "title": "算法变体"
    },
    "yaxis": {
      "title": "相对中位数性能得分"
    },
    "autosize": true,
    "margin": {
      "l": 60,
      "r": 30,
      "t": 50,
      "b": 80
    }
  }
}
```

</details>



> 相对性能得分，说明了向 DQN 添加组件所带来的渐进式改进，最终形成了 Rainbow。实际得分在不同环境中差异很大。

> 图示了 Rainbow DQN 代理在学习过程中不同组件的运作关系。

尽管 Rainbow DQN 代表了基于值的深度强化学习的一大进步，但它也带来了相当大的复杂性。实现和调试这样一个代理需要细致管理多个彼此关联的部分，并且调整与每个组件相关的超参数 (hyperparameter)（例如，PER 的 alpha 和 beta，分布强化学习的原子数量，N 步长度）可能具有挑战性。Nonetheless, Rainbow 仍然是一个有力的例子，说明了如何将不同研究方向的见解结合起来，可以带来显著的性能提升，并为许多强化学习任务提供了坚实的基准。

## 参考资料

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/11796) — Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver (2018)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 32; Pages: 3215-3222; DOI: [10.1609/aaai.v32i1.11796](https://doi.org/10.1609/aaai.v32i1.11796)
  介绍完整的 Rainbow DQN 智能体，详细说明多种技术的集成及其性能优势。
- [A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html) — Marc G. Bellemare, Will Dabney, Rémi Munos (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 449-458
  介绍分布式强化学习，这是 Rainbow 中用于建模回报分布的关键组件。
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Schaul, Tom, John Quan, Ioannis Antonoglou, and David Silver (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1511.05952](https://doi.org/10.48550/arXiv.1511.05952)
  描述优先经验回放，它提高了样本效率并与分布式方法协同。
- [Dueling Network Architectures for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/wangf16.html) — Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, Nando Freitas (2016)
  Journal: Proceedings of The 33rd International Conference on Machine Learning; Publisher: PMLR; Volume: 48; Pages: 1995-2003
  介绍对偶网络架构以改进策略评估，该架构可适应分布式环境。
- [Deep Reinforcement Learning with Double Q-learning](https://doi.org/10.1609/aaai.v30i1.10295) — Hado van Hasselt, Arthur Guez, David Silver (2016)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 30; DOI: [https://doi.org/10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)
  介绍 Double DQN，一种减轻 Q 值估计中过高估计偏差的方法，已集成到 Rainbow 中。

---

[上一节](08-%E5%88%86%E5%B8%83%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%8E%9F%E7%90%86.md) · [下一节](10-DQN%E5%8F%98%E4%BD%93%E5%AE%9E%E7%8E%B0%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5.md)
