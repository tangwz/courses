# 第 4 章：使用近端策略优化（PPO）进行强化学习微调

来源：[原章节](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning)

[返回课程目录](../README.md)

在确定了模拟人类偏好的方法之后，下一步是使用这个信号直接改进语言模型的表现。本章主要关注强化学习（RL）的微调阶段，特别是采用近端策略优化（PPO）。PPO 是一种策略梯度方法，常用于 RLHF 中，以优化语言模型策略，使其符合学到的奖励模型，同时确保不会与原始的监督微调模型偏离过远。

您将学习如何在大型语言模型的环境中应用 PPO 算法。我们将分析策略网络和价值网络的设置，KL散度惩罚（$D_{KL}$）对于训练稳定的重要作用，计算广义优势估计（GAE）等优势值的方法，以及超参数调整的实际考量。我们还将查看使用 Hugging Face 的 TRL 等库进行实现的例子，并讨论训练不稳定等常见挑战。目标是让您掌握实现和管理 RLHF 流程中基于 PPO 的优化阶段的知识。

## 小节

- 1. [RLHF背景下的PPO算法](01-RLHF%E8%83%8C%E6%99%AF%E4%B8%8B%E7%9A%84PPO%E7%AE%97%E6%B3%95.md)
- 2. [策略网络与价值网络的实现](02-%E7%AD%96%E7%95%A5%E7%BD%91%E7%BB%9C%E4%B8%8E%E4%BB%B7%E5%80%BC%E7%BD%91%E7%BB%9C%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
- 3. [KL散度惩罚的作用](03-KL%E6%95%A3%E5%BA%A6%E6%83%A9%E7%BD%9A%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 4. [优势和回报的计算](04-%E4%BC%98%E5%8A%BF%E5%92%8C%E5%9B%9E%E6%8A%A5%E7%9A%84%E8%AE%A1%E7%AE%97.md)
- 5. [LLM的PPO超参数调整](05-LLM%E7%9A%84PPO%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4.md)
- 6. [常用 PPO 实现库 (TRL)](06-%E5%B8%B8%E7%94%A8%20PPO%20%E5%AE%9E%E7%8E%B0%E5%BA%93%20%28TRL%29.md)
- 7. [PPO训练不稳定性故障排除](07-PPO%E8%AE%AD%E7%BB%83%E4%B8%8D%E7%A8%B3%E5%AE%9A%E6%80%A7%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4.md)
- 8. [实践：实现PPO更新步骤](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0PPO%E6%9B%B4%E6%96%B0%E6%AD%A5%E9%AA%A4.md)
