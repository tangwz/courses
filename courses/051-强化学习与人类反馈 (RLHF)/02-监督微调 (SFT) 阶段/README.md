# 第 2 章：监督微调 (SFT) 阶段

来源：[原章节](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf)

[返回课程目录](../README.md)

基于人类反馈的强化学习 (RLHF) 过程通常从监督微调 (SFT) 开始。这一初始步骤使一个通用预训练大型语言模型 (LLM) 适应，以更好地符合目标任务或应用场景，*在*强化学习阶段*之前*。SFT 使用一个包含高质量提示-响应示例的数据集，为模型提供对所需行为和输出格式的扎实基础认知。

本章侧重于SFT阶段。您将学习：

*   SFT 在为 RLHF 初始化策略时的作用。
*   收集合适演示数据集的方法。
*   重要的实施方面，包括训练配置和超参数。
*   评估 SFT 模型表现的技巧。

我们将通过一个实际练习来结束本章，演示如何在语言模型上进行 SFT。理解 SFT 对于构建一个高效的 RLHF 流程非常重要。

## 小节

- 1. [SFT在RLHF流程中的作用](01-SFT%E5%9C%A8RLHF%E6%B5%81%E7%A8%8B%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 2. [高质量SFT数据集的整理](02-%E9%AB%98%E8%B4%A8%E9%87%8FSFT%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E6%95%B4%E7%90%86.md)
- 3. [SFT 实现细节](03-SFT%20%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82.md)
- 4. [评估 SFT 模型表现](04-%E8%AF%84%E4%BC%B0%20SFT%20%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0.md)
- 5. [动手实践：SFT 执行](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9ASFT%20%E6%89%A7%E8%A1%8C.md)
