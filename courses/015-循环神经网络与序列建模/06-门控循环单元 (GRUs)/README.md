# 第 6 章：门控循环单元 (GRUs)

来源：[原章节](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru)

[返回课程目录](../README.md)

虽然长短期记忆 (LSTM) 网络提供了一种使用多个门控来获取长期依赖关系的有效方法，但门控循环单元 (GRUs) 提供了一种相关且通常更简单的替代方案。

本章介绍 GRU 架构。我们将查看它的组成部分，特别是**更新门** ($z_t$) 和**重置门** ($r_t$)，并了解它们如何共同作用以控制信息流动并更新隐藏状态。我们将了解候选隐藏状态是如何计算并与先前的隐藏状态结合的。

我们还将直接比较 GRU 和 LSTM，讨论它们的结构差异、相对计算效率，并为特定序列建模任务选择合适的门控单元提供实用指导。

## 小节

- 1. [介绍GRU：一种更简洁的门控架构](01-%E4%BB%8B%E7%BB%8DGRU%EF%BC%9A%E4%B8%80%E7%A7%8D%E6%9B%B4%E7%AE%80%E6%B4%81%E7%9A%84%E9%97%A8%E6%8E%A7%E6%9E%B6%E6%9E%84.md)
- 2. [GRU 单元结构](02-GRU%20%E5%8D%95%E5%85%83%E7%BB%93%E6%9E%84.md)
- 3. [更新门](03-%E6%9B%B4%E6%96%B0%E9%97%A8.md)
- 4. [重置门](04-%E9%87%8D%E7%BD%AE%E9%97%A8.md)
- 5. [计算候选隐藏状态](05-%E8%AE%A1%E7%AE%97%E5%80%99%E9%80%89%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81.md)
- 6. [计算最终隐藏状态](06-%E8%AE%A1%E7%AE%97%E6%9C%80%E7%BB%88%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81.md)
- 7. [GRU与LSTM的比较](07-GRU%E4%B8%8ELSTM%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 8. [计算效率考量](08-%E8%AE%A1%E7%AE%97%E6%95%88%E7%8E%87%E8%80%83%E9%87%8F.md)
- 9. [何时选择 GRU 或 LSTM](09-%E4%BD%95%E6%97%B6%E9%80%89%E6%8B%A9%20GRU%20%E6%88%96%20LSTM.md)

章节测验：[在线测验](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/quiz)
