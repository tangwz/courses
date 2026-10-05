# 第 3 章：自相关性与模型识别

来源：[原章节](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id)

[返回课程目录](../README.md)

在讨论了时间序列的组成部分和其平稳性后，接下来主要关注理解数据的内部相关结构。$t$ 时刻的一个值与之前时刻（例如 $t-1$、$t-2$ 等）的值有什么关系呢？回答这个问题对于选择合适的预测模型非常重要。

本章介绍分析这种时间依赖关系的主要工具：

*   **自相关函数 (ACF):** 学习它如何衡量时间序列与其滞后版本之间的相关性。
*   **偏自相关函数 (PACF):** 理解它在排除中间滞后值的影响后，如何衡量时间序列与某个滞后版本之间的相关性。
*   **绘图与解读:** 查看如何使用 Python 库生成 ACF 和 PACF 图，并学习那些提示特定模型类型（如 AR 或 MA 模型）的常见模式。

到本章结束时，你将能够计算并解读 ACF 和 PACF 图，以帮助识别适用于你的平稳时间序列数据的潜在候选模型。

## 小节

- 1. [自动相关函数 (ACF)](01-%E8%87%AA%E5%8A%A8%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0%20%28ACF%29.md)
- 2. [偏自相关函数 (PACF)](02-%E5%81%8F%E8%87%AA%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0%20%28PACF%29.md)
- 3. [在 Python 中绘制 ACF 和 PACF](03-%E5%9C%A8%20Python%20%E4%B8%AD%E7%BB%98%E5%88%B6%20ACF%20%E5%92%8C%20PACF.md)
- 4. [解读ACF/PACF以选择模型](04-%E8%A7%A3%E8%AF%BBACF-PACF%E4%BB%A5%E9%80%89%E6%8B%A9%E6%A8%A1%E5%9E%8B.md)
- 5. [动手实践：ACF/PACF 图的绘制与解读](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AACF-PACF%20%E5%9B%BE%E7%9A%84%E7%BB%98%E5%88%B6%E4%B8%8E%E8%A7%A3%E8%AF%BB.md)

章节测验：[在线测验](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/quiz)
