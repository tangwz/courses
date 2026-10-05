# 第 8 章：监控与调试模型

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models)

[返回课程目录](../README.md)

构建和训练模型是重要的进展，但通常，初次尝试并不能达到预期效果或运行不顺畅。模型可能收敛缓慢、生成无意义的输出，或遇到运行时错误。本章讨论监控训练过程和调试 PyTorch 应用程序的实际需求。

我们将介绍诊断和解决常见问题的系统方法，包含张量形状不匹配以及与 CPU/GPU 设备分配相关的错误。你将学习如何检查梯度，以发现训练稳定性问题，比如梯度消失或梯度爆炸。此外，本章介绍监控训练动态的方法，特别是使用 TensorBoard 可视化例如损失和准确率随时间的变化。我们还将讨论集成基本日志记录，以及使用 Python 调试器 (`pdb`) 进行逐步代码检查。到本章结束时，你将掌握一个工具集，能够有效地排查问题并观察你的 PyTorch 模型。

## 小节

- 1. [PyTorch开发中常见错误](01-PyTorch%E5%BC%80%E5%8F%91%E4%B8%AD%E5%B8%B8%E8%A7%81%E9%94%99%E8%AF%AF.md)
- 2. [调试张量形状不匹配](02-%E8%B0%83%E8%AF%95%E5%BC%A0%E9%87%8F%E5%BD%A2%E7%8A%B6%E4%B8%8D%E5%8C%B9%E9%85%8D.md)
- 3. [检查设备放置 (CPU/GPU)](03-%E6%A3%80%E6%9F%A5%E8%AE%BE%E5%A4%87%E6%94%BE%E7%BD%AE%20%28CPU-GPU%29.md)
- 4. [检查梯度问题（消失/爆炸）](04-%E6%A3%80%E6%9F%A5%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%EF%BC%88%E6%B6%88%E5%A4%B1-%E7%88%86%E7%82%B8%EF%BC%89.md)
- 5. [使用 TensorBoard 可视化训练进度](05-%E4%BD%BF%E7%94%A8%20TensorBoard%20%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%9B%E5%BA%A6.md)
- 6. [训练/评估期间的指标记录](06-%E8%AE%AD%E7%BB%83-%E8%AF%84%E4%BC%B0%E6%9C%9F%E9%97%B4%E7%9A%84%E6%8C%87%E6%A0%87%E8%AE%B0%E5%BD%95.md)
- 7. [在 PyTorch 中使用 Python 调试器 (pdb)](07-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Python%20%E8%B0%83%E8%AF%95%E5%99%A8%20%28pdb%29.md)
- 8. [实践：调试与可视化](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%B0%83%E8%AF%95%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/quiz)
