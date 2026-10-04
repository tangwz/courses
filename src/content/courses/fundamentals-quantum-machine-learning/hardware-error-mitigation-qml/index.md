---
course: "fundamentals-quantum-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml"
sourceId: 380
chapter: "hardware-error-mitigation-qml"
title: "量子机器学习中的硬件考量与误差缓解"
order: 7
description: "处理在NISQ硬件上运行量子机器学习算法所面临的挑战，包括噪声识别、误差缓解和线路优化。"
hasQuiz: false
---

在奠定高级量子机器学习的理论基础和算法结构之后，我们现在转向在物理量子硬件上执行这些算法的实际问题。当前量子处理器运行在近期中等规模量子 (NISQ) 时代，其特点是量子比特数量有限、连接性受限且极易受到噪声影响。退相干、门操作缺陷和读出误差等因素会在理想计算与实际结果之间造成偏差。这些不完美之处直接影响量子机器学习模型的训练过程和最终性能，可能掩盖量子计算带来的任何优势。

本章着重处理这些硬件实际情况。您将学习如何：

*   识别不同类型的量子硬件噪声。
*   分析噪声对量子机器学习算法执行的影响。
*   应用量子误差缓解技术，例如零噪声外推 (ZNE) 和概率误差消除 (PEC)。
*   优化和转译量子线路以符合硬件限制。
*   设计硬件高效的电路拟设。
*   了解在量子设备上对量子机器学习算法进行基准测试的方法。

熟练掌握这些方面对于开发能在当前和未来量子计算机上有效运行的量子机器学习应用来说是必要的。
