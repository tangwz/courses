---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-3-reverse-diffusion-process"
sourceId: 993
chapter: "reverse-diffusion-process"
title: "逆向扩散过程"
order: 3
description: "说明扩散模型如何学习反转加噪过程，从而从噪声生成数据。"
hasQuiz: true
---

在上一章中，我们讨论了正向扩散过程，在该过程中，我们经过一系列时间步逐步向数据添加高斯噪声，直到只剩下噪声。现在，我们关注其逆向操作：逆向扩散过程。这是生成部分，我们旨在从纯噪声开始并逐步去噪，以生成一个与原始数据分布相似的样本。

本章介绍这种反转是如何实现的。由于直接计算真实逆向概率 $p(x_{t-1}|x_t)$ 通常难以处理，我们将学习如何对其进行近似。

您将学习到：

*   逆向过程的目标：估计转移概率 $p(x_{t-1}|x_t)$。
*   为什么直接计算很困难，以及如何使用神经网络进行近似。
*   我们如何对逆向过程进行参数化，通常是通过训练一个网络来预测时间步 $t$ 时添加的噪声 $\epsilon$。
*   定义了单一步去噪的数学表述，使我们能够从 $x_t$ 得到 $x_{t-1}$ 的估计。

在本章结束时，您将明白扩散模型如何通过系统地反转加噪过程来学习生成数据。
