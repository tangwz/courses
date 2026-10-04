---
course: "generative-adversarial-networks-gans"
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-1-gan-foundations-revisited"
sourceId: 622
chapter: "gan-foundations-revisited"
title: "重温GAN基本原理"
order: 1
description: "回顾GAN核心思想、最小最大博弈和常见失败模式，为学习高级技术做好准备。"
hasQuiz: false
---

为了有效构建生成对抗网络，我们首先巩固对核心原理的理解。本章将重点回顾应对更复杂GAN所必需的基本思想。

我们将重新审视：
*   **生成器**和**判别器**网络的基本作用及其配合方式。
*   原始的**最小最大目标函数**，通常表示为 $$ \min_G \max_D V(D, G) $$，及其理论依据。
*   常见的**训练不稳定问题**，例如模式坍塌和梯度消失，这些问题会阻碍收敛。
*   初始GAN形式的**固有限制**，这促使了对高级技术的需求。
*   对**深度卷积生成对抗网络 (DCGAN)**所引入的架构改进进行简要回顾。

本次回顾确立了共同的起点，以确保在学习高级架构和稳定方法之前，大家对基本机制和挑战有清晰的理解。
