---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process"
sourceId: 990
chapter: "forward-diffusion-process"
title: "前向扩散过程"
order: 2
description: "详述扩散模型中前向加噪过程的数学步骤和性质。"
hasQuiz: true
---

本章讲解扩散模型的第一个组成部分：前向过程。这个过程系统地在初始数据点（例如一张图片）上添加噪声，经过许多步直到其接近纯高斯噪声。理解这种受控的退化很重要，因为扩散模型的生成能力来自于学习逆转这个过程。

我们将首先将前向过程定义为马尔可夫链，每一步都依据预定的方差计划（$\beta_t$）添加少量高斯噪声。你将学到描述单个加噪步骤的数学方程，通常写为 $q(x_t | x_{t-1})$，以及一个有用的闭式表达式，它允许我们从初始数据$x_0$在任意时间步$t$直接采样得到噪声状态$x_t$，无需遍历所有中间步骤。我们将分析这个前向机制的性质，并以一个模拟噪声添加的实践练习作为结尾，其中包含代码示例。
