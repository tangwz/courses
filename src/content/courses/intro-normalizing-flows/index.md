---
course: "intro-normalizing-flows"
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows"
sourceId: 262
title: "归一化流简介"
description: "使用 PyTorch 构建并训练可逆神经网络，实现精确密度估计与数据生成。"
category: "Machine Learning"
level: 2
duration: 8
order: 130
prerequisites: "微积分、概率论及 Python 基础"
color: "pink"
chapterCount: 5
lessonCount: 25
outcomes: [{"topic": "数学基础", "description": "应用变量替换定理和雅可比行列式计算精确的概率密度。"}, {"topic": "流模型架构", "description": "动手实现耦合层、掩码自回归流和平面流等模型结构。"}, {"topic": "密度估计", "description": "使用精确极大似然估计方法训练流模型。"}, {"topic": "生成式建模", "description": "利用逆向过程从学习到的复杂概率分布中高效采样。"}]
hasProject: false
---

本课程讲解生成模型中归一化 (normalization)流的运行机制。你将学习如何通过可逆变换，将简单的基础分布转化为复杂的概率分布。课程内容涵盖变量替换定理的数学原理、雅可比行列式，以及 RealNVP 和掩码自回归 (autoregressive)流等标准流模型的结构。通过 PyTorch 代码实现，你将构建并训练流模型，用于生成数据及进行精确的密度估计。
