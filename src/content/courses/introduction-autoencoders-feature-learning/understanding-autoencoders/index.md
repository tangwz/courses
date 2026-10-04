---
course: "introduction-autoencoders-feature-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-1-understanding-autoencoders"
sourceId: 1146
chapter: "understanding-autoencoders"
title: "理解自动编码器"
order: 1
description: "学习自动编码器的基本原理、其组成部分（编码器、解码器、瓶颈）以及它们在无监督学习中的作用。"
hasQuiz: true
---

自动编码器是一类主要用于无监督学习的人工神经网络。它们的主要作用是学习输入数据的压缩表示，通常称为“编码”，然后尽可能准确地从这些编码重建原始输入。此过程使其能找到数据中潜在的结构。

本章为理解自动编码器奠定基本认识。我们首先会将自动编码器置于无监督学习的背景下，并简要提及神经网络的基础知识，后者是它们的构成部分。您将学习自动编码器背后的核心思想：学习重建数据。接着我们会将其架构分解为编码器、瓶颈（或称潜在空间）和解码器。我们还将讨论学习有意义的数据表示为何重要，用一个简单的类比说明其核心机制，并指明自动编码器最初解决的问题类型。本章结束时，您将清楚地掌握什么是自动编码器以及它们为何是机器学习中一个有用的工具。
