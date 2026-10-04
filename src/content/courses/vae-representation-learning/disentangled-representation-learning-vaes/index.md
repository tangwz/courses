---
course: "vae-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes"
sourceId: 1141
chapter: "disentangled-representation-learning-vaes"
title: "解耦表示学习与变分自编码器 (VAEs)"
order: 5
description: "学习使用变分自编码器实现和评估解耦表示的技术与指标。"
hasQuiz: false
---

在前几章中，我们研究了变分自编码器 (VAEs) 如何学习数据的潜在表示。表示学习的一个重要目标是找出数据中潜在的变动因素，这些因素不仅紧凑而且有意义。本章讨论 *解耦表示学习*，这是一个关注训练模型（尤其是 VAEs）的方面，使其学习到的表示中，每个潜在维度都对应于数据中存在的独立、可解释的生成因素。例如，对于人脸图像，解耦表示可以将头发颜色、姿态或情感等因素分离到独立的潜在变量中。

我们将首先考虑定义解耦的不同方法以及实现解耦的相应难点。您将了解到用于量化解耦程度的常用评估指标，例如互信息间隙 (MIG)、独立属性可预测性 (SAP) 以及解耦性、完备性和信息性 (DCI)。本章接下来将研究 VAE 目标函数中的 $KL$ 散度项如何影响解耦，从而产生了像 $\beta$-VAEs 这样的模型。我们还将讲解旨在提高解耦效果的方法，包括 FactorVAEs 和总相关性 VAEs (TCVAEs)，这些方法更直接地处理潜在编码中的统计独立性。将讨论与信息瓶颈理论和群论观点的理论关联，以及该方面固有的局限性和可识别性问题。最后，您将通过训练用于解耦的 VAEs 并使用已确立的评估指标来评估它们的性能，从而获得实践经验。
