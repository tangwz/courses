---
course: "evaluating-synthetic-data-quality"
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-4-privacy-assessment-techniques"
sourceId: 1035
chapter: "privacy-assessment-techniques"
title: "隐私评估方法"
order: 4
description: "实施方法以量化合成数据中的隐私风险，例如成员推断和属性泄露。"
hasQuiz: false
---

生成合成数据通常旨在保护原始数据集中个体的隐私。然而，仅仅生成数据是不够的；我们必须严格评估实际保留了多少隐私。如果不这样做，可能导致重构敏感信息或识别原始数据中个体等风险。

本章侧重于量化这些隐私风险的实用方法。您将学到：

*   **潜在的隐私安全问题：** 了解合成数据如何无意中泄露信息。
*   **成员推断攻击 (MIAs)：** 判断特定个体数据是否被用于训练生成模型的方法。
*   **属性推断攻击：** 评估推断个体敏感属性可能性的方法，即使其成员身份未被证实。
*   **基于距离的隐私度量：** 使用合成数据与真实记录之间的接近程度测量作为隐私风险的指标。
*   **差分隐私考量：** 评估在数据生成过程中是否使用了差分隐私等特定机制时的隐私状况。

在本章结束时，您将能够实施并理解若干主要方法，用于衡量您的合成数据集的隐私特点。
