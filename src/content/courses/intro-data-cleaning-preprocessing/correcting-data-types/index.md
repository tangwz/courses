---
course: "intro-data-cleaning-preprocessing"
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-4-correcting-data-types"
sourceId: 801
chapter: "correcting-data-types"
title: "修正数据类型"
order: 4
description: "了解常见数据类型（数值型、字符串型、布尔型、日期时间型）以及如何确保列采用正确的格式。"
hasQuiz: true
---

处理了缺失和重复数据后，我们现在关注确保数据以正确的格式存储，即其数据类型。数据分析工具和算法非常依赖数据类型。尝试对存储为文本的数据进行加法等数学运算，或者对被视为简单字符串的日期进行排序，可能导致错误或产生不可靠的结果。例如，如果 '5' 是文本而不是数字，像 $5 + 10$ 这样的操作会有不同的表现。

本章介绍数据集中常见的数据类型，例如数值型（如整数 $100$，浮点数 $98.6$）、字符串（文本）、布尔型（$True$/$False$）和日期时间格式。您将学习如何检查分配给数据列的当前类型，并且，重要的是，如何将它们转换为适当的格式。我们将介绍将数据转换为数值、日期时间以及字符串或类别类型的技术，包括处理转换过程中可能出现问题的方法。正确设置数据类型是准确分析和为后续任务做准备的必要一步。
