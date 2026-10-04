---
course: "data-governance-quality-observability-production"
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-2-data-quality-dimensions-testing"
sourceId: 1413
chapter: "data-quality-dimensions-testing"
title: "数据质量维度与测试"
order: 2
description: "实施数据质量测试。内容涉及准确性、完整性、一致性及自动化验证技术。"
hasQuiz: false
---

治理层设定规范，工程团队则需贯彻执行。软件工程通过单元测试核实逻辑，数据工程则通过断言确认数据状态。若无自动化测试，质量问题常隐而不显，直至影响下游报告或应用运行。

本章侧重于数据质量检查的技术实现。我们将从抽象的数据标准定义，转为具体的程序化断言。您将学会如何使用标准维度来量化质量：即准确性、完整性、一致性与有效性。

我们将审视数据断言的结构，编写逻辑以验证数据模式，并对数据类型进行强制规范。本课程内容还包含统计分析，这种方法能找出数据分布中，简单基于规则检查可能忽略的异常。例如，验证某列均值 $\mu$ 是否处于预期范围 $[\mu_{min}, \mu_{max}]$ 内，便可实现动态质量控制。

本模块学习完毕，您将明白如何构建用作关卡的验证套件，以防止不合规数据流入生产环境。
