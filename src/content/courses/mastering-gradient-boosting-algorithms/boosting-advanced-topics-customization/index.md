---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-7-boosting-advanced-topics-customization"
sourceId: 541
chapter: "boosting-advanced-topics-customization"
title: "高级主题与定制"
order: 7
description: "审视模型可解释性技术（SHAP）、概率校准，以及在提升框架中实现自定义损失函数。"
hasQuiz: false
---

在已对XGBoost、LightGBM和CatBoost等梯度提升算法的工作原理和实现方法有了扎实了解后，我们将转向那些能提升其实际应用效果和可靠性的技术。构建模型往往只是第一步；理解其决策过程、确保其输出经过良好校准，并根据具体问题限制进行调整，对于实际部署来说常常是必需的。

本章将集中讨论这些高级方面。您将学习如何解释复杂提升模型的预测结果，使用SHAP（SHapley Additive exPlanations）等方法，特别是TreeSHAP变体，它专为树集合模型设计。我们将讲解校准分类模型概率输出的技术，使其更具可靠性。此外，您还将获得定制提升框架的能力，通过实现自定义目标函数和评估指标，这使您能够直接针对特定的业务目标进行优化或处理独特的数据特征，例如类别不平衡。最后，我们将通过实践案例巩固这些内容，涵盖自定义目标实现和模型解释。
