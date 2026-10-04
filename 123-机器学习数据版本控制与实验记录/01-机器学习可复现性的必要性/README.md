# 第 1 章：机器学习可复现性的必要性

来源：[原章节](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-1-ml-reproducibility-challenges)

[返回课程目录](../README.md)

机器学习项目常涉及代码、数据集、模型配置和环境的频繁变动。重现特定结果，无论是为了调试、协作还是部署，都可能成为一个不小的难题。简单的Git代码版本控制有所帮助，但无法解决追踪大型数据集或记录特定模型训练运行所用精确参数等问题。

本章将让您明白，为什么系统化管理这些要素是必要的。我们会研究重现机器学习实验时常遇到的常见困难。您将了解为什么仅靠标准版本控制工具不足以应对机器学习工作流程的独特要求，尤其是在数据方面。我们将明确可复现性在此处的意思，并指出需要追踪的基本组成部分。最后，我们会介绍数据版本控制和实验追踪的初步设想，这些设想我们将在整个课程中，使用DVC和MLflow等工具进行后续讲解。

## 小节

- 1. [机器学习项目管理中的难题](01-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE%E7%AE%A1%E7%90%86%E4%B8%AD%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 2. [为何单独使用 Git 无法满足需求](02-%E4%B8%BA%E4%BD%95%E5%8D%95%E7%8B%AC%E4%BD%BF%E7%94%A8%20Git%20%E6%97%A0%E6%B3%95%E6%BB%A1%E8%B6%B3%E9%9C%80%E6%B1%82.md)
- 3. [定义机器学习中的可复现性](03-%E5%AE%9A%E4%B9%89%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E5%8F%AF%E5%A4%8D%E7%8E%B0%E6%80%A7.md)
- 4. [可复现机器学习工作流程的组成部分](04-%E5%8F%AF%E5%A4%8D%E7%8E%B0%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%E7%9A%84%E7%BB%84%E6%88%90%E9%83%A8%E5%88%86.md)
- 5. [数据版本管理的基本思想](05-%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E7%AE%A1%E7%90%86%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md)
- 6. [实验追踪的基本理念](06-%E5%AE%9E%E9%AA%8C%E8%BF%BD%E8%B8%AA%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%90%86%E5%BF%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-1-ml-reproducibility-challenges/quiz)
