---
course: "introduction-to-machine-learning"
chapter: "what-is-machine-learning"
lesson: "tools-you-might-use"
sourceId: 1460
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-1-what-is-machine-learning/tools-you-might-use"
title: "你会用到的工具"
description: "简要提及机器学习中常用的编程语言和库，如 Python。"
order: 7
plots: []
sourceHash: "fadc5c149cf105f8241836b4d916f90f860f55caacb7da69322aaa39b7623f7f"
sourceCorrections: []
---

理解机器学习 (machine learning)的基本原理和工作流程很重要，但将这些理念付诸实践需要特定的软件工具。你无需一夜之间成为编程专家，但了解所涉及的常用工具将帮助你在机器学习领域中更好地应用和发展。

### Python的主导地位

**Python** 是许多机器学习 (machine learning)任务中，特别是在学习阶段，使用最广泛的编程语言。这有几个原因：

1. **简洁性与可读性：** Python 的语法相对直接，通常类似于日常英语，这使得初学者比学习 C++ 或 Java 等其他语言更容易上手。
2. **丰富的库：** 这是最主要的优势。Python 拥有专门用于科学计算、数据分析和机器学习的丰富库系统。这些库提供预构建函数来处理复杂操作，从而省去了你从零开始编写所有内容的麻烦。
3. **庞大社区：** 庞大且活跃的社区意味着网上有大量的教程、文档、论坛（如 Stack Overflow）和预编写的代码示例。如果你遇到问题，很可能其他人已经解决了。

### 机器学习 (machine learning)的主要Python库

Python 本身提供基本支撑，但专门的库承担了机器学习项目中的繁重工作。你很可能会经常遇到以下这些：

- **NumPy (Numerical Python)：** 这是 Python 中进行数值计算的基本包。它支持大型、多维数组和矩阵，并提供一系列数学函数来高效地操作这些数组。许多其他数据科学库都建立在 NumPy 之上。可以把它看作是处理数值数据的根本。
- **Pandas：** Pandas 建立在 NumPy 之上，提供高性能、易用的数据结构和数据分析工具。它的主要数据结构 DataFrame，在 Python 中就像电子表格或 SQL 表，非常适合加载、操作、清理和分析结构化数据（例如来自 CSV 文件或数据库的数据）。在将数据输入机器学习模型之前，你会广泛使用 Pandas 来准备数据。
- **Scikit-learn：** 这可以说是 Python 中最重要的通用机器学习库，特别是对初学者而言。Scikit-learn 提供简单高效的工具来完成数据分析和机器学习任务。它包含了分类、回归、聚类、降维、模型选择和数据预处理（例如特征缩放和处理缺失值，我们稍后会介绍）等广泛算法的实现。它在不同模型之间提供一致的接口，使实验更加容易。
- **Matplotlib 和 Seaborn：** 理解数据和模型结果通常需要可视化。Matplotlib 是 Python 中一个基本的绘图库，能够创建静态、动画和交互式可视化图表。Seaborn 建立在 Matplotlib 之上，提供更高级的接口来绘制美观且富有信息的统计图形。这些库可以帮助你创建直方图、散点图、热力图及其他图表，以便查看数据模式和评估模型表现。

### 其他工具和环境

虽然 Python 及其核心库构成了许多人的主要工具集，但你也可能会听说以下这些：

- **R：** 另一种常用于统计分析和可视化的编程语言。它拥有强大的统计生态系统，但通常比 Python 更少用于构建端到端的机器学习 (machine learning)系统。
- **Jupyter Notebooks / Google Colab：** 这些是交互式环境，允许你在一个文档中编写和执行代码（如 Python）、显示可视化图表并添加解释性文本。它们在数据查看、实验和结果分享方面非常受欢迎。Google Colab 是一个免费的云端版本，无需设置。
- **SQL (结构化查询语言)：** 通常，机器学习所需的数据存储在数据库中。SQL 是与这些数据库交互的标准语言，用于在你将其加载到 Python 之前检索、筛选和汇总数据。
- **云平台 (AWS, Google Cloud, Azure)：** 对于更大规模的项目，公司通常使用云平台，这些平台提供专门的机器学习服务（如 Amazon SageMaker、Google AI Platform、Azure Machine Learning）。它们提供在大数据集上训练复杂模型所需的基础设施，但这些通常在你掌握基本知识后才会接触。

对于本入门课程，我们将主要侧重于使用 Python 以及 NumPy、Pandas 和 Scikit-learn 等库，通常在 Jupyter Notebooks 或 Google Colab 等环境中进行。不必担心一次性掌握所有这些工具。我们将在后续章节开始进行实际案例时，根据需要逐步介绍它们。这里的目的只是让你了解用于实现机器学习理念的常用工具。

## 参考资料

- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://books.google.com/books/about/Python_for_Data_Analysis.html?id=zDCoDwAAQBAJ) — Wes McKinney (2017)
  Publisher: O'Reilly Media, Inc.; Pages: 550
  使用 Python、NumPy 和 Pandas 进行数据处理和分析的主要参考资料。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供使用 Scikit-learn 和其他框架实现机器学习模型的实践指导。
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Scikit-learn developers (2024)
  Scikit-learn 的官方文档，详细介绍了其功能、算法和使用方法。
- [Project Jupyter Official Documentation](https://docs.jupyter.org/en/latest/) — Project Jupyter (2024)
  交互式计算环境的官方项目网站和文档。
