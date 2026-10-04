---
course: "data-versioning-experiment-tracking"
chapter: "ml-reproducibility-challenges"
lesson: "reproducible-ml-workflow-components"
sourceId: 4054
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-1-ml-reproducibility-challenges/reproducible-ml-workflow-components"
title: "可复现机器学习工作流程的组成部分"
description: "确定机器学习工作流程中需要管理以实现复现性的必要部分。"
order: 4
plots: []
sourceHash: "7371423111903855b07598d389d86a9bff554e2a852344904cf3c9d63d3aa185"
sourceCorrections: []
---

在机器学习 (machine learning)中实现复现性并非只靠单一工具或技巧；而是系统地管理项目的所有动态要素。当您运行一个实验时，结果取决于代码、数据、配置及其运行环境的特定组合。为了以后能复现该结果，您需要能够重构出这种精确的组合。让我们来分析您需要管理的必要组成部分。

### Source Code

这通常是人们首先想到的，也是理所当然的。您的源代码包括用于数据处理、特征工程、模型训练、评估以及任何实用功能的脚本。

- **重要性：** 代码在开发过程中不断变化。预处理步骤中的微小修改或不同的模型实现都可能导致非常不同的结果。
- **管理：** Git 等版本控制系统是跟踪代码变化的常规做法。将特定代码版本（例如，Git 提交哈希值）与实验运行关联起来是实现复现性的根本。

### Data

机器学习 (machine learning)模型根本上是数据驱动的。用于训练和评估的数据状态与代码同等重要。这包括：

- **原始数据：** 任何处理之前的初始数据集。
- **处理过的数据：** 清理、转换或特征工程之后的数据集。
- **重要性：** 数据集会演变。可能添加新数据、修正错误或更改处理步骤。在*不同*版本的数据上使用*相同*的代码进行训练会产生不同结果。简单地将数据存储在共享驱动器中而不进行版本控制，很快就会导致对哪个数据集用于哪个实验产生混淆。如前所述，大型数据集对标准 Git 工作流程构成挑战。
- **管理：** 这需要专门的数据版本控制工具（如 DVC，我们将在第2章介绍），这些工具能够高效处理大文件并与代码版本控制集成。目标是将数据的特定版本与使用它的代码和实验关联起来。

### Configuration

模型和数据处理步骤通常有许多控制其行为的设置和超参数 (parameter) (hyperparameter)。示例包括学习率、正则化 (regularization)强度、树深度、图像增强参数或数据清洗阈值。

- **重要性：** 机器学习 (machine learning)模型的性能通常对其配置高度敏感。如果不知道特定运行使用的精确参数，就无法复现结果或理解其表现方式。
- **管理：** 配置应明确定义并被跟踪。这可以是将参数存储在与 Git 进行版本控制的配置文件中（如 YAML 或 JSON），也可以是使用实验跟踪工具（如 MLflow，将在第3章讨论）系统地记录它们。

### Environment

代码运行的软件环境的作用虽然不明显，但其影响是根本的。这包括操作系统、Python 版本、特定版本的库（如 scikit-learn、TensorFlow、PyTorch、pandas），以及可能的硬件细节（如 GPU 类型和驱动程序，尽管除非性能有主要影响，否则较少被跟踪）。

- **重要性：** 库更新可能引入破坏性更改或改变算法实现，即使代码、数据和配置相同，也可能导致不同的结果。使用 `scikit-learn 1.0` 训练的模型可能与使用 `scikit-learn 1.1` 训练的模型表现不同。
- **管理：** 使用 `requirements.txt` (pip) 或 `environment.yml` (conda) 等文件明确定义依赖关系是必要措施。这些文件应与代码一同进行版本控制。Docker 等容器化工具可以捕获整个环境，以实现更强的复现性。实验跟踪系统也能记录重要的库版本。

### Results and Artifacts

复现性也意味着能够验证实验的输出。这包括：

- **指标：** 性能的定量度量（例如，准确率、F1 分数、RMSE）。
- **模型：** 训练好的模型文件本身。
- **图表和可视化：** 显示性能、数据分布等的图。
- **重要性：** 您需要将复现运行的结果与原始结果进行比较。跟踪这些输出以及输入（代码、数据、配置）可以形成闭环。对于调试或部署而言，访问特定运行生成的*精确*模型工件是必要的。
- **管理：** 实验跟踪工具就是为此而设计的。它们允许您记录指标、存储模型文件（工件），并将图表直接附加到特定实验运行的记录中。

> 影响机器学习 (machine learning)工作流程复现性的核心要素。已版本化的输入（数据、代码、配置、环境）进入执行过程，产生应被跟踪的输出（模型、指标、图表）。

单独管理这些组成部分是必要的，但真正的复现性通常需要了解整个工作流程、这些部分如何连接，以及一个部分的改变如何影响其他部分。我们即将介绍的 DVC 和 MLflow 等工具提供了管理这些组成部分及其相互依赖关系的机制，形成了可复现机器学习工作流程的主体。

## 参考资料

- [Reproducibility in Machine Learning: A Review of Tools and Practices](https://ieeexplore.ieee.org/document/9906649) — Jean-Pierre C. de Siqueira, Luis Fernando R. C. Carmezim, Rodrygo L. T. Santos, Marcelo G. S. da Rocha, and Bruno F. L. Azevedo (2022)
  Journal: IEEE Access; Publisher: IEEE; Volume: 10; Pages: 109867-109886; DOI: [10.1109/ACCESS.2022.3211516](https://doi.org/10.1109/ACCESS.2022.3211516)
  本文对机器学习中确保可重现性的方法和技术进行了全面概述，涵盖了本节讨论的各个方面。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本备受推崇的书籍，涵盖了ML系统的整个生命周期，包括管理代码、数据、配置和环境的实用策略，以实现稳定和可重现的MLOps。
