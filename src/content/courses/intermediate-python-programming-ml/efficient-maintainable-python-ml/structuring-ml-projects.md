---
course: "intermediate-python-programming-ml"
chapter: "efficient-maintainable-python-ml"
lesson: "structuring-ml-projects"
sourceId: 2214
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/structuring-ml-projects"
title: "机器学习项目结构化"
description: "了解在机器学习项目目录中组织代码、数据和模型的通用方法。"
order: 2
plots: []
sourceHash: "16511ddfab376f7a444a6e0b820c953f718a71a3a316e7bac82b569e0257f448"
sourceCorrections: []
---

随着机器学习 (machine learning)项目超出简单的脚本和探索性笔记本的范畴，您组织文件和代码的方式变得愈发重要。清晰的项目结构不仅关乎整洁；它对构建易于理解、可复现、可维护和可扩展的系统起到基础作用。尽管没有一个“完美”的结构强制适用于所有项目，但既定的惯例和模式能提供很多益处，特别是在与他人协作或一段时间后重新审视自己的工作时。

设想一下，如果没有清晰的结构，您可能会面临哪些问题：训练最终模型的脚本在哪里？那个实验使用了哪个版本的数据？同事如何能快速理解如何运行您的分析？一个合理的结构有助于高效地回答这些问题。

### 标准化结构的好处

采用一致的目录布局带来多种优势：

1. **可复现性：** 清楚代码、数据、配置和结果的位置，大大简化了重新运行分析或模型训练过程，确保您或他人能得到相同的结果。
2. **可维护性：** 当代码逻辑上分离（例如，数据处理区别于模型定义）时，查找特定组件进行调试或更新变得简单。
3. **协作：** 对项目布局的共同理解让团队成员能够浏览代码库，有效贡献，并无需大量解释即可理解工作流的不同部分。
4. **可扩展性：** 随着项目复杂性增加，在一个有组织的框架内添加新功能、模型或数据源，比理清一堆无序的文件要简单。

### 常见目录布局

许多机器学习 (machine learning)项目的一个典型、有效的结构通常如下所示。这可作为一个稳固的起点，您可以根据项目需求进行调整。

> 机器学习项目的一种常见目录结构，促进关注点分离。

让我们了解各自的主要组成部分的作用：

- **`data/`**：存放所有项目数据。最佳做法是进一步细分，例如：

  - `data/raw/`：原始的、不可更改的数据。切勿直接修改此处的任何文件。
  - `data/interim/`：处理过程中生成的中间数据文件。
  - `data/processed/`：最终的、清理过的数据集，可用于建模。
  - **注意：** 避免将大型数据文件直接提交到 Git。使用Git LFS（大型文件存储）或数据版本管理工具（例如DVC），或将数据存储在由配置引用的专用存储（如云存储桶）中。
- **`docs/`**：项目文档。这可包含方法的详细解释、数据字典，或从代码注释生成的文档（使用Sphinx等工具）。
- **`notebooks/`**：Jupyter笔记本，主要用于探索、实验和可视化。最好以数字前缀命名文件，以指示工作流顺序（例如，`01-initial-data-exploration.ipynb`、`02-feature-engineering-ideas.ipynb`）。避免将可复用、核心的逻辑代码完全放在笔记本中；将有用的代码重构为`src/`目录下的Python模块。
- **`src/`**（或项目特定的名称，如`my_ml_project/`）：包含主要的Python源代码，组织成模块，并可能包含子包。这促进代码复用和可测试性。

  - 示例模块：`data_processing.py`、`feature_engineering.py`、`model_training.py`、`utils.py`。
  - 包含一个`__init__.py`文件使该目录成为一个Python包。
  - 如果您打算将项目代码安装为一个包，请考虑添加`setup.py`文件，这能简化导入和部署。
- **`scripts/`**：存放独立的Python脚本，用于运行项目的特定阶段，例如下载数据、预处理数据、训练模型或运行评估。这些脚本通常从`src/`目录导入函数和类。示例：`train_final_model.py`。
- **`tests/`**：包含`src/`中代码的单元测试和集成测试。维护测试套件有助于确保代码正确性并防止更改时出现回归问题。`tests/`内部的结构通常与`src/`的结构相对应。
- **`models/`**：专门用于保存训练好的机器学习模型、缩放器、编码器或训练管道产生的其他序列化对象。对于大型模型文件，再次考虑Git LFS或其他替代方案。
- **`reports/`**：生成的分析输出，例如图表、汇总统计或项目报告。`figures/`子目录常用于存储可视化文件。
- **`config/`**：配置文件（例如，使用YAML或JSON格式）。在此处存储文件路径、超参数 (parameter) (hyperparameter)、特征列表等参数，与代码本身分离。这样可以轻松更改设置，而无需修改脚本。
- **`README.md`**：项目的首页。它应提供清晰的概述，说明项目目标，列出依赖项，并提供关于如何设置环境和运行主要工作流的说明。
- **`requirements.txt` 或 `environment.yml`**：列出运行项目所需的所有Python包依赖及其版本。这对可复现性很重要，并由环境管理工具使用（在下一节关于虚拟环境的部分讨论）。

### 结构化原则

这些常见布局的背后是一些重要的原则：

- **关注点分离：** 保持项目不同方面之间的区分。代码应与数据分离，配置与脚本分离，探索性笔记本与可复用模块分离。
- **模块化：** 将复杂过程分解为模块内更小、可管理和可复用的函数或类。
- **配置管理：** 将参数 (parameter)和设置外部化到配置文件中，而不是在脚本中硬编码。
- **自动化：** 设计结构以支持工作流的自动化执行（例如，运行一个脚本来预处理数据并使用`src/`中的模块训练模型）。

### 运用项目模板

启动一个新项目通常需要重复创建这些目录和文件。[Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/)等工具结合预定义模板，例如流行的[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)，可以自动化此设置过程。这些模板提供了一个稳固的、经过社区验证的起始结构，节省时间并从一开始就强制执行良好实践。尽管您不必使用模板，但理解它们提倡的结构很有价值。

最终，最好的结构是适合您的项目和团队的结构。一致性通常比完美遵循特定模板更重要。从上述概述的合理结构开始，并随着项目的演变进行调整。早期投入少量时间进行组织，从长远来看会带来丰厚回报，使您的机器学习 (machine learning)项目更清晰、易懂和协作性强。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本全面书籍，为机器学习项目提供关于结构、MLOps、可复现性和部署策略的指导。
- [Cookiecutter Data Science: A Logical, Reproducible, and Collaborative Project Structure](https://cookiecutter-data-science.drivendata.org/) — DrivenData (2023)
  一个开源项目模板的官方文档，有助于为数据科学和机器学习项目建立一致且可靠的目录结构。
- [Data Version Control (DVC) Documentation](https://dvc.org/doc) — Iterative.ai (2023)
  DVC 的官方文档，DVC 是一个用于数据和模型版本控制的工具，对于确保机器学习项目的可复现性和促进协作至关重要。
