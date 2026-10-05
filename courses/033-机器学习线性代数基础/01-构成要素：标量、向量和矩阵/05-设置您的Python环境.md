# 设置您的Python环境

来源：[原文](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/python-numpy-setup)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了实际应用数学对象，例如向量 (vector)和矩阵，我们需要一个计算环境。尽管许多编程语言都能处理数字，Python 已成为机器学习 (machine learning)和数据科学的普遍选择。它简洁的语法，结合强大的专业库，使其成为将数学理论转化为实际应用的绝佳工具。

在本节中，我们将讲解如何设置一个完整的Python数值计算环境。我们将侧重于使用Anaconda发行版，因为它简化了包和环境的管理，让您能专注于学习线性代数，而不是解决安装难题。

### Anaconda发行版

新手开始使用Python进行数据科学的最佳方法是安装Anaconda。Anaconda不仅仅是Python；它是一个完整的发行版，将Python解释器与一个包管理器（称为`conda`）以及一系列最受欢迎的科学计算库捆绑在一起，包括NumPy、SciPy、pandas和Matplotlib。

> **为什么要用Anaconda？**
>
> 您可以直接从python.org安装Python，然后使用`pip`等工具单独安装每个库。然而，管理库之间的依赖关系有时会比较复杂。Anaconda通过在协调统一的环境中为您管理所有包，从而大大简化了这一过程。它确保您安装的库彼此兼容，从而避免了许多常见的设置问题。

以下是在您的系统上安装Anaconda的步骤：

1. **下载安装程序**：访问[Anaconda发行版网站](https://www.anaconda.com/products/distribution)。下载适用于您的操作系统（Windows、macOS或Linux）的安装程序。您应该选择最新的Python 3版本。
2. **运行安装程序**：找到下载的文件并运行它。您将由安装向导引导。对于大多数用户来说，默认设置即可。除非您是了解其中含义的高级用户，否则无需更改安装位置或将Anaconda添加到系统PATH变量中。
3. **完成安装**：按照屏幕上的指示完成安装。这可能需要几分钟。

完成后，您将拥有Python、`conda`包管理器以及所有必需的库，随时可用。

> Anaconda生态系统提供了本课程所需的核心工具。

### 核心工具：NumPy和Jupyter Notebook

安装Anaconda后，您将自动拥有本课程的两个主要工具：

- **NumPy（数值Python）**：这是Python中进行数值计算的基础库。它提供了一个高性能的多维数组对象，我们将使用它来创建向量 (vector)和矩阵。Python中几乎所有数据科学库都构建于NumPy之上。
- **Jupyter Notebook**：这是一个交互式、基于浏览器的应用程序，允许您创建和共享包含实时代码、公式、可视化内容和解释性文本的文档。它是学习和实验的理想环境，因为您可以一次运行一小段代码并立即查看结果。我们将把Jupyter Notebook用于所有动手实践。

### 验证您的设置

让我们确保一切正常运行。最直接的开始方法是启动Jupyter Notebook。

1. **打开Anaconda Navigator**：找到并打开Anaconda Navigator应用程序。这是一个图形界面，显示了随安装程序附带的所有应用程序。
2. **启动Jupyter Notebook**：从Navigator主屏幕中，找到“Jupyter Notebook”应用程序并点击“Launch”按钮。这将在您的网络浏览器中打开一个新标签页，显示Jupyter文件浏览器。
3. **创建新Notebook**：在文件浏览器的右上方，点击“New”并选择“Python 3”（或类似的内核名称）。这将打开一个新的Notebook，它就是您的交互式编码环境。

现在，在您的新notebook的第一个单元格中，输入以下代码，然后按 `Shift + Enter` 运行它：

```python
import numpy as np

print(f"NumPy 版本：{np.__version__}")

# 创建一个简单的向量进行测试
my_vector = np.array([5, 10, 15])

print(f"成功创建了一个NumPy数组：{my_vector}")
```

如果您的安装成功，您应该会看到类似以下内容的输出：

```
NumPy 版本：1.23.5
成功创建了一个NumPy数组：[ 5 10 15]
```

版本号可能不同，但看到版本号和确认消息表明您的环境已正确配置。您已成功安装Python，启动了Jupyter Notebook，并确认NumPy库已可使用。完成此设置后，您现在已准备好从理论转向实践，并开始在代码中创建向量 (vector)和矩阵。

## 参考资料

- [Anaconda Distribution Documentation](https://docs.anaconda.com/) — Anaconda, Inc. (2024)
  Publisher: Anaconda, Inc.
  Anaconda 发行版及 conda 包管理器安装、配置和环境管理的官方资源。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2024)
  Publisher: NumPy Foundation
  NumPy 官方文档，提供了关于其基本数组对象和高性能数值计算函数的详细信息。
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/) — Project Jupyter (2024)
  Publisher: Project Jupyter
  Jupyter Notebooks 的官方指南，用于理解和使用这个结合代码、文本和可视化的交互式网络环境。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本内容丰富的书籍，涵盖了使用 Python 进行数据分析的工具和技术，包括 NumPy 及其与其他库交互的详细说明。

---

[上一节](04-%E7%9F%A9%E9%98%B5%EF%BC%9A%E4%BB%A5%E7%BD%91%E6%A0%BC%E5%BD%A2%E5%BC%8F%E7%BB%84%E7%BB%87%E6%95%B0%E6%8D%AE.md) · [下一节](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%88%9B%E5%BB%BA%E5%90%91%E9%87%8F%E5%92%8C%E7%9F%A9%E9%98%B5.md)
