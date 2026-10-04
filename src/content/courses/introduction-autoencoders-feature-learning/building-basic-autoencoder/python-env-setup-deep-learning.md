---
course: "introduction-autoencoders-feature-learning"
chapter: "building-basic-autoencoder"
lesson: "python-env-setup-deep-learning"
sourceId: 6428
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder/python-env-setup-deep-learning"
title: "深度学习的 Python 环境配置"
description: "为带有自编码器的深度学习项目配置 Python 环境（Anaconda、Jupyter）的指南。"
order: 1
plots: []
sourceHash: "0f27cb100a5490749893a6b534465b018343ffa94253c2374cb7ff7315c2e77b"
sourceCorrections: []
---

配置一个专用的 Python 工作区对于高效编写和运行深度学习 (deep learning)代码是必不可少的。这个过程类似于在新项目开始前准备好一个工作坊。一个组织良好的环境可确保所需的软件库正确安装，并避免与其他开发工作产生冲突。

针对深度学习，特别是使用 PyTorch 和 Keras 等库时，管理软件依赖项很重要。不同的项目可能需要这些库的不同版本，而全局安装所有内容可能导致冲突。为避免这种情况，我们将使用虚拟环境。虚拟环境是一个独立的 Python 配置，允许您为每个项目拥有特定的库集和版本。

## 为何需要专用环境？

设想您正在处理两个不同的 Python 项目。项目 A 需要某个库的 1.0 版本，而项目 B 需要 2.0 版本。如果您在系统范围内安装这些库，一个项目很可能会出问题。虚拟环境通过为每个项目创建一个独立的目录来解决此问题，该目录拥有自己的 Python 解释器和已安装的包。

对于本课程，我们建议使用 **Anaconda**。Anaconda 是一个广受欢迎的 Python 和 R 发行版，用于科学计算和数据科学。它简化了包管理和部署。它附带 `conda`，一个功能强大的包和环境管理器，以及数百个预安装的科学包。

如果您尚未安装 Anaconda，可以从[Anaconda 官方网站](https://www.anaconda.com/products/distribution)下载。选择适合您操作系统的安装程序（Windows、macOS 或 Linux），并按照安装说明进行操作。我们建议选择 Python 3.x 版本。

## 使用 Conda 创建您的项目环境

Anaconda 安装完成后，您可以为本课程创建一个新的虚拟环境。打开您的终端（Windows 上的 Anaconda Prompt，或 macOS/Linux 上的 Terminal），并使用以下命令：

```bash
conda create --name autoencoder_env python=3.9
```

我们来分析一下这个命令：
\*   `conda create`：这是用于创建新环境的 `conda` 命令。
\*   `--name autoencoder_env`：这指定了我们环境的名称。我们称之为 `autoencoder_env`，但如果您愿意，可以选择其他名称。
\*   `python=3.9`：这告诉 `conda` 在此环境中安装 Python 3.9 版本。尽管可能有更新的版本，但 3.9 对于深度学习 (deep learning)库来说是一个稳定的选择。如果您愿意，可以查看更近期的稳定版本。

然后 Conda 会显示将要安装的包，并请求您的确认。输入 `y` 并按 Enter 键。

环境创建后，您需要激活它。激活环境意味着您的终端会话现在将使用该特定环境里安装的 Python 解释器和包。

要激活您的新环境，请使用：

```bash
conda activate autoencoder_env
```

您的终端提示符应随之改变，以表明 `autoencoder_env` 已激活，通常表现为提示符前缀为 `(autoencoder_env)`。

当您在此环境中工作完毕并希望返回系统默认 Python 时，可以停用它：

```bash
conda deactivate
```

目前，请保持 `autoencoder_env` 激活状态，因为接下来我们将安装所需的库。

## 安装必要库

在 `autoencoder_env` 激活状态下，我们现在可以安装构建和训练自编码器所需的 Python 库。本课程要用的主要库有：

1.  **PyTorch**：一个开源机器学习 (machine learning)框架，以其灵活性和易用性而闻名。Keras，我们用于构建神经网络 (neural network)的高级 API，可以使用 PyTorch 作为其后端。
2.  **NumPy**：Python 中用于数值计算的基本包。它对于处理神经网络所需的数据数组非常重要。
3.  **Matplotlib**：Python 中用于创建静态、动画和交互式可视化的绘图库。我们将使用它来显示图像和绘制结果。

您可以使用 `conda install` 或 `pip install` 来安装这些库。通常建议在 conda 环境中使用 `conda install`，因为 conda 能很好地处理依赖关系。然而，如果某个包无法通过 conda 渠道获得，或者您需要一个非常具体的版本，也可以使用 `pip`（Python 的默认包安装程序）。

我们来安装 PyTorch 并配置 Keras 3 使用 PyTorch 后端。

```bash
pip install torch torchvision
pip install "keras-core~=0.1.0"
pip install "keras~=3.0"
```

接下来，安装 NumPy 和 Matplotlib：

```bash
conda install numpy matplotlib
```

Conda（或 pip）将下载并安装这些包及其依赖项。同样，您可能会被要求确认安装。

## 使用 Jupyter Notebooks 或 JupyterLab（推荐）

对于交互式编写和运行 Python 代码，尤其是在数据科学和机器学习 (machine learning)方面，Jupyter Notebooks 或 JupyterLab 是非常出色的工具。它们允许您创建包含实时代码、公式、可视化内容和叙述文本的文档。

Anaconda 通常附带 Jupyter Notebook。如果您偏好 JupyterLab（一个更高级的界面），或者它尚未安装，您可以将其添加到您的环境中：

```bash
conda install jupyterlab
```

或者对于经典的 Jupyter Notebook：

```bash
conda install notebook
```

要启动 JupyterLab，请确保您的 `autoencoder_env` 已激活，然后键入：

```bash
jupyter lab
```

或者对于 Jupyter Notebook：

```bash
jupyter notebook
```

这通常会在您的网络浏览器中打开一个新标签页，显示 Jupyter 界面。在此处，您可以创建新的 notebook 并开始编写代码。当您创建新 notebook 时，请确保它使用与您的 `autoencoder_env` 关联的内核，以便它能访问我们刚刚安装的库。

## 验证您的配置

检查 PyTorch 是否正确安装的一个快速方法是打开 Python 解释器或新的 Jupyter Notebook 并运行以下命令。我们还将设置 Keras 后端。

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
print(torch.__version__)

import keras
print(keras.__version__)

import numpy as np
print(np.__version__)

import matplotlib
print(matplotlib.__version__)
```

如果这些命令运行无误并打印出库的版本号，则您的环境已准备就绪！

您现在已成功为您的深度学习 (deep learning)项目设置了一个专用 Python 环境。这种独立的配置将帮助您保持项目井然有序，并确保您拥有所有必要工具的正确版本。在下一节中，我们将更仔细地查看 PyTorch 和 Keras，这些是我们构建自编码器将使用的工具。

## 参考资料

- [Installing Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) — Anaconda, Inc. (2024)
  Publisher: Anaconda, Inc.
  提供安装Anaconda发行版和理解`conda`环境管理器的详细说明。
- [PyTorch Documentation: Get Started - Installation](https://pytorch.org/get-started/locally/) — PyTorch Project (2024)
  PyTorch的官方安装指南，涵盖各种操作系统和包管理器。
- [Keras Documentation: Get started with Keras 3](https://keras.io/keras_3/) — Keras Team (2024)
  Keras 3的官方安装和后端配置指南，包括PyTorch。
- [JupyterLab Documentation: Installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) — Project Jupyter (2024)
  安装JupyterLab的官方说明，这是一个推荐的交互式开发环境。
- [\`venv\` - Creation of virtual environments (Python documentation)](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  解释Python虚拟环境的基本概念和用法，为隔离实践提供背景。
