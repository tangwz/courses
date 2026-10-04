# 配置Keras环境

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-1-intro-deep-learning-concepts/setting-up-keras-environment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

准备必要的软件环境。拥有一个一致且配置正确的环境，对于顺利运行示例和练习非常重要。我们将侧重于安装Keras 3及其首选后端PyTorch。

### 独立环境的重要性

在安装任何软件包之前，强烈建议为本项目创建一个独立的Python环境。这可以避免不同项目所需的包依赖项之间的冲突。Python内置的`venv`模块或`conda`包管理器都是实现此目的的很好选择。

**使用`venv`：**

1. 在终端中导航到您的项目目录。
2. 创建一个虚拟环境（例如，命名为`.venv`）：

   ```bash
   python -m venv .venv
   ```
3. 激活环境：
   - 在macOS/Linux上：`source .venv/bin/activate`
   - 在Windows上（命令提示符）：`.venv\Scripts\activate.bat`
   - 在Windows上（PowerShell）：`.venv\Scripts\Activate.ps1`

您的终端提示符现在应显示您已位于虚拟环境内。

**使用`conda`：**

1. 创建一个新环境（例如，命名为`keras-env`），并指定Python版本：

   ```bash
   conda create --name keras-env python=3.9 # 或您偏好的Python 3版本
   ```
2. 激活环境：

   ```bash
   conda activate keras-env
   ```

所有后续的软件包安装都将限制在此激活的环境之内。

### 安装Keras和PyTorch

在激活虚拟环境后，您现在可以使用Python包安装器`pip`安装所需的库。Keras 3设计用于支持多个后端（PyTorch、TensorFlow、JAX）。本课程中，我们将主要使用PyTorch后端。

使用以下命令安装Keras和PyTorch：

```bash
pip install keras torch torchvision torchaudio
```

此命令将安装：

- `keras`：Keras核心库（版本3或更高）。
- `torch`：PyTorch库，它将作为计算后端。
- `torchvision`、`torchaudio`：PyTorch的配套库，常用于数据加载和转换，尤其在视觉和音频任务中。

Keras 3通常配置为自动检测并使用像PyTorch这样的可用后端。如果您安装了多个后端（例如TensorFlow），并希望确保使用PyTorch，您可以在运行Python脚本或Jupyter笔记本之前设置`KERAS_BACKEND`环境变量：

- 在macOS/Linux上：

  ```bash
  export KERAS_BACKEND="torch"
  python your_script.py
  ```
- 在Windows上（命令提示符）：

  ```bash
  set KERAS_BACKEND=torch
  python your_script.py
  ```
- 在Windows上（PowerShell）：

  ```bash
  $env:KERAS_BACKEND="torch"
  python your_script.py
  ```

您也可以在导入Keras**之前**在Python代码中进行配置，尽管通常更推荐设置环境变量：

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
# 现在Keras将使用PyTorch后端
```

### 关于GPU加速的说明（可选）

深度学习 (deep learning)通常涉及计算密集型操作，使用图形处理单元（GPU），特别是NVIDIA GPU，可以显著加速。如果您有兼容的NVIDIA GPU并希望充分利用它：

1. **安装NVIDIA驱动：** 确保您已从NVIDIA网站安装了特定GPU的最新驱动。
2. **安装CUDA Toolkit和cuDNN：** PyTorch需要特定版本的CUDA Toolkit和cuDNN库。通常最简单的方法是安装一个捆绑了这些的PyTorch版本。访问[PyTorch网站](https://pytorch.org/get-started/locally/)，使用其配置工具获取适合您系统和所需CUDA版本的正确`pip`或`conda`命令。该命令可能与上述基本命令略有不同，会指定CUDA版本。
3. **验证安装：** 安装后，如果设置正确，PyTorch应自动检测并使用GPU。您可以在Python中验证这一点（我们很快会讲到检查方法）。

虽然GPU加速可以显著加速大型模型和数据集的训练，但对于学习本课程中的基本原理而言，并非严格要求。大多数示例在现代CPU上也能运行良好。

安装好Keras及其PyTorch后端后，您的环境现已准备就绪。在下一节中，我们将进行一个快速验证步骤，以确认Keras已正确安装并能与后端通信。

## 参考资料

- [Keras documentation](https://keras.io/getting_started/) — Keras team (2024)
  Keras 3安装、后端配置和使用的官方指南，与开发环境的搭建直接相关。
- [PyTorch documentation](https://pytorch.org/get-started/locally/) — PyTorch team (2024)
  安装PyTorch的必要说明，包括利用CUDA和cuDNN进行GPU加速的特定指导，与本节内容相符。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python (2024)
  关于使用`venv`创建和管理隔离Python环境的官方Python文档，本节对此方法有所讨论。
- [Managing environments](https://docs.conda.io/en/latest/conda-essentials/managing-environments.html) — Anaconda, Inc. (2017)
  Publisher: Anaconda, Inc.
  使用`conda`创建和管理虚拟环境的官方说明，这是本节中提出的另一种环境隔离方法。

---

[上一节](04-Keras%20%E7%AE%80%E4%BB%8B.md) · [下一节](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%8E%AF%E5%A2%83%E9%AA%8C%E8%AF%81.md)
