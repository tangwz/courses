---
course: "getting-started-with-jax"
chapter: "introduction-to-jax"
lesson: "jax-installation-setup"
sourceId: 1682
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/jax-installation-setup"
title: "安装与设置"
description: "安装 JAX 并支持 CPU、GPU 或 TPU 的指南。"
order: 4
plots: []
sourceHash: "6567819714d823f7f860b35c70cf2667c8c0097cdad9c1922125c79a90852bcd"
sourceCorrections: []
---

JAX 是一个为高性能数值计算而设计的框架，主要通过自动微分和函数变换实现。要开始使用 JAX，您需要设置您的环境。安装 JAX 通常很简单，但根据您是想在 CPU、NVIDIA GPU 还是 Google TPU 上运行，需要略微不同的步骤。

本课程假设您已安装了可用的 Python (3.7 或更高版本)，并熟悉使用 `pip` 或 `conda` 进行包管理。强烈建议使用虚拟环境，以避免与其他项目或您系统的 Python 安装发生冲突。

### 创建虚拟环境（推荐）

在安装 JAX 之前，请创建并激活一个虚拟环境。

**使用 `venv`：**

```bash
# 创建一个名为 .venv（或您喜欢的其他名称）的虚拟环境
python -m venv .venv

# 激活环境
# 在 Linux/macOS 上：
source .venv/bin/activate
# 在 Windows (命令提示符) 上：
.\.venv\Scripts\activate.bat
# 在 Windows (PowerShell) 上：
.\.venv\Scripts\Activate.ps1
```

**使用 `conda`：**

```bash
# 创建一个名为 'jax-env' 的 conda Python 环境
conda create -n jax-env python=3.9  # 或您想要的 Python 版本

# 激活环境
conda activate jax-env
```

一旦您的环境被激活，您就可以继续进行 JAX 的安装。

### 为 CPU 安装 JAX

如果您只打算使用 CPU，安装很简单。JAX 本身是一个纯 Python 包，而 `jaxlib` 包含编译后的后端 (XLA) 和平台特定代码。

使用 pip 安装 `jax` 和 `jaxlib` 的最新版本：

```bash
pip install --upgrade jax jaxlib
```

此命令会安装针对您系统 CPU 架构优化的 `jaxlib` 构建。

### 为 NVIDIA GPU 安装 JAX

要使用 NVIDIA GPU，您的系统需要安装兼容的 NVIDIA 驱动、CUDA 工具包以及可选的 cuDNN。JAX 需要特定版本的 CUDA 和 cuDNN，这些版本与预构建的 `jaxlib` wheels 相对应。

安装正确支持 GPU 的 `jaxlib` 的**最可靠方法**是遵循 JAX 官方 GitHub 仓库上的具体说明，因为所需的版本和安装命令经常变化。

**访问：** <https://github.com/google/jax#installation>

查找适合您特定 CUDA 和 cuDNN 版本的命令。命令通常看起来像这样（但**请勿**在未查看官方说明的情况下运行此精确命令）：

```bash
# 仅供示例 - 请检查 JAX 官方 GitHub 以获取正确命令！
pip install --upgrade jax jaxlib[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

`[cuda<version>_cudnn<version>]` 部分指定了目标构建。使用错误的版本很可能导致 JAX 无法检测或使用您的 GPU。请始终参考 JAX 官方安装指南以获取最新命令。

### 为 Google TPU 安装 JAX

在 TPU 上使用 JAX 主要通过 Google Cloud Platform (GCP) 或 Google Colaboratory (Colab) 完成。

- **Google Colab:** 如果您在 Colab 中选择 TPU 运行时 (`运行时` -> `更改运行时类型` -> `TPU`)，JAX 通常预装，或者可以使用标准 CPU 安装命令进行安装。Colab 管理 TPU 驱动程序和底层软件栈。

  ```bash
  pip install --upgrade jax jaxlib
  ```
- **Google Cloud TPU 虚拟机：** 在 GCP 上使用 TPU 虚拟机时，环境通常已预配置。您通常可以在 TPU 虚拟机环境中使用标准 pip 命令安装 JAX。如有需要，请参考 GCP 的具体文档来设置 TPU 虚拟机。

### 验证安装

安装后，您可以验证 JAX 是否正确安装，并检查它能检测到哪些设备（CPU、GPU、TPU）。打开 Python 解释器或运行以下代码的脚本：

```python
import jax
import jax.numpy as jnp

try:
  devices = jax.devices()
  print("JAX 检测到以下设备：")
  for i, device in enumerate(devices):
      print(f"{i}: {device.platform.upper()} ({device.device_kind})")

  # 测试一个简单计算
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (10,))
  y = jnp.dot(x, x)
  print(f"\n成功执行了一个简单的 JAX 操作。结果：{y}")

  # 检查默认设备
  print(f"\n默认设备：{jax.default_backend()}")

except Exception as e:
  print("JAX 验证过程中发生错误：")
  print(e)
  print("\n请检查您的安装步骤，特别是 GPU 驱动/CUDA 版本（如果适用）。")
```

**示例输出（仅 CPU）：**

```text
JAX 检测到以下设备：
0: CPU (cpu)

成功执行了一个简单的 JAX 操作。结果：10.811179161071777

默认设备：cpu
```

**示例输出（单 GPU）：**

```text
JAX 检测到以下设备：
0: GPU (NVIDIA GeForce RTX 3090) # 您的 GPU 型号会有所不同
1: CPU (cpu)

成功执行了一个简单的 JAX 操作。结果：10.811179161071777

默认设备：gpu
```

**示例输出（多 GPU 或 TPU）：**

您会看到列出多个 GPU 或 TPU 设备。

```text
JAX 检测到以下设备：
0: TPU (TPU v3)
1: TPU (TPU v3)
... # (可能还有更多 TPU)
N: CPU (cpu) 

成功执行了一个简单的 JAX 操作。结果：10.811179161071777

默认设备：tpu
```

如果脚本运行无误并列出预期设备，您的 JAX 安装就绪。如果遇到错误，特别是与 CUDA 或 GPU 相关的错误，请再次检查您的驱动程序、CUDA 工具包和 `jaxlib` 版本是否兼容，方法是查阅 JAX 官方安装指南。

JAX 安装完成后，您就可以直接使用 JAX 数组并了解其核心功能了。

## 参考资料

- [JAX Installation](https://github.com/google/jax#installation) — Google (2023)
  Publisher: Google
  提供JAX安装的官方和最新说明，包括CPU、GPU和TPU的平台特定指南。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  官方Python文档，解释如何使用内置的`venv`模块创建和管理虚拟环境。
- [Managing environments](https://docs.conda.io/en/latest/user-guide/tasks/manage-environments.html) — Anaconda Inc. (2023)
  使用Conda创建、激活和管理独立Python环境的官方指南，这是包管理的常见做法。
