# 配置 PyTorch 和 CUDA

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup/configuring-pytorch-and-cuda)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了高效地训练神经网络 (neural network)，你的系统需要一个能够并行执行数百万次矩阵乘法的底层支持。PyTorch 正是承担了这一角色。它是一个开源机器学习 (machine learning)框架，提供了名为张量（tensor）的专用数据结构，以及用于在训练期间计算梯度的自动求导引擎。

虽然 PyTorch 可以在标准 CPU 上运行，但在 CPU 上微调 (fine-tuning)语言模型的速度极慢，并不实用。你需要硬件加速。计算统一设备架构（Compute Unified Device Architecture），通常被称为 CUDA，是 NVIDIA 的并行计算平台和编程模型。它允许 PyTorch 直接将繁重的数学运算卸载到 GPU 上执行。

> 硬件加速模型训练的软件栈。

安装 PyTorch 时，需要将其二进制文件与系统中安装的 CUDA 驱动程序相匹配。在运行任何安装命令之前，必须确定本地的 CUDA 版本。在 Linux 或 Windows 上，可以通过在终端运行特定命令来检查 NVIDIA 驱动程序和支持的 CUDA 版本。

```bash
nvidia-smi
```

该命令的输出将显示一个包含 GPU 统计数据的表格。在表格的右上角，你会看到一个标为 "CUDA Version" 的数值。这个数字决定了显卡驱动程序支持的最高 CUDA 工具包版本。

当你在 PyTorch 官方网站配置安装命令时，必须选择一个等于或低于系统显示的计算平台版本。例如，如果你的系统支持 CUDA 12.1，你将使用类似于以下的 pip 命令来安装兼容的 PyTorch 软件包。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装完成后，验证 PyTorch 是否能与 GPU 正常通信十分必要。自动回退到 CPU 是常见的安装问题，这会导致后续训练流程中性能大幅下降并出现内存错误。你可以通过运行一段简单的 Python 脚本来确认配置。

```python
import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

if cuda_available:
    # 打印设备名称和 PyTorch 的 CUDA 版本
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA 版本: {torch.version.cuda}")
```

如果脚本为 CUDA 可用性打印出 `True` 并显示了显卡名称，则说明基础环境已配置正确。你现在拥有了一个可以正常工作的张量后端，能够直接在 GPU 上处理如 $Y = WX + b$ 之类的运算。

值得一提的是，虽然 NVIDIA GPU 是训练语言模型的标准配置，但 PyTorch 也支持其他后端。如果你使用的是 Apple Silicon，可以通过检查 `torch.backends.mps.is_available()` 来使用 Metal Performance Shaders 后端。然而，更广泛的微调工具和量化 (quantization)库生态系统通常以 CUDA 环境为前提。对于中级机器学习任务，使用 NVIDIA 硬件是相对省心的选择。

## 参考资料

- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://papers.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf) — Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala (2019)
  Journal: Advances in Neural Information Processing Systems 32; Publisher: Curran Associates, Inc.; Pages: 8024-8035
  该框架的设计理念和实现的原始研究论文。
- [CUDA: A Programming Model for General Purpose Parallel Computing](https://ieeexplore.ieee.org/document/4487024) — John Nickolls, Ian Buck, Michael Garland, and Kevin Skadron (2008)
  Journal: IEEE Micro; Publisher: IEEE; Volume: 28; Pages: 39-51; DOI: [10.1109/MM.2008.31](https://doi.org/10.1109/MM.2008.31)
  提供了有关 NVIDIA GPU 硬件加速的并行计算架构的概述。
- [PyTorch Documentation: CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) — PyTorch Contributors (2024)
  关于 GPU 使用、内存管理和最佳实践的官方技术文档。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (2020)
  Publisher: Manning Publications
  使用该框架构建和训练神经网络的综合指南。

---

[上一节](../02-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E6%A0%BC%E5%BC%8F%E5%8C%96/05-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E6%B5%81%E6%B0%B4%E7%BA%BF.md) · [下一节](02-Hugging%20Face%20Transformers%20%E5%BA%93%E7%AE%80%E4%BB%8B.md)
