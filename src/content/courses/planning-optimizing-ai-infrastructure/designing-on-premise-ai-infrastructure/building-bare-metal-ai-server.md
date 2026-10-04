---
course: "planning-optimizing-ai-infrastructure"
chapter: "designing-on-premise-ai-infrastructure"
lesson: "building-bare-metal-ai-server"
sourceId: 6989
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure/building-bare-metal-ai-server"
title: "构建裸机AI服务器"
description: "用于机器学习任务的裸机服务器组装和配置逐步考量。"
order: 7
plots: []
sourceHash: "5d28c15ef699f30de7dd52838d1a9f33599e4f5736eedfc573c42fd304cf0c86"
sourceCorrections: []
---

将精心挑选的组件转换为可用的服务器，需要一个系统化的组装和配置流程。虽然与组装一台标准台式电脑类似，但构建裸机AI服务器涉及处理高功率GPU、确保最大数据吞吐量 (throughput)以及配置系统以使其在持续高负载下保持稳定的特别考量。这里将介绍一些主要步骤，从物理组装到初始软件设置。

### 兼容性预检

在您拿起螺丝刀之前，进行最后的兼容性检查。此处出错可能导致购买错误的零件，或者发现组件在机箱内发生物理干涉。

- **主板和CPU：** 确认CPU插槽类型（例如LGA 1700、AM5）与主板匹配。
- **主板和GPU：** 这是一个常见错误点。确认主板x16 PCIe插槽的物理间距是否能容纳您的GPU宽度。许多高性能GPU会占用三甚至四个插槽，这意味着一个有四个x16插槽的主板可能只能安装两张这样的卡。
- **机箱和组件：** 检查机箱规格，了解最大GPU长度和CPU散热器高度。如果您使用的是一体式液冷，请确保机箱有适合所选散热器尺寸（例如240mm、360mm）的安装点。
- **电源（PSU）：** 检查总瓦数。计算电源上可用的PCIe 8针或12VHPWR电源连接器数量，并与您的GPU需求进行比较。一台有四个GPU，每个GPU需要两个8针连接器的服务器，将需要电源提供八个专用连接器。

### 组装流程

组装过程应遵循逻辑顺序，以避免为了给其他组件腾出空间而重新卸载已安装的组件。以下流程列出了AI服务器构建中的重要阶段。

> 裸机AI服务器的典型组装和配置顺序。

对于AI服务器来说，这个过程中最需细心处理的部分是GPU安装和随后的线缆管理。安装多个GPU时，将它们逐一牢固地插入PCIe插槽。如果使用NVLink，请在GPU固定好后连接桥接器。特别注意线缆布线。糟糕的线缆管理不仅仅是美观问题；它会严重阻碍气流，这对于防止GPU在长时间训练中出现热节流非常重要。

### 必要的BIOS/UEFI配置

在安装操作系统之前，您必须配置主板BIOS/UEFI中的一些设置。这些对于多GPU系统正常运行必不可少。

- **启用4G以上解码（Above 4G Decoding）：** 此设置允许64位操作系统正确映射多GPU所需的大内存地址空间。没有它，系统可能无法启动或只识别单个GPU。
- **启用可调整大小的BAR（Base Address Register）：** 作为一项相关功能，ReBAR允许CPU一次性访问整个GPU显存（VRAM），而不是分成更小的256MB块。这可以在某些工作负载中提升性能。
- **设置PCIe链路速度：** 找到PCIe插槽的设置，确保它们没有被设置为较低的代数（例如，当插槽和GPU都支持“Gen 4.0”时设置为“Gen 3.0”）。强制使用最高支持的速度可以避免在CPU和GPU之间产生不必要的数据瓶颈。

### 操作系统和驱动安装

一个稳定、精简的操作系统是最佳的根基。Ubuntu Server等Linux发行版的长期支持（LTS）版本是常见且可靠的选择。一旦操作系统安装完成，最重要的一步软件设置是安装专有的NVIDIA驱动。这些驱动包含CUDA工具包，这是机器学习 (machine learning)框架使用GPU的必要条件。

安装驱动后，您可以验证系统是否正确识别了所有硬件。主要工具是NVIDIA系统管理界面。打开终端并运行以下命令：

```shell
nvidia-smi
```

成功的构建将生成类似这样的输出，确认所有GPU都已检测到，温度正常，并已准备好工作。

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  名称        持久模式| 总线ID        显示器A | 易失性非ECC |
| 风扇 温度 性能 功耗:使用/上限|         显存使用 | GPU利用率  计算模式 |
|                               |                      |               MIG模式 |
|===============================+======================+======================|
|   0  NVIDIA RTX 3090      On   | 00000000:01:00.0 Off |                  N/A |
| 30%   35C    P8    23W / 350W |      2MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
|-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX 3090      On   | 00000000:21:00.0 Off |                  N/A |
| 30%   34C    P8    21W / 350W |      2MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-----------------------------------------------------------------------------+
```

在此列表中看到所有已安装的GPU，是对硬件构建成功的最终确认。这台机器现在就像一张白纸，已为软件堆栈（包括Docker和Kubernetes）做好准备，您将用它来运行机器学习工作负载。

## 参考资料

- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  提供在Linux上安装NVIDIA驱动和CUDA工具包的详细说明，这对于启用机器学习工作负载的GPU加速至关重要。
- [Ubuntu Server Guide](https://ubuntu.com/server/docs) — Canonical Ltd. (2023)
  官方Ubuntu服务器安装和配置文档，是裸机AI基础设施的稳定且广泛使用的操作系统基础。
- [Deep Learning Workstation Guides](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHP0XsNyN0qBIBZzI0aE6-qOrcbSKQI1n9P58EC_wJlVi4PCN1W977bDGAk5bAK_hNjBSr21C2PSB6LGQXYTEoP2vv5tmUUKCSQltkwI5jEXldMu7LiNZ3I5YaJfaMtwYYLWKSM4XblqwzQo1BUKdYZSpFTNte48KWV-kxK9ToEWDiUWenSuk-v5VRpWwGE) — Puget Systems (2023)
  Publisher: Puget Systems
  一系列关于组件选择、兼容性检查和高性能深度学习工作站物理组装的实用指南。
- [PCIe Resizable BAR (ReBAR) Support in NVIDIA GPUs](https://developer.nvidia.com/blog/resizable-bar-support-in-nvidia-gpus/) — Scott Gelburd (2020)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  解释了可调整大小的BAR（Resizable BAR）的技术优势和实现，这是优化现代多GPU系统性能的关键BIOS设置。
