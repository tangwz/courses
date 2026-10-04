---
course: "getting-started-local-llms"
chapter: "preparing-local-environment"
lesson: "checking-system-specs"
sourceId: 4219
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-2-preparing-local-environment/checking-system-specs"
title: "检查您的系统规格"
description: "查找您电脑在不同操作系统上的 CPU、RAM 和 GPU 信息的简单步骤。"
order: 4
plots: []
sourceHash: "11cb0d3cb44109759d84e68c306f78e0d18c67d3a54108babc8cef06bb15d4ec"
sourceCorrections: []
---

电脑硬件组件，包括中央处理器 (CPU)、随机存取存储器 (RAM) 和图形处理器 (GPU) 及其专属视频内存 (VRAM)，非常重要。这些组件显著影响您在本地运行大型语言模型的效果和速度。确定您系统的具体规格，是判断您的电脑能运行哪些类型模型的第一步。

幸运的是，您无需专业工具即可找到这些信息。您的操作系统内置了实用程序，可以提供这些详细信息。以下是在 Windows、macOS 和 Linux 上检查系统规格的方法。

### 检查 Windows 上的规格

Windows 提供了多种查看系统硬件的方法。以下是两种常用方法：

**1. 使用系统信息：**

此工具提供硬件和软件的全面概览。

- 按 `Windows + R` 键打开“运行”对话框。
- 输入 `msinfo32`，然后按 Enter 键。
- “系统摘要”页面将打开。请查看：
  - **处理器：** 这显示您的 CPU 型号和速度。
  - **已安装的物理内存 (RAM)：** 这显示您的系统上已安装的 RAM 总量（例如，16.0 GB）。
- 要查找您的 GPU 详细信息，请在左侧窗格中导航到 `组件` > `显示`。请查看：
  - **名称：** 这显示您的显卡型号（例如，NVIDIA GeForce RTX 4070、AMD Radeon RX 7800 XT、Intel Arc A770）。
  - **适配器 RAM：** 这通常指示您的专用 GPU 上可用的 VRAM 量。请注意，集成显卡（通常是 CPU 的一部分）可能会显示与系统 RAM 共享的内存。

**2. 使用任务管理器：**

任务管理器提供您硬件使用情况和规格的快速、实时视图。

- 右键单击屏幕底部的任务栏并选择“任务管理器”，或按 `Ctrl + Shift + Esc`。
- 如果您看到简化视图，请单击“更多详细信息”。
- 转到“性能”选项卡。
- 单击：
  - **CPU：** 在右上角显示您的处理器型号名称。
  - **内存：** 在右上角显示您的总安装 RAM（例如，32.0 GB）。
  - **GPU 0、GPU 1 等：** 如果您有一个或多个 GPU，它们将在此处列出。单击每个 GPU。
    - 型号名称显示在右上角。
    - 在 GPU 视图中向下滚动以找到“**专用 GPU 内存**”。这就是您的 VRAM 量。如果显示 0.0 GB 或非常小的量，您可能正在查看与系统 RAM 共享的集成显卡。如果您有专用显卡，请查找另一个 GPU 条目。

### 检查 macOS 上的规格

在 macOS 上查找硬件详细信息使用“关于本机”实用程序非常简单。

- 单击屏幕左上角的“**苹果菜单**”()。
- 选择“**关于本机**”。
- 将出现一个概览窗口，显示：
  - **处理器**或**芯片：** 显示您的 CPU 类型（例如，Intel Core i7、Apple M2）。
  - **内存：** 显示您的总安装 RAM（例如，16 GB）。
- 要获取更详细的显卡信息，包括 VRAM：
  - 在“关于本机”窗口中，单击“**系统报告...**”（在较旧的 macOS 版本上）或转到“**系统设置**”>“**通用**”>“**关于**”，然后向下滚动并单击“**系统报告...**”（在 Ventura 和 Sonoma 等较新 macOS 版本上）。
  - 在“系统报告”窗口中，导航到左侧边栏中的 `硬件` > `图形/显示器`。
  - 选择“显卡”下列出的显卡。查找标有“**VRAM（总量）**”或类似字样的信息。这显示了专用视频内存的量。
  - 注意：对于配备 Apple 芯片（M1、M2、M3 系列芯片）的 Mac，内存是“统一”的，这意味着 CPU 和 GPU 共享相同的内存池。系统报告将显示统一内存的总量，该总量同时用作 RAM 和 VRAM。

### 检查 Linux 上的规格

Linux 提供了灵活性，这意味着确切的步骤可能因您的发行版（如 Ubuntu、Fedora、Mint）和桌面环境（如 GNOME、KDE、XFCE）而异。但是，命令行提供了通用方法。

**1. 使用终端（命令行）：**

打开您的终端应用程序。

- **CPU：** 运行以下命令之一：
  - `lscpu | grep "Model name"` (通常输出最清晰)
  - `cat /proc/cpuinfo | grep "model name" | uniq`
    这些命令将输出您的处理器型号名称。
- **RAM：** 运行以下命令：
  - `free -h`
    查找 `Mem:` 行中的 `total` 值。`-h` 标志使输出更易读（例如，`15G` 代表 16 GB，考虑到系统表示）。
- **GPU 和 VRAM：** 识别 GPU 及其 VRAM 可能稍微复杂一些。
  - 要识别 GPU 型号，请使用：
    - `lspci | grep -i vga`
    - 或者如果您知道供应商，可以更具体：`lspci | grep -i nvidia` 或 `lspci | grep -i amd`
      此命令列出您的显卡控制器。
  - 要查找 VRAM：
    - **NVIDIA GPU：** 如果您安装了 NVIDIA 驱动程序，命令 `nvidia-smi` 提供详细信息，包括总 VRAM（“内存使用情况”部分通常显示已用/总量）。
    - **AMD/Intel GPU：** 信息可能通过 `radeontop`（适用于 AMD，可能需要安装）等命令，或通过检查系统日志（`dmesg | grep -i vram`）或特定显卡实用程序的输出来获得。通过单个通用命令识别非 NVIDIA 显卡的 VRAM 有时会很困难。检查您的 VGA 设备 `lspci -v` 的输出也可能在“可预取内存”下提供内存大小详细信息。

**2. 使用图形用户界面系统监控工具：**

大多数桌面环境都附带图形系统监控器。

- 在您的应用程序菜单中搜索“系统监控器”、“使用情况”或类似名称（例如，GNOME 系统监控器、KDE 中的 KSysGuard）。
- 这些工具通常具有“资源”或“系统”选项卡或部分，以图形方式显示 CPU 型号、总 RAM，有时还显示基本的 GPU 信息。VRAM 的具体信息可能仍然需要上面提到的命令行方法。

一旦您收集到关于 CPU、RAM 和 GPU/VRAM 的信息，您将更好地了解在运行不同本地 LLM 时可以预期的性能，并选择适合的模型和工具，以备后续章节使用。在接下来的学习中，请将这些规格信息放在手边。

## 参考资料

- [Computer Organization and Design RISC-V Edition: The Hardware/Software Interface](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6) — David A. Patterson, John L. Hennessy (2020)
  Publisher: Morgan Kaufmann; Pages: 736
  一本权威教材，涵盖计算机硬件的基本原理，包括CPU、内存层次结构（RAM、VRAM）和I/O设备（GPU）。
- [NVIDIA System Management Interface (nvidia-smi)](https://developer.nvidia.com/nvidia-system-management-interface) — NVIDIA Corporation (2023)
  提供`nvidia-smi`的信息和用法，这是一个在Linux上监控NVIDIA GPU性能和VRAM的工具。
