# 设置 Ollama

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/setting-up-ollama)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Ollama 是管理本地大型语言模型最直接的选项之一。它通过将模型权重 (weight)、配置以及运行模型所需的软件打包成一个易于管理的捆绑包，简化了在个人计算机上运行开源大型语言模型的过程。它是一个方便的命令行工具，旨在帮你快速开始使用。

Ollama 是一个受欢迎的选择，特别是对初学者来说，因为它简化了许多技术步骤。它支持 macOS、Windows 和 Linux 系统，其简单的命令使得模型下载和使用变得非常方便。尽管它主要通过命令行（macOS/Linux 上的终端，Windows 上的命令提示符或 PowerShell）运行，但其易于安装的特点使其成为一个很好的起点。请注意，Ollama 是作为后台服务运行的，而不是一个完整的图形窗口；点击应用程序图标只会启动该后台服务，并在菜单栏或系统托盘中添加一个图标。

在继续之前，请回顾第 2 章中讨论的硬件考虑事项。Ollama 可以仅使用你的 CPU 和系统内存运行模型，但如果你有兼容的 GPU（图形处理单元）和足够的 VRAM（显存 (VRAM)），性能，尤其是推理 (inference)速度（模型生成文本的速度），将明显更好。Ollama 会尝试自动检测并使用兼容的硬件。

### 安装 Ollama

安装过程因你的操作系统而略有不同。请按照以下针对你具体系统的步骤进行操作。

#### macOS

1. **下载：** 访问 Ollama 官方网站 (<https://ollama.com>)，点击“Download”（下载）按钮，然后选择“Download for macOS”（下载 macOS 版）。这会下载一个包含应用程序的 `.zip` 文件。
2. **安装：** 打开下载的 `.zip` 文件（通常双击即可）。将 `Ollama` 应用程序拖到你的 `Applications`（应用程序）文件夹中。
3. **运行：** 从 `Applications`（应用程序）文件夹打开 `Ollama` 应用程序。你将在菜单栏中看到一个小图标，表示 Ollama 正在后台运行。它不会打开传统的软件主窗口。首次启动时也可能会提示你安装其命令行工具。如果出现请求，请允许安装。

#### Windows

1. **下载：** 访问 Ollama 网站 (<https://ollama.com>)，点击“Download”（下载），然后选择“Download for Windows”（下载 Windows 版）。这会下载一个 `.exe` 安装程序文件。
2. **安装：** 双击下载的 `.exe` 文件以启动安装程序。按照屏幕上的提示操作。安装程序会设置 Ollama 并将所需的命令行工具添加到你系统的 PATH 环境变量中，使其可以通过命令提示符或 PowerShell 访问。启动后，Ollama 会在后台运行并常驻在你的系统托盘中。
3. **GPU 驱动程序（重要）：** 为了让 Ollama 使用你的 NVIDIA GPU（如果你有的话），请确保已安装最新的 NVIDIA 驱动程序。你通常可以通过 NVIDIA GeForce Experience 应用程序或直接从 NVIDIA 官方网站获取这些驱动程序。Ollama 依赖这些驱动程序来获得 CUDA 支持。

#### Linux

1. **下载和安装：** 在 Linux 上安装 Ollama 的推荐方法是通过命令行脚本。打开你的终端并运行以下命令：

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   此命令会下载安装脚本并执行它。脚本会检测你的系统并正确安装 Ollama。
2. **GPU 驱动程序（重要）：**
   - **NVIDIA：** 请确保你已安装官方 NVIDIA 驱动程序和 NVIDIA Container Toolkit，以便 Ollama 使用你的 GPU。安装方法因发行版而异（例如，Debian/Ubuntu 使用 `apt`，Fedora 使用 `dnf`）。有关你 Linux 发行版的具体说明，请查阅 Ollama 的文档或 NVIDIA 的指南。
   - **AMD：** 对 AMD ROCm 的实验性支持可能可用。这通常需要为你使用的发行版安装特定的 ROCm 驱动程序。请查阅 Ollama 文档以获取 AMD GPU 支持的最新状态和说明。
3. **权限（潜在问题）：** 根据你的设置，特别是对于 GPU 访问，你可能需要将用户帐户添加到特定组（如 `render` 或 `docker`）。如果遇到权限问题，安装脚本或 Ollama 的文档可能会提供指导。

### 验证安装

安装完成后，Ollama 通常作为后台服务运行。要确认它已安装并可从命令行访问：

1. 打开你的终端（macOS/Linux 上的终端，Windows 上的命令提示符或 PowerShell）。
2. 输入命令 `ollama` 并按回车键。

如果安装成功，你将看到一个帮助消息，列出可用的 Ollama 命令，类似于这样：

```
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       启动 ollama
  create      从 Modelfile 创建模型
  show        显示模型信息
  run         运行模型
  pull        从注册表拉取模型
  push        将模型推送到注册表
  list        列出模型
  cp          复制模型
  rm          移除模型
  help        获取任何命令的帮助

Flags:
  -h, --help      获取 ollama 的帮助
  -v, --version   显示 ollama 的版本

Use "ollama [command] --help" 获取有关命令的更多信息。
```

看到这个输出确认你的系统识别 `ollama` 命令并且应用程序已准备好使用。如果你收到“command not found”（命令未找到）错误，请仔细检查安装步骤，确保 Ollama 正在运行（特别是在 macOS/Windows 上，它可能是一个菜单栏/系统托盘应用程序），并可能需要重启你的终端或计算机。

Ollama 安装并验证后，你现在已具备从终端直接下载和运行大型语言模型所需的条件。接下来的章节将引导你使用 `ollama` 命令拉取你的第一个模型并开始使用它。

## 参考资料

- [Ollama: Run large language models locally](https://ollama.com) — Ollama (2024)
  Ollama的官方网站, 提供本地部署大型语言模型的安装说明、文档和模型访问。
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  CUDA工具包的官方文档, 是NVIDIA GPU加速（Ollama及其他本地大型语言模型工具所用）并行计算平台和编程模型的关键资源。
- [AMD ROCm Documentation](https://rocm.docs.amd.com/) — Advanced Micro Devices, Inc. (2025)
  Publisher: Advanced Micro Devices, Inc.
  AMD ROCm开源平台的官方文档, 该平台支持GPU计算, 与Ollama等应用程序中AMD GPU的使用相关。

---

[上一节](01-%E6%9C%AC%E5%9C%B0LLM%E8%BF%90%E8%A1%8C%E5%99%A8%E4%BB%8B%E7%BB%8D.md) · [下一节](03-%E7%94%A8%20Ollama%20%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.md)
