---
course: "getting-started-local-llms"
chapter: "running-first-local-llm"
lesson: "setting-up-lm-studio"
sourceId: 4265
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/setting-up-lm-studio"
title: "设置 LM Studio"
description: "安装 LM Studio 的分步指南，LM Studio 是一款用于运行本地 LLM 的 GUI 应用程序。"
order: 5
plots: []
sourceHash: "80f4b3696bbd2fbec548170a56d3a26ab6168508868c2226501d94cd89786c86"
sourceCorrections: []
---

虽然像 Ollama 这样的命令行工具提供了使用本地大型语言模型（LLMs）的有效方式，但许多用户更喜欢使用图形用户界面（GUI）来管理软件。LM Studio 提供了这样的环境，它是一款用户友好的应用程序，可以直接在您的电脑上查找、下载和运行大型语言模型。它将模型管理和执行的复杂操作简化为可视化界面，如果您不太习惯命令行或仅仅喜欢点击操作方式，那么它会是一个很好的选择。

### 下载 LM Studio

要开始使用，您需要下载 LM Studio 应用程序。务必直接从官方来源下载，以确保您获得正版和最新版本。

1. 打开您的网页浏览器，前往 LM Studio 官方网站：<https://lmstudio.ai/>
2. 在主页上，您会看到针对不同操作系统的醒目下载链接（Windows、macOS、Linux）。
3. 点击与您电脑操作系统对应的下载按钮。这将下载安装文件。

### 安装 LM Studio

安装过程因操作系统而异。

#### Windows

1. 在您的“下载”文件夹中找到已下载的 `.exe` 文件（例如 `LM-Studio-Setup-....exe`）。
2. 双击 `.exe` 文件以启动安装向导。
3. 您可能会看到来自 Windows 的安全警告；请确认您要运行此应用程序。
4. 按照屏幕上的提示操作。通常可以接受默认的安装位置设置。安装程序会复制所需文件，并可能提供创建桌面或开始菜单快捷方式的选项。
5. 安装完成后，您可以从“开始”菜单或（如果已创建）桌面快捷方式启动 LM Studio。

#### macOS

1. 在您的“下载”文件夹中找到已下载的 `.dmg` 文件（例如 `LM.Studio-....dmg`）。
2. 双击 `.dmg` 文件将其打开。会弹出一个新窗口，通常显示 LM Studio 应用程序图标以及一个指向“应用程序”文件夹的快捷方式。
3. 将 LM Studio 图标拖动到该窗口中的“应用程序”文件夹快捷方式内。这会将应用程序复制到您的系统。
4. 现在您可以弹出 `.dmg` 文件（将其图标从桌面或 Finder 边栏拖到“废纸篓”/“弹出”图标）并根据需要删除原始下载文件。
5. 首次从“应用程序”文件夹打开 LM Studio 时，macOS 可能会显示安全警告，因为它是在互联网上下载的。请确认您信任此应用程序并希望打开它。如果出现提示，您可能需要前往“系统设置”>“隐私与安全性”来允许其运行。

#### Linux

1. 适用于 Linux 的 LM Studio 通常以 AppImage 文件（例如 `LM_Studio-....AppImage`）的形式分发。在您的“下载”文件夹中找到此文件。
2. AppImage 文件是独立的应用程序，不需要传统安装。但是，您需要先使该文件可执行。打开您的终端，导航到您下载文件的目录（例如 `cd ~/Downloads`），然后运行以下命令，将 `LM_Studio-....AppImage` 替换为实际的文件名：

   ```bash
   chmod +x LM_Studio-....AppImage
   ```
3. 现在，您可以直接从终端运行 LM Studio：

   ```bash
   ./LM_Studio-....AppImage
   ```
4. 或者，在使其可执行后，您通常可以在文件管理器中双击 AppImage 文件来运行它。一些桌面环境可能会询问您是否要将 AppImage 集成到您的系统中（创建菜单项），这会很方便。
5. *注意*：一些 Linux 发行版可能需要安装 `libfuse` 库才能使 AppImage 正常工作。如果您遇到问题，请搜索有关为您的特定发行版安装 `fuse` 或 `libfuse2` 的说明（例如，在基于 Debian/Ubuntu 的系统上运行 `sudo apt install libfuse2`）。

### 首次启动

首次启动 LM Studio 时，它将进行初始化并显示主界面。您可能会看到一条消息或一些初始提示。现在应用程序已准备好供您开始使用和下载模型。

LM Studio 界面通常包含几个主要区域：

- 一个用于查找模型的搜索部分（通常连接到 Hugging Face）。
- 一个用于与加载模型交互的聊天界面。
- 一个用于管理已下载模型的区域。
- 硬件加速和其他选项的配置设置。

LM Studio 安装完成后，您现在有了一个可视工具集。接下来的步骤是使用其界面寻找合适的模型，将它们下载到您的电脑上，最后加载它们以开始生成文本。请记住，尽管 LM Studio 简化了流程，但第 2 章中讨论的硬件要求（足够的内存，可能还有一块性能好的 GPU）对于高效运行模型仍然适用。

## 参考资料

- [LM Studio Official Website](https://lmstudio.ai/) — The LM Studio Team (2024)
  下载LM Studio应用程序以及获取其最新信息和更新的主要权威来源。
- [Model Hub Documentation](https://huggingface.co/docs/hub/index) — Hugging Face (2024)
  提供Hugging Face Model Hub的全面文档，该Hub是LM Studio等应用程序可发现和使用的许多LLM的模型库。
- [llama.cpp - Inference of LLaMA model in pure C/C++](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and llama.cpp contributors (2024)
  提供高性能C/C++推理引擎的开源项目，常作为LM Studio等本地LLM GUI的基础技术。
