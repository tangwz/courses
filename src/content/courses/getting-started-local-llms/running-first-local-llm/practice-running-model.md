---
course: "getting-started-local-llms"
chapter: "running-first-local-llm"
lesson: "practice-running-model"
sourceId: 4273
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/practice-running-model"
title: "动手实践：运行模型"
description: "跟随步骤下载并运行一个小型LLM，可选择Ollama或LM Studio。"
order: 9
plots: []
sourceHash: "926582e5522cbe80a13e0fad0746ca5a536969feedae33f36dfc872ff93c66c9"
sourceCorrections: []
---

Ollama和LM Studio是可在您的电脑上运行大型语言模型的有用辅助工具。下面将实际运行一个模型，并观察它根据您的输入生成文本。

本节提供具体步骤，帮助您开始。我们将使用一个相对较小的模型来做这个首次练习，以确保它能在更多样的硬件上顺利运行。请记住，根据前面章节的说明，您应该已经安装了Ollama或LM Studio。选择您已安装的工具，然后按照下面相应的步骤操作。

### 使用Ollama运行模型（命令行）

如果您选择了Ollama，您将通过终端或命令提示符与其进行操作。

1. **下载模型（如果尚未下载）：**
   让我们下载一个小型且性能不错的模型。我们将使用`phi3:mini`，这是一个由微软开发，以其在同等大小模型中表现良好而闻名的模型。打开您的终端并运行：

   ```bash
   ollama pull phi3:mini
   ```

   您会看到下载进度指示。这可能需要几分钟时间，具体取决于您的网络速度。下载完成后，模型将保存在本地。
2. **交互式运行模型：**
   现在，使用以下命令与模型开始一个交互式会话：

   ```bash
   ollama run phi3:mini
   ```

   Ollama会加载模型（这可能需要一些时间，特别是第一次加载时），然后会向您显示一个提示符，通常看起来像 `>>> Send a message (/? for help)`。
3. **与模型交互：**
   在 `>>>` 后面直接输入您的提示并按回车键。让我们尝试让它创作一些内容：

   ```text
   >>> Write a short story about a robot who discovers gardening.
   ```

   模型会处理您的请求并生成回复，将文本输出直接流式传输到您的终端。
4. **继续对话：**
   您可以继续交互。模型会记住对话在其上下文 (context)窗口中的先前部分（如第5章所说明）。尝试提出一个后续问题。
5. **退出会话：**
   完成操作后，您可以退出Ollama交互式会话。输入 `/bye` 并按回车键，或者在大多数系统上，您可以按 `Ctrl+D`。

   > 使用命令行通过Ollama交互式运行模型的基本流程图。

### 使用LM Studio运行模型（图形界面）

如果您更喜欢图形界面，LM Studio让运行模型变得简单明了。

1. **启动LM Studio：** 打开您之前安装的LM Studio应用程序。
2. **下载模型（如果需要）：**

   - 前往“搜索”选项卡（通常由放大镜图标 🔍 表示）。
   - 在搜索栏中，输入 `phi3 mini instruct`。您会看到不同的版本。请找一个GGUF格式的模型，最好是名称中带有`Q4_K_M`或`Q4_0`的，因为它们在大小和质量之间提供了良好的平衡。例如，您可能会找到 `Phi-3-mini-4k-instruct-q4_0.gguf`。
   - 点击您选择的模型旁边的“下载”按钮。您可以在应用程序底部查看下载进度。
3. **加载模型以进行聊天：**

   - 点击“聊天”选项卡（通常由对话气泡图标 💬 表示）。
   - 在顶部，您可能会看到一个标有“选择要加载的模型”的下拉菜单。点击它并选择您刚刚下载的 `Phi-3` 模型。
   - LM Studio会将模型加载到内存中。这可能需要一些时间，您通常可以在屏幕上（通常在右侧）看到进度或状态信息。等待它指示模型已准备就绪。
4. **与模型交互：**

   - 模型加载完成后，您会看到一个类似于在线聊天机器人的聊天界面。
   - 找到底部的文本输入框。在那里输入您的提示。让我们尝试一个简单的指令：

     ```text
     Explain what a Large Language Model is in one sentence.
     ```
   - 按回车键或点击“发送”按钮。
   - 模型会处理您的请求，其回复将显示在上面的聊天窗口中。
5. **继续聊天：**
   您可以输入更多提示并继续对话，就像使用命令行版本一样。LM Studio会为您管理交互历史。

   > 使用LM Studio图形界面交互式运行模型的基本流程图。

**关于性能的简要说明：** 加载模型（将其从存储驱动器传输到计算机的RAM或VRAM中）通常是最慢的部分，特别是在启动应用程序后首次运行某个特定模型时。一旦加载完成，生成文本的速度应该会相对快一些，但速度仍然很大程度上取决于您的硬件（CPU、GPU、RAM）以及模型的大小。

恭喜！您已成功在本地下载并运行了您的第一个大型语言模型。您向它发出了提示，它根据您的输入生成了文本，所有这些都完全在您自己的机器上运行。在下一章中，我们将更仔细地研究如何通过提示与这些模型进行有效交流。

## 参考资料

- [Ollama Docs](https://ollama.com/docs) — The Ollama Team (2024)
  使用Ollama在本地运行和管理语言模型的官方指南。
- [LM Studio](https://lmstudio.ai/) — GGUF.AI (2024)
  LM Studio的官方来源，一个方便本地运行大型语言模型的桌面应用程序。
- [ggerganov/llama.cpp: Port of Facebook's LLaMA model to C/C++](https://github.com/ggerganov/llama.cpp) — Georgi Gerganov and contributors (2024)
  支持LLM高效CPU推理并引入GGUF模型格式的项目。
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — Microsoft (2024)
  Journal: Hugging Face Model Hub; Publisher: Microsoft
  该模型卡提供Microsoft Phi-3 mini instruct模型的详细信息，包括其性能和能力。
