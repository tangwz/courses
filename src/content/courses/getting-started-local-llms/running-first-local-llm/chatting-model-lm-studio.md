---
course: "getting-started-local-llms"
chapter: "running-first-local-llm"
lesson: "chatting-model-lm-studio"
sourceId: 4270
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/chatting-model-lm-studio"
title: "在LM Studio中加载模型并进行聊天"
description: "如何在LM Studio中加载已下载的模型并开始聊天会话。"
order: 7
plots: []
sourceHash: "7d3e98758c8733eb21b37569082cff8d077d7f14e799b8cb8ed162768a520fbe"
sourceCorrections: []
---

与本地大型语言模型交互需要将模型文件加载到内存中。LM Studio为此过程提供了一个用户友好的图形界面，让你能够轻松地开始首次对话。

### 找到聊天界面

首先，请确保LM Studio正在运行。在应用程序窗口中，寻找主要交互区域。它通常由一个看起来像语音气泡的图标表示，或者标记 (token)为“聊天”。它通常位于左侧导航面板中。点击此区域以打开聊天界面。

### 选择并加载你的模型

在聊天界面的顶部或侧面，你通常会找到一个下拉菜单或一个标记 (token)为“选择要加载的模型”的按钮。

1. **选择模型：** 点击此下拉菜单。你应该会看到你在上一步下载的模型列表（例如，`Mistral 7B Instruct Q4_K_M GGUF`）。选择你想使用的模型。
2. **加载过程：** 选定后，LM Studio将开始将模型加载到你的电脑RAM中（如果你已配置并启用了GPU加速，还会加载到VRAM）。你经常会看到一个进度指示器或状态消息，例如“正在加载模型...”或显示已加载的百分比。

加载可能需要几秒到几分钟不等的时间，主要取决于：

- **模型大小：** 更大的模型（更多参数 (parameter)、更高量化 (quantization)级别）需要更多资源，加载时间也更长。
- **系统RAM：** 你需要有足够的可用RAM来容纳整个模型。如果你的系统开始使用交换内存（将磁盘空间用作虚拟RAM），加载速度将明显变慢。
- **磁盘速度：** 模型文件需要从你的存储驱动器读取。更快的SSD（固态硬盘）相比传统HDD（机械硬盘）能带来更快的加载时间。

请耐心等待加载过程完成。LM Studio通常会在模型准备就绪时发出提示，通常通过启用聊天输入框或显示“模型已加载”的状态。

### 通过聊天窗口交互

模型加载后，你会看到主要聊天区域变为活动状态。它通常包括：

1. **输出区域：** 这个大区域显示对话历史，包括你的提示和模型的回复。
2. **输入框：** 位于底部，这是你向LLM输入消息（提示）的地方。
3. **发送按钮：** 通常在输入框旁边，点击此按钮（或通常只需按`Enter`键）即可将你的提示发送给模型。
4. **配置面板（可选）：** 通常在右侧，你可能会看到诸如温度、上下文 (context)长度等设置。目前，你通常可以保持这些设置为默认值。

> 在LM Studio中加载模型并进行聊天的基本流程。

### 发送你的第一个提示

让我们开始一个简单的对话。

1. 点击聊天窗口底部文本输入框内。
2. 输入一个简单的问候或问题。例如：
   - `你好！你能告诉我一个有趣的知识吗？`
   - `用一句话描述月亮。`
   - `2加2是多少？`
3. 按下`Enter`键或点击“发送”按钮。

### 观察回复

LM Studio现在将你的文本提示发送给已加载的LLM。模型处理你的输入并开始逐个生成回复。你将在输出区域看到回复逐步显示。

生成速度很大程度上取决于：

- **模型大小：** 较小的模型通常响应更快。
- **硬件：** 强大的CPU有所帮助，但如果LM Studio设置中配置了卸载功能，兼容的GPU能显著加速生成。

等待模型完成回复生成。它可能看起来像这样：

```text
你：
你好！你能告诉我一个有趣的知识吗？

模型：
当然！一个有趣的知识是：蜂蜜永不腐败。考古学家在古埃及墓穴中发现的蜂蜜罐，即使已有3000多年的历史，仍然完全可以食用！
```

恭喜！你刚刚完成了与完全在你自己的电脑上运行的大型语言模型的首次交互。

### 继续对话

你可以通过输入后续提示来继续聊天。模型通常会记住对话的最近部分（在其“上下文 (context)窗口”内，我们将在下一章更多地讨论），以提供相关回复。尝试根据之前的回复提出一个后续问题。

### 卸载模型

当你聊完或想释放系统资源（尤其是RAM）时，你可以卸载模型。在模型选择下拉菜单附近寻找一个“弹出”按钮，或者有时从下拉菜单中选择“未加载模型”或类似选项也能实现此目的。卸载模型会将其从你电脑的内存中移除。

你现在已在LM Studio中成功加载了一个LLM并进行了基本的聊天会话。这确认了你的设置运行正常，并为后续研究更复杂的交互和提示技巧提供了基础，我们将在接下来介绍。

## 参考资料

- [LM Studio Official Website](https://lmstudio.ai/) — LM Studio Team (2024)
  提供 LM Studio 的官方软件、指南和社区资源，对于下载、管理和与本地大型语言模型交互至关重要。
- [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp) — Georgi Gerganov and the llama.cpp contributors (2024)
  高效进行大型语言模型 CPU/GPU 推理的基础项目，包括 LM Studio 使用的 GGUF 文件格式的开发。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  全面概述了大型语言模型，涵盖其架构、训练、能力和应用，有助于理解其底层技术。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了量化大型语言模型的高效微调技术，提供了关于量化如何减少内存占用并影响本地部署性能的见解。
