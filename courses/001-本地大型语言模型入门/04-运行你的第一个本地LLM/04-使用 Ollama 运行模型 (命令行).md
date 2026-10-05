# 使用 Ollama 运行模型 (命令行)

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/running-model-ollama-cli)

[返回章节目录](README.md) · [返回课程目录](../README.md)

成功使用 `ollama pull` 命令下载模型后，就可以直接从终端或命令提示符与它进行交流了。Ollama 提供了一种简单的方法来运行模型并开始对话。

要开始与你选择的模型进行互动会话，你需要使用 `ollama run` 命令，后面跟上模型名称。例如，如果你在上一步下载了 `llama3:8b` 模型，你可以输入以下命令并按回车键：

```bash
ollama run llama3:8b
```

执行此命令后，Ollama 会将指定的模型加载到你电脑的内存中（RAM，如果你有兼容的 GPU，也可能加载到 VRAM）。这个加载过程可能需要几秒钟，尤其是你首次运行某个模型或重启电脑后。当你看到一个提示符出现时，你就知道模型已准备就绪，它通常看起来像这样：

```
>>> Send a message (/? for help):
```

这个提示符表示大型语言模型正在等待你的输入。你现在可以直接在终端中输入你的问题或指令。我们来试着问一个简单的问题：

```
>>> Send a message (/? for help): What is the primary function of a CPU in a computer?
```

输入消息后按回车键。大型语言模型会处理你的输入并生成回复，以流式传输文本输出到你的终端。输出可能看起来像这样（具体措辞会因模型而异）：

```text
计算机中中央处理器（CPU）的主要作用是执行程序指令。它根据指令执行基本的算术、逻辑、控制以及输入/输出（I/O）操作。它本质上是计算机的“大脑”，执行系统和应用程序运行所需的各项任务。
```

你可以在新的提示符处输入另一条消息来继续对话。模型通常会保持当前会话的上下文 (context)，让你能够提出后续问题或在之前的交流基础上提问，直至其上下文窗口的限制（我们稍后会提到这个）。

### 会话中的操作

在交互式 `ollama run` 会话中，你可以使用一些有用的命令，方法是输入一个正斜杠 (`/`) 后跟命令词。要查看可用选项，输入 `/?` 并按回车键：

```
>>> Send a message (/? for help): /?
可用命令：
  /?, /help        命令帮助
  /bye, /exit      退出 Ollama
  /set             设置会话变量
  /show            显示会话信息
  /save            将会话保存到文件
  /load            从文件加载会话
... 以及其他可能的命令 ...
```

你最初可能用到的一些最常用命令有：

- `/bye` 或 `/exit`：使用其中任何一个可以停止当前的模型互动并退出 Ollama 会话，让你返回到常规终端提示符。
- `/show info`：显示当前加载模型的详细信息。
- `/show license`：显示模型的许可证信息。

### 退出会话

当你与模型互动完毕后，只需输入 `/bye` 或 `/exit` 并按回车键即可。

```
>>> Send a message (/? for help): /exit
再见！
$
```

你将返回到标准命令提示符（根据你的系统，可能显示为 `$` 或 `>` 或类似符号）。你通常也可以使用键盘快捷键 `Ctrl+D` 来退出会话。

### 运行非交互式提示

有时，你可能只想从模型那里快速获得一个答案，而不想开始一个完整的交互式聊天会话。你可以通过在模型名称后直接在命令行上提供提示来实现这一点。

例如，要直接向 `llama3:8b` 模型询问量化 (quantization)的简短定义，你可以运行：

```bash
ollama run llama3:8b "Briefly explain model quantization"
```

Ollama 会加载模型，处理这个单一的提示（“Briefly explain model quantization”），将输出打印到终端，然后立即退出回到你的命令提示符。这对于简单的、一次性任务或将大型语言模型回复整合到脚本中非常有用。

你现在已经成功地使用 Ollama 的命令行界面，直接在你的电脑上运行并与大型语言模型进行了互动。可以随意尝试运行你下载的不同模型或尝试各种类型的提示。

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  提供了自然语言处理和Transformer架构的基础知识，这些是大型语言模型的基础。
- [Meta Llama 3](https://ai.meta.com/llama/) — Meta AI (2024)
  Publisher: Meta AI
  Meta Llama 3大型语言模型系列的官方信息来源，包含其功能和变体。

---

[上一节](03-%E7%94%A8%20Ollama%20%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.md) · [下一节](05-%E8%AE%BE%E7%BD%AE%20LM%20Studio.md)
