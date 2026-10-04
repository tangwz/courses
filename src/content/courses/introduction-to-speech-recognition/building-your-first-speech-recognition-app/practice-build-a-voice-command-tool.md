---
course: "introduction-to-speech-recognition"
chapter: "building-your-first-speech-recognition-app"
lesson: "practice-build-a-voice-command-tool"
sourceId: 7149
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/practice-build-a-voice-command-tool"
title: "实践：构建一个简单的语音命令工具"
description: "一个动手练习环节，您将构建一个能监听特定语音命令并执行操作的简单工具。"
order: 7
plots: []
sourceHash: "e205d9e65b21803e042536465057e0ab288d5f2ec2ff3c2baf97b9867a66cf44"
sourceCorrections: []
---

通过音频输入和转录，能够构建一个交互式应用程序。我们将构建一个简单的语音命令工具，它会监听一个“唤醒词”，然后响应特定的指令。该程序结合了音频输入、转录和基本的Python逻辑，以创建一个能响应您声音的实用程序。

### 应用程序的逻辑

在编写任何代码之前，勾勒出程序的流程会很有帮助。我们的语音命令工具将在一个连续循环中运行，遵循以下步骤：

1. **监听唤醒词**：程序会被动监听一个特定短语，比如“嘿，电脑”。
2. **激活并确认**：一旦唤醒词被检测到，程序会发出它已准备好接收命令的信号，可能通过说“什么事？”。
3. **监听命令**：程序随后会监听用户的下一个指令，例如“现在几点了？”。
4. **执行命令**：它会处理转录的文本以识别命令，并执行相应的操作。
5. **返回监听**：执行命令后，程序会回到第一步，再次等待唤醒词。

这个循环为语音控制应用程序创建了一个简单而有效的交互模型。

> 该程序的执行流程。它保持监听状态，直到唤醒词激活它，之后它处理单个命令并返回初始状态。

### 设置工具

对于这个应用程序，我们需要一些库。您应该已经从之前的章节安装了 `SpeechRecognition` 和 `PyAudio`。我们还会增加 `pyttsx3`，这是一个文本转语音库，它允许我们的程序与我们对话。

如果您尚未安装 `pyttsx3`，可以使用 pip 来安装：

```bash
pip install pyttsx3
```

在 Linux 系统上，如果 `espeak` 或其他 TTS 引擎尚未安装，您可能还需要安装它们。

我们还将使用 Python 的两个内置库：`datetime` 用于获取当前时间，`webbrowser` 用于打开网页浏览器。

### 逐步构建工具

让我们来编写代码。我们将创建一个Python脚本。其结构将包含识别器和文本转语音引擎的初始化阶段，然后是驱动应用程序的主循环。

#### 1. 初始化

首先，我们导入必要的库，并设置我们的识别器和文本转语音引擎。该引擎允许程序将文本字符串转换为语音。

```python
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser

# 初始化识别器
r = sr.Recognizer()

# 初始化文本转语音引擎
engine = pyttsx3.init()

# 定义唤醒词
WAKE_WORD = "hey computer"
```

#### 2. 主应用程序循环

我们程序的核心是一个 `while` 循环，它持续监听唤醒词。在这个循环中，我们使用 `try...except` 块来优雅地处理麦克风未能捕获任何清晰语音的情况。

当在转录文本中检测到唤醒词时，我们激活命令监听阶段。

```python
def listen_for_audio():
    """监听音频并将其作为转录文本返回。"""
    with sr.Microphone() as source:
        print("监听中...")
        # 调整环境噪音以提高准确性
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        # 使用谷歌引擎识别语音
        text = r.recognize_google(audio).lower()
        print(f"您说的是: {text}")
        return text
    except sr.UnknownValueError:
        # 语音不清楚
        print("抱歉，我没听懂。")
        return ""
    except sr.RequestError:
        # API无法访问或无响应
        print("抱歉，我的语音服务出问题了。")
        return ""

# 主循环
while True:
    text = listen_for_audio()

    if WAKE_WORD in text:
        engine.say("什么事？")
        engine.runAndWait()

        # 现在监听实际命令
        command_text = listen_for_audio()

        # 在这里我们将处理命令
        # (稍后实现)
```

在这个结构中，程序会打印“监听中...”并等待。您说话后，它会转录您的语音。如果短语“嘿，电脑”存在，它会响应“什么事？”，然后立即再次调用 `listen_for_audio()` 以捕获后续的命令。

#### 3. 处理命令

现在，让我们添加处理不同命令的逻辑。我们将使用一组简单的 `if/elif/else` 语句来检查转录命令中的关键字。

以下是包含命令处理逻辑的完整脚本。

```python
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser

# 初始化识别器
r = sr.Recognizer()

# 初始化文本转语音引擎
engine = pyttsx3.init()

# 定义唤醒词
WAKE_WORD = "hey computer"

def speak(text):
    """将文本转换为语音的函数。"""
    print(f"电脑: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_for_audio():
    """监听音频并将其作为转录文本返回。"""
    with sr.Microphone() as source:
        # 我们在开始监听时调整一次环境噪音
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("监听命令中...")
        audio = r.listen(source)

    try:
        # 使用谷歌在线服务识别语音
        text = r.recognize_google(audio).lower()
        print(f"您说的是: {text}")
        return text
    except sr.UnknownValueError:
        # 此错误表示库无法理解音频
        return ""
    except sr.RequestError:
        # 此错误表示API请求有问题
        speak("抱歉，我连接语音服务时遇到问题。")
        return ""

def process_command(command):
    """处理命令并执行相应的操作。"""
    if "what time is it" in command:
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"当前时间是 {now}。")

    elif "open browser" in command:
        speak("正在打开您的网页浏览器。")
        webbrowser.open("https://www.google.com")

    elif "stop listening" in command or "goodbye" in command:
        speak("再见！")
        return True # 表示退出循环

    else:
        speak("我不确定如何帮助您处理这个。")

    return False # 表示继续监听

# --- 主应用程序循环 ---
speak("我已准备就绪。请说唤醒词开始。")

while True:
    print("\n监听唤醒词中...")
    text_input = listen_for_audio()

    if WAKE_WORD in text_input:
        speak("什么事？我能帮您什么？")
        command = listen_for_audio()

        if command:
            should_exit = process_command(command)
            if should_exit:
                break
```

### 运行您的语音命令工具

1. 将上面的代码保存到名为 `voice_assistant.py` 的文件中。
2. 打开您的终端或命令提示符。
3. 进入您保存文件的目录。
4. 使用命令运行脚本：`python voice_assistant.py`

程序会打印“我已准备就绪”并开始监听。尝试说“嘿，电脑”。它应该响应“什么事？我能帮您什么？”。然后，给它一个命令，例如“现在几点了？”或“打开浏览器”。要停止程序，请说“嘿，电脑”，等待响应，然后说“停止监听。”

### 改进建议

这个简单的工具是一个很棒的起点。以下是在此基础上进行扩展的几点建议：

- **增加更多命令**：与其他的库或API结合，以增加更多功能。您可以让它获取天气、读取新闻网站的头条，或者控制智能家居设备。
- **改进命令解析**：您可以不再使用简单的 `if "关键字" in command` 检查，而是使用正则表达式或更高级的意图解析库来实现更灵活的命令识别。
- **离线识别**：为了在没有互联网连接的情况下提高隐私和性能，您可以尝试使用 `recognize_sphinx()` 这样的离线识别器，尽管它可能需要更多配置，并且准确性可能低于在线服务。
- **可自定义唤醒词**：允许用户设置自己的唤醒词，而不是将其硬编码在脚本中。

## 参考资料

- [SpeechRecognition documentation](https://speechrecognition.readthedocs.io/) — Kory Northrup (2024)
  详细说明Python SpeechRecognition库的使用方法，包括麦克风输入、转录方法和错误处理。
- [pyttsx3 documentation](https://pypi.org/project/pyttsx3/) — Natesh M Bhat (2023)
  pyttsx3 Python库的官方指南，提供文本转语音转换和引擎配置的方法。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  第7-9章涵盖自动语音识别（ASR）和第24章文本转语音（TTS）的原理，对于理解这些系统如何运行至关重要。
- [Designing Voice User Interfaces: Principles of Conversational Experiences](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGonKFzxIGes_yVIYmPRY2__VB4K_wQTfWqOv3l_nnTpfi-IisbycXlv5PUsTUSmmfJRIrhBL_t-2_iNvcKWt45uBCHVZBU-XUEa2aQDuwZNm7FdEtylE6xmfEUhF2pReHfuMNimqgM_goF_qCdMakZkwym97lvdQHq4b8oB0KHzA4fMEIR6HH6TqWapOYKXcsm24OB) — Cathy Pearl (2016)
  Publisher: O'Reilly Media
  介绍了设计有效语音控制应用程序的原则，包括唤醒词激活和用户交互模型。
