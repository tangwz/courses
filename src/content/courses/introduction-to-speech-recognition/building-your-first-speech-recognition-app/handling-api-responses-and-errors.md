---
course: "introduction-to-speech-recognition"
chapter: "building-your-first-speech-recognition-app"
lesson: "handling-api-responses-and-errors"
sourceId: 7148
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/handling-api-responses-and-errors"
title: "处理 API 响应和错误"
description: "处理语音识别服务输出和妥善处理潜在错误的最佳实践。"
order: 6
plots: []
sourceHash: "dfda7c5f4d8104e6df89d08a330ec5d61a1143ea8c65ae9ea32b399054241b99"
sourceCorrections: []
---

当您调用语音识别服务时，您正在通过网络发送音频数据并等待响应。这个过程并非即时，并且可能因为程序无法控制的原因而失败。服务可能暂时不可用，您的网络连接可能中断，或者您发送的音频可能不包含任何可识别的语音。一个设计良好的应用程序必须预见这些问题并妥善处理，而不是崩溃。

处理 ASR 服务的不同响应对于构建可靠的语音识别应用程序非常重要。这包括成功的转录以及可能出现的各种错误。通常使用 Python 的 `try...except` 块来管理这些情况，从而确保应用程序能够优雅地处理潜在问题。

### 成功响应的结构

当识别尝试成功时，大多数库和 API 返回的不仅仅是转录的文本。它们通常提供一个结构化响应，通常以字典或对象形式，包含额外的元数据。虽然确切的结构在不同服务之间（如 Google Web Speech API 与 Wit.ai）有所不同，但它们通常包含相似的信息。

例如，服务可能会返回多个可能的转录，每个都有一个置信度分数。

```python
# ASR 服务的一个潜在响应对象
{
  "transcriptions": [
    {
      "transcript": "what time is it",
      "confidence": 0.94
    },
    {
      "transcript": "what time was it",
      "confidence": 0.05
    }
  ],
  "is_final": True,
  "language_code": "en-US"
}
```

在此响应中，最可能的转录是“what time is it”，置信度分数为 94%。获取这些额外信息对于更复杂的应用程序很有用。例如，如果最高置信度分数很低，您可能会要求用户重复。目前，我们的主要目标是稳定获取主要的 `transcript`。

### 常见失败情况

在编程中，我们不能只为一切顺利的“顺利路径”做计划。我们还必须处理不可避免的错误。使用 Python 中流行的 `SpeechRecognition` 库时，API 调用在出现问题时会引发异常。您会遇到的两个最常见的异常是 `UnknownValueError` 和 `RequestError`。

#### 音频无法识别时

有时，音频已成功发送到 ASR 服务，但服务无法在其中找到任何可识别的语音。如果麦克风只捕捉到背景噪音，或者说话者含糊不清，或者音频是静音的，都可能发生这种情况。

在这种情况下，`SpeechRecognition` 库会引发 `UnknownValueError`。您的程序应该捕获此异常并告知用户音频无法理解。

#### 服务不可达时

另一个常见问题是您的程序根本无法与 ASR 服务通信。这可能由于以下几个原因：

- **无互联网连接：** 您的计算机已离线。
- **API 服务中断：** 提供商（例如，谷歌、微软）遇到技术难题。
- **无效 API 密钥：** 您正在使用不正确或已过期的凭据。
- **API 限制：** 您已超出允许的免费请求数量。

对于这些情况，库会引发 `RequestError`。捕获此异常可以让您提供一条有用的消息，例如“无法连接到语音识别服务”，这比让程序因晦涩的网络错误而终止要好得多。

### 使用 try...except 构建健壮的代码

为了处理这些潜在故障，您应该将 API 调用包裹在一个 `try...except` 块中。这告诉 Python：“尝试运行这段代码，但如果发生特定错误，不要崩溃。而是运行这段其他代码。”

让我们看一下上一节中的一个简单的转录脚本，并使其更强大。

这是原始的“顺利路径”代码：

```python
# 警告：此代码未处理错误！
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("请说点什么！")
    audio = r.listen(source)

# 如果语音未被理解或 API 不可用，此行将导致程序崩溃
text = r.recognize_google(audio)
print("您说的是：" + text)
```

如果您运行此代码并保持沉默，程序将因 `UnknownValueError` 而崩溃。

现在，让我们添加错误处理：

```python
# 此代码包含错误处理
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("请说点什么！")
    audio = r.listen(source)

# 使用 try...except 块来处理潜在错误
try:
    # 尝试识别语音
    text = r.recognize_google(audio)
    print("您说的是：" + text)
except sr.UnknownValueError:
    # 如果语音无法识别，则运行此块
    print("Google 语音识别无法理解音频")
except sr.RequestError as e:
    # 如果服务出现问题，则运行此块
    print(f"无法从 Google 语音识别服务请求结果；{e}")
```

这个更新版本更加用户友好。它为每种常见故障模式提供清晰的反馈，而不是崩溃。这个流程的逻辑可以可视化为决策路径。

> 此图说明了语音识别尝试的控制流程。程序尝试执行识别，并根据成功与否或是否遇到特定错误而遵循不同的路径。

### 处理响应的最佳实践

当您在下一节继续构建语音命令工具时，请记住这些指导原则：

1. **始终包裹 API 调用：** 任何向 ASR 服务发起网络请求的代码都应该在 `try` 块内。
2. **捕获特定异常：** 分别捕获 `speech_recognition.UnknownValueError` 和 `speech_recognition.RequestError`，以便向用户提供明确、有用的反馈。
3. **查阅文档：** 如果您使用不同的库或直接使用云服务提供商的 SDK（例如 AWS 或 Azure），请阅读其文档以理解它们引发的特定异常以及响应对象的结构。
4. **提供清晰的用户反馈：** 告诉用户发生了什么。“无法理解音频”或“服务不可用”等信息有助于他们知道下一步该怎么做，例如说话更清晰或检查他们的互联网连接。

通过处理成功和失败，您可以确保您的应用程序稳定并提供良好的用户体验，这是从简单的脚本转向功能性应用程序的重要一步。

## 参考资料

- [The try statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement) — Python Software Foundation (2024)
  官方文档，解释Python的try...except块，用于处理异常和创建可靠的应用程序。
- [SpeechRecognition Library Reference](https://github.com/Uberi/speech_recognition) — Anthony Zhang (2017)
  Publisher: Anthony Zhang
  SpeechRecognition库的官方文档，详细介绍了其API、支持的服务以及UnknownValueError和RequestError等特定异常。
- [Effective Python: 90 Specific Ways to Write Better Python](https://www.oreilly.com/library/view/effective-python-90/9780134853982/) — Brett Slatkin (2019)
  Publisher: Addison-Wesley Professional
  提供了编写结构良好和可维护Python代码的实用建议，包括处理错误和与API交互的模式。
