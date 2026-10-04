---
course: "introduction-to-speech-recognition"
chapter: "building-your-first-speech-recognition-app"
lesson: "transcribing-audio-from-a-file"
sourceId: 7144
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/transcribing-audio-from-a-file"
title: "从文件中转录音频"
description: "编写一个Python脚本，从磁盘加载音频文件并将其口语内容转换为文本。"
order: 4
plots: []
sourceHash: "697e3d35a025b998d7a258bdcd5abc79103d8315b2185694217bd371f7aad3c5"
sourceCorrections: []
---

将预先录制的音频文件转换为文本是语音识别中的一项基础任务。这个过程需要Python环境，它构成了许多应用的根基，例如转录访谈、为视频生成字幕或分析客户服务通话。我们将使用 `SpeechRecognition` 库，它为多个流行的ASR服务提供了方便的接口。

对于本次练习，你需要一个音频文件。WAV格式是理想的，因为它未经压缩且得到广泛支持，无需额外软件。如果你没有，可以轻松找到或创建一段包含清晰语句（例如“hello, this is a test”）的短WAV文件。将此文件保存在你存放Python脚本的同一目录下，并命名为 `test-audio.wav`。

### 转录流程

使用 `SpeechRecognition` 库转录音频文件的过程涉及几个不同的步骤。首先，你的脚本会初始化一个 `Recognizer` 对象。该对象随后会打开并读取音频文件，将其数据加载为库可以处理的格式。最后，这些音频数据被发送到一个外部ASR引擎，该引擎返回转录的文本。

> 使用Python脚本和ASR API转录音频文件的步骤图示。

### 编写转录脚本

让我们编写Python代码来实现这一点。创建一个名为 `transcribe_file.py` 的新文件，并添加以下代码。我们将逐一讲解脚本的每个部分。

```python
import speech_recognition as sr

# 1. 初始化识别器
r = sr.Recognizer()

# 2. 定义音频文件路径
audio_file = "test-audio.wav"

# 3. 打开音频文件并处理
with sr.AudioFile(audio_file) as source:
    # 从文件中读取音频数据
    audio_data = r.record(source)

    # 4. 执行识别
    print("正在转录音频...")
    try:
        # 使用谷歌的免费网络语音API
        text = r.recognize_google(audio_data)
        print(f"转录结果: {text}")

    except sr.UnknownValueError:
        # API无法理解音频
        print("Google语音识别无法理解该音频。")

    except sr.RequestError as e:
        # API无法访问或无响应
        print(f"无法从Google语音识别服务请求结果; {e}")
```

### 理解代码

让我们将脚本分解为主要部分。

#### 1. 初始化识别器

```python
import speech_recognition as sr

r = sr.Recognizer()
```

首先，我们导入库，使用 `sr` 作为标准别名以保持代码简洁。然后，我们创建一个 `Recognizer` 类的实例。这个 `r` 对象是我们应用程序的核心部分。它负责处理音频输入并与ASR服务通信。

#### 2. 打开音频文件

```python
audio_file = "test-audio.wav"

with sr.AudioFile(audio_file) as source:
    # ... 处理文件的代码在此处 ...
```

在这里，我们指定音频文件的名称。然后，我们在 `with` 语句中使用 `sr.AudioFile()`。这是一种重要的实践，因为它会自动处理文件的打开和关闭，确保资源得到正确管理。这个 `AudioFile` 对象，我们称之为 `source`，代表我们已打开的音频文件。

#### 3. 读取音频数据

```python
audio_data = r.record(source)
```

在 `with` 代码块内部，我们调用 `r.record(source)`。这个方法接收 `source` 对象，读取音频文件的全部内容，并将其存储在一个 `AudioData` 对象中。现在，这个 `audio_data` 变量以识别器可以处理的格式保存着音频。

#### 4. 执行识别并处理错误

```python
try:
    text = r.recognize_google(audio_data)
    print(f"转录结果: {text}")

except sr.UnknownValueError:
    print("Google语音识别无法理解该音频。")

except sr.RequestError as e:
    print(f"无法从Google语音识别服务请求结果; {e}")
```

这就是实际转录发生的地方。因为我们通过网络向外部服务发送请求，所以可能会出现问题。音频可能嘈杂，服务可能停止运行，或者你的互联网连接可能失败。使用 `try...except` 块可以使我们的脚本更具韧性。

- `r.recognize_google(audio_data)`: 这是完成工作的方法。它将 `audio_data` 发送到谷歌的网络语音API并等待响应。如果成功，它将转录的文本作为字符串返回。
- `except sr.UnknownValueError`: 当语音识别器无法理解所说内容时，会引发此错误。这可能发生在音频只有静默、包含太多背景噪音，或者语言不符合API预期的情况下。
- `except sr.RequestError`: 如果与API的网络连接出现问题，例如没有互联网访问或服务本身存在问题，则会发生此错误。

### 运行你的首次转录

要运行脚本，请保存它并从你的终端执行，确保 `test-audio.wav` 文件位于同一目录中。

```bash
python transcribe_file.py
```

如果一切正常，你应该看到类似于这样的输出：

```
Transcribing audio...
Transcription: hello this is a test
```

恭喜。你已成功编写了一个程序，可以将音频文件中的口语转换为文本。这个简单的脚本是一个有力的起点。在下一节中，我们将调整此代码以直接从你的麦克风处理实时音频输入。

## 参考资料

- [SpeechRecognition Library Documentation](https://pypi.org/project/SpeechRecognition/) — Anthony Zhang and collaborators (2025)
  Python SpeechRecognition 库的官方文档，详细介绍了其音频输入 API 和与各种 ASR 服务的集成。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本广泛使用的教科书，涵盖了语音识别、自然语言处理和计算语言学的基本原理。
- [Google Cloud Speech-to-Text Documentation](https://cloud.google.com/speech-to-text/docs) — Google Cloud (2024)
  Publisher: Google
  Google 商业语音转文本 API 的官方文档，为高级、可扩展的 ASR 解决方案提供了背景信息。
