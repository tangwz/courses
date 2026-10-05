# 使用 Gradio 构建语音转文本应用

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems/building-stt-app-gradio)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当机器学习 (machine learning)模型被开发和评估之后，最后一步是使其可用。尽管 Hugging Face 的 `pipeline` 提供了一个简单的编程接口，但分享模型通常需要一种更易用、交互性更强的格式。Gradio 这样的工具在此非常实用，它让您能够直接从 Python 代码构建简单的网页应用。

### 使用 Gradio 从模型到应用

Gradio 是一个 Python 库，旨在快速构建和分享机器学习 (machine learning)模型的网页应用。它的主要优点是简洁性。您只需几行代码即可创建一个功能性的用户界面，让同事或用户可以通过网页浏览器与您的模型交互，而无需自己运行任何代码。

一个 Gradio 应用围绕三个主要组成部分构建：

1. **处理函数：** 这是一个普通的 Python 函数，它接收一些输入（如音频文件路径）并返回一个输出（如转录文本）。
2. **输入组件：** 用于用户提供数据的用户界面元素，例如麦克风录音机或文件上传器。
3. **输出组件：** 用于显示结果的用户界面元素，例如文本框。

这三个部分被封装在一个 `gradio.Interface` 对象中，该对象自动生成并启动网页应用。

### 环境设置

为了开始，您需要安装 Gradio，如果您正在跟着上一节的 Hugging Face pipeline 例子操作，还需要安装 `transformers` 和 `torch` 库。

```bash
pip install gradio transformers torch
```

### 构建 ASR 网页应用

让我们为 ASR 模型构建一个基于网页的界面。我们将使用 Hugging Face 上的预训练 (pre-training) Whisper 模型进行此次演示，因为它提供了一个高质量、即用型的转录引擎。

#### 第一步：定义转录逻辑

首先，我们需要加载 ASR pipeline 并创建一个 Python 函数来执行转录。这个函数将作为我们 Gradio 应用的核心逻辑。它必须接受输入组件提供的参数 (parameter)，并返回输出组件可以显示的值。

```python
import gradio as gr
from transformers import pipeline

# 加载 ASR pipeline
# 使用更小的模型如 whisper-tiny.en 以获得更快的推理速度
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

def transcribe_audio(audio):
    """
    一个使用 Hugging Face ASR pipeline 转录音频的函数。
    Gradio 会将音频输入作为 NumPy 数组和采样率提供。
    """
    if audio is None:
        return "未提供音频文件。请上传或录制音频。"

    # pipeline 返回一个字典，我们从中提取转录文本
    result = asr_pipeline(audio)
    return result["text"]
```

在 `transcribe_audio` 函数中，`audio` 参数将由 Gradio 的音频输入组件提供。然后函数将此音频传递给我们的 `asr_pipeline`，并返回生成的文本转录。

#### 第二步：创建和配置 Gradio 界面

接下来，我们使用 `gradio.Interface` 定义用户界面。我们将 `transcribe_audio` 函数连接到用户友好的输入和输出组件。

对于我们的输入，`gradio.Audio` 是一个非常适合的选择。我们可以配置它来接受上传的文件和直接的麦克风录音。对于输出，一个简单的 `gradio.Textbox` 就足以显示转录的文本。

```python
# 定义 Gradio 界面
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=gr.Textbox(label="转录"),
    title="简单 ASR 系统",
    description="上传音频文件或使用麦克风录制音频，将语音转录成文本。该模型基于 OpenAI 的 Whisper-tiny。"
)
```

注意 `inputs` 参数。我们指定了 `gr.Audio` 并配置了 `sources`，以允许文件上传和麦克风输入，为用户提供了灵活性。`type="filepath"` 参数告诉 Gradio 将输入音频保存到临时文件，并将路径传递给我们的函数，这是一种处理音频数据的常见且可靠的方法。

数据从用户流向模型再返回的流程在下图所示。

> 用户与 Gradio 音频组件交互。Gradio 将音频数据传递给我们的 Python 函数，该函数使用 Hugging Face pipeline 执行转录。生成的文本被返回到 Gradio 界面并显示给用户。

#### 第三步：启动应用

最后一步是启动网页服务器。这是通过在 `Interface` 对象上调用 `launch()` 方法完成的。

```python
# 启动应用
if __name__ == "__main__":
    demo.launch()
```

当您运行这个 Python 脚本时，Gradio 会输出一个本地 URL（通常是 `http://127.0.0.1:7860`）。在您的网页浏览器中打开这个 URL，将显示您功能完善的语音转文本应用。

### 分享您的演示

Gradio 最有用的功能之一是它能够为您的应用创建一个临时的、可分享的公共链接。这非常适合演示或从不在您本地网络上的合作者那里获取反馈。

要启用此功能，只需在 `launch()` 方法中将 `share` 参数 (parameter)设置为 `True`：

```python
# 使用公共链接启动应用
demo.launch(share=True)
```

Gradio 会生成一个公共 URL（例如 `https://xxxx.gradio.app`），该 URL 将保持活跃 72 小时。请注意，这会使您的应用公开可访问，因此，请勿将其用于处理敏感数据的应用。

您现在已经看到如何以最少的工作量将复杂的 ASR 模型封装到交互式网页应用中。Gradio 是一个出色的工具，用于创建演示、交互式测试模型和分享结果。然而，对于需要低延迟和连续处理的应用，例如实时字幕，则需要不同的架构。下一节将介绍构建此类实时流媒体系统所面临的挑战和方法。

## 参考资料

- [Gradio Documentation](https://www.gradio.app/docs/) — Gradio Team (2024)
  提供关于构建和分享机器学习应用的详细信息。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/) — Hugging Face Team (2024)
  Publisher: Hugging Face
  Transformers库的官方文档，包含模型使用、流水线等内容。
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever (2022)
  Journal: arXiv preprint arXiv:2212.04356; DOI: [10.48550/arXiv.2212.04356](https://doi.org/10.48550/arXiv.2212.04356)
  介绍 Whisper 自动语音识别模型的原始研究论文。

---

[上一节](04-%E4%BD%BF%E7%94%A8Hugging%20Face%E6%B5%81%E6%B0%B4%E7%BA%BF%E8%BF%9B%E8%A1%8CASR.md) · [下一节](06-%E5%AE%9E%E6%97%B6%E6%B5%81%E5%BC%8FASR%E7%9A%84%E8%80%83%E8%99%91%E4%BA%8B%E9%A1%B9.md)
