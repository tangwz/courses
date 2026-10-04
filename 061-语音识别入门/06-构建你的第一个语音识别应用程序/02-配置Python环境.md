# 配置Python环境

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/setting-up-python-environment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在您编写任何语音识别代码之前，必须首先准备好您的开发环境。此过程包括安装Python以及提供语音识别功能的特定库。您可以把它想象成收集工具和布置工作台。通过为本项目创建一个干净、独立的区域，可以确保为语音识别安装的库不会干扰您系统上的其他Python项目，反之亦然。

### Python和虚拟环境

本章的所有示例都将使用Python。如果您尚未安装Python，请从官方[Python 网站](https://www.python.org/downloads/)下载并安装最新稳定版本（3.8或更高）。

现代Python开发中，一种标准做法是为每个项目使用一个虚拟环境。虚拟环境是一个独立的目录，其中包含特定版本的Python和单个项目所需的所有库。这可以避免需要同一库不同版本的项目之间发生冲突。

现在，我们来创建一个。打开您的终端或命令提示符，导航到您的项目文件夹，然后运行以下命令：

```shell
# 在macOS或Linux上
python3 -m venv asr_env

# 在Windows上
py -m venv asr_env
```

此命令会创建一个名为 `asr_env` 的新目录，其中包含Python解释器及其标准库的副本。要开始使用此环境，您需要“激活”它。

**在macOS或Linux上激活环境：**

```shell
source asr_env/bin/activate
```

**在Windows上激活环境：**

```shell
.\asr_env\Scripts\activate
```

激活后，您的命令提示符通常会显示活动环境的名称，例如 `(asr_env)`。现在，您安装的任何库都将放置在此环境中，使您的全局Python安装保持整洁。

> 虚拟环境 `asr_env` 在您的项目文件夹中创建了一个独立空间，包含其自身的Python解释器和库，与系统的主Python安装分离。

### 安装所需库

虚拟环境激活后，您现在可以安装库了。本章中，我们将主要使用 `SpeechRecognition` 库。这是一个出色的封装库，提供了一个一致的接口，用于处理多种不同的语音识别引擎和API，包括在线和离线。

要安装它，请使用Python的包安装器 `pip`。

```shell
pip install SpeechRecognition
```

此命令会将 `SpeechRecognition` 库及其直接依赖项下载并安装到您的 `asr_env` 环境中。

### 麦克风访问特别说明

本章后面，我们将编写一个脚本来直接从麦克风捕获音频。为此，`SpeechRecognition` 依赖于另一个名为 `PyAudio` 的库。不幸的是，`PyAudio` 有时安装起来可能比较麻烦，因为它依赖于系统级的音频库。

如果 `PyAudio` 的简单 `pip install` 命令失败，或者您稍后在访问麦克风时遇到错误，您可能需要手动安装它。

- **在Debian/Ubuntu Linux上：** 您可能需要先安装系统依赖项： `sudo apt-get install portaudio19-dev python3-pyaudio`
- **在macOS上：** 使用像Homebrew这样的包管理器会有所帮助： `brew install portaudio`，然后是 `pip install pyaudio`。
- **在Windows上：** 最简单的方法通常是从Christoph Gohlke的[Python扩展包的非官方Windows二进制文件](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)页面等可靠来源下载与您的Python版本和系统架构（32位或64位）匹配的预编译 `PyAudio` wheel (`.whl`) 文件。然后，您可以使用 `pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl` 来安装它（请调整文件名以匹配您下载的文件）。

目前，`SpeechRecognition` 库是您处理音频文件所需的一切。您可以在进入实时转录部分时再处理 `PyAudio` 的安装。

### 验证您的设置

让我们确保一切正常运行。创建一个名为 `verify_install.py` 的新Python文件，并添加以下行：

```python
import speech_recognition as sr

try:
    print(f"SpeechRecognition 库版本: {sr.__version__}")
    print("设置成功！您的环境已准备就绪。")
except Exception as e:
    print(f"发生错误: {e}")
    print("请检查您的安装。")
```

在您的虚拟环境仍处于活动状态时，在您的终端中运行此脚本：

```shell
python verify_install.py
```

如果您的设置正确，您应该会看到一条输出消息，确认库版本和成功消息。您的环境现已正确配置，您可以继续编写您的第一个脚本来转录音频文件了。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — The Python Core Developers (2023)
  关于创建和使用独立 Python 环境的官方指南。
- [Installing Packages](https://packaging.python.org/en/latest/guides/installing-packages/) — The Python Packaging Authority (2025)
  Publisher: The Python Packaging Authority
  使用 pip 管理 Python 包的官方文档。
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) — Anthony Zhang (2025)
  提供 SpeechRecognition Python 库的官方信息和使用说明。
- [PyAudio](https://pypi.org/project/PyAudio/) — Hubert Pham (2023)
  提供 PyAudio 的官方信息，包括麦克风访问的安装说明。

---

[上一节](01-%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%ABAPI%E4%B8%8E%E5%BA%93%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%BD%AC%E5%BD%95.md)
