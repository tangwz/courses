# 第 2 章：配置用于大型语言模型 (LLM) 开发的 Python 环境

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm)

[返回课程目录](../README.md)

在构建与大型语言模型交互的应用前，建立一个统一且易于管理的开发环境是起步阶段的第一项要务。LLM 项目通常依赖特定版本的库，并需要细致处理如 API 密钥这类敏感信息。从一开始就正确配置，可避免在开发后期出现问题。

本章将重点讲述如何配置专为 LLM 工作而设的 Python 环境。我们将涵盖以下内容：

*   选择合适的 Python 版本。
*   使用 `venv` 或 `conda` 等虚拟环境来隔离项目依赖。
*   用 `pip` 和 `requirements.txt` 文件有效管理所需软件包。
*   安装 LLM 工作流程中常用的库。
*   实施安全的方法来存储和访问 API 密钥。

学完本章后，您将对如何设置和维护一个整洁、可复现的开发环境有实际的认识，该环境专为用 Python 构建 LLM 应用而定制。

## 小节

- 1. [选择你的Python版本](01-%E9%80%89%E6%8B%A9%E4%BD%A0%E7%9A%84Python%E7%89%88%E6%9C%AC.md)
- 2. [虚拟环境 (venv, conda)](02-%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%20%28venv%2C%20conda%29.md)
- 3. [使用pip和requirements.txt管理依赖](03-%E4%BD%BF%E7%94%A8pip%E5%92%8Crequirements.txt%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96.md)
- 4. [核心库安装](04-%E6%A0%B8%E5%BF%83%E5%BA%93%E5%AE%89%E8%A3%85.md)
- 5. [安全配置 API 密钥](05-%E5%AE%89%E5%85%A8%E9%85%8D%E7%BD%AE%20API%20%E5%AF%86%E9%92%A5.md)
- 6. [开发环境配置实践](06-%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E5%AE%9E%E8%B7%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/quiz)
