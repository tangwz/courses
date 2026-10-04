---
course: "getting-started-with-llm-toolkit"
chapter: "foundational-text-generation"
lesson: "course-overview-and-setup"
sourceId: 7748
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-1-foundational-text-generation/course-overview-and-setup"
title: "课程概览与设置"
description: "了解课程结构概况，并学习如何为Kerb工具包设置您的开发环境。"
order: 1
plots: []
sourceHash: "b81e1e1cc5303244e9b5f48e45b75eb7c72977fc2d0cd1219f2b67b9ad2f413b"
sourceCorrections: []
---

学习如何使用Kerb工具包构建由大型语言模型驱动的精巧应用程序。学习过程从基本的文本生成开始，逐步达到复杂的、多步骤的自主代理。您将学习管理提示词 (prompt)、处理外部数据以实现检索增强生成（RAG）、实现对话记忆，并为生产环境优化您的应用程序。实际操作是主要重心，使您具备构建、测试和部署现代AI系统的能力。

### 学习路径

本文档不仅是Kerb工具包的说明，而且还构建了构建大型语言模型应用的基本知识。我们从与大型语言模型通信的核心组件开始，并逐步将它们组合成完整的应用程序。以下图表描绘了前进的路线。

> 课程从基本技能开始，逐步构建完整的RAG系统、自主代理，并为应用程序的生产环境做好准备。

### 环境设置

在我们开始之前，需要进行一些设置步骤，以确保您的环境已准备就绪。本文档假定您具备Python的实用知识，并对大型语言模型是什么以及API如何工作有大致了解。

#### 安装

第一步是安装工具包。您可以使用 `pip` 直接从Python包索引（PyPI）安装它。打开您的终端并运行以下命令：

```bash
pip install kerb[all]
```

此命令安装核心库以及RAG、代理和评估等高级主题所需的所有模块。某些特定的第三方工具（如某些向量 (vector)数据库或文档加载器）可能需要额外的包，这将在相关章节中说明。

#### 获取API密钥

该工具包提供了一个统一的接口，以与各种大型语言模型提供商（如OpenAI、Anthropic和Google）进行交互。要使用这些服务，您需要从它们各自的平台获取API密钥。

1. **OpenAI:** 从[OpenAI API密钥页面](https://platform.openai.com/api-keys)获取您的密钥。
2. **Anthropic:** 访问您的[Anthropic控制台](https://console.anthropic.com/settings/keys)。
3. **Google AI:** 从[Google AI Studio](https://aistudio.google.com/app/apikey)获取您的密钥。

在本文档中，我们将使用这些提供商的模型。建议至少拥有一个OpenAI密钥，以便跟随所有示例操作。

#### 配置API密钥

为了安全性和灵活性，您不应将API密钥直接硬编码到您的源代码中。最佳做法是将其存储为环境变量。该工具包设计为自动从这些变量中加载密钥，这也是为其各自的大型语言模型包设置API密钥的标准变量约定。

在您的系统中设置以下环境变量。

对于Linux和macOS：

```bash
export OPENAI_API_KEY="您的OpenAI API密钥"
export ANTHROPIC_API_KEY="您的Anthropic API密钥"
export GOOGLE_API_KEY="您的Google API密钥"
```

对于Windows（命令提示符）：

```bash
set OPENAI_API_KEY="您的OpenAI API密钥"
set ANTHROPIC_API_KEY="您的Anthropic API密钥"
set GOOGLE_API_KEY="您的Google API密钥"
```

对于Windows（PowerShell）：

```powershell
$Env:OPENAI_API_KEY="您的OpenAI API密钥"
$Env:ANTHROPIC_API_KEY="您的Anthropic API密钥"
$Env:GOOGLE_API_KEY="您的Google API密钥"
```

通过设置这些变量，工具包的配置模块可以安全地访问您的凭据，而不会在您的代码中暴露它们。我们将在本章后续部分更详细地介绍此配置系统。您的环境现已配置完毕，您可以开始构建了。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5591/978-1-57766-508-2.6](https://doi.org/10.5591/978-1-57766-508-2.6)
  介绍了Transformer架构，这是大多数现代大型语言模型的基础组成部分。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  介绍了生成式智能体，这是一种利用大型语言模型模拟可信人类行为的计算架构。
