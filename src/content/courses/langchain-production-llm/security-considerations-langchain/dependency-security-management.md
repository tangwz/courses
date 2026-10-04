---
course: "langchain-production-llm"
chapter: "security-considerations-langchain"
lesson: "dependency-security-management"
sourceId: 3938
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-8-security-considerations-langchain/dependency-security-management"
title: "依赖项安全管理"
description: "扫描和管理LangChain项目中使用的第三方库中的安全弱点。"
order: 7
plots: []
sourceHash: "480c8fbbfa426cbda5e35d32f539196ae931b5bf3d5bfaed1ba0e1af0e5b11da"
sourceCorrections: []
---

现代应用，包括用LangChain构建的那些，很少是单一的。它们是通过组合许多第三方库和包构建的，每个都可能带来自己的一套依赖项。尽管这加速了开发，但它也引出了一个重要的安全考量：从这些外部组件继承的弱点。管理项目依赖项的安全性是构建有弹性、值得信赖的LangChain应用的一个必要方面。

你的LangChain项目不仅依赖于`langchain`本身，还依赖于它使用的库（例如`langchain-core`、`langchain-community`、`langsmith`）、用于特定LLM提供商的库（如`openai`、`anthropic`）、向量 (vector)数据库（`pinecone`、`chromadb`）、数据加载器（`unstructured`）、API交互（`requests`）以及核心Python环境。任何一个依赖项中的弱点，即使是传递性依赖（一个依赖项的依赖项），都可能危及你的整个应用。

与不安全依赖项相关的常见风险有：

- **远程代码执行 (RCE)：** 某个弱点可能使攻击者能在你的应用环境中运行任意代码。
- **数据泄露：** 你的应用处理或存储的敏感数据可能会被泄露。
- **拒绝服务 (DoS)：** 针对有弱点依赖项的恶意输入可能使你的应用崩溃。
- **权限提升：** 攻击者可能在托管你的应用的系统中获得更高权限。

### 识别和评估弱点

第一步是了解你的项目使用了哪些依赖项。这通常通过`requirements.txt` (pip) 或 `pyproject.toml` (Poetry, PDM) 等文件进行管理。但是，仅仅列出直接依赖项还不够；你需要全面查看整个依赖项树，包括传递性依赖项。

有些工具可以自动识别你的依赖项中已知的弱点：

1. **`pip-audit`：** 这是Python打包管理机构 (PyPA) 的一个工具，它根据弱点数据库（主要是Python打包建议数据库 - PyPI）检查你已安装的环境或需求文件。你可以这样运行它：

   ```bash
   # 检查当前环境
   pip-audit

   # 检查需求文件
   pip-audit -r requirements.txt
   ```
2. **`safety`：** 另一个常用的命令行工具，它根据精心维护的已知安全弱点数据库检查已安装的依赖项。

   ```bash
   # 扫描当前环境
   safety check

   # 扫描需求文件
   safety check -r requirements.txt
   ```
3. **平台专用工具：** GitHub (Dependabot)、GitLab (依赖项扫描) 等服务以及商业产品（Snyk、Sonatype Nexus Lifecycle）提供集成化的依赖项扫描。这些通常直接集成到你的源代码仓库和CI/CD流水线中，提供自动警报，有时甚至自动创建拉取请求来修正弱点。

这些工具将你项目中列出的库的特定版本与常见的弱点和暴露 (CVE) 列表以及平台专用建议等数据库进行比较。它们会报告查到的弱点，通常包括严重程度（例如：致命、高、中、低）以及关于弱点详情和可能修正方法的链接。

### 依赖项安全管理方案

积极主动的管理对于降低依赖项相关风险是必要的。

1. **固定依赖项和使用锁定文件：**
   避免在你的主要需求文件（`requirements.txt`、`pyproject.toml`）中使用如`>=1.0.0`这样不限定的N版本说明符。尽管这方便获取最新更新，但未经明确审查，它可能引入意料之外的破坏性改变或新的弱点。相反，应固定特定版本（例如，`langchain==0.3.0`）。
   另外，使用锁定文件（`requirements.lock`、`poetry.lock`、`pdm.lock`）。这些文件记录了安装期间解析的*所有*依赖项（包括传递性依赖项）的精确版本。将锁定文件提交到你的仓库可以确保每个开发人员和部署环境都使用完全相同的依赖项集，从而实现可复现的构建并防止依赖项树出现意外变动。`pip-tools`（使用`pip-compile`）之类的工具或Poetry和PDM等包管理器会自动生成并管理这些锁定文件。
2. **定期扫描和更新：**
   依赖项安全并非一次性检查。新的弱点会持续被查到。将依赖项扫描整合到你的开发流程中：

   - **本地开发：** 鼓励开发人员在本地运行`pip-audit`之类的工具。
   - **CI/CD流水线：** 在你的持续集成流水线中添加一个专门的步骤，以便在每次提交或拉取请求时自动扫描依赖项。如果查到致命或高严重性弱点，则使构建失败。
   - **计划性扫描：** 配置自动化工具（如Dependabot）以定期重新扫描你的主分支，并提醒你已固定依赖项中新查到的弱点。
     建立一个流程来审查弱点报告并更新依赖项。这包括评估弱点在你特定应用环境中的风险，并测试更新后的依赖项以确保其不会引入问题。
3. **减少依赖项占用：**
   每增加一个依赖项都会增加潜在的攻击面。定期审查项目的依赖项，并移除不再需要的。在选择提供类似功能的库时，考虑它们的依赖项树——一个拥有更少、维护良好的依赖项的库，从安全角度来看可能更具优势。
4. **审查依赖项：**
   在添加新依赖项之前，进行一些必要的检查：

   - 该库是否积极维护？检查提交历史和发布频率。
   - 它是否有安全问题记录？检查弱点数据库和项目的议题追踪器。
   - 它是否被社群广泛使用和信任？
   - 它自己的依赖项有哪些？

### LangChain专用说明

LangChain本身是一个快速发展的框架。保持你的`langchain`、`langchain-core`及相关包更新到合理的新版本通常是明智的，因为更新通常包含错误修正、性能提升和安全补丁。密切留意发布说明。

另外，考虑特定LangChain集成引入的依赖项。如果你使用Pinecone向量 (vector)数据库，你依赖于`pinecone`。如果你使用Anthropic模型，你依赖于`anthropic`库。这些底层SDK和驱动程序的安全性与LangChain本身一样必要。对这些集成专用依赖项应用相同的扫描和管理原则。

### 整合到CI/CD中

在你的CI/CD流水线中自动化依赖项检查提供了一个持续的安全保障。这是一个使用GitHub Actions和`pip-audit`的例子：

```yaml
name: 安全检查

on: [push, pull_request]

jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: 设置 Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11' # 使用你项目的Python版本

    - name: 安装依赖项
      run: |
        python -m pip install --upgrade pip
        pip install pip-audit
        # 假设你使用 requirements.txt 和锁定文件机制
        pip install -r requirements.txt # 或者使用你的锁定文件安装（例如 poetry install）

    - name: 运行 pip-audit
      run: |
        # 如果查到任何弱点则失败。根据需要调整参数。
        # 例如，使用 --ignore-vuln ID 忽略特定弱点
        # 或者在哈希检查模式下使用 --require-hashes 进行更严格的检查。
        pip-audit
```

> 运行`pip-audit`的GitHub Actions工作流程步骤示例。

此步骤确保引入有弱点依赖项的代码更改在合并或部署前被标记 (token)出来。

有效管理依赖项安全需要持续的警惕。通过实施自动化扫描、使用锁定文件维护可复现的环境、定期更新软件包并减少项目依赖项占用，你可以大大提升生产LangChain应用的安全状态。

## 参考资料

- [pip-audit - The PyPA-recommended tool for auditing Python environments for known vulnerabilities](https://pypa.github.io/pip-audit/) — Python Packaging Authority (PyPA) (2024)
  `pip-audit` 的官方文档，详细说明其用于识别 Python 依赖项中已知漏洞的用法。
- [OWASP Software Supply Chain Security Project](https://github.com/OWASP/Software-Component-Verification-Standard) — OWASP Foundation (2024)
  Publisher: OWASP Foundation
  一个关于理解和减轻整个软件供应链（包括第三方依赖项）风险的综合资源。
- [Security audits](https://packaging.python.org/en/latest/guides/tool-recommendations/#security-audits) — Python Packaging Authority (PyPA) (2024)
  一份关于理解 Python 打包生态系统中安全审计工具和实践的指南。
