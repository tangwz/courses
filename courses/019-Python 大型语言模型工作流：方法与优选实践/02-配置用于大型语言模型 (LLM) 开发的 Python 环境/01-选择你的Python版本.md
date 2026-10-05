# 选择你的Python版本

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/choosing-python-version)

[返回章节目录](README.md) · [返回课程目录](../README.md)

选择合适的Python版本是环境设置中的主要决策点。尽管Python 3是标准版本，但具体的次要版本（如3.9、3.10、3.11、3.12）可能会对与快速更新的LLM生态系统库的兼容性产生影响。

### Python版本为何重要

你选择的Python版本会影响开发流程的多个方面：

1. **库兼容性：** 这通常是最重要的因素。像LangChain、LlamaIndex、Transformers（来自Hugging Face）、TensorFlow和PyTorch等主要的LLM库都明确规定了它们支持的Python版本。使用不受支持的版本可能导致安装失败或意外的运行时错误。通常，库会要求一个最低版本（例如Python 3.8+），并且可能不会立即支持最新的发布版本。
2. **语言特性：** 新的Python版本会引入有用的语言特性。例如，结构化模式匹配（3.10中引入的`match`语句）或类型提示的改进可以使你的代码更简洁、更易于维护。尽管对于LLM工作流程并非严格必需，但应用现代特性可以改进开发。
3. **性能：** Python的核心开发者不断致力于性能改进。新版本通常比旧版本执行代码更快，这对于有时在数据预处理或模型交互循环中出现的计算密集型任务可能很重要。
4. **安全与维护：** 旧的Python版本最终会达到生命周期结束（EOL），并停止接收安全更新。坚持使用积极维护的版本对于安全和项目的长期稳定性很重要。

### 检查你的版本并做出选择

在继续之前，你可以通过打开终端或命令提示符并运行以下命令来检查当前安装的默认Python版本：

```bash
python --version
# 或者有时
python3 --version
```

如果你有多个Python安装，请确保你检查的是你打算用于LLM项目的那个。

那么，你应该选择哪个版本呢？这里有一个实用的方法：

- **查阅库文档：** 查阅你计划使用的主要库（例如LangChain、LlamaIndex、OpenAI的Python客户端）的文档。它们通常会清楚地说明其支持的Python版本。
- **选择近期稳定版本：** 通常来说，请选择LLM生态系统广泛支持的、近期且稳定的Python版本之一。截至2023年末/2024年初，Python 3.9、3.10和3.11等版本通常是稳妥的选择，它们在现代特性和广泛的库兼容性之间提供了良好的平衡。Python 3.12正在获得越来越多的支持。
- **避免使用EOL版本：** 不要使用不再接收安全更新的版本（例如Python 3.7及更早版本）。
- **对最新版本保持谨慎：** 尽管诱人，但立即跳到最新的Python发布版本（例如最近的.0发布或测试版）可能会导致兼容性问题，因为库需要时间来适应。在Python主要版本发布后，通常更明智的做法是等待几个月，让生态系统跟上。

下图说明了Python版本典型的状态演进。

> Python版本生命周期阶段。LLM开发应选择处于“稳定发布版”或“仅安全修复”早期阶段的版本。

本课程推荐使用**Python 3.9或更高版本**，以确保与我们将使用的示例和库兼容。如果你已经安装了略微不同的版本（例如3.10、3.11），它很可能也能正常工作。最重要的是保持一致性，这将在接下来讨论虚拟环境时提及。

## 参考资料

- [Python Release Schedule](https://www.python.org/downloads/) — Python Software Foundation (Current)
  Publisher: Python Software Foundation
  提供Python官方版本发布信息，包括生命周期结束（EOL）日期和维护状态，这对于项目的长期稳定性和安全性至关重要。
- [What's New In Python 3.12](https://docs.python.org/3/whatsnew/3.12.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  介绍Python 3.12中新增的语言特性、性能改进和变化，有助于了解Python的最新功能。
- [Python Packaging User Guide](https://packaging.python.org/en/latest/) — Python Packaging Authority (Current)
  一份权威的Python包和环境管理指南，为处理开发工作流中的库兼容性和版本控制问题提供背景知识。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广受认可的Python数据科学书籍，涵盖环境设置和依赖管理等实用方面，与数据密集型和机器学习应用相关。

---

[上一节](../01-Python%20LLM%20%E5%B7%A5%E4%BD%9C%E6%B5%81%E5%85%A5%E9%97%A8/05-%E8%AF%BE%E7%A8%8B%E7%BB%93%E6%9E%84%E4%B8%8E%E7%9B%AE%E6%A0%87.md) · [下一节](02-%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%20%28venv%2C%20conda%29.md)
