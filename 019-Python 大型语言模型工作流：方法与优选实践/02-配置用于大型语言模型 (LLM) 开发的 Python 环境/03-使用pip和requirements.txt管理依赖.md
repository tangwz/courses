# 使用pip和requirements.txt管理依赖

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/managing-dependencies-pip)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虚拟环境为项目依赖提供了一个独立的、隔离的空间。在此环境中，安装必要的Python包是一项基本任务。这个过程主要依赖于Python的标准包安装器`pip`。有效管理这些依赖对于创建可重现和可维护的LLM应用很重要。

### 使用pip安装包

`pip`是一个命令行工具，它从Python包索引（PyPI）获取包，并将其安装到你当前的环境中。最基本的用法很简单。例如，要安装常用的`requests`库，你可以运行：

```bash
pip install requests
```

此命令会下载`requests`包及其依赖的最新可用版本并进行安装。

在LLM开发中，你通常会使用快速变化的库，或者这些库对LLM提供商API或其他工具（如LangChain或LlamaIndex）有特定的兼容性要求。为了确保不同开发设置和部署之间的一致性，通常需要安装特定版本的包。你可以使用`==`运算符来完成此操作：

```bash
# 安装特定版本的OpenAI库
pip install openai==1.3.5

# 安装最低版本（对库有用，对应用则不那么常用）
pip install langchain>=0.0.350
```

使用`==`将依赖固定到确切的版本，从而最大限度地提高可重现性。这通常是应用开发（包括LLM工作流）的推荐做法，因为它能避免上游库更新导致的不预期行为。

要查看当前环境中安装了哪些包及其版本，你可以使用：

```bash
pip list
```

或者，对于专门为requirements文件设计的格式：

```bash
pip freeze
```

### 使用requirements.txt声明依赖

手动使用`pip install`安装每个包适用于小型实验，但对于实际项目来说，很快就会变得繁琐且容易出错。你如何与协作者共享你的项目？如何确保你的部署环境具有完全相同的依赖？标准解决方案是`requirements.txt`文件。

这是一个简单的文本文件，通常放在项目的根目录中，它列出了所有所需的包，通常每行一个，并且经常带有特定的版本号。

一个LLM项目的`requirements.txt`示例可能如下所示：

```text
# requirements.txt

langchain==0.1.0
openai==1.6.1
python-dotenv==1.0.0
requests==2.31.0
fastapi==0.108.0
uvicorn==0.25.0
chromadb==0.4.22 # 向量存储客户端示例
```

请注意使用`==`来固定版本。这可以确保任何使用此文件设置项目的人都将获得你在开发期间使用的确切版本，从而避免“在我机器上能跑”的问题。可以使用`#`符号添加注释。

### 生成和使用requirements.txt

虽然你可以手动创建`requirements.txt`，但从你的工作环境生成它是一种常见做法。安装所有必需的包后，你可以使用`pip freeze`捕获当前状态：

```bash
pip freeze > requirements.txt
```

此命令将`pip freeze`的输出（列出所有已安装的包及其确切版本）重定向到`requirements.txt`文件中。

**注意：** `pip freeze`会捕获环境中*所有*已安装的包，也包含你的直接依赖的依赖。这有时可能导致`requirements.txt`文件过长。对于大多数应用，这是可接受的，甚至对于最大程度的可重现性来说是更好的。但是，请注意它也会锁定子依赖。

一旦你有了`requirements.txt`文件（无论是生成的还是手动创建的），在其他地方设置项目环境就变成了一个简单的命令。在另一台机器上或在部署脚本中创建并激活新的虚拟环境后，只需运行：

```bash
pip install -r requirements.txt
```

`pip`将读取该文件并安装所有列出的包及其指定版本。这是复制Python项目环境的标准方法。

### 依赖管理的好做法

1. **激活环境：** 务必确保你的项目虚拟环境在安装包或生成`requirements.txt`*之前*处于激活状态。
2. **版本固定：** 在应用的`requirements.txt`中使用`==`来指定特定版本，以确保可重现性。
3. **版本控制：** 将你的`requirements.txt`文件提交到版本控制系统（如Git）。这会将你的依赖历史与代码一起跟踪。
4. **定期更新：** 定期检查并更新你的依赖（`pip install --upgrade package_name`，然后进行全面测试）并重新生成`requirements.txt`。这有助于整合安全补丁和新功能，但需要测试以确保兼容性。
5. **纯净环境：** 如果可能，从一个只包含项目所需依赖的纯净环境生成`requirements.txt`，以避免包含无关的包。

通过系统地使用`pip`和`requirements.txt`，你为管理LLM应用所依赖的库建立了一个可靠的体系，使协作和部署显著更顺畅。

## 参考资料

- [pip documentation](https://pip.pypa.io/en/stable/) — The pip developers (2024)
  Python包安装程序的官方文档，涵盖所有命令和使用细节。
- [Installing packages](https://packaging.python.org/en/latest/guides/installing-packages/) — Python Packaging Authority (2025)
  Publisher: Python Packaging Authority
  来自Python打包用户指南的权威资料，关于使用pip安装包。
- [Managing application dependencies](https://packaging.python.org/tutorials/managing-dependencies/) — Python Packaging Authority (2024)
  Publisher: Python Packaging Authority
  一份关于Python项目依赖管理的指南，侧重于使用`requirements.txt`实现可重复性。

---

[上一节](02-%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%20%28venv%2C%20conda%29.md) · [下一节](04-%E6%A0%B8%E5%BF%83%E5%BA%93%E5%AE%89%E8%A3%85.md)
