---
course: "python-for-beginners"
chapter: "reusing-code-modules-packages"
lesson: "python-pip-install"
sourceId: 2360
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/python-pip-install"
title: "使用pip查找和安装外部包"
description: "学习如何使用`pip`这个Python的标准包安装器，从Python包索引（PyPI）查找并安装第三方库。"
order: 5
plots: []
sourceHash: "4743eee469ceecce37f86280cd2cffa2ca6b2150ec17e42989b2dbf27ca23abd"
sourceCorrections: []
---

Python标准库为常见任务提供了丰富的内置模块，但Python生态系统更为广泛。成千上万的开发者和组织贡献了专门的库和框架，它们处理从Web开发、数据分析到机器学习 (machine learning)和游戏制作等各种事务。这些外部代码集合被称为第三方包。

但你如何在自己的项目中找到并使用这些包呢？这时，Python包索引（PyPI）和`pip`工具就发挥作用了。

### Python包索引（PyPI）

可以将Python包索引（通常称为PyPI，发音为“派-P-爱”）看作Python软件的官方中央存储库。它就像一个专门为Python库而设的应用商店。开发者可以将他们的包上传到PyPI，供其他人下载和使用。PyPI托管着数十万个项目，为各种编程问题提供了解决方案。

你可以直接通过其网站[pypi.org](https://pypi.org)浏览PyPI。它是查找可能有助于你特定需求的包的重要资源。

### `pip`介绍：包安装器

`pip`是Python的标准包管理器。它是一个命令行工具，允许你安装、更新和移除PyPI上的Python包。当你安装最新版本的Python时，`pip`通常会自动包含在内。它的主要作用是从PyPI获取包文件，并将它们安装到你的Python环境中，这样你的脚本就可以像导入标准库模块一样导入它们了。

### 检查`pip`安装情况

在安装包之前，最好检查一下`pip`是否已安装并被你的系统识别。打开你的终端或命令提示符：

- **Windows：** 搜索 `cmd` 或 `PowerShell`。
- **macOS：** 搜索 `终端`（通常在“应用程序”>“实用工具”中）。
- **Linux：** 通常 `Ctrl+Alt+T` 会打开终端，或者在你的应用程序菜单中找到它。

打开终端后，输入以下命令之一并按回车键：

```bash
pip --version
```

或者，如果上述命令不起作用（可能由于存在多个Python版本），请尝试：

```bash
python -m pip --version
```

如果`pip`安装正确，你将看到显示`pip`版本及其位置的输出，类似于此（确切的版本和路径会有所不同）：

```
pip 23.3.1 from /path/to/your/python/lib/python3.11/site-packages/pip (python 3.11)
```

如果出现“command not found”之类的错误消息，你可能需要重新查看（第一章中的）Python安装步骤，或者查阅Python关于安装`pip`的文档。

### 在PyPI上查找包

除了浏览PyPI网站，有时你也可以使用`pip`进行搜索，尽管网站通常更用户友好。网站提供了描述、文档链接、版本历史和使用统计，这些都有助于你评估一个包是否合适且维护良好。

选择包时，请考虑：

- 它是否能解决你的特定问题？
- 它是否活跃维护（检查发布日期）？
- 它是否有良好文档？
- 它是否被广泛使用（通常通过下载统计数据反映）？

### 使用`pip`安装包

一旦你确定了要使用的包，安装它就很直接了。基本命令格式是：

```bash
pip install <package_name>
```

将`<package_name>`替换为PyPI上列出的包的实际名称。我们来尝试安装一个名为`colorama`的简单包，它使在终端应用程序中添加彩色文本输出变得容易。

在你的终端中，运行：

```bash
pip install colorama
```

`pip`将连接到PyPI，找到`colorama`包，下载它（以及`colorama`依赖的任何其他包，称为依赖项），并将它们安装到你的Python环境中。你将看到详细描述此过程的输出。

### 使用已安装的包

安装完成后，该包就可以在你的Python脚本中使用了。以下是你使用`colorama`的方式：

```python
# 从 colorama 包导入特定项
from colorama import Fore, Back, Style, init

# 初始化 colorama（Windows 上需要）
init(autoreset=True)

print(Fore.RED + '这段文字是红色的')
print(Fore.GREEN + Back.YELLOW + '这是黄色背景上的绿色文字')
print(Style.BRIGHT + Fore.BLUE + '这段文字是亮蓝色的')
print('这段文字已恢复默认颜色。')
```

将此代码保存为Python文件（例如，`color_test.py`），并从你的终端运行它（`python color_test.py`）。你应该会看到输出显示为不同的颜色！

### 包管理

`pip`提供了其他几个有用的命令：

- **列出已安装的包：** 查看当前环境中所有已安装的包：

pip list
\`\`\`

- **显示包详情：** 获取有关特定已安装包的更多信息：

pip show <package\_name>
\`\`\`
示例：`pip show colorama`

- **升级包：** 将已安装的包更新到最新版本：

pip install --upgrade <package\_name>
\`\`\`
示例：`pip install --upgrade colorama`

- **安装特定版本：** 如果你需要某个特定版本的包：

pip install <package\_name>==<version\_number>
\`\`\`
示例：`pip install colorama==0.4.4`

- **卸载包：** 移除不再需要的包：

pip uninstall <package\_name>
\`\`\`
示例：`pip uninstall colorama`（它会要求确认。）

### 关于虚拟环境的说明

当你开始处理不同项目时，你可能会发现项目A需要某个库的1.0版本，而项目B需要2.0版本。将包直接安装到你的主Python安装环境中（就像我们上面所做的那样）可能会导致冲突。

为了解决这个问题，Python开发者使用**虚拟环境**。虚拟环境是一个隔离的目录，其中包含特定的Python解释器及其自己的一套已安装包。这使得每个项目都可以拥有自己独立的依赖项，而不会相互干扰。

Python包含一个名为`venv`的内置模块，用于创建虚拟环境。虽然创建和管理虚拟环境的详细步骤不在本介绍部分讲解范围之内，但了解它们的存在并被认为是几乎所有Python项目的最佳实践很重要。我们强烈建议你在后续学习中学习和使用虚拟环境。

使用`pip`和PyPI大大扩展了你作为Python程序员可用的能力。通过借助更广泛社区的工作成果，你可以比从头开始编写一切要快得多地构建更复杂和强大的应用程序。

## 参考资料

- [The Python Package Index](https://pypi.org/) — Python Packaging Authority (2023)
  Python 软件的官方中心仓库，用于浏览和发现包。
- [pip documentation](https://pip.pypa.io/en/stable/) — pip developers (2023)
  `pip` 包安装程序的官方文档，提供详细的命令参考和使用指南。
- [Python Packaging User Guide](https://packaging.python.org/en/latest/) — Python Packaging Authority (2023)
  Publisher: Python Packaging Authority
  Python 打包的综合指南，涵盖 `pip`、PyPI 和环境管理的良好实践。
- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  Python 内置 `venv` 模块的官方文档，用于创建独立的虚拟环境。
- [An Introduction to pip and Python Virtual Environments](https://realpython.com/what-is-pip-env-venv/) — Martin Heinz (2023)
  Publisher: Real Python
  来自 Real Python 的实用教程，解释了 `pip` 和使用 Python 虚拟环境的好处。
