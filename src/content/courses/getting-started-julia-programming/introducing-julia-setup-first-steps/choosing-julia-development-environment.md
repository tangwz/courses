---
course: "getting-started-julia-programming"
chapter: "introducing-julia-setup-first-steps"
lesson: "choosing-julia-development-environment"
sourceId: 6762
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/choosing-julia-development-environment"
title: "选择您的Julia开发环境"
description: "了解流行的Julia开发集成开发环境（IDE）和文本编辑器，例如VS Code和Juno。"
order: 6
plots: []
sourceHash: "3fcf22ea3cb7aac85337bbb6d770a0a91d29e146156137ac8f442b0e2f890668"
sourceCorrections: []
---

Julia REPL（读取-评估-打印循环）是进行快速测试和交互式试用的极佳工具，但当您开始编写不止几行的程序时，会希望有一个更具结构性和功能齐全的环境。这时，选择一个好的开发设置变得很重要。它能大幅提升您的效率，并让编程变得更愉快。

### 了解您的选择：文本编辑器与IDE

大体上，您有两大类工具用于编写Julia代码：

1. **文本编辑器**：这些是用于编写和编辑纯文本（包括源代码）的程序。现代文本编辑器通常支持语法高亮（对代码的不同部分，如关键字、变量和注释进行着色，以提高可读性），并且有些可以通过插件进行扩展，以增加更多编程专用功能。例子包括Sublime Text、Atom、Notepad++（Windows）或gedit（Linux）。更高级、高度可配置的编辑器，如Vim和Emacs也属于此类，尽管它们通常学习曲线更陡峭。
2. **集成开发环境（IDE）**：IDE将文本编辑器与一套其他工具捆绑在一起，旨在简化开发过程。常见的IDE功能包括：

   - **代码补全**：在您输入时，提供函数、变量和关键字的建议。
   - **调试器**：一种工具，可让您逐行执行代码、检查变量并查找错误。
   - **集成REPL/终端**：直接在环境中访问命令行或Julia的REPL。
   - **项目管理**：用于组织文件和管理项目依赖的工具。
   - **版本控制集成**：通常内置对Git等系统的支持。

对于Julia来说，界限可能变得模糊，因为带有合适扩展的强大文本编辑器也能提供类似IDE的体验。

### 首选方案：带Julia扩展的Visual Studio Code

对于大多数Julia用户，特别是初学者而言，带官方**Julia扩展**的\*\*Visual Studio Code (VS Code)\*\*是受推荐最广泛且功能完善的开发环境。

- **为何选择VS Code？** 它是由微软开发的一款免费、开源且非常流行的代码编辑器，可在Windows、macOS和Linux上运行。其优势在于其扩展生态系统，为多种语言和工具增加了支持。
- **Julia扩展**：此扩展由JuliaLang开发者维护，可将VS Code转变为一个强大的Julia IDE。主要功能包括：
  - **出色的REPL集成**：您可以将代码直接从编辑器运行到集成的Julia REPL中，发送代码行或代码块，并立即查看结果。
  - **语法高亮和代码补全**：让您的代码更易于阅读和编写。
  - **行内结果**：直接在编辑器中查看代码执行结果。
  - **绘图面板**：如果您正在进行数据可视化，图表可以直接显示在VS Code内。
  - **调试工具**：图形化调试器，帮助您排除程序故障。
  - **工作区查看器**：检查当前Julia会话中的变量、已定义函数和模块。
  - **文档浏览器**：轻松访问Julia文档。
  - **代码规范检查集成**：在您输入时，帮助捕捉常见错误和风格问题。

**设置方法：**

1. 从VS Code官方网站下载并安装VS Code。
2. 打开VS Code。
3. 前往“扩展”视图（通常是左侧边栏中看起来像堆叠方块的图标）。
4. 搜索“Julia”并安装由`julialang`提供的扩展。

### 其他环境和工具

虽然VS Code是首要推荐，但了解您可能遇到的其他选择也很有用：

- **Juno（历史说明）**：Juno曾是一个流行的Julia独立IDE，基于Atom编辑器构建。虽然Atom的开发已停止，但Juno核心团队在Julia专用功能方面的精力已转移到VS Code扩展上。因此，Juno许多出色的功能和理念现在已体现在VS Code Julia扩展中。您可能会在较旧的教程或文章中看到Juno被提及。
- **基本文本编辑器 + 终端**：如果您喜欢极简设置，当然可以在任何纯文本编辑器（如Sublime Text、Gedit、Notepad++）中编写Julia脚本（`.jl`文件），然后使用`julia yourscript.jl`命令从系统终端运行它们。这种方法让您有很大的控制权，但缺少IDE的集成便利性。
- **Notebook界面**：对于交互式工作、数据试用以及创建混合代码、文本和输出的文档，Notebook界面非常流行。

  - **Jupyter Notebook/JupyterLab**：一个广泛使用的基于网络的交互式计算平台。您可以通过**IJulia.jl**包在Jupyter中使用Julia，该包提供Julia内核。这非常适合数据分析和科学计算。
  - **Pluto.jl**：一个专为Julia设计的响应式Notebook环境。“响应式”表示当您更改变量或函数时，Pluto会自动重新运行依赖于它的单元格。这带来了非常流畅且富有启发性的编码体验。Pluto还在Notebook内部处理包管理。

此图阐释了IDE如何集成各种工具来辅助程序员：

> IDE提供了一个集中位置，用于编写代码、使用Julia解释器运行代码、调试问题以及管理项目文件。

### 做出您的选择

对于本课程以及作为Julia开发的普遍起点，**我们强烈推荐使用带Julia扩展的VS Code**。它在功能、易用性和社区支持之间提供了最佳平衡。

随着您经验的增加，您可能会根据自己的特定需求或偏好尝试其他工具。例如，Pluto.jl Notebook非常适合某些类型的交互式工作或教育目的。要点是，从一个能帮助您学习并提高效率、不增加不必要复杂性的环境开始。

不必过于担心立即做出“完美”选择。您在编写Julia代码中学到的技能是可迁移的，您总是可以在以后切换或尝试不同的环境。目前，熟悉VS Code将为您提供一个坚实的基础。在下一节中，我们将指导您编写并运行您的第一个Julia脚本，很可能就是使用此设置。

## 参考资料

- [Julia Documentation](https://docs.julialang.org/en/v1/) — JuliaLang Developers (2024)
  提供Julia编程语言、其REPL及核心组件的全面文档。
- [Visual Studio Code Documentation](https://code.visualstudio.com/docs) — Microsoft (2024)
  提供Visual Studio Code（推荐代码编辑器）的安装、配置和使用的官方指南。
- [Julia for VS Code Documentation](https://www.julia-vscode.org/) — JuliaLang Developers (2024)
  Visual Studio Code官方Julia扩展的专用资源，详细说明了安装、功能和使用方法。
- [Project Jupyter](https://jupyter.org/) — Project Jupyter (2024)
  介绍Jupyter交互式计算平台及其通过IJulia.jl内核与Julia的集成。
- [Pluto.jl: Reactive Notebooks for Julia](https://plutojl.org/) — Fons van der Plas and Pluto.jl Contributors (2024)
  Pluto.jl的官方文档，一个专为Julia设计的响应式交互式笔记本环境。
