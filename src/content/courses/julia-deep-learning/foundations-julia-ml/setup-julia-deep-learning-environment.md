---
course: "julia-deep-learning"
chapter: "foundations-julia-ml"
lesson: "setup-julia-deep-learning-environment"
sourceId: 6873
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/setup-julia-deep-learning-environment"
title: "配置您的Julia深度学习环境"
description: "安装Julia、Flux.jl及其他深度学习开发所需包的分步指南。"
order: 7
plots: []
sourceHash: "07157bd75d64606a2a9a4eee7d410eecddfd757cef40cd70bac0ac7b8f49dbcd"
sourceCorrections: []
---

拥有一个正确配置的开发环境，是高效使用Julia进行深度学习 (deep learning)的必要条件。这个配置确保您拥有Julia编译器、Flux.jl等必需的库，以及编写和管理代码的工具。内容包括Julia的安装、集成开发环境（IDE）的选择，以及示例和项目所需核心包的安装。

### 安装Julia

第一步是在您的系统上安装Julia。Julia语言官方网站（julialang.org）是获取下载链接和Windows、macOS、Linux安装说明的最佳来源。

1. **下载Julia**：前往[julialang.org](https://julialang.org)的“下载”部分。您会找到适用于各种平台的安装程序和二进制文件。
2. **安装方法**：

   - **官方二进制文件**：下载适用于您操作系统的适合的安装程序或压缩包，并按照提供的说明操作。这通常需要将Julia的`bin`目录添加到系统的PATH环境变量中，以便您可以在任何终端或命令提示符下运行Julia。
   - **使用`juliaup`**：一种强烈推荐的方法，特别是如果您预计需要多个Julia版本或希望轻松更新，就是使用`juliaup`。这是一个Julia版本管理器，类似于Rust的`rustup`或Python的`pyenv`。`juliaup`的安装说明可在其[GitHub仓库](https://github.com/JuliaLang/juliaup)中找到。`juliaup`安装完成后，您可以使用 `juliaup add stable` 这样的命令来安装特定的Julia版本（例如，最新稳定版本）。
3. **验证安装**：安装完成后，打开一个新的终端或命令提示符，然后输入：

   ```bash
   julia --version
   ```

   您应该会看到已安装的Julia版本被打印出来，例如 `julia version 1.9.3`。

### 选择和配置IDE

虽然Julia代码可以在任何文本编辑器中编写，但使用具有良好Julia支持的IDE或代码编辑器将显著提升您的工作效率。

- **Visual Studio Code (VS Code)**：这是目前Julia开发最受欢迎且功能丰富的环境。您需要为VS Code安装Julia扩展。

  1. 从[code.visualstudio.com](https://code.visualstudio.com)安装VS Code。
  2. 打开VS Code，前往扩展视图（Ctrl+Shift+X 或 Cmd+Shift+X）。
  3. 搜索“Julia”并安装由`julialang`提供的扩展。
  4. 该扩展提供语法高亮、代码补全（IntelliSense）、集成Julia REPL、调试工具和绘图库集成等功能。它通常会自动检测您的Julia安装。如果不能，您可以在扩展设置中配置Julia可执行文件的路径。
- **Jupyter Notebooks/Lab with IJulia.jl**：对于交互式数据分析、创建教程或分享结果，Jupyter是一个很棒的选择。

  1. 您需要安装带有Jupyter Notebook或JupyterLab的Python。
  2. 在Julia中安装`IJulia`包：打开Julia REPL（在终端中输入`julia`），然后按`]`进入Pkg REPL模式。接着输入：

     ```
     pkg> add IJulia
     ```
  3. 一旦`IJulia`安装完成，您就可以在终端中运行`jupyter notebook`或`jupyter lab`，并且能够创建带有Julia内核的新Notebook。

其他文本编辑器，如Sublime Text、Atom或Vim，也有Julia插件，但VS Code通常为Julia开发提供最全面的体验。

### 安装深度学习 (deep learning)必需的Julia包

Julia的功能通过包进行扩展。内置的包管理器`Pkg`负责这些包的安装、更新和管理。您可以通过其REPL模式或通过编程方式与`Pkg`交互。

要进入`Pkg` REPL模式，请启动Julia并在`julia>`提示符下输入`]`。提示符将变为`pkg>`。

以下是我们将在整个课程中使用的主要包：

- **Flux.jl**：Julia中核心的深度学习库。它提供构建和训练神经网络 (neural network)的工具。
- **Zygote.jl**：一个自动微分包。Flux.jl依赖Zygote来计算梯度，这对模型训练是必要的。（Zygote通常作为Flux的依赖项安装）。
- **CUDA.jl**：用于利用NVIDIA GPU加速计算。如果您有AMD GPU，可以考虑`AMDGPU.jl`，尽管`CUDA.jl`目前在生态系统中具有更广泛的支持。
- **MLUtils.jl**：包含用于机器学习 (machine learning)任务的实用工具，如数据迭代器、批处理和数据分割。
- **DataFrames.jl**：用于处理表格数据，类似于Python中的Pandas。
- **CSV.jl**：用于读取和写入CSV文件。
- **Plots.jl**：一个通用绘图库。您也可以尝试`Makie.jl`以获得更高级或交互式的可视化。
- **BSON.jl**：用于序列化Julia数据结构，包括保存和加载已训练的Flux模型。

您可以在`Pkg` REPL中输入以下命令来安装这些包：

```
pkg> add Flux CUDA MLUtils DataFrames CSV Plots BSON
```

按下回车，`Pkg`将下载并安装这些包及其依赖项。首次安装可能需要几分钟。要返回Julia REPL，请按退格键（Backspace）或Ctrl+C。

### 项目专用环境

为了提高重现性并管理不同项目的依赖项，Julia支持项目专用环境。每个项目都可以有自己的`Project.toml`文件（列出直接依赖项）和`Manifest.toml`文件（列出所有依赖项的确切版本）。

要为新项目创建并激活环境：

1. 为您的项目创建一个新目录，例如 `MyDLProject`。
2. 在终端中进入此目录。
3. 启动Julia。
4. 进入`Pkg` REPL（输入`]`）。
5. 输入`activate .`（点号表示当前目录）：

   ```
   pkg> activate .
   ```

   如果当前目录中不存在`Project.toml`和`Manifest.toml`文件，此命令将创建它们。随后添加的任何包都将特定于此项目环境。

激活环境可以确保您的项目使用一组一致的包版本，从而使您的工作更易于分享和重现。

### 验证您的配置

让我们编写一个小型Julia脚本或在Julia REPL中运行一些命令，以确保核心组件正常运行。

启动Julia（如果您在一个项目环境中，提示符中会显示，例如 `(MyDLProject) julia>`）。

```julia
using Flux
using DataFrames
using MLUtils

println("Flux.jl、DataFrames.jl和MLUtils.jl加载成功！")

# 测试一个简单的Flux层
model = Dense(10, 2) # 一个密集层，将10个输入映射到2个输出
println("成功创建Flux密集层：", model)

# 测试创建一个简单的DataFrame
df = DataFrame(A = 1:3, B = ['x', 'y', 'z'])
println("成功创建DataFrame：")
println(df)
```

如果这些命令无错误运行，那么您的深度学习 (deep learning)基本Julia环境就绪。

### GPU支持验证（可选）

如果您安装了`CUDA.jl`并拥有安装了适当驱动程序和CUDA工具包的NVIDIA GPU，您可以检查Julia是否可以访问GPU：

```julia
using CUDA

if CUDA.functional()
    println("CUDA运行正常。GPU可用且Julia可访问。")
    CUDA.versioninfo() # 打印有关您的CUDA设置和GPU的信息
    # 示例：将一个简单数组移动到GPU
    cpu_array = rand(Float32, 2, 2)
    gpu_array = cu(cpu_array)
    println("成功将数组移动到GPU：", typeof(gpu_array))
else
    println("CUDA无法运行。请检查您的NVIDIA驱动程序和CUDA工具包安装。")
    println("GPU加速将不可用。训练将在CPU上运行。")
end
```

如果CUDA在此阶段无法运行，请不要担心，特别是如果您没有NVIDIA GPU或尚未配置驱动程序和工具包。大多数初始示例在CPU上也能正常运行。第5章将更详细地讲解GPU计算。

随着Julia的安装、IDE的配置以及必要包添加到您的环境中，您现在已充分准备好学习如何使用Julia强大的生态系统来构建和训练深度学习 (deep learning)模型。接下来的章节将在此配置的基础上，向您介绍数据处理和算法实现的实际方面。

## 参考资料

- [Julia Documentation](https://docs.julialang.org/) — The Julia Language (2024)
  Publisher: The Julia Language
  官方且最全面的资源，用于理解 Julia 语言、其安装、Pkg 包管理器和项目环境。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Developers (2025)
  Flux.jl 的官方文档，它是 Julia 中主要的深度学习库，对于构建和训练神经网络至关重要。
- [Julia extension for Visual Studio Code](https://www.julia-vscode.org/) — The Julia Language Developers (2024)
  设置和高效使用推荐的 Julia Visual Studio Code 扩展的官方指南。
