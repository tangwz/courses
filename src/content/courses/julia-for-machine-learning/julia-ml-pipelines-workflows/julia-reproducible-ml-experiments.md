---
course: "julia-for-machine-learning"
chapter: "julia-ml-pipelines-workflows"
lesson: "julia-reproducible-ml-experiments"
sourceId: 6894
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-5-julia-ml-pipelines-workflows/julia-reproducible-ml-experiments"
title: "Julia 中可重现机器学习实验的策略"
description: "使用 Julia 环境和版本控制，确保机器学习实验可重现性的最佳实践。"
order: 5
plots: []
sourceHash: "48e94427c1e470dc6a25a7ed1952ce2f67326921571065ad1004e651c11ce876"
sourceCorrections: []
---

随着机器学习 (machine learning)管道复杂度增加，包含多个数据处理步骤和模型训练程序，确保实验可重现变得越来越重要。可重现性意味着你或其他人可以重新运行实验并获得相同或非常相似的结果。这对于验证结果、调试问题以及建立对模型的信任是基础。良好的实践和专用工具的结合可以在 Julia 中帮助实现这一点。

### 使用 Pkg.jl 进行环境管理

Julia 内置的包管理器 `Pkg.jl` 是创建可重现环境的核心部分。每个 Julia 项目都可以有自身独立的包依赖项及其确切版本，这些都被精确记录。这通过两个主要文件进行管理：

- **`Project.toml`**：此文件列出项目的直接依赖项及其兼容版本范围。
- **`Manifest.toml`**：此文件记录了依赖图（包括间接依赖项）中所有包的精确版本，这些版本是在特定项目状态下解析并使用的。它确保任何使用此 `Manifest.toml` 的人都能获得完全相同的包版本。

要创建或使用项目专用环境，你通常在 Julia REPL 中进入项目目录并运行：

```julia
using Pkg
Pkg.activate(".") 
Pkg.instantiate() 
```

`Pkg.activate(".")` 命令告诉 Julia 使用当前目录中定义的环境。然后 `Pkg.instantiate()` 会下载并安装 `Manifest.toml` 中指定的所有包（如果 `Manifest.toml` 不存在或不同步，则根据 `Project.toml` 解析它们）。

将 `Project.toml` 和 `Manifest.toml` 都提交到你的版本控制系统是一种标准做法。这使得协作者（或你未来的自己）能够完全重现项目环境。

### 代码和实验的版本控制

追踪代码、实验配置以及环境文件（`Project.toml`、`Manifest.toml`）的变化是很要紧的。版本控制系统，Git 是应用最广的，对此不可或缺。

定期提交更改可以让你：

- 在出现问题时，回滚到之前的工作状态。
- 了解项目历史以及结果随时间的变化情况。
- 通过合并不同贡献者的更改与他人协作。

在 Git 中使用分支也是管理不同实验的好方法。你可以为每个实验设想创建一个新分支，保持主代码库稳定，同时尝试不同变体。

### 数据版本控制的考量

尽管 `Pkg.jl` 处理代码依赖，但你的模型训练数据也可能变化。为了实现真正的端到端可重现性，特别是当数据集演变时，你需要数据版本控制的策略。

- 对于**小型数据集**，你可以直接将其包含在 Git 仓库中。
- 对于**大型数据集**，这通常不切实际。像数据版本控制（DVC）这样的工具，可以与 Git 协作，追踪大型数据文件而无需将其存储在 Git 仓库中。另外，使用数据文件的校验和（例如 SHA256 哈希）或将数据存储在不可变存储位置（如版本化对象存储）可以帮助确保你使用的正是实验运行时所用的数据集版本。

### 代码组织和模块化

结构良好、模块化的代码本质上更容易理解、调试和重现。强烈建议使用函数和 Julia 模块将你的机器学习 (machine learning)工作流分解为不同的、可管理的部分。

MLJ.jl 管道是本章的一个重点，它们是促进模块化的优秀例子。MLJ 管道中的每一步（例如数据缩放器、编码器、模型）都是一个独立的组件。这使得整个工作流透明且更易于管理，从而有助于可重现性。如果你能清楚地看到每个步骤，就更容易验证和复现。

### 使用种子管理随机性

许多机器学习 (machine learning)算法和过程包含一定程度的随机性：

- 将数据拆分为训练集和测试集。
- 初始化神经网络 (neural network)中的权重 (weight)。
- 随机优化算法。
- 像随机森林或 K 均值聚类这样的算法。

为确保这些随机过程在每次运行代码时产生相同的结果，你必须设置一个**随机种子**。在 Julia 中，你可以使用 `Random` 标准库设置全局随机种子：

```julia
using Random
Random.seed!(123) # 将 123 替换为你选择的整数种子
```

在脚本开头设置种子可以确保任何后续依赖 Julia 默认随机数生成器的操作都将表现出确定性。大多数 MLJ.jl 模型和操作都遵守这个全局种子。为了更细致的控制，某些特定的模型或包可能允许你直接向它们传递种子。始终记录实验所用的种子。

### 日志记录和文档

全面的日志记录和清晰的文档对可重现性很重要。你应该记录：

- **实验参数 (parameter)**：模型的超参数 (hyperparameter)、特征工程步骤、使用的数据子集。
- **性能指标**：验证集和测试集上的准确率、F1 分数、损失值等。
- **环境信息**：Julia 版本、`Pkg.jl` 清单（或重要包及其版本）。
- **使用的随机种子**。
- 实验的**时间戳和持续时间**。

Julia 的标准库 `Logging` 提供 `@info`、`@warn` 和 `@error` 等宏，用于结构化日志记录。

```julia
using Logging

# 例子：配置一个日志器写入文件
global_logger(SimpleLogger(open("experiment_log.txt", "w")))

@info "实验开始..."
@info "随机种子: $(123)"
# ... 运行实验 ...
@info "模型准确率: 0.85"
```

除了自动化日志，人类可读的文档（例如 `README.md` 文件或注释良好的代码），用于说明项目结构、数据源、预处理步骤以及重大决策背后的原因，价值非凡。

### 自动化测试

自动化测试有助于确保机器学习 (machine learning)管道的各个组件（数据预处理函数、特征转换器、模型训练脚本）按预期运行。尽管测试可能不总是保证随机算法产生位对位相同的数值结果，但它们对于捕获由代码更改引入的可能导致先前结果失效的回归或错误很有用。

Julia 通过 `Test` 标准库提供了一个内置测试框架。你可以编写测试来：

- 验证数据清洗函数是否正确处理缺失值或异常值。
- 检查特征转换输出是否具有预期的维度或数据类型。
- 确保模型训练脚本对于少量数据能够无错误地完整运行。
- 确认已保存的模型可以加载并用于预测。

```julia
using Test

# 在测试脚本中，例如 test/runtests.jl
@testset "数据预处理测试" begin
    # 假设 preprocess_data 是你的 MyProject 模块中的一个函数
    # data = MyProject.load_data("sample.csv")
    # processed_data = MyProject.preprocess_data(data)
    # @test size(processed_data, 2) == 10 # 示例断言
    # @test sum(ismissing, processed_data.target_column) == 0
end
```

### 借助 DrWatson.jl 进行科学项目

除这些通用做法外，Julia 生态系统还提供了专用工具来简化可重现的科学项目。在这方面，一个重要的包是 `DrWatson.jl`。它提供了一种标准化但灵活的项目结构，以及一套旨在使你的科学计算更具组织性和可重现性的辅助函数。

使用 `DrWatson.jl` 的主要好处包括：

- **标准化项目结构**：`DrWatson.jl` 帮助你建立一致的目录布局（例如 `scripts/`、`data/`、`plots/`、`results/`）。
- **安全保存和加载**：像 `safesave` 和 `wsave` 这样的函数可以自动将元数据（例如 Git 提交哈希或脚本参数 (parameter)）包含在结果中，从而更容易追溯结果的来源。
- **路径管理**：`projectdir()`、`datadir()`、`scriptsdir()` 等实用程序提供了引用项目文件的方式，无论脚本从何处运行。
- **参数管理**：用于运行不同参数的实验，并根据这些参数系统地命名输出文件的工具。

尽管 `DrWatson.jl` 的完整指南内容很多，但其核心理念是鼓励以可追溯的方式关联代码、数据和结果。例如，你可以用它根据生成脚本和所用参数来命名输出文件，从而确保在相同设置下重新运行实验时，将检索或一致地覆盖之前的结果。

### 可重现性工作流的可视化

下图展示了这些不同组件如何协同工作，以支持 Julia 中可重现的机器学习 (machine learning)实验：

> 概述了在 Julia 中促进机器学习实验可重现性的组成部分。版本控制管理代码和环境文件，这些文件定义了 Julia 执行环境，管道在此环境中运行，最终产生一致的结果。

在你的 Julia 机器学习项目中采用这些可重现性策略，起初可能看似额外的付出，但从长远来看会带来显著回报。这会带来更可靠的研究、更轻松的协作、更快的调试，并最终形成更值得信赖的机器学习系统。通过将 Julia 优秀的包管理与版本控制、细致的种子设置、良好的编码实践以及 `DrWatson.jl` 等工具结合，你可以构建可靠的机器学习工作流。

## 参考资料

- [Pkg.jl Documentation](https://pkgdocs.julialang.org/) — Julia Language Developers (2024)
  Julia包管理器的官方指南，对于创建隔离且可复现的项目环境至关重要。
- [The Turing Way: A Handbook for Reproducible Research](https://the-turing-way.netlify.app/) — The Turing Way Community (2024)
  Publisher: Zenodo; DOI: [10.5281/zenodo.3233853](https://doi.org/10.5281/zenodo.3233853)
  一份关于可复现研究原则和实践的综合开源指南，与机器学习实验高度相关。
- [DrWatson.jl Documentation](https://juliadynamics.github.io/DrWatson.jl/dev/) — Datseris, George (2024)
  DrWatson.jl的官方文档，这是一个旨在简化和标准化可复现科学项目的Julia包。
- [Data Version Control (DVC) Documentation](https://dvc.org/doc) — Iterative.ai (2024)
  数据版本控制（DVC）的官方文档，这是一个用于在ML工作流中与代码一起管理数据和模型版本的流行工具。
- [Random Numbers (Julia Documentation)](https://docs.julialang.org/en/v1/stdlib/Random/) — Julia Language Developers (2024)
  Julia `Random` 标准库的官方文档，详细介绍了如何设置随机种子以确保随机过程的确定性执行。
