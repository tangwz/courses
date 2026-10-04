---
course: "julia-deep-learning"
chapter: "foundations-julia-ml"
lesson: "julia-ml-ecosystem-overview"
sourceId: 6872
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/julia-ml-ecosystem-overview"
title: "Julia 机器学习包集合概述"
description: "了解 Julia 机器学习的整体情况，明确深度学习在其内部的位置。"
order: 6
plots: []
sourceHash: "fcb8b0eb2580792807c04d791e1a10c700ee631a4b667f1904b4988624d87e13"
sourceCorrections: []
---

Julia 在机器学习 (machine learning)和深度学习 (deep learning)方面的优势不只在于其核心语言功能。一个活跃且不断壮大的包集合提供了构建复杂模型、管理数据和加速计算所需的专业工具。了解这个包集合有助于您选择可用的选项，并认识到不同组成部分如何共同构成一个完整的深度学习流程。提供了一张导图，将深度学习工具置于 Julia 机器学习能力的整体背景中。

Julia 机器学习包集合的核心是其鼓励可组合性的设计理念。Julia 包通常设计得专注且能与其他包良好协作，这使得开发者可以为流程的每个部分选择最合适的工具。这与一些其他语言中的单体框架不同，提供了更大的灵活性。

### Julia 机器学习 (machine learning)包集合的核心组成部分

该包集合可大致分为几个部分，每个部分都有重要的包：

1. **数据处理与操作：**
   在训练任何模型之前，数据需要被加载、清洗和转换。

   - `DataFrames.jl`: Julia 中处理表格数据的标准，类似于 Python 的 Pandas。它提供高效的数据结构和丰富的操作集，用于数据处理。
   - `CSV.jl`: 一个快速灵活的包，用于读写 CSV 文件。
   - `MLUtils.jl`: 为机器学习任务提供实用工具，包括数据迭代、批处理、数据集分割以及其他在将数据送入模型前通常需要的预处理步骤。在为深度学习 (deep learning)框架准备数据时，这尤其有用。
2. **数值计算与统计：**
   机器学习，特别是深度学习，高度依赖数值操作。

   - `LinearAlgebra` (标准库): Julia 内置的线性代数功能广泛且性能好，是许多机器学习算法的核心。
   - `Statistics` (标准库): 提供基本的统计函数。
   - `Distributions.jl`: 提供多种概率分布和统计建模工具。
   - `StatsBase.jl`: 包含更高级的统计工具，包括抽样、计数和排序。
3. **自动微分 (AD)：**
   正如前一节所述，自动微分对于通过自动计算梯度来训练神经网络 (neural network)非常重要。Julia 有几个优秀的自动微分包。

   - `Zygote.jl`: 一个流行的源到源自动微分系统，特别适合深度学习。它以其灵活性、能够对 Julia 代码（包括控制流）的很大一部分进行微分以及与 `Flux.jl` 的集成而闻名。
   - `ForwardDiff.jl`: 实现前向模式自动微分，对于输入数量少而输出数量多的函数来说很高效。
   - `ReverseDiff.jl`: 提供反向模式自动微分，对于输入多而输出少的函数通常更高效，这在神经网络训练中很常见。
4. **通用机器学习框架：**
   对于经典机器学习任务，Julia 提供全面的框架。

   - `MLJ.jl` (Machine Learning Julia): 一个用于各种机器学习任务的统一框架。它为来自不同 Julia 包的许多模型（从线性回归、决策树到聚类和降维）提供通用接口。`MLJ.jl` 侧重于模型组合、调优和评估，为完整的机器学习流程提供高级 API。虽然本课程侧重于深度学习，但 `MLJ.jl` 是 Julia 机器学习发展中的重要一环。
5. **深度学习框架：**
   为了构建和训练神经网络，可以使用专门的库。

   - `Flux.jl`: 本课程的主要关注点。`Flux.jl` 是一个极简且灵活的深度学习库。它完全用 Julia 编写，并与语言的特性深度集成，包括其强大的类型系统和多重分派。`Flux.jl` 旨在感觉像是 Julia 的自然延伸，便于定制和研究。它使用 `Zygote.jl` 来实现其自动微分功能。
   - `Knet.jl`: Julia 中另一个成熟的深度学习框架，由 Koç 大学开发。它也提供自动微分 (AutoGrad) 和 GPU 操作支持。
6. **GPU 计算：**
   训练大型深度学习模型通常需要 GPU 的计算能力。

   - `CUDA.jl`: 使用 Julia 编程 NVIDIA GPU 的主要包。它允许 Julia 代码在支持 CUDA 的 GPU 上运行，为数组操作和模型训练提供显著加速。`Flux.jl` 与 `CUDA.jl` 集成以实现 GPU 加速。
   - `AMDGPU.jl`: 通过 ROCm 为 AMD GPU 提供支持。虽然 `CUDA.jl` 目前在深度学习包集合中更常用，但 `AMDGPU.jl` 正在将 Julia 的应用范围扩展到其他硬件。
7. **互操作性：**
   有时，您可能需要使用其他包集合中的库或工具。

   - `PyCall.jl`: 允许您直接从 Julia 调用 Python 函数和使用 Python 库。这对于访问基于 Python 的机器学习工具或可能没有直接 Julia 等效项的预训练 (pre-training)模型很有用。
   - `RCall.jl`: 提供与 R 交互的类似功能。
8. **绘图与可视化：**
   可视化数据、模型架构和训练进度非常重要。

   - `Plots.jl`: 一个流行的绘图元包，为多个后端绘图库提供统一接口。
   - `Makie.jl`: 一个强大的、高性能绘图库，能够创建交互式 2D 和 3D 可视化。

下面的图表说明了这些组成部分之间如何相互关联，特别是在使用 `Flux.jl` 的深度学习流程中：

> Julia 机器学习包集合中主要组成部分的关系，侧重于支持基于 `Flux.jl` 的深度学习流程的元素。

正如您所见，`Flux.jl` 并非独立存在。它建立在 Julia 的核心优势之上，使用 `Zygote.jl` 等强大的自动微分引擎，并与数据处理和 GPU 加速库集成，以提供一个全面的深度学习环境。

尽管 Julia 通过 `MLJ.jl` 等包支持多种机器学习任务，但本课程将侧重于使用 `Flux.jl` 进行深度学习。您在本章中构建的基本内容，包括对这个包集合的理解，旨在使您能够专注于该特定方向。考虑到这一概述，下一步是配置您的本地环境，以便您可以使用这些工具。

## 参考资料

- [Flux: Elegant Deep Learning in Julia](https://proceedings.juliacon.org/papers/10.21105/jcon.00033) — Mike Innes, Elliot Saba, Keno Fischer, Dhairya Gandhi, Viral Shah (2018)
  Journal: Proceedings of the JuliaCon Conferences; Publisher: The Open Journal; DOI: [10.21105/jcon.00033](https://doi.org/10.21105/jcon.00033)
  介绍Flux.jl的设计原则和特性，一个Julia中轻量级的深度学习库。
- [Zygote: Differentiable Programming from First Principles](https://proceedings.juliacon.org/papers/10.21105/jcon.00096) — Mike Innes, Alan Edelman, David P. Sanders (2019)
  Journal: Proceedings of the JuliaCon Conferences; Publisher: The JuliaCon Proceedings; DOI: [10.21105/jcon.00096](https://doi.org/10.21105/jcon.00096)
  介绍了Zygote.jl，一个对Julia深度学习能力很重要的源到源自动微分系统。
- [MLJ: A Julia package for composing and tuning machine learning models](https://joss.theoj.org/papers/10.21105/joss.02704) — Anthony D. Blaom, Franz Kiraly, Thibaut Lienart, Yiannis Simillides, Diego Arenas, Sebastian J. Vollmer (2020)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 5; Pages: 2704; DOI: [10.21105/joss.02704](https://doi.org/10.21105/joss.02704)
  描述了MLJ.jl，一个Julia中用于通用机器学习任务的统一框架，展示了其组合性。
- [The Julia Language Documentation](https://docs.julialang.org/) — The Julia Language Community (2024)
  Julia编程语言的官方且全面的文档，包含其与高性能计算相关的核心特性。
- [CUDA.jl Documentation](https://cuda.juliagpu.org/) — The Julia GPU community (2024)
  CUDA.jl的官方文档，为Julia代码实现GPU加速，对深度学习至关重要。
