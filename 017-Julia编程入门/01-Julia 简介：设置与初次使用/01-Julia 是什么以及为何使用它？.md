# Julia 是什么以及为何使用它？

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/what-is-julia-why-use-it)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Julia 是一种现代编程语言，专门为高性能数值分析和计算科学而设计。它提供了一种独特的结合：既有 Python 或 R 等动态语言的易用性，又具备与 C 或 Fortran 等静态类型语言相媲美的性能。这使得 Julia 成为需要快速开发和高效运行的任务的理想选择。

多年来，开发人员和科学家们常遇到“双语言问题”。他们可能会使用一种以快速开发周期著称的高级交互式语言来构建想法原型。然后，为了在大规模计算中达到所需的速度，他们需要用一种低级编译语言重写代码。这个重写步骤既耗时又易出错，并且在研究与生产之间制造了隔阂。

Julia 的创建正是为了解决这个问题。它的目标是成为一种兼具这两种用途的单一语言。它是如何做到这一点的？

- **动态类型和可选的静态类型：** Julia 是动态类型的，这意味着你无需总是声明变量的类型。这使得快速编写脚本和尝试变得容易。不过，你可以添加类型注解，Julia 可以借助它们生成高度优化的代码。
- **即时 (JIT) 编译：** 与传统的解释型语言不同，Julia 代码使用基于 LLVM 的 JIT 编译器，在运行前被编译成适用于各种平台的高效本地机器码。这是其速度的主要原因。
- **为性能而构建：** 该语言的语法及其标准库从一开始就以性能为目标构建的。

那么，你为什么会选择学习和使用 Julia 呢？

- **速度：** Julia 运行很快。在数值任务中，Julia 代码的速度通常可以接近甚至达到 C 代码的速度，而无需 C 语言通常需要的手动内存管理或复杂的构建系统。
- **生产力：** 它的语法简洁直观，特别是对于有其他脚本语言经验的人。这让你能够简洁地表达复杂的想法。
- **多重分派：** 这是一项核心特性，被调用的具体函数取决于其所有参数 (parameter)的类型。它允许编写非常通用的代码，并以灵活和可扩展的方式进行专门化。我们将在第 5 章中更详细地了解这一点。
- **互操作性：** Julia 与其他语言兼容良好。你可以无需“粘合”代码即可直接调用 C、Fortran 和 Python 库，这使得集成现有代码库或专用工具变得容易。
- **开源和社区：** Julia 是开源的，拥有一个不断壮大且活跃的全球社区，贡献着软件包和支持。

Julia 非常适合科学计算、机器学习 (machine learning)、数据挖掘、大规模线性代数和并行计算。如果你的工作涉及大量计算、复杂的算法，或需要快速迭代分析模型，Julia 提供了一种令人耳目一新且实用的选择。本课程将指导你使用这些功能，从最基础的部分开始。

## 参考资料

- [Julia: A Fresh Approach to Numerical Computing](https://doi.org/10.1137/141000671) — Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah (2017)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 59; Pages: 65-98; DOI: [10.1137/141000671](https://doi.org/10.1137/141000671)
  这篇开创性论文介绍了 Julia 语言，阐述了其设计原则、动机（包括解决“双语言问题”）以及多重派发和 JIT 编译等实现高性能数值计算的关键特性。
- [The Julia Language Documentation](https://docs.julialang.org/) — The Julia Language Developers (2024)
  Julia 编程语言的官方综合指南，详细介绍了其语法、标准库和核心功能。对于理解该语言至关重要。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045021/) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本介绍 Julia 编程基础知识的入门书籍，适合初学者和从其他语言转型的人，侧重于解决问题和计算思维。

---

[下一节](02-Julia%20%E5%9C%A8%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E6%96%B9%E9%9D%A2%E7%9A%84%E4%BC%98%E5%8A%BF.md)
