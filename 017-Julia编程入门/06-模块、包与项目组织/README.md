# 第 6 章：模块、包与项目组织

来源：[原章节](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization)

[返回课程目录](../README.md)

随着程序变得更复杂，有效管理代码变得非常重要。本章会介绍一些方法，用来组织你的 Julia 代码以及管理外部依赖。

你会了解模块，它们能帮助你把相关函数和数据归类，避免名称冲突并提升代码可读性。接着，我们会讲解 Julia 内置的包管理器 Pkg。这个工具对于将外部库引入你的项目非常重要，能帮助你管理这些库的不同版本，以及搭建可重复使用的项目环境。

学完本章，你将能够有条理地组织你的 Julia 项目，使用模块来构建可重复使用的部分，并熟练地管理包，以扩展 Julia 的功能。

## 小节

- 1. [使用模块组织代码](01-%E4%BD%BF%E7%94%A8%E6%A8%A1%E5%9D%97%E7%BB%84%E7%BB%87%E4%BB%A3%E7%A0%81.md)
- 2. [模块内容的导入与导出](02-%E6%A8%A1%E5%9D%97%E5%86%85%E5%AE%B9%E7%9A%84%E5%AF%BC%E5%85%A5%E4%B8%8E%E5%AF%BC%E5%87%BA.md)
- 3. [运用 Julia 的标准库](03-%E8%BF%90%E7%94%A8%20Julia%20%E7%9A%84%E6%A0%87%E5%87%86%E5%BA%93.md)
- 4. [Pkg 介绍：Julia 的包管理器](04-Pkg%20%E4%BB%8B%E7%BB%8D%EF%BC%9AJulia%20%E7%9A%84%E5%8C%85%E7%AE%A1%E7%90%86%E5%99%A8.md)
- 5. [查找、添加和管理包](05-%E6%9F%A5%E6%89%BE%E3%80%81%E6%B7%BB%E5%8A%A0%E5%92%8C%E7%AE%A1%E7%90%86%E5%8C%85.md)
- 6. [在项目中集成外部库](06-%E5%9C%A8%E9%A1%B9%E7%9B%AE%E4%B8%AD%E9%9B%86%E6%88%90%E5%A4%96%E9%83%A8%E5%BA%93.md)
- 7. [设置一个基本的 Julia 项目](07-%E8%AE%BE%E7%BD%AE%E4%B8%80%E4%B8%AA%E5%9F%BA%E6%9C%AC%E7%9A%84%20Julia%20%E9%A1%B9%E7%9B%AE.md)
- 8. [实践：组织代码与管理依赖](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%BB%84%E7%BB%87%E4%BB%A3%E7%A0%81%E4%B8%8E%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization/quiz)
