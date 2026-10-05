# 第 1 章：高级 JAX 变换与控制流

来源：[原章节](https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow)

[返回课程目录](../README.md)

你已经使用过 JAX 的核心变换：`jit` 用于编译，`grad` 用于微分，以及 `vmap` 用于向量化。尽管这些是核心要素，但许多精密的模型需要更复杂的计算结构，例如循环步骤、条件执行或动态循环。

本章介绍 JAX 的函数式控制流原语，它们使你能够在 JAX 可追踪和可编译的框架内表达这些复杂的模式。我们将探讨以下内容：

*   使用 `lax.scan` 有效实现序列操作，例如循环神经网络 (RNN) 中的操作。
*   使用 `lax.cond` 在编译函数内部实现条件逻辑。
*   使用 `lax.while_loop` 处理动态迭代。
*   理解这些控制流操作如何与 `jit`、`grad` 和 `vmap` 配合。
*   应用遮蔽技术进行选择性计算，这通常是可变长度数据所必需的。
*   JAX 在追踪过程中如何处理涉及闭包的 Python 代码。

学完本章，你将能够构建和分析包含循环、条件和序列依赖的 JAX 函数，为你构建更复杂的模型和算法做好准备。

## 小节

- 1. [核心转换回顾：jit, grad, vmap](01-%E6%A0%B8%E5%BF%83%E8%BD%AC%E6%8D%A2%E5%9B%9E%E9%A1%BE%EF%BC%9Ajit%2C%20grad%2C%20vmap.md)
- 2. [精通 lax.scan 处理序列操作](02-%E7%B2%BE%E9%80%9A%20lax.scan%20%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E6%93%8D%E4%BD%9C.md)
- 3. [使用 lax.cond 进行条件执行](03-%E4%BD%BF%E7%94%A8%20lax.cond%20%E8%BF%9B%E8%A1%8C%E6%9D%A1%E4%BB%B6%E6%89%A7%E8%A1%8C.md)
- 4. [使用 lax.while_loop 进行循环](04-%E4%BD%BF%E7%94%A8%20lax.while_loop%20%E8%BF%9B%E8%A1%8C%E5%BE%AA%E7%8E%AF.md)
- 5. [结合控制流与变换](05-%E7%BB%93%E5%90%88%E6%8E%A7%E5%88%B6%E6%B5%81%E4%B8%8E%E5%8F%98%E6%8D%A2.md)
- 6. [高级掩码技术](06-%E9%AB%98%E7%BA%A7%E6%8E%A9%E7%A0%81%E6%8A%80%E6%9C%AF.md)
- 7. [了解闭包和JAX暂存](07-%E4%BA%86%E8%A7%A3%E9%97%AD%E5%8C%85%E5%92%8CJAX%E6%9A%82%E5%AD%98.md)
- 8. [实践：实现复杂的循环逻辑](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%A4%8D%E6%9D%82%E7%9A%84%E5%BE%AA%E7%8E%AF%E9%80%BB%E8%BE%91.md)
