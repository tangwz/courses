# 第 4 章：进阶自动微分方法

来源：[原章节](https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation)

[返回课程目录](../README.md)

自动微分是JAX的核心组成部分，它使得机器学习模型能够进行基于梯度的优化。虽然`jax.grad`提供了获取梯度的便捷接口，但了解并控制其内部运作将为复杂任务带来更大的灵活性和更高的效率。

本章将基于`jax.grad`的初步知识进行拓展。你将学习自动微分的两种基本模式：正向模式（雅可比向量积，JVP）和反向模式（向量雅可比积，VJP）。我们将介绍如何直接使用`jax.jvp`和`jax.vjp`计算它们，这些构成了`jax.grad`及其他变换的构建块。

主要内容包括：

*   计算通过组合微分变换得到的高阶导数。
*   高效计算完整雅可比矩阵和海森矩阵的方法。
*   使用`jax.custom_vjp`和`jax.custom_jvp`为函数定义自定义微分规则，这可用于提高数值稳定性、优化性能或处理非JAX代码。
*   理解自动微分在`lax.scan`、`lax.cond`和`lax.while_loop`等控制流原语中如何运作。
*   处理不可微分函数或需要明确停止梯度传播的函数的方法，例如使用`jax.lax.stop_gradient`。

在本章结束时，你将对JAX的自动微分系统有更全面的理解，并掌握在进阶场景中有效应用它的工具。

## 小节

- 1. [前向和反向模式自动微分回顾](01-%E5%89%8D%E5%90%91%E5%92%8C%E5%8F%8D%E5%90%91%E6%A8%A1%E5%BC%8F%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E5%9B%9E%E9%A1%BE.md)
- 2. [雅可比向量积 (JVPs) 与 jax.jvp](02-%E9%9B%85%E5%8F%AF%E6%AF%94%E5%90%91%E9%87%8F%E7%A7%AF%20%28JVPs%29%20%E4%B8%8E%20jax.jvp.md)
- 3. [向量-雅可比积 (VJPs) 与 jax.vjp](03-%E5%90%91%E9%87%8F-%E9%9B%85%E5%8F%AF%E6%AF%94%E7%A7%AF%20%28VJPs%29%20%E4%B8%8E%20jax.vjp.md)
- 4. [高阶导数](04-%E9%AB%98%E9%98%B6%E5%AF%BC%E6%95%B0.md)
- 5. [计算完整的雅可比矩阵和海森矩阵](05-%E8%AE%A1%E7%AE%97%E5%AE%8C%E6%95%B4%E7%9A%84%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5%E5%92%8C%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5.md)
- 6. [使用 jax.custom_vjp 的自定义微分规则](06-%E4%BD%BF%E7%94%A8%20jax.custom_vjp%20%E7%9A%84%E8%87%AA%E5%AE%9A%E4%B9%89%E5%BE%AE%E5%88%86%E8%A7%84%E5%88%99.md)
- 7. [jax.custom_jvp 自定义求导规则](07-jax.custom_jvp%20%E8%87%AA%E5%AE%9A%E4%B9%89%E6%B1%82%E5%AF%BC%E8%A7%84%E5%88%99.md)
- 8. [通过控制流原语求导](08-%E9%80%9A%E8%BF%87%E6%8E%A7%E5%88%B6%E6%B5%81%E5%8E%9F%E8%AF%AD%E6%B1%82%E5%AF%BC.md)
- 9. [处理不可微分函数](09-%E5%A4%84%E7%90%86%E4%B8%8D%E5%8F%AF%E5%BE%AE%E5%88%86%E5%87%BD%E6%95%B0.md)
- 10. [实践：实现自定义梯度](10-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A2%AF%E5%BA%A6.md)
