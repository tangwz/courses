# 第 3 章：使用 `grad` 进行自动微分

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad)

[返回课程目录](../README.md)

许多计算任务，特别是在机器学习模型训练中，需要计算函数的梯度。自动微分提供了一种高效的方法来计算这些导数。本章将介绍 `jax.grad`，它是JAX的核心变换，用于从处理数值输入的Python代码中获取梯度函数。

您将学习如何：
*   应用 `jax.grad` 计算标量值函数 $f$ 的梯度 $∇f(x)$。
*   理解 `grad` 所使用的反向模式自动微分的基本原理。
*   控制对特定函数参数的微分。
*   通过组合 `grad` 计算高阶导数。
*   使用 `jax.value_and_grad` 高效地同时获取函数的输出值及其梯度。
*   了解控制流如何与微分关联，并识别潜在的局限性。

到本章结束时，您将能够有效使用 `jax.grad` 在JAX框架内对您的数值函数进行求导。

## 小节

- 1. [理解梯度](01-%E7%90%86%E8%A7%A3%E6%A2%AF%E5%BA%A6.md)
- 2. [介绍 \`jax.grad\`](02-%E4%BB%8B%E7%BB%8D%20%60jax.grad%60.md)
- 3. [自动微分的工作方式：反向模式](03-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%96%B9%E5%BC%8F%EF%BC%9A%E5%8F%8D%E5%90%91%E6%A8%A1%E5%BC%8F.md)
- 4. [关于参数求导](04-%E5%85%B3%E4%BA%8E%E5%8F%82%E6%95%B0%E6%B1%82%E5%AF%BC.md)
- 5. [高阶导数（\`grad\`的\`grad\`）](05-%E9%AB%98%E9%98%B6%E5%AF%BC%E6%95%B0%EF%BC%88%60grad%60%E7%9A%84%60grad%60%EF%BC%89.md)
- 6. [值和梯度 (\`jax.value_and_grad\`)](06-%E5%80%BC%E5%92%8C%E6%A2%AF%E5%BA%A6%20%28%60jax.value_and_grad%60%29.md)
- 7. [求导与控制流](07-%E6%B1%82%E5%AF%BC%E4%B8%8E%E6%8E%A7%E5%88%B6%E6%B5%81.md)
- 8. [局限性与注意事项](08-%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9.md)
- 9. [动手实践：计算梯度](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%A2%AF%E5%BA%A6.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/quiz)
