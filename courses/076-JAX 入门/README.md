# JAX 入门

来源：[JAX 入门](https://apxml.com/zh/courses/getting-started-with-jax)

学习 JAX，用于高性能数值计算和机器学习 (machine learning)研究。本课程涵盖 JAX 核心知识，包括其 NumPy 接口、`jit`、`grad`、`vmap` 和 `pmap` 等函数变换，以及用于状态管理的函数式编程模式。您将获得适配现代硬件（GPU/TPU）来加速和求导 Python 代码的实践经验。

预计学时：12 小时

先修要求：熟悉 Python 和 NumPy。

## 课程目录

### 1. [JAX 简介](01-JAX%20%E7%AE%80%E4%BB%8B/README.md)

- 1. [JAX 是什么？](01-JAX%20%E7%AE%80%E4%BB%8B/01-JAX%20%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md)
- 2. [JAX 对比 NumPy](01-JAX%20%E7%AE%80%E4%BB%8B/02-JAX%20%E5%AF%B9%E6%AF%94%20NumPy.md)
- 3. [核心设计理念：函数变换](01-JAX%20%E7%AE%80%E4%BB%8B/03-%E6%A0%B8%E5%BF%83%E8%AE%BE%E8%AE%A1%E7%90%86%E5%BF%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%8F%98%E6%8D%A2.md)
- 4. [安装与设置](01-JAX%20%E7%AE%80%E4%BB%8B/04-%E5%AE%89%E8%A3%85%E4%B8%8E%E8%AE%BE%E7%BD%AE.md)
- 5. [使用 JAX 数组](01-JAX%20%E7%AE%80%E4%BB%8B/05-%E4%BD%BF%E7%94%A8%20JAX%20%E6%95%B0%E7%BB%84.md)
- 6. [设备管理：CPU、GPU、TPU](01-JAX%20%E7%AE%80%E4%BB%8B/06-%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%EF%BC%9ACPU%E3%80%81GPU%E3%80%81TPU.md)
- 7. [动手练习：基本数组操作](01-JAX%20%E7%AE%80%E4%BB%8B/07-%E5%8A%A8%E6%89%8B%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%9F%BA%E6%9C%AC%E6%95%B0%E7%BB%84%E6%93%8D%E4%BD%9C.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/quiz)

### 2. [通过 JIT 编译加速函数](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/README.md)

- 1. [速度提升：为何需要编译？](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/01-%E9%80%9F%E5%BA%A6%E6%8F%90%E5%8D%87%EF%BC%9A%E4%B8%BA%E4%BD%95%E9%9C%80%E8%A6%81%E7%BC%96%E8%AF%91%EF%BC%9F.md)
- 2. [介绍 \`jax.jit\`](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/02-%E4%BB%8B%E7%BB%8D%20%60jax.jit%60.md)
- 3. [JIT 工作原理：追踪与编译](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/03-JIT%20%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%EF%BC%9A%E8%BF%BD%E8%B8%AA%E4%B8%8E%E7%BC%96%E8%AF%91.md)
- 4. [Python 控制流与 \`jit\`](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/04-Python%20%E6%8E%A7%E5%88%B6%E6%B5%81%E4%B8%8E%20%60jit%60.md)
- 5. [静态值与跟踪值](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/05-%E9%9D%99%E6%80%81%E5%80%BC%E4%B8%8E%E8%B7%9F%E8%B8%AA%E5%80%BC.md)
- 6. [\`jit\` 的常见问题](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/06-%60jit%60%20%E7%9A%84%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98.md)
- 7. [动手实践：应用 \`jit\`](02-%E9%80%9A%E8%BF%87%20JIT%20%E7%BC%96%E8%AF%91%E5%8A%A0%E9%80%9F%E5%87%BD%E6%95%B0/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%20%60jit%60.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/quiz)

### 3. [使用 \`grad\` 进行自动微分](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/README.md)

- 1. [理解梯度](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/01-%E7%90%86%E8%A7%A3%E6%A2%AF%E5%BA%A6.md)
- 2. [介绍 \`jax.grad\`](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/02-%E4%BB%8B%E7%BB%8D%20%60jax.grad%60.md)
- 3. [自动微分的工作方式：反向模式](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/03-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%96%B9%E5%BC%8F%EF%BC%9A%E5%8F%8D%E5%90%91%E6%A8%A1%E5%BC%8F.md)
- 4. [关于参数求导](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/04-%E5%85%B3%E4%BA%8E%E5%8F%82%E6%95%B0%E6%B1%82%E5%AF%BC.md)
- 5. [高阶导数（\`grad\`的\`grad\`）](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/05-%E9%AB%98%E9%98%B6%E5%AF%BC%E6%95%B0%EF%BC%88%60grad%60%E7%9A%84%60grad%60%EF%BC%89.md)
- 6. [值和梯度 (\`jax.value_and_grad\`)](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/06-%E5%80%BC%E5%92%8C%E6%A2%AF%E5%BA%A6%20%28%60jax.value_and_grad%60%29.md)
- 7. [求导与控制流](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/07-%E6%B1%82%E5%AF%BC%E4%B8%8E%E6%8E%A7%E5%88%B6%E6%B5%81.md)
- 8. [局限性与注意事项](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/08-%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9.md)
- 9. [动手实践：计算梯度](03-%E4%BD%BF%E7%94%A8%20%60grad%60%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%A2%AF%E5%BA%A6.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/quiz)

### 4. [使用 \`vmap\` 实现自动向量化](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/README.md)

- 1. [向量化的原理](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/01-%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [介绍 \`jax.vmap\`](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/02-%E4%BB%8B%E7%BB%8D%20%60jax.vmap%60.md)
- 3. [对特定参数进行映射（\`in_axes\`，\`out_axes\`）](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/03-%E5%AF%B9%E7%89%B9%E5%AE%9A%E5%8F%82%E6%95%B0%E8%BF%9B%E8%A1%8C%E6%98%A0%E5%B0%84%EF%BC%88%60in_axes%60%EF%BC%8C%60out_axes%60%EF%BC%89.md)
- 4. [处理多个批处理参数](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/04-%E5%A4%84%E7%90%86%E5%A4%9A%E4%B8%AA%E6%89%B9%E5%A4%84%E7%90%86%E5%8F%82%E6%95%B0.md)
- 5. [嵌套 \`vmap\`](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/05-%E5%B5%8C%E5%A5%97%20%60vmap%60.md)
- 6. [结合 \`vmap\` 与 \`jit\` 和 \`grad\`](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/06-%E7%BB%93%E5%90%88%20%60vmap%60%20%E4%B8%8E%20%60jit%60%20%E5%92%8C%20%60grad%60.md)
- 7. [\`vmap\`的性能考量](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/07-%60vmap%60%E7%9A%84%E6%80%A7%E8%83%BD%E8%80%83%E9%87%8F.md)
- 8. [动手实践：函数向量化](04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%90%91%E9%87%8F%E5%8C%96.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/quiz)

### 5. [使用 \`pmap\` 在多设备上并行计算](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/README.md)

- 1. [数据并行 (SPMD) 介绍](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/01-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%20%28SPMD%29%20%E4%BB%8B%E7%BB%8D.md)
- 2. [介绍 \`jax.pmap\`](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/02-%E4%BB%8B%E7%BB%8D%20%60jax.pmap%60.md)
- 3. [将数据映射到设备 (\`in_axes\`, \`out_axes\`)](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/03-%E5%B0%86%E6%95%B0%E6%8D%AE%E6%98%A0%E5%B0%84%E5%88%B0%E8%AE%BE%E5%A4%87%20%28%60in_axes%60%2C%20%60out_axes%60%29.md)
- 4. [设备网格与轴名称](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/04-%E8%AE%BE%E5%A4%87%E7%BD%91%E6%A0%BC%E4%B8%8E%E8%BD%B4%E5%90%8D%E7%A7%B0.md)
- 5. [集体操作（\`lax.psum\`、\`lax.pmean\`等）](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/05-%E9%9B%86%E4%BD%93%E6%93%8D%E4%BD%9C%EF%BC%88%60lax.psum%60%E3%80%81%60lax.pmean%60%E7%AD%89%EF%BC%89.md)
- 6. [将 \`pmap\` 与其他变换结合使用](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/06-%E5%B0%86%20%60pmap%60%20%E4%B8%8E%E5%85%B6%E4%BB%96%E5%8F%98%E6%8D%A2%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8.md)
- 7. [调试 \`pmap\` 化的函数](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/07-%E8%B0%83%E8%AF%95%20%60pmap%60%20%E5%8C%96%E7%9A%84%E5%87%BD%E6%95%B0.md)
- 8. [动手实践：并行计算](05-%E4%BD%BF%E7%94%A8%20%60pmap%60%20%E5%9C%A8%E5%A4%9A%E8%AE%BE%E5%A4%87%E4%B8%8A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/quiz)

### 6. [JAX 中的状态管理](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/README.md)

- 1. [函数纯粹性与副作用](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/01-%E5%87%BD%E6%95%B0%E7%BA%AF%E7%B2%B9%E6%80%A7%E4%B8%8E%E5%89%AF%E4%BD%9C%E7%94%A8.md)
- 2. [函数式代码中的状态挑战](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/02-%E5%87%BD%E6%95%B0%E5%BC%8F%E4%BB%A3%E7%A0%81%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E6%8C%91%E6%88%98.md)
- 3. [模式：显式状态传递](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/03-%E6%A8%A1%E5%BC%8F%EF%BC%9A%E6%98%BE%E5%BC%8F%E7%8A%B6%E6%80%81%E4%BC%A0%E9%80%92.md)
- 4. [使用 PyTree 管理分层状态](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/04-%E4%BD%BF%E7%94%A8%20PyTree%20%E7%AE%A1%E7%90%86%E5%88%86%E5%B1%82%E7%8A%B6%E6%80%81.md)
- 5. [示例：有状态计数器](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/05-%E7%A4%BA%E4%BE%8B%EF%BC%9A%E6%9C%89%E7%8A%B6%E6%80%81%E8%AE%A1%E6%95%B0%E5%99%A8.md)
- 6. [例子：简单的优化器状态](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/06-%E4%BE%8B%E5%AD%90%EF%BC%9A%E7%AE%80%E5%8D%95%E7%9A%84%E4%BC%98%E5%8C%96%E5%99%A8%E7%8A%B6%E6%80%81.md)
- 7. [将状态管理与变换结合](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/07-%E5%B0%86%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86%E4%B8%8E%E5%8F%98%E6%8D%A2%E7%BB%93%E5%90%88.md)
- 8. [实践：实现有状态函数](06-JAX%20%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E7%AE%A1%E7%90%86/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E6%9C%89%E7%8A%B6%E6%80%81%E5%87%BD%E6%95%B0.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/quiz)

## 学习目标

- **JAX 基础知识**：理解 JAX 的核心知识、它与 NumPy 的关联及其函数式编程方法。
- **函数变换**：应用 JAX 的主要变换：`jit` 用于编译，`grad` 用于自动求导，`vmap` 用于向量化，`pmap` 用于并行化。
- **高性能代码**：编写能高效运用 GPU 和 TPU 等现代加速器的 JAX 代码。
- **自动求导**：使用 `grad` 自动计算 Python 函数的梯度。
- **状态管理**：使用适合 JAX 的函数式编程模式实现有状态计算。
- **调试与性能分析**：识别调试 JAX 代码时遇到的常见问题和基本方法。
