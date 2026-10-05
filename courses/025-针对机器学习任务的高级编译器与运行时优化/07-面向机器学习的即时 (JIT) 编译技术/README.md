# 第 7 章：面向机器学习的即时 (JIT) 编译技术

来源：[原章节](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-7-jit-compilation-ml)

[返回课程目录](../README.md)

机器学习框架中常见的执行模式，如即时执行或预先 (AOT) 编译，在灵活性和优化潜力之间有各自的取舍。即时 (JIT) 编译的工作方式则不同，它将最终的代码生成推迟到运行时进行。这种策略能够根据动态运行时环境的信息进行优化，例如实际观测到的张量形状 (例如，$B \times C \times H \times W$) 或数值。

本章将介绍与机器学习工作负载相关的 JIT 编译方法。我们将分析图获取技术，例如追踪 (tracing) 和脚本化 (scripting)，审视 JIT 专用中间表示的需求，并研究运行时特化和自适应编译策略。TensorFlow XLA 和 PyTorch JIT (TorchScript) 等重要 JIT 系统的架构和功能将作为具体实例进行说明。

## 小节

- 1. [ML中JIT编译的动因](01-ML%E4%B8%ADJIT%E7%BC%96%E8%AF%91%E7%9A%84%E5%8A%A8%E5%9B%A0.md)
- 2. [追踪与脚本方法](02-%E8%BF%BD%E8%B8%AA%E4%B8%8E%E8%84%9A%E6%9C%AC%E6%96%B9%E6%B3%95.md)
- 3. [JIT系统中的中间表示](03-JIT%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA.md)
- 4. [运行时专门化与多态性](04-%E8%BF%90%E8%A1%8C%E6%97%B6%E4%B8%93%E9%97%A8%E5%8C%96%E4%B8%8E%E5%A4%9A%E6%80%81%E6%80%A7.md)
- 5. [即时编译器（JIT）中的配置文件引导优化（PGO）](05-%E5%8D%B3%E6%97%B6%E7%BC%96%E8%AF%91%E5%99%A8%EF%BC%88JIT%EF%BC%89%E4%B8%AD%E7%9A%84%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E5%BC%95%E5%AF%BC%E4%BC%98%E5%8C%96%EF%BC%88PGO%EF%BC%89.md)
- 6. [自适应与多层编译](06-%E8%87%AA%E9%80%82%E5%BA%94%E4%B8%8E%E5%A4%9A%E5%B1%82%E7%BC%96%E8%AF%91.md)
- 7. [案例分析：TensorFlow XLA](07-%E6%A1%88%E4%BE%8B%E5%88%86%E6%9E%90%EF%BC%9ATensorFlow%20XLA.md)
- 8. [案例研究：PyTorch JIT (TorchScript)](08-%E6%A1%88%E4%BE%8B%E7%A0%94%E7%A9%B6%EF%BC%9APyTorch%20JIT%20%28TorchScript%29.md)
- 9. [实践操作：分析JIT编译代码](09-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%88%86%E6%9E%90JIT%E7%BC%96%E8%AF%91%E4%BB%A3%E7%A0%81.md)
