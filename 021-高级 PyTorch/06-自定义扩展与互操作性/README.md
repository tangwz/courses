# 第 6 章：自定义扩展与互操作性

来源：[原章节](https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability)

[返回课程目录](../README.md)

PyTorch 提供了丰富的工具集，但特定应用场景常常需要标准库中没有的操作或性能优化。
本章将介绍如何将 PyTorch 的功能扩展到其标准 Python 应用编程接口之外。

我们将构建自定义运算符，使用 C++ 和 CUDA 来应对对计算效率要求高或需要专用算法的场合。您将获得直接操作 PyTorch C++ 后端 (ATen) 的经验，学习管理 PyTorch 张量和 NumPy 数组之间的数据传输，并通过扩展 `torch.nn.Module` 和 `torch.optim.Optimizer` 来组织自定义网络组件和优化方法。此外，还将介绍使用外部函数接口 (FFI) 与现有 C 库进行交互的技术。学完本章，您将能够把自定义代码和外部库集成到您的 PyTorch 工作流程中。

## 小节

- 1. [构建定制C++扩展](01-%E6%9E%84%E5%BB%BA%E5%AE%9A%E5%88%B6C%2B%2B%E6%89%A9%E5%B1%95.md)
- 2. [构建自定义 CUDA 扩展](02-%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%20CUDA%20%E6%89%A9%E5%B1%95.md)
- 3. [使用 ATen 库](03-%E4%BD%BF%E7%94%A8%20ATen%20%E5%BA%93.md)
- 4. [PyTorch 与 NumPy 的连接](04-PyTorch%20%E4%B8%8E%20NumPy%20%E7%9A%84%E8%BF%9E%E6%8E%A5.md)
- 5. [使用自定义模块扩展 torch.nn](05-%E4%BD%BF%E7%94%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97%E6%89%A9%E5%B1%95%20torch.nn.md)
- 6. [扩展 torch.optim，使用自定义优化器](06-%E6%89%A9%E5%B1%95%20torch.optim%EF%BC%8C%E4%BD%BF%E7%94%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 7. [外部函数接口 (FFI)](07-%E5%A4%96%E9%83%A8%E5%87%BD%E6%95%B0%E6%8E%A5%E5%8F%A3%20%28FFI%29.md)
- 8. [实践：构建一个简单的 CUDA 扩展](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%20CUDA%20%E6%89%A9%E5%B1%95.md)
