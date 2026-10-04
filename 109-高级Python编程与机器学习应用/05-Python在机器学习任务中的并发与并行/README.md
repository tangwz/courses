# 第 5 章：Python在机器学习任务中的并发与并行

来源：[原章节](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml)

[返回课程目录](../README.md)

机器学习工作流，从数据预处理到模型训练，常会涉及大量计算或I/O操作，这些操作适合通过并发方式执行。尽管Python的全局解释器锁（GIL）对某些并行类型带来挑战，但多种技术仍能大幅提升性能。

本章着重讲解如何在Python中为机器学习任务实现并发与并行方案。我们会考察线程和多进程的区别及其使用场景，使用`concurrent.futures`模块来简化管理，了解`asyncio`在I/O密集型任务中的应用，并讨论进程间通信、同步以及并发代码的调试方法等重要知识点。学完本章后，您将能够选择并应用适合的并发模型，以加速您的基于Python的机器学习应用。

## 小节

- 1. [机器学习任务中的多线程与多进程](01-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E5%A4%9A%E7%BA%BF%E7%A8%8B%E4%B8%8E%E5%A4%9A%E8%BF%9B%E7%A8%8B.md)
- 2. [用于并行执行的 \`multiprocessing\` 模块](02-%E7%94%A8%E4%BA%8E%E5%B9%B6%E8%A1%8C%E6%89%A7%E8%A1%8C%E7%9A%84%20%60multiprocessing%60%20%E6%A8%A1%E5%9D%97.md)
- 3. [进程间通信 (IPC) 技术](03-%E8%BF%9B%E7%A8%8B%E9%97%B4%E9%80%9A%E4%BF%A1%20%28IPC%29%20%E6%8A%80%E6%9C%AF.md)
- 4. [使用 \`concurrent.futures\` 实现高级并发](04-%E4%BD%BF%E7%94%A8%20%60concurrent.futures%60%20%E5%AE%9E%E7%8E%B0%E9%AB%98%E7%BA%A7%E5%B9%B6%E5%8F%91.md)
- 5. [asyncio 异步机器学习操作简介](05-asyncio%20%E5%BC%82%E6%AD%A5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%93%8D%E4%BD%9C%E7%AE%80%E4%BB%8B.md)
- 6. [同步原语（锁、信号量、事件）](06-%E5%90%8C%E6%AD%A5%E5%8E%9F%E8%AF%AD%EF%BC%88%E9%94%81%E3%80%81%E4%BF%A1%E5%8F%B7%E9%87%8F%E3%80%81%E4%BA%8B%E4%BB%B6%EF%BC%89.md)
- 7. [调试并发 Python 应用](07-%E8%B0%83%E8%AF%95%E5%B9%B6%E5%8F%91%20Python%20%E5%BA%94%E7%94%A8.md)
- 8. [动手实践：数据预处理并行化](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E5%B9%B6%E8%A1%8C%E5%8C%96.md)
