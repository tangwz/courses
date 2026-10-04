---
course: "advanced-python-programming-ml"
chapter: "python-performance-optimization-ml"
lesson: "understanding-python-gil"
sourceId: 2279
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/understanding-python-gil"
title: "理解Python的全局解释器锁（GIL）"
description: "理解全局解释器锁（GIL）对Python中CPU密集型机器学习任务的影响。"
order: 6
plots: []
sourceHash: "a6b3f75e53f53ad6c27266c539672bfc5bb7dcc02aabdaa2ffbf56af75297c26"
sourceCorrections: []
---

在讨论Python的性能优化时，特别是在并发编程的背景下，全局解释器锁（通常称为GIL）是无法绕开的话题。GIL是CPython（最常用的Python实现）的一个核心特征，它对多线程Python程序的运行方式有着重要的影响，尤其是在机器学习 (machine learning)中常见的CPU密集型任务。

### 什么是GIL？

本质上，GIL是一个互斥锁，它保护对Python对象的访问，阻止同一个进程内的多个线程同时执行Python字节码。即使在多核处理器上，在任何给定时间点，也只有一个线程可以持有GIL并执行Python字节码。

需要说明的是，GIL是CPython特有的实现细节。其他Python实现，例如Jython（运行在JVM上）、IronPython（运行在.NET上）或PyPy（特别是其软件事务内存版本PyPy-STM）没有GIL，可以为CPU密集型任务实现真正的线程并行。然而，由于CPython是参考实现且使用最广，理解GIL对大多数Python开发者来说非常重要。

### GIL为何存在？

GIL存在的主要历史原因是为了简化CPython中的内存管理。Python使用引用计数进行内存管理。GIL通过防止竞争条件（即多个线程可能同时修改同一个对象的计数）来确保这些引用计数始终保持一致。这种设计使得CPython更易于实现，也更易于集成现有C库，因为扩展作者最初不必担心Python层面的线程安全问题。

### GIL如何影响执行

在一个多线程CPython程序中：

1. 线程在执行任何Python字节码之前必须获得GIL。
2. 当前运行的线程持有GIL。
3. 线程会定期释放GIL。在现代Python版本中，这大约每执行几毫秒发生一次，或者更重要的是，当线程执行阻塞I/O操作（例如从文件、网络套接字读取数据，或等待用户输入）时。
4. 当GIL被释放时，另一个等待的线程可以获得它并开始执行其Python字节码。

这种切换发生得很快，产生了并行执行的错觉。然而，对于纯计算且只涉及Python字节码执行的任务（CPU密集型任务），GIL有效地串行化了它们的执行。

> 此图说明了多个Python线程如何争用单个全局解释器锁（GIL）。即使有多个核心可用，也只有当前持有GIL的线程（线程1）才能在CPU核心上执行Python字节码。其他线程必须等待。

### 对机器学习 (machine learning)工作负载的影响

GIL的影响在很大程度上取决于工作负载的类型：

- **CPU密集型任务：** 如果您的ML任务涉及主要用纯Python代码实现的繁重计算（例如，复杂的特征工程循环、未使用优化库的自定义算法实现），那么使用`threading`模块**不会**通过使用多个CPU核心来提升性能。GIL确保一次只有一个线程运行Python代码，使线程化对于并行化此类计算无效。在某些情况下，线程管理的开销甚至可能略微降低性能。
- **I/O密集型任务：** 对于主要等待输入/输出操作的任务（例如，从网络源获取数据批次、从磁盘读写大文件、查询数据库、与API交互进行数据收集或模型服务），`threading`可以提供显著的并发优势。当线程执行阻塞I/O调用时，它通常会释放GIL，允许其他线程运行。这使您的程序能够同时处理多个I/O操作，从而提高整体吞吐量 (throughput)和响应性。
- **C扩展的作用（NumPy、SciPy、Pandas、Scikit-learn）：** 这是机器学习中的一个重要考量。许多核心ML库是高度优化的C、C++或Fortran代码的封装。这些库通常在执行计算密集型操作之前释放GIL（例如，NumPy执行大型矩阵乘法，Scikit-learn训练某些模型）。当这些底层库释放GIL时，其他Python线程*可以*并行运行，即使从Python的角度看任务是CPU密集型的。这意味着，如果瓶颈存在于这些优化且能释放GIL的C扩展中，那么`threading`*有时*可以为数值工作负载提供并行能力。性能分析对于判断您的特定工作负载是否如此非常重要。

### 克服CPU密集型任务的GIL限制

由于GIL限制了`threading`中CPU密集型任务的并行能力，标准的Python解决方案是使用`multiprocessing`模块。`multiprocessing`通过创建独立的进程来绕过GIL，每个进程都有自己的Python解释器和内存空间。这允许CPU密集型Python代码在多个核心上真正并行执行。然而，它伴随着一些权衡：更高的内存消耗（因为数据可能需要被复制）以及如果进程需要共享数据，会有进程间通信（IPC）的开销。

另一种方法是，像Cython（允许`nogil`块）和Numba这样的工具可以将Python代码编译成C，通常为编译后的部分释放GIL，实现类似C扩展的线程并行。这些技术在本章的其他部分有所介绍。

总之，GIL是CPython中的一个限制，它阻止了多核系统上Python字节码在多个线程间的真正并行执行。虽然它不妨碍I/O密集型并发，且其影响可以通过C扩展释放锁来减轻，但它使`threading`不适合加速纯Python编写的CPU密集型任务。掌握GIL有助于您选择正确的并发方法（`threading`用于I/O或可释放GIL的扩展，`multiprocessing`用于CPU密集型Python代码）以有效优化您的ML应用程序。

## 参考资料

- [multiprocessing - Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html) — Python Software Foundation (2024)
  Python `multiprocessing`模块的官方文档，提供了实现真正的CPU密集型并行计算的主要方法。
