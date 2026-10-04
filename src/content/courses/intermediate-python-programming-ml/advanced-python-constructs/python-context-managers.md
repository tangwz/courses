---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "python-context-managers"
sourceId: 2096
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/python-context-managers"
title: "用于资源管理的上下文管理器"
description: "了解并实现使用 'with' 语句的上下文管理器，以实现可靠的资源处理。"
order: 6
plots: []
sourceHash: "a9ba43c60925425b3e8fba6f32a4dbf82b7f4b0c63b7b983c24259119c496b45"
sourceCorrections: []
---

在处理大型数据集和更复杂的机器学习 (machine learning)管道时，有效管理资源变得日益重要。文件、网络连接或数据库会话等资源需要可靠地设置，并且最重要的是，无论您的代码是成功运行还是遇到错误，都需要可靠地清理。未能释放资源可能导致内存泄漏、文件损坏或连接池耗尽。Python 为此提供了一个简洁的解决方案：结合 `with` 语句使用的**上下文 (context)管理器**。

### 问题：确保资源清理

考虑一个常见任务：从文件中读取数据。您需要打开文件，处理其内容，然后确保文件被关闭。一种直接的方法可能如下所示：

```python
# 潜在问题：如果发生错误，文件可能不会被关闭
f = open('my_data.csv', 'r')
data = f.read()
# ... 处理数据 ...
# 如果这里发生错误怎么办？f.close() 将被跳过！
f.close()
```

如果在数据处理期间发生错误，`f.close()` 这行可能永远不会执行，导致文件保持打开状态。在上下文 (context)管理器出现之前，处理这种问题的标准可靠方法是使用 `try...finally` 块：

```python
f = None # 在 try 块外部初始化 f
try:
    f = open('my_data.csv', 'r')
    data = f.read()
    # ... 处理数据 ...
except IOError as e:
    print(f"读取文件时发生错误：{e}")
finally:
    if f: # 检查文件是否成功打开
        f.close()
        print("文件已关闭。")
```

尽管这种方法有效并保证 `finally` 块会执行，但它相当冗长，特别是当您需要管理多个资源时。

### 解决方案：`with` 语句

Python 的 `with` 语句极大地简化了资源管理。它抽象了 `try...finally` 的样板代码，确保清理代码总是运行。下面是使用 `with` 的文件读取示例：

```python
try:
    with open('my_data.csv', 'r') as f:
        data = f.read()
        # ... 处理数据 ...
        print("在 'with' 块内处理完成。")
    # 文件在这里自动关闭，即使块内部发生错误
    print("文件现在已关闭。")
except IOError as e:
    print(f"发生错误：{e}")

# 如果您尝试在这里访问 f，它将是关闭状态：
# print(f.closed) # 输出：True
```

语法是 `with expression as variable:`。`expression` 返回的对象（在本例中是 `open()` 返回的文件对象）必须支持*上下文 (context)管理协议*。`with` 语句保证了此对象的某些方法在进入和退出块时会被调用。

### 上下文 (context)管理器的工作原理：协议

与 `with` 语句配合使用的对象称为上下文管理器。它们必须实现两个特殊方法：

1. `__enter__(self)`: 在进入 `with` 块时执行。其返回值被赋值给 `as` 之后指定的变量（如果有的话）。此方法通常处理资源的获取（如打开文件）。
2. `__exit__(self, exc_type, exc_val, exc_tb)`: 在退出块时执行，无论是正常退出还是因为异常。它处理资源清理（如关闭文件）。
   - `exc_type`: 如果块内发生异常，则为异常类，否则为 `None`。
   - `exc_val`: 如果发生异常，则为异常实例，否则为 `None`。
   - `exc_tb`: 如果发生异常，则为追溯对象，否则为 `None`。

如果 `__exit__` 返回 `True`，则表明已处理了发生的任何异常，并且应抑制该异常。如果它返回 `False`（或隐式返回 `None`），则在 `__exit__` 完成后，任何异常将重新引发。对于文件对象，`__exit__` 确保调用 `close()` 并返回 `None`，从而传播任何异常。

### 创建自定义上下文 (context)管理器

尽管许多内置对象（如文件）和库对象（如数据库连接或锁）可以作为上下文管理器，但您可以创建自己的。这对于管理自定义资源或在代码中设置/清理特定状态很有用。

#### 使用类

您可以定义一个包含 `__enter__` 和 `__exit__` 方法的类。让我们创建一个简单的计时器上下文管理器：

```python
import time

class SimpleTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        # 如果您希望上下文管理器对象可以通过 'as' 在 'with' 块中可用，则返回 'self'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        print(f"代码块执行耗时 {elapsed:.4f} 秒。")
        # 返回 False（或 None）以传播异常
        return False

# 示例用法
with SimpleTimer():
    # 模拟一些工作
    sum(x*x for x in range(1000000))
    print("计算完成。")

# 输出可能看起来像：
# Calculation finished.
# Block executed in 0.0987 seconds.
```

#### 使用 `contextlib.contextmanager`

对于简单的设置/清理逻辑，编写一个完整的类有时会感觉过于繁琐。`contextlib` 模块提供了一个方便的装饰器 `@contextmanager`，它允许您从生成器函数创建上下文管理器。

生成器应该：

1. 在 `yield` 之前执行设置操作。
2. 仅 `yield` 一次。yield 的值绑定到 `with` 语句中的 `as` 变量（如果使用）。此时控制权传递给 `with` 块。
3. 在 `yield` 之后执行清理操作。此代码在 `with` 块完成或其中发生异常时运行。

下面是使用 `@contextmanager` 实现的计时器：

```python
import time
from contextlib import contextmanager

@contextmanager
def simple_timer_gen():
    start_time = time.perf_counter()
    try:
        # yield 之前的任何内容都类似于 __enter__
        yield # 控制权转到 'with' 块
    finally:
        # yield 之后的任何内容都类似于 __exit__
        # 'finally' 确保即使有异常发生也会进行清理
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"生成器代码块执行耗时 {elapsed:.4f} 秒。")

# 示例用法
with simple_timer_gen():
    # 再次模拟一些工作
    sum(x*x for x in range(1000000))
    print("生成器计算完成。")

# 输出可能看起来像：
# Generator calculation finished.
# Generator block executed in 0.0991 seconds.
```

这种基于生成器的方法对于更简单的上下文管理器来说通常更简洁、更易读。

### 在数据科学和机器学习 (machine learning)中的相关性

上下文 (context)管理器在许多数据相关场景中都很有价值：

- **文件处理：** 如前所示，确保数据文件（CSV、JSON、HDF5 等）正确关闭至关重要。
- **数据库连接：** 管理数据库连接以确保它们被关闭或返回到连接池。
- **临时状态：** 临时更改设置（例如，NumPy 打印选项、Pandas 显示设置）并在之后恢复它们。
- **API 会话：** 管理与 Web API 交互的会话。
- **模型加载：** 确保与加载大型模型相关的资源被释放。

始终如一地使用 `with` 可以通过清晰地定义资源生命周期并保证清理，从而减少与资源泄漏相关的难以发现的错误风险，进而编写出更可靠、更易于维护的代码。在构建更复杂的数据处理管道时，采用上下文管理器是编程的规范做法。

## 参考资料

- [The with statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  官方语言规范，详细说明了用于资源管理的 'with' 语句的语法和执行模型。
- [contextlib - Utilities for with Statement Contexts](https://docs.python.org/3/library/contextlib.html) — Python Software Foundation (2024)
  'contextlib' 模块的文档，提供了像 '@contextmanager' 这样的工具，用于简化上下文管理器的创建。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG9BYTJcfd13uj6KyKl4htgQYoRqdO3uNfai6a5vrgeSyf4diS2yYU-dqGQ38DDG7DuIYnx9v89ywcKu9_qTvVAubZLQONW0Xr18XFzS7KO_sSd5QcQi_FX3YH9Svt-z3itxsDhXixPTedSHf7pR1y5Ib59Kum6z9ajIu4nVGo=) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本权威指南，探讨了 Python 的高级特性，包括对上下文管理器及其实际应用的详细讨论。
- [PEP 343 -- The "with" Statement](https://peps.python.org/pep-0343/) — Guido van Rossum, Alyssa Coghlan (2005)
  引入 'with' 语句和上下文管理器协议的 Python 基础增强提案 (PEP)。
