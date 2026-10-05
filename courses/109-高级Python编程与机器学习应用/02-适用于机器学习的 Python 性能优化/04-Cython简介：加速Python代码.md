# Cython简介：加速Python代码

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/introduction-to-cython)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy和Pandas操作优化能大幅提高性能，但你可能遇到核心算法、复杂循环或自定义数值例程仍是瓶颈的情况。纯Python的动态类型和解释执行虽然灵活，但在计算密集型机器学习 (machine learning)任务中会带来明显的开销。当向量 (vector)化不直接或需要从关键代码路径中压榨出最大性能时，你需要能弥合Python易用性与C等编译语言原始速度之间差距的工具。

Cython是为此目的专门设计的一个强大工具。它最好被理解为两部分：

1. **一种编程语言：** Cython是Python的超集。这意味着几乎所有有效的Python代码也都是有效的Cython代码。不过，Cython通过允许你添加可选的静态类型声明来扩展Python，这些声明类似于C语言中的用法。
2. **一个编译器：** Cython编译器将Cython代码（通常写在`.pyx`文件中）转换成高度优化的C或C++代码。这些生成的C/C++代码直接与Python C API交互。

生成的C/C++代码随后由标准C编译器（如GCC或MSVC）编译成一个Python扩展模块（在Linux/macOS上是`.so`文件，在Windows上是`.pyd`文件）。这个编译后的模块可以直接导入到你的Python会话中，就像任何普通Python模块一样，为其中定义的函数提供潜在的巨大加速。

### Cython如何实现性能提升

Cython的性能优势主要来自**静态类型**。当你声明变量类型时（例如，`cdef int count`或`cdef double learning_rate`），Cython可以绕过Python较慢的动态对象系统。它无需操作通用Python对象，而是生成直接对C级别数据类型（整数、浮点数、指针）进行操作的C代码，处理器处理这些类型效率更高。

考虑一个简单的Python循环：

```python
# 纯Python
def sum_values(data):
    total = 0.0
    for x in data:
        total += x # 每个'x'都是一个Python浮点对象
    return total
```

在Python中，每个`x`都是一个完整的Python `float`对象，`+=`操作涉及到Python的对象协议（类型检查、潜在的方法调用如`__add__`、引用计数）。

在Cython中，你可以添加类型：

```python
# Cython (.pyx 文件)
# 注意：需要先编译！
import cython

# 假设'data'是可迭代的，生成C双精度浮点数
# （例如，高效传递的NumPy数组）
@cython.ccall # 用于潜在更快C级调用约定的装饰器
def sum_values_cython(double[:] data_view): # 用于高效访问的类型化memoryview
    cdef double total = 0.0 # C双精度浮点变量
    cdef Py_ssize_t i, n
    n = data_view.shape[0]

    # 使用C整数并直接访问C双精度浮点数的循环
    for i in range(n):
        total += data_view[i] # 直接访问底层数据
    return total
```

通过将`total`声明为`cdef double`并使用类型化的`memoryview`（`double[:]`）来高效访问输入数据（尤其适用于NumPy数组），Cython生成一个C循环，直接对机器级的双精度浮点数进行操作。这避免了循环内大部分Python对象开销，从而为计算密集型代码带来了显著的速度提升。

### Cython开发流程

使用Cython会在你的开发过程中引入一个编译步骤：

1. **编写Cython代码：** 创建一个`.pyx`扩展名的文件（例如，`my_module.pyx`）。你可以先将你的慢速Python代码复制到其中。
2. **添加静态类型（可选但推荐）：** 识别性能关键的变量和函数参数 (parameter)，并使用`cdef`添加C类型声明。对于函数，你可以使用`cdef`定义主要从其他Cython代码调用的函数（最快），或使用`cpdef`创建高效的C和Python可调用版本。标准Python `def`函数仍然可以从Python调用，但在调用时会产生开销。
3. **创建`setup.py`脚本：** 使用Python的`setuptools`库来告诉Python如何构建你的Cython代码。

   ```python
   # setup.py
   from setuptools import setup
   from Cython.Build import cythonize
   import numpy # 机器学习中经常需要

   setup(
       name="我的优化模块",
       ext_modules=cythonize("my_module.pyx"),
       include_dirs=[numpy.get_include()] # 如果使用NumPy C API特性，这是必需的
   )
   ```
4. **构建扩展：** 在与`setup.py`相同的目录下，从你的终端运行编译过程：

   ```bash
   python setup.py build_ext --inplace
   ```

   此命令调用Cython编译器生成C代码，然后调用系统的C编译器在当前目录中创建最终的扩展模块（`my_module.so`或`my_module.pyd`）。
5. **导入和使用：** 在你的Python脚本或解释器中，你现在可以导入编译后的模块：

   ```python
   import my_module
   import numpy as np

   # 假设 sum_values_cython 定义在 my_module.pyx 中
   data = np.random.rand(1_000_000)
   result = my_module.sum_values_cython(data) # 调用快速Cython函数
   print(result)
   ```

### Cython与NumPy

Cython与NumPy数组配合得非常好。它提供了一些机制，特别是**memoryviews**，可以直接访问NumPy数组的原始数据缓冲区而没有Python开销。这使得你可以在NumPy数组中保存的大型数据集上编写C速度的循环，这对许多机器学习 (machine learning)算法和数据预处理步骤都非常有益。上面的`sum_values_cython`示例正是为此使用了memoryview（`double[:] data_view`）。



[交互图表：Python与Cython性能对比](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/introduction-to-cython#plot-166j23w)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "Python与Cython性能对比",
    "xaxis": {
      "title": "实现方式"
    },
    "yaxis": {
      "title": "相对执行时间（越低越好）",
      "range": [
        0,
        11
      ],
      "tickvals": [
        0,
        2,
        4,
        6,
        8,
        10
      ]
    },
    "template": "plotly_white",
    "bargap": 0.4,
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "纯Python循环",
        "使用类型的Cython"
      ],
      "y": [
        10,
        1
      ],
      "marker": {
        "color": [
          "#ff6b6b",
          "#4263eb"
        ]
      },
      "name": "执行时间",
      "width": [
        0.5,
        0.5
      ]
    }
  ]
}
```

</details>



> 针对循环密集型数值任务的相对执行时间。带有静态类型的Cython可以比等效的纯Python代码提供数量级的加速。

### 何时考虑使用Cython

Cython特别适用于：

- **CPU密集型算法：** 主要由循环和数值计算而非I/O操作主导的函数。
- **逻辑复杂的代码：** NumPy向量 (vector)化难以或无法清晰应用的情况。
- **迭代算法：** 优化例程、模拟或自定义模型训练循环。
- **集成C/C++库：** Cython提供了一种简洁的方式来封装现有的C或C++代码，以便在Python中使用。

### 考量因素

虽然功能强大，但Cython引入了：

- **构建步骤：** 相比纯Python开发增加了复杂性。
- **学习曲线：** 需要了解C数据类型和Cython特定语法（`cdef`、memoryviews）才能达到最佳性能。
- **调试：** 调试编译后的C代码有时比调试Python更复杂，尽管有工具可以帮助。

Cython在优化能力上相比标准Python以及NumPy/Pandas等库级优化有了很大的提升。它允许你针对ML代码中特定的性能关键部分，以C语言的速度重写它们，同时将应用程序的其余部分保留在熟悉的Python中。对于需要在要求高的机器学习 (machine learning)应用中对性能进行细粒度控制的工程师来说，这是一项有价值的技术。

## 参考资料

- [Cython Documentation](https://cython.readthedocs.io/en/latest/) — The Cython Developers (2024)
  学习和参考 Cython 语言、编译器和功能的官方综合资源。
- [Cython: A Guide for Python Programmers](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHtoL9ppdQFpzsPQZeL2iDixKQIqwbg1nfvkpYAN1jayeLdP4gtUv3fYkccQ5N2MLVl44-fCllj-Bx2br26HF1PfhKohb5R5qIWQDOB7GOgTXTASkmdvNxRFE678330nBmMbJ1qLfD94hWmGsAJxf3hu_ZP) — Kurt W. Smith (2015)
  Publisher: O'Reilly Media; Pages: 254
  一本实用且详细的书籍，为寻求性能优化的 Python 开发人员介绍 Cython。
- [Extending and Embedding the Python Interpreter](https://docs.python.org/3/extending/index.html) — The Python Development Team (2024)
  提供对 Python C API 的基本见解，这是 Cython 如何与 Python 交互的基础。
- [NumPy C-API Reference](https://numpy.org/doc/stable/reference/c-api/index.html) — The NumPy Developers (2024)
  解释 NumPy 数组的底层 C 接口，Cython 内存视图利用该接口实现高效数据访问。

---

[上一节](03-%E5%A4%A7%E5%9E%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84Pandas%E9%AB%98%E6%95%88%E7%94%A8%E6%B3%95.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20Numba%20%E8%BF%9B%E8%A1%8C%E5%8D%B3%E6%97%B6%E7%BC%96%E8%AF%91.md)
