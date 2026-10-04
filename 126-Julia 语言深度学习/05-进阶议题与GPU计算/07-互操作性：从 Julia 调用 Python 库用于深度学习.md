# 互操作性：从 Julia 调用 Python 库用于深度学习

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/interoperability-python-julia-dl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 Julia 在深度学习 (deep learning)方面拥有一个快速扩大的生态系统，但 Python 拥有一个成熟且丰富的库、预训练 (pre-training)模型和工具集，这些都是经过多年发展而来的。从 Julia 访问这些资源可以显著扩展您的能力，使您能够将 Python 的专门功能整合到基于 Julia 的深度学习项目中，而无需重复造轮子。`PyCall.jl` 是 Julia 与 Python 进行互操作性的主要包，它能实现这种整合。

### PyCall.jl 的作用

`PyCall.jl` 允许您直接从 Julia 调用 Python 函数，在两种语言之间传递数据，并在 Julia 环境中管理 Python 对象。它充当一座桥梁，使得 Python 的库（包括机器学习 (machine learning)和数据科学中常见的 NumPy、SciPy、Pandas、Scikit-learn、PyTorch 和 TensorFlow/Keras 等）可供您的 Julia 程序使用。

### 设置 Python 互操作性环境

在调用 Python 代码之前，您需要安装 `PyCall.jl` 并确保它能找到 Python 安装。

1. **安装 PyCall.jl**：
   打开 Julia REPL 并使用包管理器：

   ```julia
   using Pkg
   Pkg.add("PyCall")
   ```
2. **Python 安装**：
   默认情况下，`PyCall.jl` 将尝试使用它通过 `Conda.jl` 自动管理的 Conda Python 安装。当您首次 `using PyCall` 或尝试导入 Python 模块时，如果 `PyCall` 未找到合适的 Python 安装，它会提示您安装 Miniconda。这通常是开始使用的最简单方法。

   另外，如果您有希望 `PyCall.jl` 使用的现有 Python 安装（例如系统 Python 或虚拟环境），您可以通过在 Julia 会话中首次加载 `PyCall` 之前设置 `PYTHON` 环境变量来配置它。例如，在您的 Julia 脚本或 REPL 中：

   ```julia
   ENV["PYTHON"] = "/path/to/your/python/executable"
   using PyCall
   ```

   确保此路径指向 Python 可执行文件本身，而不仅仅是目录。

一旦 `PyCall.jl` 设置完成，您就可以开始导入和使用 Python 模块了。

### 基本 Python 交互

通过 `PyCall.jl` 与 Python 库进行交互非常直接。

1. **导入 Python 模块**：
   `pyimport` 函数用于导入 Python 模块。此函数返回一个 Julia 对象，该对象充当 Python 模块的代理。

   ```julia
   using PyCall

   # 导入 Python 的 'math' 模块
   math = pyimport("math")

   # 导入 NumPy
   np = pyimport("numpy")
   ```
2. **调用 Python 函数和访问属性**：
   您可以使用熟悉的 Julia 点语法调用这些代理对象的函数和访问其属性。

   ```julia
   # 调用 Python math 模块的 sqrt 函数
   py_sqrt_val = math.sqrt(25.0)
   println("Python math.sqrt(25.0): $py_sqrt_val") # 输出: Python math.sqrt(25.0): 5.0

   # 创建一个 NumPy 数组
   py_array = np.array([1, 2, 3, 4])
   println("NumPy array: $py_array")

   # 访问一个属性（例如，NumPy 的 pi 常量）
   py_pi = np.pi
   println("NumPy pi: $py_pi")
   ```

`PyCall.jl` 会自动处理许多数据类型转换。例如，Julia 数字会转换为 Python 数字，Julia 字符串转换为 Python 字符串，Julia 数组通常转换为 NumPy 数组，反之亦然，当 Python 函数返回值时。

下面的图表展示了您的 Julia 代码使用 `PyCall.jl` 调用 Python 库时的典型交互路径。

> 当 Julia 通过 PyCall.jl 调用 Python 库时的交互流程。数据和函数调用在 Julia 和 Python 环境之间进行调度。

### 借鉴 Python 深度学习 (deep learning)库

`PyCall.jl` 在深度学习中的一个重要用例是访问 Python 丰富的深度学习库，例如用于 NLP 模型的 `transformers`、用于图像处理工具的 `scikit-image`，甚至是来自 `PyTorch` 或 `TensorFlow` 的特定层或优化器，如果 Julia 中没有直接可用或合适的等效项。

例如，您可能希望使用 Hugging Face `tokenizers` 库中的先进分词 (tokenization)器 (tokenizer)。

首先，请确保 Python 库已安装在 `PyCall.jl` 所使用的 Python 环境中。如果 `PyCall` 管理自己的 Conda 环境，您可以使用 `Conda.jl` 将包安装到其中：

```julia
using Conda
Conda.add("tokenizers", channel="huggingface") # Hugging Face tokenizers 的示例
Conda.add("torch") # PyTorch 的示例
```

然后，您可以导入并使用它：

```julia
using PyCall

# 从 Hugging Face Transformers 导入 AutoTokenizer（假设已安装）
# 注意：此处使用的是 Python 包名
try
    transformers = pyimport("transformers")
    AutoTokenizer = transformers.AutoTokenizer

    # 加载预训练分词器
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 对一些文本进行分词
    julia_text = "Hello, Julia and Python working together!"
    encoded_input = tokenizer(julia_text, return_tensors="pt") # pt for PyTorch tensors

    println("输入文本: $julia_text")
    println("分词结果 (ID): $(encoded_input["input_ids"])")
    # 输出将是一个封装了 PyTorch 张量的 PyObject。
    # 您可能需要进一步转换它以在 Julia/Flux 中使用。
catch e
    println("导入或使用 Python 库出错: $e")
    println("请确保 'transformers' 和 'torch' Python 包已安装在 PyCall 的 Python 环境中。")
end
```

### 数据转换和管理

`PyCall.jl` 在 Julia 和 Python 之间的数据类型转换方面做得很好。

- Julia `Array` 通常会自动转换为 NumPy 数组，反之亦然。由于 Flux.jl 张量通常是 Julia `Array`，这非常方便。
- 基本类型，如数字、字符串、布尔值、`Dict`（转换为 Python `dict`）和 `Vector`（转换为 Python `list`）通常都能处理。

然而，有时您会从 Python 调用中接收到 `PyObject`。这是围绕 Python 对象的通用 Julia 封装器。您通常可以直接在后续调用其他 Python 函数时使用此 `PyObject`。如果您需要将 `PyObject` 转换为特定的 Julia 类型，可以使用 `convert(JuliaType, py_object)`。

```julia
np = pyimport("numpy")
py_list = np.array([10, 20, 30]) # 这会返回一个封装了 NumPy 数组的 PyObject
julia_vector = convert(Vector{Int}, py_list)
println("转换后的 Julia 向量: $julia_vector") # 输出: 转换后的 Julia 向量: [10, 20, 30]
```

请注意数据传输。尽管 `PyCall.jl` 效率高，但 Julia 和 Python 内存空间之间频繁的大量数据传输可能会引入性能开销。对于兼容类型和内存布局的 NumPy 数组和 Julia 数组，`PyCall.jl` 有时可以避免数据复制，使它们共享相同的底层内存。这对于大型数值数组尤其有用。

### 将 Python 组件整合到 Flux.jl 工作流中

您可以在 Flux.jl 流水线的各个阶段整合 Python 组件：

- **数据预处理**：使用 Python 库进行专门的数据加载、增强（例如，图像的 `albumentations`）或特征提取。
- **模型组件**：虽然在核心模型架构中不太常见（因为您可能会损失 Julia 在 Python 部分的一些性能优势），但理论上您可以将 Python 函数作为自定义 Flux 层的一部分进行调用，前提是处理好数据转换。
- **评估**：如果 Python 提供的度量或可视化工具有特定优势，则可以使用它们。
- **使用预训练 (pre-training)模型**：例如，通过 `PyCall.jl` 加载 PyTorch 模型，向其传递 Julia 数据（转换为 NumPy 数组或 PyTorch 张量），并获取预测结果。

**示例：将 Python 工具与 Flux 数据结合使用**

假设您有一个来自 Flux 的张量，并希望在其上使用 NumPy 函数：

```julia
using Flux
using PyCall

np = pyimport("numpy")

# 一个 Flux 张量（它只是一个 Julia 数组）
flux_tensor = rand(Float32, 2, 3)
println("Flux 张量 (Julia 数组):\n$flux_tensor")

# PyCall 自动将 Julia 数组转换为 NumPy 数组以供 NumPy 函数使用
numpy_sum = np.sum(flux_tensor, axis=0) # 直接传递 Julia 数组
println("沿轴 0 求和（通过 NumPy）:\n$numpy_sum")

# 结果 'numpy_sum' 是一个 PyObject（封装了 NumPy 数组）。
# 如果后续 Flux 操作需要，请将其转换回 Julia 数组。
julia_sum_vector = convert(Vector{Float32}, numpy_sum)
println("转换回 Julia 向量:\n$julia_sum_vector")
```

如果您通过 `PyCall.jl` 将此 `flux_tensor` 传递给 PyTorch 模型，您通常需要先将其转换为 PyTorch 张量，这通常通过 NumPy 数组进行：

```julia
# 假设已导入 'torch' 且 'flux_tensor' 是您的 Julia 数组
# 1. 将 Julia 数组转换为封装 NumPy 数组的 PyObject（通常是自动的，或使用 PyObject(flux_tensor)）
# 2. 将 NumPy PyObject 转换为 PyTorch Tensor PyObject
# py_numpy_array = np.asarray(flux_tensor) # 确保它是一个 NumPy 数组
# py_torch_tensor = torch.from_numpy(py_numpy_array)
# 现在 py_torch_tensor 可以馈送给 PyTorch 模型了。
```

### 考虑事项和建议

尽管 `PyCall.jl` 提供了很大的灵活性，但请记住以下几点：

- **性能开销**：从 Julia 调用 Python 函数会带来一定的开销，原因在于数据转换以及 Julia 和 Python 运行时之间的上下文 (context)切换。对于性能敏感的循环或调用非常频繁的函数，这种开销会变得显著。如果可能，对于此类部分，请优先选择原生的 Julia 解决方案。
- **依赖管理**：同时管理 Julia（`Project.toml`、`Manifest.toml`）和 Python（例如 Conda 环境、`requirements.txt`）的依赖会增加项目的复杂性。请清楚地记录两者的设置。
- **调试**：调试跨 Julia 和 Python 的问题可能比在单一语言中工作更具挑战性。堆栈跟踪可能涉及两种语言的调用。
- **类型稳定性**：Julia 的性能通常依赖于类型稳定性。与 Python 交互时，Python 调用返回的类型可能不会总是被 Julia 编译器精确推断，除非明确转换，这可能会影响使用这些结果的 Julia 代码的性能。
- **何时使用**：在以下情况中，互操作性最具优势：
  - 一个特定且成熟的 Python 库提供了 Julia 中尚不可用或不那么全面的功能。
  - 您需要将现有的 Python 代码或预训练 (pre-training)模型整合到 Julia 工作流中。
  - Python 部分的计算成本足够高，使得 `PyCall.jl` 的开销可以忽略不计，或者便利性高于性能成本。

`PyCall.jl` 是连接 Julia 和 Python 生态系统的一个强大工具。通过了解如何有效地使用它，您可以在深度学习 (deep learning)项目中借鉴两种语言的优势，提高您的生产力，并扩展您可以处理的问题范围。然而，请始终权衡其优势与潜在的复杂性和性能考量，当原生 Julia 解决方案能高效满足您的需求时，应优先选择它们。

## 参考资料

- [PyCall.jl Documentation](https://github.com/JuliaPy/PyCall.jl) — Steven G. Johnson, et al. (2023)
  Publisher: JuliaPy
  PyCall.jl的官方文档，详细介绍了安装、用法、类型转换和Julia-Python互操作性的高级功能。
- [Calling C and Fortran Code (from the Julia Manual)](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/) — The Julia Language Developers (2024)
  提供了Julia外部函数接口（FFI）的基础知识，这是PyCall.jl等包与外部语言库交互的底层机制。
- [Julia for Data Science](https://www.packtpub.com/product/julia-for-data-science/) — Huda Nassar and Bogdan Oancea (2023)
  Publisher: Packt Publishing
  一本实用指南，涵盖了Julia中用于数据科学的数据处理、机器学习以及与Python等其他语言（通过PyCall.jl）的互操作性。

---

[上一节](06-Julia%E5%85%B6%E4%BB%96%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%93%E7%AE%80%E8%A6%81%E6%A6%82%E8%A7%88.md) · [下一节](08-Julia%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F%E7%9A%84%E9%83%A8%E7%BD%B2%E6%96%B9%E5%BC%8F.md)
