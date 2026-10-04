---
course: "advanced-python-programming-ml"
chapter: "metaprogramming-python-internals-ml"
lesson: "dynamic-code-generation-execution"
sourceId: 2308
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-3-metaprogramming-python-internals-ml/dynamic-code-generation-execution"
title: "动态代码生成与执行"
description: "了解为自适应ML系统动态生成和执行Python代码的技术。"
order: 4
plots: []
sourceHash: "88479d80c366a6311e4871df1dd01700b0f37ab7a43b7dae3a6568ecaa012fd2"
sourceCorrections: []
---

虽然像装饰器和元类这样的元编程技术允许修改现有结构，但Python也提供了在运行时生成和执行全新代码构造的机制。这种能力被称为动态代码生成与执行，为构建高度自适应和配置驱动的机器学习 (machine learning)系统提供了一个主要途径。您无需对每种可能的行为进行硬编码，而是可以根据用户输入、配置文件，甚至是ML流程中的中间结果来构建和运行代码。

然而，这种能力也伴随着重大责任。执行动态生成的代码，特别是当它受到外部源影响时，会引入潜在的安全风险，并可能使调试更具挑战性。理解这些工具及其影响对于开发复杂的ML框架很有必要。

### 使用`exec()`执行语句

内置的`exec()`函数动态执行Python代码。它接受包含Python语句的字符串，或一个预编译的代码对象，并执行它。

```python
import numpy as np

# 定义一个字符串形式的处理步骤
code_string = """
import numpy as np # 在执行范围内可能需要导入
def dynamic_transform(data):
    # 示例：根据某些运行时参数应用缩放
    scale_factor = 2.5
    print(f"Applying dynamic scaling with factor: {scale_factor}")
    return data * scale_factor
"""

# 准备一个字典来存放exec的命名空间
execution_namespace = {}

# 在指定的命名空间内执行代码字符串
exec(code_string, execution_namespace)

# 访问由exec创建的函数
transform_func = execution_namespace['dynamic_transform']

# 使用动态创建的函数
data = np.array([1, 2, 3, 4])
transformed_data = transform_func(data)
print(f"Transformed data: {transformed_data}")
# 预期输出：
# 应用动态缩放因子：2.5
# 转换后的数据：[ 2.5  5.   7.5 10. ]
```

`exec()`可以选择性地接受`globals`和`locals`字典来控制代码执行的命名空间。如果只提供了`globals`，它将同时用于两者。如果两者都没有提供，代码将在当前作用域中运行，这可能是有风险的，因为它可能会无意中修改局部或全局变量。在示例中使用专用的字典（`execution_namespace`）通常更安全。

**在ML中的应用：** 设想一个特征工程步骤在配置文件中定义的情景。`exec()`可以用来定义和执行与这些步骤对应的Python函数，从而允许用户自定义流程，而无需修改核心框架代码。

**警告：** 切勿将`exec()`与来自不可信源（如通过网络的的用户输入）的字符串一起使用。攻击者可能会注入恶意代码以在您的系统上执行。

### 使用`eval()`评估表达式

与`exec()`类似，`eval()`函数也从字符串或代码对象执行代码，但它仅限于评估单个*表达式*，而不是语句（如赋值或`def`）。它返回评估表达式的结果。

```python
import math

expression_string = "math.log(data_point * factor + 1)"

# 评估上下文
evaluation_context = {
    'math': math,
    'data_point': 10,
    'factor': 0.5
}

result = eval(expression_string, evaluation_context)
print(f"Result of evaluating '{expression_string}': {result}")
# 预期输出：
# 评估 'math.log(data_point * factor + 1)' 的结果：1.791759469228055
```

`eval()`也接受`globals`和`locals`参数 (parameter)用于命名空间控制。与`exec()`一样，如果与不可信输入一起使用，它也会带来安全风险，因为表达式仍然可以调用有害函数或访问敏感数据。

**在ML中的应用：** `eval()`可用于根据以字符串形式存储的公式动态计算指标，评估配置中定义的模型性能阈值，或解析简单的基于规则的系统。

### 使用`compile()`预编译代码

使用`exec()`或`eval()`从字符串执行代码，每次都涉及字符串的解析和编译。如果您需要多次执行相同的动态代码，这种开销可能会变得很大。`compile()`函数允许您将字符串预编译成一个*代码对象*。然后，这个代码对象可以被`exec()`或`eval()`高效执行。

`compile()`接受三个主要参数 (parameter)：

1. `source`：包含代码的字符串或AST对象。
2. `filename`：一个字符串，表示代码读取的源文件名（用于回溯；可以是描述性名称，如`<string>`或`<generated_code>`）。
3. `mode`：指定要编译的代码类型：
   - `'exec'`：用于编译一系列语句（与`exec()`一起使用）。
   - `'eval'`：用于编译单个表达式（与`eval()`一起使用）。
   - `'single'`：用于编译单个交互式语句（打印表达式的结果）。

```python
# 将被多次执行的代码
operation_string = "result = input_value ** power"
filename_tag = "<dynamic_power_calc>"

# 编译代码字符串以便与exec()一起执行
compiled_code = compile(operation_string, filename_tag, 'exec')

# 在不同上下文中多次执行编译后的代码
context1 = {'input_value': 5, 'power': 2}
exec(compiled_code, context1)
print(f"Result 1: {context1['result']}") # 输出：结果 1: 25

context2 = {'input_value': 3, 'power': 3}
exec(compiled_code, context2)
print(f"Result 2: {context2['result']}") # 输出：结果 2: 27
```

通过一次编译，后续的`exec()`调用会跳过解析和编译步骤，可能提高循环或频繁调用的动态函数的性能。提供有意义的`filename`有助于调试，因为错误回溯会指向这个标签。

### 动态生成函数和类

执行预定义字符串时，您可以程序化地构建字符串本身，或使用其他机制来即时创建函数和类。

**函数工厂：** 函数可以创建并返回其他函数，通常借助闭包来捕获状态。

```python
def create_polynomial_func(coefficients):
    """生成一个评估多项式的函数。"""
    def polynomial(x):
        res = 0
        for i, coeff in enumerate(coefficients):
            res += coeff * (x ** i)
        return res
    return polynomial

# 动态创建一个二次函数：2 + 3x + 1x^2
quadratic = create_polynomial_func([2, 3, 1])

print(f"quadratic(0) = {quadratic(0)}") # 输出：quadratic(0) = 2
print(f"quadratic(2) = {quadratic(2)}") # 输出：quadratic(2) = 12
```

**使用`type()`动态创建类：** 内置函数`type()`在以三个参数 (parameter)`type(name, bases, dict)`调用时，可作为类工厂。

```python
# 动态创建一个简单的数据持有类
DynamicDataHolder = type(
    'DynamicDataHolder',  # 类名
    (object,),            # 基类（元组）
    {                     # 属性和方法字典
        '__init__': lambda self, value: setattr(self, 'data', value),
        'get_data': lambda self: self.data,
        '__repr__': lambda self: f"DynamicDataHolder(data={self.data})"
    }
)

instance = DynamicDataHolder(value=100)
print(instance)          # 输出：DynamicDataHolder(data=100)
print(instance.get_data()) # 输出：100
```

这允许您根据运行时信息定义类结构，这对于生成表示特定数据模式或在程序执行期间发现的配置的类很有用。

> 此图说明了ML框架中从配置动态生成代码的流程。

### ML中的应用与考量

动态代码技术使ML框架中可以实现多种模式：

1. **配置驱动行为：** 在外部文件（YAML、JSON）中定义流程阶段、特征转换乃至模型结构。框架读取配置并动态生成/执行相应的Python代码。
2. **领域特定语言（DSLs）：** 创建针对特定任务定制的简单语言，例如在强化学习 (reinforcement learning)中定义实验变体或复杂奖励函数。这些DSL字符串可以被动态解析和执行。
3. **自适应系统：** 实现能够根据性能或数据特征修改其处理逻辑的系统（尽管这需要非常仔细的设计和测试）。
4. **用于优化的代码生成：** 在某些情况下，可以为特定硬件或数据形状生成并运行时编译代码（类似于Numba或TensorFlow XLA等库的内部运作方式，尽管通常通过比`exec`更复杂的方式）。

**重要考量：**

- **安全性：** 最大风险所在。避免执行来自不可信输入的代码。对用于构建代码字符串的任何外部数据进行清理和验证。尽可能使用受限命名空间（`globals`/`locals`）。
- **调试：** 动态生成的代码中的错误可能更难追踪。在`compile()`中使用有意义的`filename`参数 (parameter)，在执行前记录生成的代码字符串，并编写彻底的单元测试。
- **可维护性：** 动态生成的代码比静态代码可读性更差，也更难理解。在获得的灵活性超过所增加的复杂性时，请谨慎使用。清晰地记录生成过程。
- **性能：** 尽管`compile()`有所帮助，但动态执行通常比运行静态定义代码的开销更大。如果性能是关注点，请对性能敏感部分进行分析。

动态代码生成与执行是高级工具。谨慎使用时，它们为构建适应性强且可配置的机器学习 (machine learning)系统提供了显著的灵活性。然而，始终优先考虑安全性、清晰性和可维护性，仅当问题确实受益于运行时代码操作时才采用这些技术。

## 参考资料

- [Built-in Functions (exec, eval, compile, type)](https://docs.python.org/3/library/functions.html) — Python Software Foundation (2024)
  理解Python动态代码执行和类型创建官方内置函数的主要来源。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056340/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本内容翔实的书籍，解释了Python的数据模型、元编程技术和运行时自省，这些都是动态代码生成的必要条件。
- [Numba: A Dynamic Python Compiler for Multicore and GPU Architectures](https://dl.acm.org/doi/10.1145/2833157.2833162) — Siu Kwan Lam, Antoine Pitrou, Stanley Seibert (2015)
  Journal: Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC; Publisher: ACM; Pages: 1-6; DOI: [10.1145/2833157.2833162](https://doi.org/10.1145/2833157.2833162)
  描述了Numba，一个先进的Python动态编译系统，展示了动态代码生成如何在科学计算和ML环境中用于性能优化。
