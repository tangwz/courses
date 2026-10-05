# 导入特定名称：`from ... import`

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/python-from-import)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准的 `import module_name` 语句会将整个模块的功能引入到你的程序中，但你总是需要在使用时，在各项前面加上模块名作为前缀（例如 `math.pi` 或 `random.randint()`）。这通常是一个好习惯，因为它能保持名称来源清晰，并避免意外的命名冲突。

不过，有时你可能只需要模块中的一两个特定函数或变量，并且计划频繁使用它们。反复输入模块名可能会感到繁琐。Python 为此提供了一种替代语法，即 `from ... import`。它允许你将特定名称直接导入到脚本的当前命名空间，让你无需模块前缀即可使用它们。

### `from ... import` 语法

基本结构如下：

```python
from module_name import name1, name2, ...
```

让我们再次查看我们的 `math` 模块示例。假设我们只需要 `sqrt` 函数（用于计算平方根）和常量 `pi`。

之前，我们会这样做：

```python
import math

radius = 5
area = math.pi * (radius ** 2)
hypotenuse = math.sqrt(3**2 + 4**2)

print(f"Area: {area}")
print(f"Hypotenuse: {hypotenuse}")
```

使用 `from ... import`，我们可以这样写：

```python
from math import pi, sqrt

radius = 5
# 现在不再需要 'math.' 前缀
area = pi * (radius ** 2)
hypotenuse = sqrt(3**2 + 4**2)

print(f"Area: {area}")
print(f"Hypotenuse: {hypotenuse}")
# 输出：
# Area: 78.53981633974483
# Hypotenuse: 5.0
```

如你所见，`pi` 和 `sqrt` 现在可以直接使用，使得代码略微简洁。

### 权衡：命名空间与清晰度

`from ... import` 的便利性带来了一个潜在代价：**命名空间污染**和清晰度降低。

**命名空间**类似于一个字典，Python 在其中存储当前已定义且可访问的名称（变量、函数、类）。当你使用 `import math` 时，Python 会为 `math` 创建一个条目，并且其所有内容都通过该 `math` 条目访问（例如 `math.pi`）。`math` 模块内部的名称不会直接与你脚本中定义的名称混淆。

然而，当你使用 `from math import pi` 时，`pi` 名称本身会被直接添加到脚本的主命名空间。如果你脚本中已经有一个名为 `pi` 的变量，它将被从 `math` 导入的名称覆盖，可能导致意外行为或错误。

考虑这个例子：

```python
# 定义我们自己的 'sqrt' 函数（可能不太好）
def sqrt(number):
  print(f"Maybe finding the square root of {number}?")
  return number / 2 # 计算错误！

# 首先计算一些不相关的东西
result1 = sqrt(100)
print(f"Our initial sqrt(100) result: {result1}")

# 现在，从 math 模块导入真正的 sqrt
from math import sqrt

# 再次尝试使用 sqrt - 哪个会运行？
result2 = sqrt(100)
print(f"Result after importing math.sqrt: {result2}")

# 输出：
# Maybe finding the square root of 100?
# Our initial sqrt(100) result: 50.0
# Result after importing math.sqrt: 10.0
```

`from math import sqrt` 语句用 `math` 模块中的 `sqrt` 函数覆盖了我们自定义的 `sqrt` 函数。虽然这个例子是人为编造的，但在包含许多变量和函数的大型项目中，如果你直接导入许多名称，意外覆盖名称成为真实可能。

### 导入所有内容：`from module import *`（谨慎使用！）

Python 提供了一种变体：`from module_name import *`。星号 (`*`) 是一个通配符，表示“导入所有内容”。

```python
# 注意：通常不推荐！
from math import *

# 现在 math 中的所有名称都直接可用
print(pi)
print(sqrt(16))
print(cos(0))
# 输出：
# 3.141592653589793
# 4.0
# 1.0
```

这会将指定模块中的所有公共名称直接导入到你当前的命名空间。虽然它对于交互式使用或非常小的脚本可能显得方便，但**在大型程序中强烈不推荐**。为什么？

1. **命名空间冲突的高风险：** 它可能会将数十或数百个名称倾倒到你的命名空间中，使意外的名称冲突几乎不可避免。
2. **可读性降低：** 很难分辨特定函数或变量的来源。看到 `sqrt(100)` 并不能立即告诉你它是内置函数、在当前文件中定义，还是从 `math`（或使用 `*` 导入的其他模块）导入的。标准的 `import math` 后跟 `math.sqrt(100)` 则没有歧义。

尽量少用 `from module import *`，甚至不用。标准实践倾向于 `import module_name` 的明确性或 `from module_name import specific_name` 的选择性。

### 使用别名导入：`from ... import ... as ...`

如果你想导入一个特定名称，但它与脚本中的现有名称冲突，或者你只是喜欢使用不同的名称，该怎么办？你可以使用 `as` 来创建别名，就像在使用标准 `import` 语句时一样。

```python
from math import pi as mathematical_pi

# 使用我们自己的名为 'pi' 的变量
pi = 3.14 # 一个精度较低的版本，可能用于特定需求

radius = 5
# 使用别名导入来获取精确值
area = mathematical_pi * (radius ** 2)

print(f"Our pi: {pi}")
print(f"Imported pi: {mathematical_pi}")
print(f"Calculated area: {area}")

# 输出：
# Our pi: 3.14
# Imported pi: 3.141592653589793
# Calculated area: 78.53981633974483
```

这让你在无需前缀的情况下直接访问，同时也能解决潜在的名称冲突或允许更具描述性的命名。

### 何时使用哪种导入方式

- **`import module_name`**：这是最常见且通常推荐的方法。
  - 保持命名空间独立和整洁。
  - 使代码明确且更易于阅读（你总是知道 `module_name.function()` 的来源）。
  - 当你需要模块中的多个项或希望获得最大清晰度时最佳。
- **`from module_name import name1, name2`**：在以下情况使用：
  - 你只从模块中需要少数特定项。
  - 你将频繁使用这些项，并且 `module_name.` 的冗余变得分散注意力。
  - 你确信没有命名冲突，或者你使用 `as` 处理了它们。
- **`from module_name import name as alias`**：在使用 `from ... import` 时，这对于防止命名冲突或缩短长名称很有用。
- **`from module_name import *`**：由于命名空间污染和可读性问题，在生产代码或任何较大规模的脚本中避免使用此方法。对于交互式 Python 会话中的快速测试，它可能是可以接受的。

了解这些不同的代码导入方式，对于有效使用模块和构建结构良好的 Python 程序来说非常重要。当你开始使用 Python 的标准库和第三方包时，你将不断做出选择，以最佳方式导入所需功能。

## 参考资料

- [The Python Tutorial, Section 6: Modules](https://docs.python.org/3/tutorial/modules.html) — Python Software Foundation (2024)
  解释模块结构、不同导入语句及其对命名空间影响的官方 Python 文档。
- [Learning Python](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  详细介绍 Python 模块、导入机制以及各种导入形式对程序结构和命名空间影响的全面指南。
- [Google's Python Class - Modules](https://developers.google.com/edu/python) — Google Developers (2024)
  Publisher: Google
  提供 Python 模块导入的实例和解释，包括命名空间管理和避免常见问题的实践。

---

[上一节](02-%E5%AF%BC%E5%85%A5%E6%A8%A1%E5%9D%97%EF%BC%9Aimport%20%E8%AF%AD%E5%8F%A5.md) · [下一节](04-Python%20%E6%A0%87%E5%87%86%E5%BA%93%E6%A6%82%E8%BF%B0.md)
