---
course: "python-for-beginners"
chapter: "introduction-object-oriented-concepts"
lesson: "python-object-methods"
sourceId: 2373
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-8-introduction-object-oriented-concepts/python-object-methods"
title: "方法：定义对象的行为"
description: "学习如何在类中定义方法（函数），以明确对象可以执行的行为或操作。"
order: 5
plots: []
sourceHash: "2033ad313e30cee93ef03c7ddec5ec7b6ea178f550ed5cc8c08125e48e474fb7"
sourceCorrections: []
---

类作为蓝图，属性存储根据该蓝图创建的每个对象的特定数据或状态。例如，一个 `Dog` 类可以规定狗有 `name`（名字）和 `breed`（品种），每个单独的 `Dog` 对象都会在其属性中保存其具体的名称和品种。

但对象不仅仅是数据的被动容器。它们经常需要执行操作或被执行操作。例如，狗可以叫，汽车可以行驶，银行账户可以存入资金。在面向对象编程中，这些操作或行为是使用**方法**定义的。

可以将方法看作是“属于”某个类的函数。它们与属性一样，定义在类块内部，但使用我们创建独立函数时学过的熟悉的 `def` 关键字。

## 定义方法

让我们为 `Dog` 类添加一个行为。我们希望任何 `Dog` 对象都能够“说话”。

```python
class Dog:
    # __init__ 方法初始化对象的属性
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
        print(f"Dog initialized: {self.name}")

    # 这是一个方法：定义在类内部的函数
    def bark(self):
        print(f"{self.name} says Woof!")

# 创建 Dog 类的一个实例（对象）
my_dog = Dog("Buddy", "Golden Retriever")

# 在 my_dog 对象上调用 bark 方法
my_dog.bark()
```

如果你运行这段代码，你会看到：

```text
Dog initialized: Buddy
Buddy says Woof!
```

注意到 `bark` 函数定义在 `Dog` 类内部。这是一个方法。

## `self` 参数 (parameter)：指代对象本身

你可能已经注意到 `__init__` 和 `bark` 中都出现了 `self` 参数。那是什么呢？

当你在类中定义一个方法时，该方法的**第一个参数**总是接收到调用该方法的*实例*（即特定对象）的引用。按照惯例，这个参数几乎总是命名为 `self`。

再次查看 `bark` 方法：

```python
    def bark(self):
        print(f"{self.name} says Woof!")
```

在 `bark` 内部，我们使用 `self.name`。这告诉 Python：“访问调用此 `bark` 方法的*特定对象*的 `name` 属性。”

当我们写 `my_dog.bark()` 时，Python 会自动将 `my_dog` 对象作为第一个参数传递给 `bark` 方法。因此，在方法调用内部，`self` *就是* `my_dog`。这就是该方法知道要打印哪只狗的名字的原因。

当你*调用*方法时（`my_dog.bark()`，而不是 `my_dog.bark(my_dog)`），你不需要显式传递 `self`。Python 会自动为你处理。但是，当你在类中*定义*方法时，你*必须*将 `self` 作为第一个参数包含在内。

## 方法可以访问和修改属性

方法是与对象数据（其属性）交互的主要方式。它们可以读取属性值，就像在 `bark` 方法中看到的那样，它们也可以改变属性值。

让我们添加一个方法来改变狗的名字：

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
        print(f"Dog initialized: {self.name}")

    def bark(self):
        print(f"{self.name} says Woof!")

    # 改变狗的名字属性的方法
    def set_name(self, new_name):
        if isinstance(new_name, str) and new_name: # 基本检查
            self.name = new_name
            print(f"My name has been changed to {self.name}")
        else:
            print("Please provide a valid name.")

# 创建实例
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Lucy", "Beagle")

# 使用方法
dog1.bark() # 输出：Buddy says Woof!
dog2.bark() # 输出：Lucy says Woof!

print(f"Dog 1's name is currently: {dog1.name}") # 输出：狗1当前的名字是：Buddy

# 在 dog1 上调用 set_name 方法
dog1.set_name("Max") # 输出：我的名字已改为 Max

print(f"Dog 1's name is now: {dog1.name}") # 输出：狗1现在的名字是：Max
dog1.bark() # 输出：Max says Woof!
```

在 `set_name` 方法中：

1. 它接受 `self`（特定的 `Dog` 对象）和 `new_name`（期望的新名字）。
2. 它执行一个简单的检查，看 `new_name` 是否为非空字符串。
3. 如果有效，它会使用 `self.name = new_name` 更新对象的 `name` 属性。
4. 注意 `dog1.set_name("Max")` 的调用只改变了 `dog1` 的名字。`dog2` 对象不受影响，因为在该特定的方法调用中，`self` 指的是 `dog1`。

## 方法与函数

把方法看作是与对象相关的函数会很有帮助。主要区别在于方法在对象的语境下运行（通过 `self`），并且可以直接访问和操作该对象的属性。独立函数不具备这种与对象状态的内在关联。

通过将数据（属性）和行为（方法）捆绑在类中，面向对象编程有助于你创建更规整、可重用且易于理解的代码，尤其当你的程序规模变大时。你正在将问题的组成部分建模为：既“了解”事物（属性），又“能做”事情（方法）的对象。

## 参考资料

- [The Python Tutorial - Classes](https://docs.python.org/3/tutorial/classes.html) — Python Software Foundation (2024)
  提供了关于Python中类、对象、属性和方法的官方基础介绍，包括`self`参数的重要性。
- [Learning Python, 5th Edition](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  一本关于Python语言的综合参考书，详细解释了面向对象编程，包括方法的定义和`self`机制。
