---
course: "python-for-beginners"
chapter: "introduction-object-oriented-concepts"
lesson: "python-self-parameter"
sourceId: 2376
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-8-introduction-object-oriented-concepts/python-self-parameter"
title: "`self` 参数说明"
description: "阐明类方法中惯用的 `self` 参数的用途，该参数指向实例本身。"
order: 7
plots: []
sourceHash: "dd4ff6475242e7fb096150f0db68be7c3233b69f918bedbaf89e06cd54caf4c8"
sourceCorrections: []
---

Python 类中定义的方法，例如 `__init__` 方法或其他自定义方法，几乎总是将 `self` 作为第一个参数 (parameter)。`self` 究竟是什么，它为什么会出现在那里？

回想一下从类创建对象（实例）的过程。你可以从同一个类蓝图创建多个对象，每个对象都有一组自己的属性值。例如：

```python
class Dog:
    def __init__(self, name, breed):
        # 为这个特定狗的属性赋值
        self.name = name
        self.breed = breed

    def describe(self):
        # 使用这个特定狗的名称和品种
        print(f"This dog is named {self.name} and is a {self.breed}.")

# 创建两个不同的 Dog 对象
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Lucy", "Poodle")
```

现在，假设你对这些对象中的一个调用 `describe` 方法：

```python
dog1.describe()
# 输出：这条狗名叫 Buddy，是金毛寻回犬。

dog2.describe()
# 输出：这条狗名叫 Lucy，是贵宾犬。
```

当调用 `dog1.describe()` 时，`describe` 方法怎么知道它应该使用名称“Buddy”和品种“Golden Retriever”？同样地，当调用 `dog2.describe()` 时，它又怎么知道使用“Lucy”和“Poodle”呢？

`self` 在这里发挥作用。当你像 `my_object.my_method(arg1, arg2)` 这样调用一个对象的方法时，Python 会自动将对象本身（在此例中是 `my_object`）作为*第一个*参数传递给该方法。按照惯例，方法定义中的这个第一个参数被命名为 `self`。

因此，在 `describe` 方法定义中：

```python
def describe(self):
    print(f"This dog is named {self.name} and is a {self.breed}.")
```

`self` 作为对调用该方法的特定实例（对象）的引用。

- 当 `dog1.describe()` 运行时，`self` 指向 `dog1` 对象。因此，`self.name` 访问 `dog1.name`（“Buddy”），`self.breed` 访问 `dog1.breed`（“Golden Retriever”）。
- 当 `dog2.describe()` 运行时，`self` 指向 `dog2` 对象。因此，`self.name` 访问 `dog2.name`（“Lucy”），`self.breed` 访问 `dog2.breed`（“Poodle”）。

简单来说，`self` 让方法的代码能够访问它所处理的特定对象实例的属性和其他方法。你可以使用 `self.attribute_name` 来获取该特定对象的属性值，或者使用 `self.method_name()` 来调用该对象的另一个方法。

**为什么叫 `self`？**

尽管 `self` 在 Python 中不是保留关键字（例如 `def`、`class` 或 `if`），但它是一个根深蒂固的约定。理论上，你可以将这个第一个参数命名为其他名称，例如 `this_object` 或 `instance`：

```python
# 理论上可行，但不推荐
class Cat:
    def __init__(this_cat, name):
        this_cat.name = name # 使用 'this_cat' 而不是 'self'

    def meow(this_cat):
        print(f"{this_cat.name} says Meow!") # 再次使用 'this_cat'

my_cat = Cat("Whiskers")
my_cat.meow()
# 输出：Whiskers 说喵！
```

然而，强烈不建议使用 `self` 以外的任何名称。遵循 `self` 的约定能让你的代码立即被其他 Python 开发者（以及你未来的自己！）识别和理解。这是定义实例方法时你应该始终遵循的规范做法。

总而言之，`self` 是 Python 类中实例方法第一个参数的惯用名称。它代表实例（对象）本身，使得方法可以访问对象的特定属性并调用其其他方法。无论何时你在对象上调用方法，Python 都会自动将该实例作为这个第一个参数传递。

## 参考资料

- [The Python Tutorial - 9. Classes](https://docs.python.org/3/tutorial/classes.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  官方文档，解释Python的类语法、实例创建和方法定义，其中演示了`self`的用法。
- [Learning Python](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQES0klGthI-24AoeX5qF3mgmL7_KBQD8pXTvjAij0qkOYwd64GnovP4a2woXhOJBRriSht-_aTipg_onbAes0F_xUJLtizgPV-OvypY5e5OMZ5C1hRt_wkXDmq3mHHuDqGSqTeTrdGfI3avXQlMzZWU-nVAIEwdkth_dBPrOtIMRSU=) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 1640
  一本关于Python的综合性书籍，涵盖面向对象编程基础知识以及通过`self`实现方法和实例属性的机制。第5版。
