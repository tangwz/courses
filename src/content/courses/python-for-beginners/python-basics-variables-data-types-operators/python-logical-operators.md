---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-logical-operators"
sourceId: 2277
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-logical-operators"
title: "使用逻辑运算符"
description: "了解如何使用逻辑运算符（and、or、not）组合布尔表达式。"
order: 7
plots: []
sourceHash: "fe33aa110a6d45bcd2a7470ba5308963049d70054e11f87b3738e394c41c4733"
sourceCorrections: []
---

诸如 `==`（等于）、`!=`（不等于）、`<`（小于）和 `>`（大于）之类的运算符用于比较值，并且总是产生布尔结果：`True` 或 `False`。逻辑运算符将这些布尔结果组合起来，从而实现更复杂的条件表达式。

通常，您需要同时检查多个条件。例如，您可能想知道用户是否*既*已登录*又*具有管理员权限，或者是否是星期六*或*星期日。这时，逻辑运算符就派上用场了。它们允许您组合布尔值（`True` 或 `False`）来创建更复杂的条件。Python 提供了三种逻辑运算符：`and`、`or` 和 `not`。

### `and` 运算符

`and` 运算符计算两个布尔表达式的值。仅当两个表达式都为 `True` 时才返回 `True`。如果其中任一表达式（或两者）为 `False`，则结果为 `False`。

可以把它想象成进入安全区域需要两种身份证明。您需要身份证 A *和* 身份证 B。只凭其中一种是不够的。

让我们看一个例子：

```python
age = 25
has_ticket = True

# 这个人年龄够了并且有票吗？
can_enter = (age >= 18) and (has_ticket == True)
print(f"Age is 25, has ticket: {can_enter}") # Output: True

age = 16
# 现在第一个条件 (age >= 18) 是 False
can_enter = (age >= 18) and (has_ticket == True)
print(f"Age is 16, has ticket: {can_enter}") # Output: False

age = 30
has_ticket = False
# 现在第二个条件 (has_ticket == True) 是 False
can_enter = (age >= 18) and (has_ticket == True)
print(f"Age is 30, no ticket: {can_enter}") # Output: False
```

比较条件 (`age >= 18`) 周围的括号 `()` 在这里并非严格必需，因为 `and` 的优先级低于比较运算符，但它们通常通过清晰地组合条件来使代码更易读。

以下是 `and` 的工作方式总结：

| 条件 1 | 条件 2 | 条件 1 `and` 条件 2 |
| --- | --- | --- |
| `True` | `True` | `True` |
| `True` | `False` | `False` |
| `False` | `True` | `False` |
| `False` | `False` | `False` |

### `or` 运算符

`or` 运算符也计算两个布尔表达式的值。如果至少一个表达式为 `True`，则返回 `True`。仅当两个表达式都为 `False` 时才返回 `False`。

想象一下，如果您是学生*或*老年人，就能获得折扣。满足其中一个条件就足够了。

请看这个例子：

```python
is_weekend = False
is_holiday = True

# 是周末还是假日？
day_off = is_weekend or is_holiday
print(f"Weekend: False, Holiday: True -> Day off: {day_off}") # Output: True

is_weekend = True
is_holiday = False
# 第一个条件是 True
day_off = is_weekend or is_holiday
print(f"Weekend: True, Holiday: False -> Day off: {day_off}") # Output: True

is_weekend = False
is_holiday = False
# 两个条件都是 False
day_off = is_weekend or is_holiday
print(f"Weekend: False, Holiday: False -> Day off: {day_off}") # Output: False
```

以下是 `or` 的总结：

| 条件 1 | 条件 2 | 条件 1 `or` 条件 2 |
| --- | --- | --- |
| `True` | `True` | `True` |
| `True` | `False` | `True` |
| `False` | `True` | `True` |
| `False` | `False` | `False` |

### `not` 运算符

`not` 运算符更简单。它作用于单个布尔表达式，并反转其值。如果表达式为 `True`，`not` 会使其变为 `False`。如果表达式为 `False`，`not` 会使其变为 `True`。

把它想象成一个电灯开关。`not on` 表示 `off`，而 `not off` 表示 `on`。

```python
is_logged_in = False
print(f"Is logged in: {is_logged_in}") # Output: False

# 检查用户是否未登录
needs_login = not is_logged_in
print(f"Needs login: {needs_login}") # Output: True

is_raining = True
print(f"Is raining: {is_raining}") # Output: True

# 检查是否没有下雨
can_go_outside = not is_raining
print(f"Can go outside: {can_go_outside}") # Output: False
```

`not` 的总结：

| 条件 | `not` 条件 |
| --- | --- |
| `True` | `False` |
| `False` | `True` |

### 组合逻辑运算符

您可以在一个表达式中组合多个逻辑运算符。Python 优先计算 `not`，然后是 `and`，最后是 `or`。然而，就像在算术中一样，您可以使用括号 `()` 来控制计算顺序，或者只是为了更清晰地表达您的意图。

```python
age = 22
is_student = True
has_coupon = False

# 如果 (age < 25 并且是学生) 或者有优惠券，则符合条件
is_eligible_discount = (age < 25 and is_student) or has_coupon
# 计算为: (True and True) or False
# -> True or False
# -> True
print(f"Eligible for discount: {is_eligible_discount}") # Output: True

# 改变优惠券状态
has_coupon = True
age = 30
is_student = False

is_eligible_discount = (age < 25 and is_student) or has_coupon
# 计算为: (False and False) or True
# -> False or True
# -> True
print(f"Eligible for discount (different criteria): {is_eligible_discount}") # Output: True
```

逻辑运算符是构建能够根据多个条件做出决定的程序的基本工具。在下一章讨论 `if` 语句等控制流结构时，您将看到它们的广泛应用，让您的程序根据复杂条件评估结果是 `True` 还是 `False` 做出不同的反应。

## 参考资料

- [Boolean Operations - and, or, not](https://docs.python.org/3/reference/expressions.html#boolean-operations) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  官方文档，详细说明了Python逻辑运算符的行为和优先级。
