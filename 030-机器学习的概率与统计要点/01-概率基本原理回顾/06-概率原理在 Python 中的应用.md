# 概率原理在 Python 中的应用

来源：[原文](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-1-probability-foundations-revisited/applying-probability-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解概率理论固然重要，但将这些原理转化为实际代码才能发挥其真正的作用，尤其是在机器学习 (machine learning)中。Python 及其丰富的科学计算库生态系统，为实验和应用概率原理提供了很好的环境。这里将指导如何使用常用的 Python 工具来实现概率概念。

我们将主要使用 NumPy，一个用于 Python 数值计算的基础软件包。它提供了高效的数组操作和强大的随机数生成模块，非常适合模拟概率情境。

### 模拟样本空间和事件

我们从基础开始：通过模拟来表示样本空间和计算事件概率。考虑掷一个标准的六面骰子。样本空间为 $\Omega = \{1, 2, 3, 4, 5, 6\}$。我们可以使用 NumPy 多次模拟掷骰子。

```python
import numpy as np

# 模拟掷一个公平的六面骰子 10,000 次
num_rolls = 10000
rolls = np.random.randint(1, 7, size=num_rolls) # 生成从 1（包含）到 7（不包含）的整数

# 计算掷出偶数的概率（事件 A = {2, 4, 6}）
# 我们统计满足事件条件的出现次数
is_even = (rolls % 2 == 0)
num_even_rolls = np.sum(is_even)

# 基于模拟的经验概率
prob_even_empirical = num_even_rolls / num_rolls

# 理论概率
prob_even_theoretical = 3 / 6 # {2, 4, 6} 占 {1, 2, 3, 4, 5, 6} 中的部分

print(f"前 10 次掷骰样本：{rolls[:10]}")
print(f"总掷骰次数：{num_rolls}")
print(f"偶数掷骰次数：{num_even_rolls}")
print(f"掷出偶数的经验概率：{prob_even_empirical:.4f}")
print(f"掷出偶数的理论概率：{prob_even_theoretical:.4f}")
```

随着 `num_rolls` 的增加，您会发现经验概率会越来越接近理论概率 0.5。这表明了大数定律的作用。

### 计算条件概率

条件概率 $P(A|B)$ 衡量的是在事件 $B$ 已经发生的情况下，事件 $A$ 发生的概率。我们可以从数据或模拟中估计这一点。

我们继续使用掷骰子的例子。设 $A$ 为掷出偶数（{2, 4, 6}）的事件， $B$ 为掷出大于 3 的数（{4, 5, 6}）的事件。我们想找到 $P(A|B)$。理论上，事件 $B$ 的结果是 {4, 5, 6}。其中，也属于 $A$ 的结果是 {4, 6}。因此，$P(A|B) = 2 / 3$。

我们通过模拟来验证这一点：

```python
# 使用上一个模拟中的 'rolls' 数组

# 定义事件 B：掷出大于 3 的数
is_greater_than_3 = (rolls > 3)
rolls_given_B = rolls[is_greater_than_3] # 筛选出事件 B 发生的掷骰结果

# 在 B 发生的条件下定义事件 A：在那些 > 3 的数中掷出偶数*
is_even_given_B = (rolls_given_B % 2 == 0)
num_A_given_B = np.sum(is_even_given_B)

# 事件 B 发生的总次数
num_B = len(rolls_given_B) # 或 np.sum(is_greater_than_3)

# 经验条件概率 P(A|B)
if num_B > 0:
    prob_A_given_B_empirical = num_A_given_B / num_B
    print(f"掷出大于 3 的次数（事件 B 发生）：{num_B}")
    print(f"掷出偶数且大于 3 的次数（事件 A 和 B 发生）：{num_A_given_B}")
    print(f"经验 P(A|B)：{prob_A_given_B_empirical:.4f}")
else:
    print("模拟中未发生事件 B。")

# 理论概率
prob_A_given_B_theoretical = 2 / 3
print(f"理论 P(A|B)：{prob_A_given_B_theoretical:.4f}")

# 检查独立性：P(A|B) == P(A) 吗？
# 我们之前计算过 P(A)（prob_even_empirical）
print(f"\n比较 P(A|B) ({prob_A_given_B_empirical:.4f}) 与 P(A) ({prob_even_empirical:.4f})")
# 由于 P(A|B) 不等于 P(A)，因此这些事件是相关的。
```

模拟结果给出的估计值接近理论值，证实了我们的理解。这种比较也使我们能够经验性地检查事件的相关性。

### 实现贝叶斯定理

贝叶斯定理是更新信念的基础。我们来实现这个公式：$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$。

考虑一个简单的例子：一种疾病的诊断测试。

- 设 $D$ 为一个人患有该疾病的事件。$P(D) = 0.01$（流行率：1% 的人口患有此病）。
- 设 $T$ 为测试结果呈阳性的事件。
- 该测试的敏感度为 $P(T|D) = 0.95$（正确识别 95% 的患病者）。
- 该测试的特异性为 $P(\neg T|\neg D) = 0.90$。这意味着假阳性概率为 $P(T|\neg D) = 1 - P(\neg T|\neg D) = 1 - 0.90 = 0.10$。

我们想找到 $P(D|T)$：给定阳性测试结果，一个人实际患有该疾病的概率。

首先，我们需要 $P(T)$，即阳性测试的总体概率。我们使用全概率定律：
$P(T) = P(T|D)P(D) + P(T|\neg D)P(\neg D)$
且 $P(\neg D) = 1 - P(D) = 1 - 0.01 = 0.99$。

让我们在 Python 中计算：

```python
# 已知概率
P_D = 0.01
P_T_given_D = 0.95 # 敏感度
P_notT_given_notD = 0.90 # 特异性

# 计算导出概率
P_notD = 1 - P_D
P_T_given_notD = 1 - P_notT_given_notD # 假阳性率

# 使用全概率定律计算 P(T)
P_T = (P_T_given_D * P_D) + (P_T_given_notD * P_notD)

# 应用贝叶斯定理计算 P(D|T)
P_D_given_T = (P_T_given_D * P_D) / P_T

print(f"P(D) = {P_D:.4f}（患病的先验概率）")
print(f"P(T|D) = {P_T_given_D:.4f}（敏感度）")
print(f"P(T|~D) = {P_T_given_notD:.4f}（假阳性率）")
print(f"P(T) = {P_T:.4f}（阳性测试的总体概率）")
print(f"P(D|T) = {P_D_given_T:.4f}（给定阳性测试结果患病的后验概率）")
```

请注意这个出人意料的结果！即使是来自一个相当敏感的测试的阳性结果，实际患病的概率仍然低于 9%。这突显了低先验概率（流行率）和假阳性率的影响。

### 模拟随机变量

我们可以使用 NumPy 模拟从表示随机变量的分布中抽取样本。虽然第 2 章将详细介绍具体的命名分布，我们可以在这里模拟简单的自定义离散随机变量。

想象一枚不均匀的硬币，正面 (H) 出现的概率为 $P(H) = 0.7$，反面 (T) 出现的概率为 $P(T) = 0.3$。我们设定 $X=1$ 代表正面，$X=0$ 代表反面。

```python
# 定义结果的概率（0 代表反面，1 代表正面）
outcomes = [0, 1]
probabilities = [0.3, 0.7] # P(X=0) = 0.3, P(X=1) = 0.7

# 模拟不均匀硬币的 1000 次抛掷
num_flips = 1000
flips = np.random.choice(outcomes, size=num_flips, p=probabilities)

# 计算经验概率
num_tails = np.sum(flips == 0)
num_heads = np.sum(flips == 1) # 或 num_flips - num_tails

prob_tails_empirical = num_tails / num_flips
prob_heads_empirical = num_heads / num_flips

print(f"模拟了 {num_flips} 次不均匀硬币抛掷。")
print(f"反面次数 (X=0)：{num_tails}，经验概率：{prob_tails_empirical:.4f}（理论值：0.3）")
print(f"正面次数 (X=1)：{num_heads}，经验概率：{prob_heads_empirical:.4f}（理论值：0.7）")
```

### 计算期望值和方差

对于具有可能值 $x_i$ 和概率 $P(X=x_i)$ 的离散随机变量 $X$，期望值是 $E[X] = \sum_i x_i P(X=x_i)$ 方差是 $Var(X) = E[(X - E[X])^2] = \sum_i (x_i - E[X])^2 P(X=x_i)$。

我们可以为我们的不均匀硬币（反面 $X=0$，正面 $X=1$）理论计算这些值：
$E[X] = (0 \times 0.3) + (1 \times 0.7) = 0.7$
$Var(X) = (0 - 0.7)^2 \times 0.3 + (1 - 0.7)^2 \times 0.7 = (-0.7)^2 \times 0.3 + (0.3)^2 \times 0.7$
$Var(X) = (0.49 \times 0.3) + (0.09 \times 0.7) = 0.147 + 0.063 = 0.21$

现在，我们使用 NumPy 从模拟中计算样本均值和方差，这应该近似于理论值：

```python
# 使用不均匀硬币模拟中的 'flips' 数组

# 计算样本均值（近似期望值）
sample_mean = np.mean(flips)

# 计算样本方差（近似方差）
# 注意：np.var 默认计算总体方差。
# 对于样本方差，请使用 ddof=1，这在推断中通常更受青睐，
# 但对于大型模拟，差异很小。
sample_variance = np.var(flips) # 在样本上使用总体方差公式
sample_variance_unbiased = np.var(flips, ddof=1) # 样本方差（无偏估计量）

print(f"\n理论 E[X] = 0.7")
print(f"样本均值 = {sample_mean:.4f}")

print(f"\n理论 Var(X) = 0.21")
print(f"样本方差 (ddof=0) = {sample_variance:.4f}")
print(f"样本方差 (ddof=1) = {sample_variance_unbiased:.4f}")
```

从模拟抛掷中计算出的样本均值和方差与我们推导出的理论期望值和方差非常接近。这种模拟与理论的联系非常重要。未来，使用 NumPy 和 SciPy 等 Python 库将使我们能够高效处理更复杂的概率分布和对机器学习 (machine learning)很重要的统计分析。

## 参考资料

- [Think Stats: Exploratory Data Analysis in Python](http://greenteapress.com/wp/think-stats-2e/) — Allen B. Downey (2014)
  Publisher: Green Tea Press
  使用Python介绍计算概率和统计，包含模拟和数据分析示例。
- [Introduction to Probability](http://probabilitybook.net) — Joe Blitzstein and Jessica Hwang (2019)
  Publisher: Chapman & Hall/CRC
  提供概率论的严谨学术基础，支持本节的理论内容。
- [NumPy Reference Manual](https://numpy.org/doc/stable/reference/index.html) — NumPy Developers (2025)
  NumPy库的官方文档，详细介绍了示例中使用的数值计算和随机数生成函数。

---

[上一节](05-%E6%9C%9F%E6%9C%9B%E5%80%BC%E4%B8%8E%E6%96%B9%E5%B7%AE.md) · [下一节](../02-%E5%B8%B8%E7%94%A8%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83/01-%E4%BC%AF%E5%8A%AA%E5%88%A9%E5%88%86%E5%B8%83%E5%92%8C%E4%BA%8C%E9%A1%B9%E5%88%86%E5%B8%83.md)
