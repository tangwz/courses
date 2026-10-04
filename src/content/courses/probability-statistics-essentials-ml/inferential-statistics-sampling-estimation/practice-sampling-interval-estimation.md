---
course: "probability-statistics-essentials-ml"
chapter: "inferential-statistics-sampling-estimation"
lesson: "practice-sampling-interval-estimation"
sourceId: 1429
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation/practice-sampling-interval-estimation"
title: "动手实践：抽样模拟与区间估计"
description: "使用 Python 进行模拟，以说明中心极限定理并计算置信区间。"
order: 7
plots: ["plots/1429-0.json", "plots/1429-1.json", "plots/1429-2.json"]
sourceHash: "7574523d04350a242e3f4e14912bc094959d4a81905c7bf56431c3264ea69ada"
sourceCorrections: []
---

中心极限定理允许我们使用样本均值对总体均值进行推断，即使原始总体不服从正态分布。置信区间为总体参数 (parameter)提供了一个合理值的范围。Python 将用于模拟这些过程。

### 模拟中心极限定理

中心极限定理（CLT）是推断统计学的一个基本原理。它指出，如果您从一个总体中抽取足够大的随机样本（有放回抽样或从足够大的总体中抽样），则样本均值的分布将近似服从正态分布，无论原始总体分布的形状如何。让我们来演示一下。

假设我们的总体服从指数分布。这种分布显然*不*是正态分布；它是严重偏斜的。

```python
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 定义总体参数（指数分布）
population_scale = 10  # 指数分布的尺度参数（beta），均值即为beta
population_size = 100000

# 生成总体数据
np.random.seed(42) # 用于结果复现
population_data = np.random.exponential(scale=population_scale, size=population_size)

# 绘制总体分布图
fig_pop = go.Figure(data=[go.Histogram(x=population_data, nbinsx=100, name='总体', marker_color='#1f77b4')]) # 使用默认蓝色
fig_pop.update_layout(
    title_text='总体分布（指数）',
    xaxis_title_text='值',
    yaxis_title_text='频率',
    bargap=0.1
)
# fig_pop.show() # 在真实的notebook环境中
```

> 上图显示了代表我们总体的指数分布的典型右偏形状。该分布的真实均值是尺度参数 (parameter)，我们将其设为 10。

现在，让我们模拟从该总体中抽取大量样本并计算每个样本均值的过程。我们将选择一个样本大小，例如 $n=30$，这通常被认为足以使 CLT 开始生效。我们将抽取 1000 个这样的样本。

```python
sample_size = 30
number_of_samples = 1000
sample_means = []

np.random.seed(123) # 不同的抽样种子
for _ in range(number_of_samples):
    sample = np.random.choice(population_data, size=sample_size, replace=True)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

# 计算抽样分布的理论均值和标准误
theoretical_mean = population_scale # 指数分布的均值为beta
theoretical_std_dev_pop = population_scale # 指数分布的标准差为beta
standard_error = theoretical_std_dev_pop / np.sqrt(sample_size)

# 绘制样本均值分布图
fig_means = go.Figure()

# 样本均值的直方图
fig_means.add_trace(go.Histogram(
    x=sample_means,
    nbinsx=30,
    name='样本均值',
    marker_color='#ff7f0e', # 使用默认橙色
    histnorm='probability density' # 归一化以与PDF比较
))

# 叠加理论正态分布（CLT预测）
x_norm = np.linspace(min(sample_means), max(sample_means), 100)
y_norm = stats.norm.pdf(x_norm, loc=theoretical_mean, scale=standard_error)
fig_means.add_trace(go.Scatter(
    x=x_norm,
    y=y_norm,
    mode='lines',
    name='正态分布（CLT）',
    line=dict(color='#2ca02c', width=2) # 使用默认绿色
))

fig_means.update_layout(
    title_text=f'样本均值的分布 (n={sample_size})',
    xaxis_title_text='样本均值',
    yaxis_title_text='密度',
    bargap=0.1,
    legend_title_text='分布'
)
# fig_means.show() # 在真实的notebook环境中

# 打印比较统计数据
print(f"总体均值: {np.mean(population_data):.4f}")
print(f"样本均值平均值: {np.mean(sample_means):.4f}")
print(f"总体标准差: {np.std(population_data):.4f}")
print(f"样本均值标准差（观测值）: {np.std(sample_means):.4f}")
print(f"理论标准误（总体标准差 / sqrt(n)）: {standard_error:.4f}")
```



![总体分布（指数）](plots/1429-0.json)



> 总体数据的直方图，显示其指数形状。真实总体均值为 10。



![样本均值的分布 (n=30)](plots/1429-1.json)



> 从指数总体中抽取的 1000 个样本均值的分布（橙色直方图），样本大小为 n=30。绿线显示了中心极限定理预测的理论正态分布，其均值以总体均值 (10) 为中心，标准差等于标准误。

模拟结果显示：

1. 样本均值的分布看起来很像正态分布，尽管基础总体是指数分布。
2. 样本均值的平均值非常接近真实总体均值 (10)。
3. 样本均值分布的标准差（观测标准误）接近理论标准误 ($\sigma / \sqrt{n}$)。

此次模拟显示了 CLT 的作用。它允许我们使用正态分布的特性，仅使用一个样本均值就可以对总体均值进行推断，前提是样本量足够。

### 计算均值的置信区间

现在，让我们取一个样本，并用它通过置信区间来估计总体均值。通常，我们不会知道总体标准差 ($\sigma$)，因此我们将使用样本标准差 ($s$) 作为估计值。当 $\sigma$ 未知且样本量相对较小（通常认为是 $n < 30$，尽管随着 $n$ 增大，这种区别变得不明显）时，t 分布比正态 (Z) 分布更适合查找临界值。然而，对于 $n=30$ 或更大的样本量，t 分布与正态分布非常接近。为保持普遍性，我们在这里使用 `scipy.stats` 提供的 t 分布。

让我们使用上面模拟循环中生成的*最后一个*样本：

```python
# 使用生成的最后一个样本
last_sample = sample # 来自上面的循环

# 计算样本统计量
sample_mean = np.mean(last_sample)
sample_std_dev = np.std(last_sample, ddof=1) # 使用ddof=1计算样本标准差
n = len(last_sample)
standard_error_sample = sample_std_dev / np.sqrt(n)

# 设置置信水平
confidence_level = 0.95
alpha = 1 - confidence_level

# 计算临界t值
degrees_freedom = n - 1
critical_t = stats.t.ppf(1 - alpha / 2, df=degrees_freedom) # ppf是逆累积分布函数

# 计算误差范围
margin_of_error = critical_t * standard_error_sample

# 计算置信区间
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"样本均值 (x_bar): {sample_mean:.4f}")
print(f"样本标准差 (s): {sample_std_dev:.4f}")
print(f"标准误 (SE): {standard_error_sample:.4f}")
print(f"自由度: {degrees_freedom}")
print(f"置信水平: {confidence_level*100}%")
print(f"临界t值: {critical_t:.4f}")
print(f"误差范围: {margin_of_error:.4f}")
print(f"置信区间: ({lower_bound:.4f}, {upper_bound:.4f})")

# 使用scipy.stats内置函数验证
ci_scipy = stats.t.interval(confidence_level, df=degrees_freedom, loc=sample_mean, scale=standard_error_sample)
print(f"置信区间（使用 scipy.stats.t.interval）: ({ci_scipy[0]:.4f}, {ci_scipy[1]:.4f})")
```

运行此代码将输出计算出的样本统计量以及由此产生的 95% 置信区间。例如，您可能会得到：

```
样本均值 (x_bar): 10.1325
样本标准差 (s): 9.8536
标准误 (SE): 1.7990
自由度: 29
置信水平: 95.0%
临界t值: 2.0452
误差范围: 3.6794
置信区间: (6.4531, 13.8119)
置信区间（使用 scipy.stats.t.interval）: (6.4531, 13.8119)
```

**说明：** 基于这个特定样本，我们有 95% 的信心认为总体真实均值（我们已知为 10）大约在 6.45 到 13.81 之间。注意，我们的区间成功涵盖了真实总体均值。

如果我们用许多不同的样本重复这个过程，我们会发现，计算出的 95% 置信区间中约有 95% 将包含真实总体均值。

### 多个置信区间的可视化

为了加强对置信区间的理解，让我们模拟抽取多个样本（例如 100 个），并为每个样本计算 95% 置信区间。然后我们可以绘制这些区间，并查看有多少涵盖了真实总体均值。

```python
num_intervals = 100
intervals = []
means = []
population_mean = population_scale # 真实均值 = 10

np.random.seed(999)
for i in range(num_intervals):
    sample_i = np.random.choice(population_data, size=sample_size, replace=True)
    mean_i = np.mean(sample_i)
    std_i = np.std(sample_i, ddof=1)
    se_i = std_i / np.sqrt(sample_size)
    df_i = sample_size - 1

    # 直接使用scipy的区间函数
    ci_i = stats.t.interval(confidence_level, df=df_i, loc=mean_i, scale=se_i)

    intervals.append(ci_i)
    means.append(mean_i)

# 创建绘图
fig_intervals = go.Figure()
captured_count = 0

for i in range(num_intervals):
    lower, upper = intervals[i]
    mean_val = means[i]
    captured = lower <= population_mean <= upper
    if captured:
        captured_count += 1
        line_color = '#2ca02c' # 如果涵盖则为绿色
    else:
        line_color = '#d62728' # 如果未涵盖则为红色

    # 绘制区间线
    fig_intervals.add_trace(go.Scatter(
        x=[lower, upper],
        y=[i, i],
        mode='lines',
        line=dict(color=line_color, width=1),
        showlegend=False
    ))
    # 绘制样本均值点
    fig_intervals.add_trace(go.Scatter(
        x=[mean_val],
        y=[i],
        mode='markers',
        marker=dict(color=line_color, size=3),
        showlegend=False
    ))

# 添加真实总体均值的线
fig_intervals.add_vline(x=population_mean, line=dict(color='#1f77b4', width=2, dash='dash'), name='真实均值')

fig_intervals.update_layout(
    title=f'{num_intervals} 个置信区间 ({confidence_level*100}%)',
    xaxis_title='值',
    yaxis_title='样本编号',
    yaxis_showticklabels=False,
    xaxis_range=[min(i[0] for i in intervals if i[0] is not None)-1, max(i[1] for i in intervals if i[1] is not None)+1], # 稍微调整范围
    height=400
)

# 添加涵盖率标注
capture_rate = captured_count / num_intervals
fig_intervals.add_annotation(
    x=population_mean + 2, y=num_intervals * 0.95, # 标注位置
    text=f"涵盖率: {capture_rate:.2f} ({captured_count}/{num_intervals})",
    showarrow=False,
    font=dict(size=12, color="black"),
    bgcolor="rgba(255,255,255,0.7)"
)

# fig_intervals.show()
```



![100 个置信区间 (95.0%)](plots/1429-2.json)



> 每条水平线代表从总体中不同随机样本（n=30）计算出的 95% 置信区间。每条线上的点是样本均值。垂直虚线是真实总体均值 (10)。绿色区间成功涵盖了真实均值；红色区间则没有。我们预计约 95% 的区间将是绿色的。

这种可视化清楚地表明，尽管任何一个置信区间可能涵盖或不涵盖真实的总体参数 (parameter)，但该*方法*能确保，如果重复多次，约 95% 的 95% 置信区间将包含真实值。模拟中观测到的涵盖率（例如，94/100 = 94%）通常非常接近名义置信水平 (95%)。

这些实际练习显示了模拟如何帮助对抽象统计思想建立直观认识，例如中心极限定理和置信区间。它们还显示了如何使用标准 Python 库轻松执行这些计算，使推断统计学可用于数据分析和机器学习 (machine learning)任务。

## 参考资料

- [Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  一门基础大学课程，涵盖中心极限定理、抽样分布和置信区间，通常包含讲座和习题，以巩固理论知识。
- [SciPy Reference Guide: Statistical Functions (scipy.stats)](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy community (2023)
  `scipy.stats` 模块的官方文档，详细介绍了在实践练习中使用的概率分布、统计检验和区间估计函数。
