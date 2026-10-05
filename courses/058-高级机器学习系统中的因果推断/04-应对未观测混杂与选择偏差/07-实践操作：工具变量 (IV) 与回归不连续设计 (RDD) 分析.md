# 实践操作：工具变量 (IV) 与回归不连续设计 (RDD) 分析

来源：[原文](https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-4-addressing-hidden-bias/practice-iv-rdd-analysis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

工具变量 (IV) 和回归不连续设计 (RDD) 是处理未观测混杂因素的有效方法。使用 Python 的实践例子展示了如何在简单回归可能产生偏差结果的情形中应用这些方法来估计因果效应。假定您拥有一个可用的 Python 环境，并安装了 `pandas`、`numpy`、`statsmodels`、`linearmodels` 和 `plotly` 等库。

### 背景介绍：未观测混杂因素问题

回顾主要问题：我们希望估计处理 $T$ 对结果 $Y$ 的因果效应，但存在一个同时影响 $T$ 和 $Y$ 的未观测变量 $U$。对 $Y$ 进行 $T$（和观测协变量 $X$）的简单回归将产生对因果效应的有偏差估计，因为它未能考虑 $U$ 的影响。


$$
Y = \alpha + \beta_{causal} T + \gamma' X + \delta U + \epsilon \\
T = \pi_0 + \pi_1 Z + \pi_2' X + \pi_3 U + \nu
$$


如果 $\delta \ne 0$ 且 $\pi_3 \ne 0$，那么 $T$ 是内生变量，并且 $\beta_{causal}$ 的标准回归估计是不一致的。IV 和 RDD 在特定结构假设下提供了获取一致估计的方法。

### 实现工具变量 (IV) 估计

IV 方法运用一个*工具变量* $Z$，该变量满足三个条件：

1. **相关性：** $Z$ 必须与处理 $T$ 相关（即，在控制 $X$ 后，上述方程中的 $\pi_1 \ne 0$）。
2. **排他性：** $Z$ *仅*通过其对 $T$ 的影响来影响结果 $Y$。它没有直接影响 $Y$ 的路径（即，一旦 $T$ 和 $X$ 被包含在内，$Z$ 不出现在 $Y$ 的方程中）。
3. **独立性（可忽略性）：** $Z$ 独立于未观测混杂因素 $U$（或更正式地说，独立于给定协变量的潜在结果）。

找到一个有效工具变量常需要大量领域知识和细致的论证。

#### 情景：估计教育回报

让我们思考一个经典计量经济学问题：估计受教育年限 ($T$) 对对数工资 ($Y$) 的因果效应。'天赋能力' ($U$) 是一个可能的未观测混杂因素，它直接影响教育程度和工资。一个潜在的工具变量 $Z$ 可以是成长过程中离大学的距离，或者如 Angrist 和 Krueger (1991) 的著名研究所示，出生季度，它因义务教育法而影响受教育年限，但可认为是与天赋能力或未来工资没有直接关系的。

#### 两阶段最小二乘法 (2SLS)

最常用的 IV 估计量是两阶段最小二乘法 (2SLS)。

1. **第一阶段：** 将内生处理变量 $T$ 回归到工具变量 $Z$ 和所有观测到的外生协变量 $X$ 上。获取预测值 $\hat{T}$。
   
   $$
   T = \hat{\pi}_0 + \hat{\pi}_1 Z + \hat{\pi}_2' X + \text{residual}
   $$
   
2. **第二阶段：** 将结果 $Y$ 回归到第一阶段的*预测*处理变量 $\hat{T}$ 和观测协变量 $X$ 上。
   
   $$
   Y = \hat{\alpha} + \hat{\beta}_{2SLS} \hat{T} + \hat{\gamma}' X + \text{residual}
   $$
   

系数 $\hat{\beta}_{2SLS}$ 是因果效应的一致 IV 估计值。

#### 使用 `linearmodels` 实现

`linearmodels` 库为 IV 估计提供了方便的接口。让我们模拟一些数据，其中 OLS 失效但 IV 有效。假设 `ability` ($U$) 未被观测。`qob`（出生季度）是我们的工具变量 $Z$。`schooling` 是 $T$，`log_wage` 是 $Y$，`experience` 是一个观测协变量 $X$。

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# 模拟数据（仅作说明 - 实际分析需使用真实数据以获得真正结果）
np.random.seed(42)
n_samples = 1000
experience = np.random.uniform(5, 25, n_samples)
ability = np.random.normal(0, 1, n_samples) # 未观测
qob = np.random.randint(1, 5, n_samples) # 工具变量（例如，出生季度）

# 受教育年限受天赋能力和工具变量影响
schooling = 10 + 0.5 * qob + 1.0 * ability + 0.1 * experience + np.random.normal(0, 1, n_samples)
# 对数工资受受教育年限、天赋能力和经验影响
log_wage = 5 + 0.1 * schooling + 0.5 * ability + 0.05 * experience + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    'log_wage': log_wage,
    'schooling': schooling,
    'experience': experience,
    'qob': qob,
    'ability': ability # 保留以供比较，但在估计中视为未观测
})

# 为回归模型添加常数项
df = sm.add_constant(df, prepend=False)

# --- 简单 OLS（因省略天赋能力而有偏差）---
ols_model = sm.OLS(df['log_wage'], df[['schooling', 'experience', 'const']])
ols_results = ols_model.fit()
print("--- OLS 结果（可能存在偏差）---")
print(ols_results.summary(yname='对数工资', xname=['受教育年限', '经验', '常数项']))
print(f"\nOLS 对受教育年限影响的估计: {ols_results.params['schooling']:.4f}\n")

# --- IV (2SLS) 估计 ---
# 因变量: 对数工资
# 外生回归变量: 经验, 常数项
# 内生回归变量: 受教育年限
# 工具变量: 出生季度

iv_model = IV2SLS(dependent=df['log_wage'],
                  exog=df[['experience', 'const']],
                  endog=df['schooling'],
                  instruments=df['qob'])
iv_results = iv_model.fit(cov_type='standard') # 使用标准误差

print("\n--- IV (2SLS) 结果 ---")
print(iv_results.summary)

# 检查第一阶段 F 统计量以判断是否存在弱工具变量
print("\n--- 第一阶段诊断 ---")
print(iv_results.first_stage)

# 真实的因果效应是 0.1
# OLS 估计值向上偏差，因为天赋能力正向影响受教育年限和工资。
# 如果工具变量有效且足够强，IV 估计值应更接近 0.1。
```

**结果解读：**

比较 OLS 结果中 `schooling` 的系数与 IV 结果。OLS 估计值可能存在偏差（在我们的模拟中，向上偏差，因为更高的天赋能力导致更多的受教育年限和更高的工资）。假设工具变量有效，IV 估计值应更接近真实因果效应（在我们的模拟中为 0.1）。

请留意 `First Stage Diagnostics`。F 统计量检验工具变量的相关性。一个常用的经验法则认为 F 统计量大于 10 表明工具变量足够强，尽管具体情况有影响。弱工具变量（F 统计量低）会导致不可靠且有偏差的 IV 估计值。

### 实现回归不连续设计 (RDD)

RDD 适用于当处理 $T$ 的分配是通过一个观测到的*运行变量* $R$ 是否越过已知阈值 $c$ 来确定时，这种确定可以是精确的也可以是模糊的。

- **精确 RDD：** 处理由阈值完全确定：如果 $R \ge c$，则 $T = 1$；如果 $R < c$，则 $T = 0$。
- **模糊 RDD：** 接受处理的*概率*在阈值处不连续变化：$\lim_{r \to c^+} P(T=1|R=r) \ne \lim_{r \to c^-} P(T=1|R=r)$。

核心思想是，恰好在截止点之下 ($R = c - \epsilon$) 和恰好在截止点之上 ($R = c + \epsilon$) 的单位很可能在观测和*未观测*特征方面相似。因此，结果 $Y$ 的平均值在阈值 $c$ 处的任何不连续跳跃都可以归因于处理的因果效应。RDD 估计专门针对截止点附近的单位的局部平均处理效应 (LATE)。

#### 情景：奖学金对毕业率的影响

想象一所大学向入学考试 ($R$) 分数达到或超过阈值 $c=700$ 的学生授予奖学金 ($T=1$)，而分数低于 700 的学生则不授予奖学金 ($T=0$)。我们希望估计奖学金对毕业概率 ($Y$) 的影响。

#### 可视化不连续性

RDD 中一个重要的第一步是可视化。绘制结果 $Y$ 与运行变量 $R$ 的关系图。在截止点 $c$ 处查找跳跃或不连续点。通常，人们会绘制运行变量分箱内的平均值。

````python
import plotly.graph_objects as go

# 模拟 RDD 数据
np.random.seed(123)
n_samples = 2000
cutoff = 700
# 运行变量（考试分数）围绕截止点居中
running_var = np.random.normal(cutoff, 50, n_samples)
# 处理分配（精确 RDD）
treatment = (running_var >= cutoff).astype(int)
# 潜在结果（有/无奖学金的毕业概率）
# 假设奖学金使毕业概率增加 0.15
baseline_prob = 0.5 + 0.0005 * (running_var - cutoff) # 基线趋势
potential_outcome_0 = baseline_prob + np.random.normal(0, 0.1, n_samples)
potential_outcome_1 = potential_outcome_0 + 0.15
# 观测结果
observed_outcome = potential_outcome_0 * (1 - treatment) + potential_outcome_1 * treatment
# 将结果截断在 0 到 1 之间（概率）
observed_outcome = np.clip(observed_outcome, 0, 1)

rdd_df = pd.DataFrame({
    'score': running_var,
    'scholarship': treatment,
    'graduated': observed_outcome
})

# 将数据分箱用于可视化
bins = np.linspace(rdd_df['score'].min(), rdd_df['score'].max(), 50)
rdd_df['score_bin'] = pd.cut(rdd_df['score'], bins, right=False)
binned_data = rdd_df.groupby('score_bin', observed=True).agg(
    mean_score=('score', 'mean'),
    mean_graduated=('graduated', 'mean'),
    n_obs=('score', 'size')
).reset_index()
binned_data = binned_data.dropna(subset=['mean_score']) # 删除空分箱

# 创建 Plotly 图表
fig = go.Figure()

# 分箱平均值的散点图
fig.add_trace(go.Scatter(
    x=binned_data['mean_score'],
    y=binned_data['mean_graduated'],
    mode='markers',
    marker=dict(
        color=np.where(binned_data['mean_score'] < cutoff, '#4263eb', '#f76707'), # 下方蓝色，上方橙色
        size=np.sqrt(binned_data['n_obs']) * 1.5, # 根据观测数量调整大小
        line_width=1,
        line_color='#495057'
    ),
    name='分箱平均值'
))

# 添加截止点的垂直线
fig.add_vline(x=cutoff, line_width=2, line_dash="dash", line_color="#868e96", name=f'截止点 = {cutoff}')

# 添加局部回归线（此处为说明单独拟合 - 实际分析使用特定的 RDD 方法）
# 注意：此处显示的是简单线性拟合；局部多项式是标准方法。
score_below = binned_data[binned_data['mean_score'] < cutoff]['mean_score']
grad_below = binned_data[binned_data['mean_score'] < cutoff]['mean_graduated']
params_below = np.polyfit(score_below, grad_below, 1)
x_below = np.linspace(score_below.min(), cutoff, 50)
y_below = np.polyval(params_below, x_below)

score_above = binned_data[binned_data['mean_score'] >= cutoff]['mean_score']
grad_above = binned_data[binned_data['mean_score'] >= cutoff]['mean_graduated']
params_above = np.polyfit(score_above, grad_above, 1)
x_above = np.linspace(cutoff, score_above.max(), 50)
y_above = np.polyval(params_above, x_above)

fig.add_trace(go.Scatter(x=x_below, y=y_below, mode='lines', line=dict(color='#1c7ed6', width=2), name='拟合（分数 < 700）'))
fig.add_trace(go.Scatter(x=x_above, y=y_above, mode='lines', line=dict(color='#f76707', width=2), name='拟合（分数 >= 700）'))

fig.update_layout(
    title_text='RDD 可视化：毕业率 vs. 入学考试分数',
    xaxis_title='入学考试分数（运行变量）',
    yaxis_title='平均毕业率',
    legend_title='图例',
    plot_bgcolor='#e9ecef',
    paper_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=12, color="#495057")
)

# 显示图表 JSON（在 Web 环境中替换为实际渲染）
# fig.show()
# 用于嵌入的示例固定 JSON:
```plotly
{
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "x": [620, 635, 650, 665, 680, 695, 705, 720, 735, 750, 765, 780],
      "y": [0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80],
      "marker": {
        "color": ["#4263eb", "#4263eb", "#4263eb", "#4263eb", "#4263eb", "#4263eb", "#f76707", "#f76707", "#f76707", "#f76707", "#f76707", "#f76707"],
        "size": [8, 9, 10, 11, 10, 9, 8, 9, 10, 11, 10, 9],
        "line": { "width": 1, "color": "#495057" }
      },
      "name": "分箱平均值"
    },
    {
      "type": "scatter",
      "x": [620, 699.9],
      "y": [0.54, 0.595],
      "mode": "lines",
      "line": { "color": "#1c7ed6", "width": 2 },
      "name": "拟合（分数 < 700）"
    },
    {
      "type": "scatter",
      "x": [700, 780],
      "y": [0.745, 0.80],
      "mode": "lines",
      "line": { "color": "#f76707", "width": 2 },
      "name": "拟合（分数 >= 700）"
    }
  ],
  "layout": {
    "xaxis": { "range": [610, 790], "title": "入学考试分数（运行变量）" },
    "yaxis": { "range": [0.45, 0.85], "title": "平均毕业率" },
    "shapes": [
      { "type": "line", "x0": 700, "y0": 0.45, "x1": 700, "y1": 0.85, "line": { "color": "#868e96", "width": 2, "dash": "dash" } }
    ],
    "title": { "text": "RDD：毕业率 vs. 入学考试分数" },
    "showlegend": true,
    "legend": { "title": { "text": "图例" } },
    "plot_bgcolor": "#e9ecef",
    "paper_bgcolor": "white",
    "font": { "family": "Arial, sans-serif", "size": 12, "color": "#495057" }
  }
}
````

> 该图显示了分数分箱内的平均毕业率。在截止分数 700 附近可以清楚地看到一个跳跃，表明奖学金的积极影响。拟合线有助于显示不连续性。

#### 估计 LATE

标准方法是使用局部多项式回归。我们在截止点 $c$ 的两侧，在选定的*带宽* $h$ 内，对数据拟合单独的多项式。这些回归在截止点 $c$ 处截距的差异就是 LATE 的 RDD 估计值。


$$
\hat{\beta}_{RDD} = \lim_{r \to c^+} E[Y|R=r] - \lim_{r \to c^-} E[Y|R=r]
$$


虽然您可以使用回归包手动实现这一点，但 `rdrobust`（R 和 Python 中均有）等专用库可以自动进行带宽选择（例如，使用 Imbens-Kalyanaraman 最优带宽）以及带标准误差的估计。

```python
# 使用 statsmodels OLS 的简化实现（仅作说明）
# 正式的 RDD 需要局部多项式和最优带宽选择
# 对于严肃分析，请考虑使用 rdrobust 包

bandwidth = 30 # 示例带宽 - 需要仔细选择！
df_rdd_local = rdd_df[(rdd_df['score'] >= cutoff - bandwidth) & (rdd_df['score'] < cutoff + bandwidth)].copy()

# 为局部线性回归创建交互项
df_rdd_local['score_centered'] = df_rdd_local['score'] - cutoff
df_rdd_local['treat_x_score_centered'] = df_rdd_local['scholarship'] * df_rdd_local['score_centered']

# 添加常数项
df_rdd_local = sm.add_constant(df_rdd_local, prepend=False)

# 回归：结果 ~ 处理 + 居中分数 + 处理*居中分数
rdd_model = sm.OLS(df_rdd_local['graduated'],
                   df_rdd_local[['scholarship', 'score_centered', 'treat_x_score_centered', 'const']])
rdd_results = rdd_model.fit(cov_type='robust') # 使用稳健标准误差

print("\n--- RDD 估计（局部线性回归 - 说明性）---")
print(rdd_results.summary(yname='毕业', xname=['奖学金 (T)', '居中分数 (R-c)', 'T*(R-c)', '常数项']))

# 'scholarship' 的系数是截止点处 LATE 的 RDD 估计值
rdd_estimate = rdd_results.params['scholarship']
print(f"\n奖学金效应的 RDD 估计值 (LATE): {rdd_estimate:.4f}")

# 与模拟中 0.15 的真实效应进行比较
```

**解读与注意事项：**

- 处理虚拟变量（示例中的 `scholarship`）的系数表示结果在截止点处的估计跳跃，即 LATE。
- **带宽选择：** 带宽 $h$ 的选择很重要。过窄会导致估计值不稳定。过宽可能违反跨阈值相似性的假设（偏差-方差权衡）。请使用数据驱动方法（如 `rdrobust` 中的方法）。
- **函数形式：** 通常，局部线性（多项式阶数 1）或局部二次（阶数 2）回归优于更高阶的多项式。
- **模糊 RDD：** 如果不连续性在于处理的*概率*（而不是确定性的），则需要模糊 RDD。这通常使用 2SLS 实现，其中指示高于阈值分配的虚拟变量（$Z = 1[R \ge c]$）作为实际处理状态 $T$ 的工具变量。
- **有效性检验：** 执行检查，例如确保协变量在截止点不发生不连续跳跃（McCrary 密度检验），以及使用不同截止点进行安慰剂检验。

### 结论

本实践部分展示了如何在 Python 中实现 IV（使用 2SLS）和 RDD（使用局部回归）。这些方法在面对潜在的未观测混杂因素时，对您的工具集来说是宝贵的补充，它们使得在特定但强大的假设下进行因果效应估计成为可能。在您的具体应用情境中，请始终使用领域知识和诊断测试，严格评估 IV（相关性、排他性、独立性）和 RDD（潜在结果和协变量的连续性、运行变量未被操纵）假设的有效性。掌握这些方法能够从观测数据中得出更可信的因果结论，这是朝着构建更可靠的机器学习 (machine learning)系统迈出的重要一步。

## 参考资料

- [Does Compulsory School Attendance Affect Schooling and Earnings?](https://doi.org/10.2307/2937954) — Joshua D. Angrist and Alan B. Krueger (1991)
  Journal: The Quarterly Journal of Economics; Publisher: Oxford University Press; Volume: 106; Pages: 979-1014; DOI: [10.2307/2937954](https://doi.org/10.2307/2937954)
  一篇基础性论文，使用工具变量（出生季度）估计教育对收入的因果效应，是本节讨论的经典示例。
- [Mastering 'Metrics': The Path from Cause to Effect](https://books.google.com/books/about/Mastering_Metrics.html?id=oI99DQAAQBAJ) — Joshua D. Angrist, Jörn-Steffen Pischke (2015)
  Publisher: Princeton University Press; Pages: 304
  一本既易懂又严谨的教材，提供关于关键因果推断方法（包括工具变量和回归不连续性设计）的实践指导。
- [Regression Discontinuity Designs: A Guide to Practice](https://www.nber.org/papers/w13039) — Guido W. Imbens and Thomas Lemieux (2008)
  Journal: Journal of Econometrics; Publisher: Elsevier; Volume: 142(2); Pages: 615-635; DOI: [10.1016/j.jeconom.2007.05.004](https://doi.org/10.1016/j.jeconom.2007.05.004)
  一篇被广泛引用的综述论文，提供了回归不连续性设计的全面概述和实践建议。
- [Econometric Analysis of Cross Section and Panel Data](https://mitpress.mit.edu/books/econometric-analysis-cross-section-and-panel-data-second-edition) — Jeffrey M. Wooldridge (2010)
  Publisher: The MIT Press
  一本全面的研究生水平计量经济学教科书，涵盖了工具变量和回归不连续性设计的理论基础和实际应用。
- [linearmodels: Instrumental Variable Regression](https://bashtage.github.io/linearmodels/doc/iv/estimation.html) — Kevin Sheppard (2021)
  Python `linearmodels` 库的官方文档，提供了实现工具变量估计的详细解释和示例，与实践部分中使用的内容一致。

---

[上一节](06-%E8%BF%91%E7%AB%AF%E5%9B%A0%E6%9E%9C%E6%8E%A8%E6%96%AD%E5%8E%9F%E7%90%86.md) · [下一节](../05-%E6%97%B6%E9%97%B4%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%8A%A8%E6%80%81%E7%B3%BB%E7%BB%9F%E7%9A%84%E5%9B%A0%E6%9E%9C%E6%8E%A8%E6%96%AD/01-%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%9B%A0%E6%9E%9C%E6%8E%A8%E6%96%AD%E7%9A%84%E9%9A%BE%E7%82%B9.md)
