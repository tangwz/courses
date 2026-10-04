---
course: "optimization-techniques-ml"
chapter: "advanced-specialized-optimization"
lesson: "practice-projected-gradient-descent"
sourceId: 1331
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-7-advanced-specialized-optimization/practice-projected-gradient-descent"
title: "实践：实现投影梯度下降"
description: "动手实践环节，实现针对受限问题的投影梯度下降。"
order: 7
plots: ["plots/1331-0.json"]
sourceHash: "8b33d5affb73081869015c32cd029c54d360b20818e81d9e98d8ac2d3fa527f1"
sourceCorrections: []
---

投影梯度下降 (gradient descent)（PGD）算法提供了一种受限优化的实用方法。这种算法旨在处理解必须位于特定可行集 $\mathcal{C}$ 内的问题，这些问题常涉及拉格朗日对偶和KKT条件。PGD是标准梯度下降的一种直观扩展，能够有效处理这些约束。

核心思想很简单：执行一个标准梯度下降步骤，这可能会将您带出可行集，然后将结果点投影回 $\mathcal{C}$ 上。这确保了每次迭代 $x_k$ 都保持可行。

### 算法

回顾PGD更新规则：

1. **梯度步：** 使用标准梯度更新计算一个潜在的下一点：
   $y_{k+1} = x_k - \alpha_k \nabla f(x_k)$
   其中 $x_k$ 是当前的可行点，$\nabla f(x_k)$ 是在 $x_k$ 处目标函数 $f$ 的梯度，$\alpha_k$ 是第 $k$ 次迭代的学习率。
2. **投影步：** 将中间点 $y_{k+1}$ 投影回可行集 $\mathcal{C}$ 上：
   $x_{k+1} = \Pi_{\mathcal{C}}(y_{k+1})$
   这里，$\Pi_{\mathcal{C}}(y)$ 表示投影算子，它在 $\mathcal{C}$ 中找到与 $y$ 最接近（欧几里得距离）的点。

PGD的有效性通常取决于投影 $\Pi_{\mathcal{C}}$ 的计算效率。幸运的是，对于机器学习 (machine learning)中许多常见的约束集（例如箱式约束、范数球或概率单纯形），投影可以解析地高效计算。

### 示例：带有箱式约束的二次函数最小化

我们来考虑在箱式约束条件下最小化一个简单的二次函数。这种类型的约束确保每个变量 $x_i$ 都保持在下限 $l_i$ 和上限 $u_i$ 之间。

**问题：**
最小化 $f(x_1, x_2) = (x_1 - 5)^2 + (x_2 - 4)^2$
约束条件：
$0 \le x_1 \le 3$
$0 \le x_2 \le 2$

无约束最小值明显位于 $(5, 4)$。然而，此点位于我们可行区域（由箱体 $\mathcal{C} = [0, 3] \times [0, 2]$ 定义）之外。我们预计PGD将收敛到此箱体的边界上的一点。

梯度是 $\nabla f(x) = [2(x_1 - 5), 2(x_2 - 4)]$。

**箱式约束的投影算子：**
对于由下限 $l = [l_1, ..., l_n]$ 和上限 $u = [u_1, ..., u_n]$ 定义的箱体，投影 $\Pi_{\mathcal{C}}(y)$ 是逐元素计算的：
$(\Pi_{\mathcal{C}}(y))_i = \max(l_i, \min(y_i, u_i))$
这只是简单地截断 $y$ 的每个分量，以使其落在对应的区间 $[l_i, u_i]$ 内。

### 使用NumPy实现

让我们使用NumPy在Python中实现PGD。

```python
import numpy as np
import plotly.graph_objects as go
import json # 导入json用于最终输出格式化

# 1. 定义目标函数
def objective_function(x):
    """ 目标函数: f(x1, x2) = (x1 - 5)^2 + (x2 - 4)^2 """
    return (x[0] - 5)**2 + (x[1] - 4)**2

# 2. 定义梯度函数
def gradient(x):
    """ 梯度: nabla f(x) = [2(x1 - 5), 2(x2 - 4)] """
    return np.array([2 * (x[0] - 5), 2 * (x[1] - 4)])

# 3. 定义箱式约束的投影算子
def project_onto_box(y, lower_bounds, upper_bounds):
    """ 将点 y 投影到由下限和上限定义的箱体上。 """
    return np.maximum(lower_bounds, np.minimum(y, upper_bounds))

# 4. 实现投影梯度下降
def projected_gradient_descent(
    start_point,
    gradient_func,
    project_func,
    lower_bounds,
    upper_bounds,
    learning_rate=0.1,
    max_iterations=100,
    tolerance=1e-6
):
    """ 执行投影梯度下降。 """
    if not isinstance(start_point, np.ndarray):
        x = np.array(start_point, dtype=float)
    else:
        x = np.copy(start_point).astype(float)

    # 确保起始点是可行的
    x = project_func(x, lower_bounds, upper_bounds)
    history = [np.copy(x)] # 存储迭代历史

    print(f"PGD从以下位置开始: {x}")

    for i in range(max_iterations):
        grad = gradient_func(x)
        if np.linalg.norm(grad) < tolerance * 10: # 对梯度范数进行早期检查
             print(f"梯度范数很小 ({np.linalg.norm(grad):.2e})，可能接近最优。")
             # pass # Allow projection step anyway

        # 梯度步
        y_next = x - learning_rate * grad

        # 投影步
        x_next = project_func(y_next, lower_bounds, upper_bounds)

        # 检查收敛性（x的变化）
        step_size = np.linalg.norm(x_next - x)
        if step_size < tolerance:
            print(f"在 {i+1} 次迭代后收敛。步长: {step_size:.2e}")
            history.append(np.copy(x_next))
            break

        x = x_next
        history.append(np.copy(x)) # 存储可行点
    else: # 循环没有中断
        print(f"达到最大迭代次数 {max_iterations}。")

    return x, np.array(history)

# --- 问题设置 ---
initial_x = np.array([0.0, 0.0]) # 从一个可行的角点开始
bounds_lower = np.array([0.0, 0.0])
bounds_upper = np.array([3.0, 2.0])
learn_rate = 0.1
iterations = 50

# --- 运行算法 ---
optimal_x, trajectory = projected_gradient_descent(
    initial_x,
    gradient,
    project_onto_box,
    bounds_lower,
    bounds_upper,
    learning_rate=learn_rate,
    max_iterations=iterations
)

print(f"在以下位置找到受限最优解: {optimal_x}")
print(f"在最优解处的目标函数值: {objective_function(optimal_x):.4f}")

# --- 可视化设置 ---
# 为等高线图生成网格
x1_vals = np.linspace(-1, 6, 50)
x2_vals = np.linspace(-1, 5.5, 50)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = objective_function([X1, X2])

# 创建等高线图
fig = go.Figure(data=go.Contour(
    z=Z, x=x1_vals, y=x2_vals,
    colorscale='Blues',
    contours=dict(coloring='lines', start=0, end=50, size=2.5),
    line_width=1
))

# 将可行区域（箱式约束）添加为矩形形状
fig.add_shape(
    type="rect",
    x0=bounds_lower[0], y0=bounds_lower[1],
    x1=bounds_upper[0], y1=bounds_upper[1],
    line=dict(color="#f03e3e", width=2), # 红色边框
    fillcolor="rgba(250, 82, 82, 0.1)", # 浅红色填充
    layer='below'
)

# 添加优化轨迹
fig.add_trace(go.Scatter(
    x=trajectory[:, 0],
    y=trajectory[:, 1],
    mode='lines+markers',
    name='PGD路径',
    line=dict(color='#1c7ed6', width=2), # 蓝色线条
    marker=dict(color='#4263eb', size=5, symbol='circle-open')
))

# 突出显示起点和终点
fig.add_trace(go.Scatter(
    x=[trajectory[0, 0]], y=[trajectory[0, 1]],
    mode='markers', name='起点',
    marker=dict(color='#37b24d', size=10, symbol='circle') # 绿色起点
))
fig.add_trace(go.Scatter(
    x=[trajectory[-1, 0]], y=[trajectory[-1, 1]],
    mode='markers', name='受限最优解',
    marker=dict(color='#f76707', size=12, symbol='star') # 橙色星形终点
))

# 添加无约束最优解作为参考
fig.add_trace(go.Scatter(
    x=[5], y=[4],
    mode='markers', name='无约束最优解',
    marker=dict(color='#495057', size=10, symbol='x-thin', line=dict(width=2)) # 灰色叉
))

# 更新布局以提高清晰度
fig.update_layout(
    title_text='投影梯度下降优化路径',
    xaxis_title='参数 x1',
    yaxis_title='参数 x2',
    width=700, height=600,
    xaxis=dict(range=[-1, 6], scaleanchor='y', scaleratio=1), # 保持纵横比
    yaxis=dict(range=[-1, 5.5]),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)')
)

# 生成单行JSON用于嵌入
plotly_json_string = json.dumps(fig.to_dict()) # 使用json.dumps生成单行
```



![投影梯度下降优化路径](plots/1331-0.json)



> 投影梯度下降 (gradient descent)最小化 $f(x_1, x_2)=(x_1-5)^2 + (x_2-4)^2$ 受限于 $0 \le x_1 \le 3$ 和 $0 \le x_2 \le 2$ 的可视化图。路径从 (0,0) 开始，向无约束最小值 (5,4) 移动，但被重复投影回可行区域（红色方框）。它收敛到角点 (3,2)，这是离无约束最小值最近的可行点。

### 讨论

正如可视化图所示，PGD迭代朝着负梯度的方向移动，但每当一步将其带出可行箱体时，就会被强制返回箱体中。该算法成功找到 $(3, 2)$ 处的受限最小值，这是箱体内离无约束最小值 $(5, 4)$ 最近的点，这符合对该二次目标的预期。

此实现的重要之处：

- **可行性：** `trajectory` 中的每个点（在起始点初始投影后）都位于可行箱式约束内。
- **投影效率：** `project_onto_box` 函数的计算成本低，仅涉及逐元素的 `min` 和 `max` 操作。此投影步骤的效率对PGD的整体性能影响很大。
- **学习率：** 与标准梯度下降 (gradient descent)一样，学习率 $\alpha$ 的选择很重要。过大的学习率可能导致振荡或发散，而过小的学习率会导致收敛缓慢。线搜索或自适应学习率等方法也可以适用于PGD，尽管在投影步骤方面需要谨慎。
- **收敛性：** 对于凸目标函数和凸可行集（如我们的示例），如果学习率选择得当（例如，足够小的常数率，或满足递减步长条件），PGD保证收敛到最优解。分析通常涉及梯度的Lipschitz连续性和投影算子在凸集上的非扩张性等性质。

这个实践示例说明了投影梯度下降的核心机制。虽然简单，但它为处理更复杂的受限优化问题奠定了基础，这些问题在正则化 (regularization)回归（例如，带非负约束的LASSO）、最优控制以及训练具有特定参数 (parameter)要求的模型等方向有所出现。主要挑战通常在于为特定感兴趣的约束集 $\mathcal{C}$ 设计或找到高效的投影算子。

## 参考资料

- [Convex Optimization](http://stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  凸优化领域的经典教材，提供了约束优化、投影算子和包括PGD在内的梯度下降方法的基础概念。
- [Nonlinear Programming](https://www.athenasc.com/npbook3.html) — Dimitri P. Bertsekas (2016)
  Publisher: Athena Scientific; Pages: 880
  非线性规划的综合参考书，详细介绍了梯度投影方法及其分析，以及约束优化的各个方面。
- [First-Order Methods in Optimization](https://epubs.siam.org/doi/book/10.1137/1.9781611974997) — Amir Beck (2017)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); Pages: xii, 475 pages; DOI: [10.1137/1.9781611974997](https://doi.org/10.1137/1.9781611974997)
  本书全面阐述了一阶优化方法，包括投影梯度算法，并讨论了它们的性质和在各个领域的应用。
