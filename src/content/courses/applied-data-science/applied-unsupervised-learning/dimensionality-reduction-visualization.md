---
course: "applied-data-science"
chapter: "applied-unsupervised-learning"
lesson: "dimensionality-reduction-visualization"
sourceId: 1894
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-4-applied-unsupervised-learning/dimensionality-reduction-visualization"
title: "用于可视化的降维"
description: "应用PCA和t-SNE等方法对高维数据进行可视化。"
order: 5
plots: ["plots/1894-0.json", "plots/1894-1.json"]
sourceHash: "ffb10d39917ec4a9ed0273acf40d7d19b22c0305b512789fc8003d081ffb2230"
sourceCorrections: []
---

高维数据集在数据科学中很常见，它们给可视化带来了很大的挑战。人类感知三维现实，因此难以直接绘制和解释包含数十、数百甚至数千个特征的数据。然而，数据可视化是进行分析的有力工具，它能帮助我们识别模式、识别聚类、检测异常值，并总体把握无标签数据中的固有结构。专门为可视化设计的降维方法旨在将高维数据映射到低维空间（通常是2D或3D），同时保持数据点之间有意义的关系。

本节考察两种广泛用于可视化降维的方法：主成分分析（PCA）和t-分布随机邻域嵌入 (embedding)（t-SNE）。虽然PCA之前已作为一种通用降维方法介绍过，但我们在此专门从可视化的角度对其进行重新审视。t-SNE主要是一种用于高维数据可视化的方法。

### 使用主成分分析（PCA）进行数据可视化

PCA通过寻找一组新的正交轴（称为主成分）来实现降维，这些轴能捕获数据中的最大方差。第一个主成分解释最大的方差，第二个主成分（与第一个正交）解释次大的方差，依此类推。

为了可视化，我们通常将数据投影到前两个主成分（PC1和PC2）上。这种二维表示捕获了原始数据中最大散布的方向。虽然舍弃其余成分会丢失信息，但这种投影通常能提供数据结构的有效概览，可能显现出聚类或趋势。

请记住，PCA对特征的尺度很敏感。在应用PCA之前，对数据进行缩放（例如，使用scikit-learn中的`StandardScaler`）是标准做法。

我们来演示如何使用`scikit-learn`将数据投影到前两个主成分上：

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

# 加载样本数据（例如，鸢尾花数据集）
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 1. 数据缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 应用PCA降至2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 创建用于绘图的DataFrame
pca_df = pd.DataFrame(data = X_pca, columns = ['Principal Component 1', 'Principal Component 2'])
pca_df['target'] = y
pca_df['species'] = pca_df['target'].apply(lambda i: target_names[i])

# 3. 可视化结果
fig = px.scatter(pca_df, x='Principal Component 1', y='Principal Component 2',
                 color='species', title='PCA of Iris Dataset (2 Components)',
                 labels={'species': 'Species'},
                 color_discrete_map={ # 使用课程调色板颜色
                     'setosa': '#228be6',      # 蓝色
                     'versicolor': '#51cf66',  # 绿色
                     'virginica': '#be4bdb'   # 紫色
                 })

fig.update_layout(
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2",
    legend_title="Species",
    width=700, # 调整宽度以适应网页显示
    height=500 # 调整高度以适应网页显示
)

# 可选：显示解释方差比
print(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")
print(f"Total explained variance by 2 components: {np.sum(pca.explained_variance_ratio_):.4f}")

# 显示图表（或生成用于网页嵌入的JSON）
# fig.show()
# 要生成用于嵌入的JSON：
# print(pio.to_json(fig))
```



![鸢尾花数据集的PCA（2个成分）](plots/1894-0.json)



> 鸢尾花数据集在最初两个主成分上的PCA投影。颜色表示真实的物种标签，这显示了PCA如何根据方差区分群体。总解释方差（在此例中约为95.8%）表明这两个成分捕获了原始数据多少的变异性。

PCA提供线性投影，这种方法计算效率高，并且在方差方面易于理解。然而，它可能无法有效分离由非线性关系或局部密度定义的聚类。

### 使用t-分布随机邻域嵌入 (embedding)（t-SNE）进行数据可视化

t-SNE是一种主要用于可视化的非线性降维方法。与PCA侧重于最大化方差（保留全局结构）不同，t-SNE旨在保留数据的*局部*结构。它将高维数据点之间的相似性建模为条件概率，然后尝试找到一个低维嵌入（通常是2D或3D），使低维点之间的相似性与高维相似性密切匹配。

t-SNE在显现数据中的聚类方面特别有效。在高维空间 (high-dimensional space)中靠近的点倾向于在低维空间中映射得也靠近。

t-SNE的重要方面：

1. **概率性：** 它涉及使用概率分布（高维空间中的高斯分布，低维空间中的t-分布）来建模相似性。
2. **非线性：** 它可以捕获PCA可能遗漏的复杂非线性结构。
3. **计算密集：** 与PCA相比，t-SNE需要更多的计算时间，特别是对于大型数据集。
4. **对超参数 (parameter) (hyperparameter)敏感：** 结果很大程度上取决于超参数：
   - `perplexity`（困惑度）：大致与每个点考虑的最近邻居数量有关。典型值在5到50之间。它影响数据局部和全局特性的平衡。
   - `n_iter`（迭代次数）：优化迭代的次数。通常需要数百次迭代（例如，1000次）才能收敛。
   - `learning_rate`（学习率）：控制优化期间的步长。

需要注意的是，生成的t-SNE图主要用于视觉分析。t-SNE图中表观聚类之间的距离可能没有实际意义，并且全局排列会因运行或不同的困惑度值而异。应关注点的分组情况，而非它们的相对位置或大小。

这里演示如何使用`scikit-learn`应用t-SNE，同样使用缩放后的鸢尾花数据：

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE # 注意：与PCA模块不同
import plotly.express as px
import plotly.io as pio

# 加载并缩放数据（如前所述）
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用t-SNE
# 常用参数：perplexity=30, n_iter=1000
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 创建用于绘图的DataFrame
tsne_df = pd.DataFrame(data = X_tsne, columns = ['TSNE Component 1', 'TSNE Component 2'])
tsne_df['target'] = y
tsne_df['species'] = tsne_df['target'].apply(lambda i: target_names[i])

# 可视化结果
fig_tsne = px.scatter(tsne_df, x='TSNE Component 1', y='TSNE Component 2',
                      color='species', title='t-SNE Visualization of Iris Dataset',
                      labels={'species': 'Species'},
                      color_discrete_map={ # 使用课程调色板颜色
                          'setosa': '#228be6',      # 蓝色
                          'versicolor': '#51cf66',  # 绿色
                          'virginica': '#be4bdb'   # 紫色
                      })

fig_tsne.update_layout(
    xaxis_title="t-SNE Component 1",
    yaxis_title="t-SNE Component 2",
    legend_title="Species",
    width=700, # 调整宽度以适应网页显示
    height=500 # 调整高度以适应网页显示
)

# 显示图表（或生成用于网页嵌入的JSON）
# fig_tsne.show()
# 要生成用于嵌入的JSON：
# print(pio.to_json(fig_tsne))
```



![鸢尾花数据集的t-SNE可视化](plots/1894-1.json)



> 鸢尾花数据集的t-SNE投影。请注意，与PCA相比，t-SNE通常能生成更清晰、分离度更好的聚类，有效地捕获同一物种数据点之间的局部相似性。

### 用于可视化的PCA与t-SNE：快速比较

- **目标：** PCA侧重于保留全局方差；t-SNE侧重于保留局部相似性。
- **结构：** PCA捕获线性结构；t-SNE捕获非线性结构，更适合显现聚类。
- **计算：** PCA速度快得多。t-SNE较慢且占用内存多。
- **输出：** PCA结果是确定性的。t-SNE结果由于其概率特性和优化过程，在不同运行之间可能略有差异。t-SNE中聚类之间的全局排列和距离不如分组本身可靠。
- **用例：** 在需要快速初步概览或全局结构重要时使用PCA。当主要目的是可视化潜在聚类和局部关系时，使用t-SNE。

通常，可以先应用PCA显著降低维度（例如，降至50个成分），然后再应用t-SNE，这可以改善t-SNE的性能并减少噪声。

### 注意事项

- **缩放：** PCA和t-SNE（特别是在没有预先PCA的情况下直接应用于数据时）通常都需要对特征进行缩放。
- **解释：** 解释图表时需谨慎。PCA轴代表最大方差方向。t-SNE轴没有直接的全局解释；应侧重于点的相对分组。
- **信息损失：** 请记住，任何降维都涉及信息损失。可视化提供的是压缩视图，而非完整画面。

通过应用PCA和t-SNE等方法，您可以将复杂的高维数据转换为可解释的2D或3D图表，有助于发现原本难以发现的模式和结构。这种视觉分析是理解无标签数据的非监督学习 (supervised learning)工具包中不可或缺的一部分。

## 参考资料

- [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten and Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Publisher: JMLR.org; Volume: 9; Pages: 2579-2605; DOI: [10.1162/jmlr.2008.02579a](https://doi.org/10.1162/jmlr.2008.02579a)
  这篇基础性论文介绍了t-SNE算法，用于可视化高维数据。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/ESLII_print12.pdf) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本广受推崇的教科书，提供了各种机器学习方法的统计框架，包括关于主成分分析（PCA）等降维技术的详细章节。（第二版）
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — scikit-learn developers (2024)
  scikit-learn的PCA实现官方文档，提供参数、属性和使用示例的详细信息。
- [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) — scikit-learn developers (2024)
  scikit-learn的t-SNE实现官方文档，提供困惑度、学习率等超参数信息和实践考虑。
