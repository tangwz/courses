---
course: "building-ml-recommendation-system"
chapter: "content-based-filtering"
lesson: "practice-building-movie-recommender"
sourceId: 7461
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-2-content-based-filtering/practice-building-movie-recommender"
title: "动手实践：构建电影推荐系统"
description: "关于如何利用剧情简介和类型信息构建基于内容的电影推荐系统的分步教程。"
order: 7
plots: []
sourceHash: "c3c037c621030e6bd29059c0e8227ab6ae0a0cfe5bcb678b809db2699350940a"
sourceCorrections: []
---

为了构建一个功能完备的基于内容的电影推荐系统，本次实践练习将应用项目特征文件、TF-IDF 和余弦相似度。我们将使用来自 The Movie Database (TMDB) 的热门数据集，其中包含数千部电影的丰富元数据，包括剧情简介、类型、关键词、演职人员。本练习将引导你完成从原始数据到个性化推荐的每个步骤。

### 1. 环境准备与数据加载

首先，我们需要导入必要的库并加载数据集。我们将使用 pandas 进行数据处理，使用 scikit-learn 进行 TF-IDF 向量 (vector)化和相似度计算。

数据集分为两个 CSV 文件：`movies.csv` 包含电影元数据，`credits.csv` 包含演职人员信息。让我们加载它们并查看前几行。

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据集
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 显示每个数据框的前几行
print("电影数据框：")
print(movies.head(2))
print("\n演职人员数据框：")
print(credits.head(2))
```

为了有效地处理这些数据，我们将根据电影标题将这两个数据框合并为一个。

```python
# 根据 'title' 列合并两个数据框
movies = movies.merge(credits, on='title')
```

### 2. 特征工程与预处理

我们的目标是为每部电影创建一个单一的文本字段，其中包含其内容特征。我们将选择以下列：`genres`（类型）、`keywords`（关键词）、`overview`（简介）、`cast`（演员）和 `crew`（制作团队）。

这些列目前的格式并不理想。`genres`、`keywords`、`cast` 和 `crew` 以类似 JSON 的字符串形式存储。我们需要解析它们以提取所需信息。例如，从 `cast` 中，我们可能只取前三位演员的名字；从 `crew` 中，我们要找到导演。

让我们定义几个辅助函数来处理这些解析工作。我们将使用 Python 的 `ast` 库来安全地评估列表和字典的字符串表示形式。

```python
import ast

def extract_names(text):
    """从字典列表中提取名称，例如类型或关键词。"""
    names = []
    for i in ast.literal_eval(text):
        names.append(i['name'])
    return names

def extract_top_actors(text, n=3):
    """从演员名单中提取前 n 位演员的名字。"""
    actors = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < n:
            actors.append(i['name'])
            counter += 1
        else:
            break
    return actors

def extract_director(text):
    """从制作团队名单中提取导演姓名。"""
    directors = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            directors.append(i['name'])
    return directors
```

现在，我们将这些函数应用于数据框，并通过删除空格来清理生成的文本。这有助于向量 (vector)化工具将 "James Cameron" 视为一个整体，而不是两个独立的单词。

```python
# 应用辅助函数
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_top_actors)
movies['crew'] = movies['crew'].apply(extract_director)

# 通过删除空格来清理文本数据
def remove_spaces(text_list):
    return [i.replace(" ","") for i in text_list]

movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(remove_spaces)
```

提取并清理好特征后，我们可以将它们合并到一个 "tags"（标签）列中。这一列将作为 TF-IDF 向量化 (quantization)工具处理每部电影的文本文档。

```python
# 将特征合并到单一的 'tags' 列中
movies['tags'] = (
    movies['overview'].fillna('').apply(lambda x: x.split()) +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# 将标签列表转回单一字符串
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# 创建一个仅包含必要列的新数据框
recommender_df = movies[['movie_id', 'title', 'tags']].copy()
```

让我们看看电影《阿凡达》(Avatar) 的 `tags`：

> In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting the home he feels is his home. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron

这个组合字符串提供了电影内容的丰富描述。

### 3. 内容向量 (vector)化

我们现在拥有一个文本语料库，其中每个文档都是一部电影的 `tags` 字符串。下一步是使用 TF-IDF 将这些文本转换为数值向量。这将创建一个矩阵，其中每一行代表一部电影，每一列代表词汇表 (vocabulary)中的一个词。每个单元格中的值将是该词针对该电影的 TF-IDF 分数。

我们将使用 scikit-learn 中的 `TfidfVectorizer`。为了使矩阵大小保持在可控范围内，我们将特征（单词）数量限制为最常出现的 5,000 个，并移除常用的英文停用词。

```python
# 初始化 TF-IDF 向量化工具
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# 拟合并转换 'tags' 列
feature_vectors = tfidf.fit_transform(recommender_df['tags']).toarray()

print(f"特征向量的形状：{feature_vectors.shape}")
```

输出形状（可能是 `(4806, 5000)`）证实了我们为数据集中的 4,806 部电影中的每一部都生成了一个 5,000 维的向量。

下图显示了我们刚刚完成的工作流程。

> 从原始元数据到最终相似度矩阵的构建基于内容的推荐系统的过程。

### 4. 计算相似度

有了用向量 (vector)表示的电影，我们现在可以计算每对电影之间的余弦相似度。scikit-learn 的 `cosine_similarity` 函数非常适合这项工作。它接收我们的特征向量矩阵，并返回一个方阵，其中每个条目 `(i, j)` 是电影 `i` 和电影 `j` 之间的相似度分数。


$$
\text{相似度}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$


这项操作在整个矩阵上的计算效率很高。

```python
# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(feature_vectors)

print(f"相似度矩阵的形状：{similarity_matrix.shape}")
print("\n相似度矩阵样本（前 5x5）：")
print(similarity_matrix[:5, :5])
```

生成的 `(4806, 4806)` 矩阵包含所有两两相似度分数。你会注意到对角线全部为 1，因为每部电影与自身完全相似。

### 5. 创建推荐函数

所有准备工作已经就绪。最后一步是构建一个使用 `similarity_matrix` 提供推荐的函数。该函数将接收一个电影标题作为输入，并执行以下步骤：

1. 在数据框中找到输入电影的索引。
2. 从相似度矩阵中获取相应的行。
3. 将相似度分数按降序排列。
4. 选择相似度最高的 5 部电影的索引。
5. 返回这些电影的标题。

```python
def recommend_movies(movie_title):
    """
    推荐与输入的 movie_title 相似的电影。
    """
    # 查找与标题匹配的电影索引
    try:
        movie_index = recommender_df[recommender_df['title'] == movie_title].index[0]
    except IndexError:
        return f"在数据集中未找到电影 '{movie_title}'。"

    # 获取指定电影的两两相似度分数
    distances = similarity_matrix[movie_index]

    # 根据相似度分数对电影进行排序
    # 我们使用 enumerate 来跟踪原始索引
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    # 获取前 5 部最相似的电影，跳过第一部（电影本身）
    recommended = []
    for i in movies_list[1:6]:
        recommended.append(recommender_df.iloc[i[0]].title)

    return recommended
```

### 6. 测试推荐系统

让我们用几个例子来测试这个函数，看看它的表现如何。

**示例 1：超级英雄电影**

```python
recommend_movies('The Dark Knight Rises')
```

**输出：**

```
['The Dark Knight', 'Batman Begins', 'Batman', 'Batman Returns', 'Batman: The Dark Knight Returns, Part 2']
```

推荐结果非常出色。系统准确识别了其他蝙蝠侠系列电影，展示了其根据角色（来自演员和关键词）及类型关联电影的能力。

**示例 2：科幻大片**

```python
recommend_movies('Avatar')
```

**输出：**

```
['Aliens vs Predator: Requiem', 'Aliens', 'Falcon Rising', 'Independence Day', 'Titan A.E.']
```

这些推荐也很给力。它们共同拥有科幻、外星人和动作等主题，而这些正是我们在为《阿凡达》创建的 `tags` 中突出的特征。

**示例 3：经典动画电影**

```python
recommend_movies('Toy Story')
```

**输出：**

```
['Toy Story 3', 'Toy Story 2', 'Cars 2', 'Monsters, Inc.', 'Finding Nemo']
```

在这里，推荐系统准确找到了同一工作室（皮克斯）制作的其他动画电影，这些电影通常共享相似的关键词，甚至有共同的演职人员（如导演约翰·拉塞特）。

### 总结与后续

在本次动手实践中，你成功地从零开始构建了一个基于内容的推荐系统。你对原始电影元数据进行了特征工程，使用 TF-IDF 将文本转换为数值向量 (vector)，通过余弦相似度计算了相似度分数，并利用这些分数生成了相关的电影推荐。

此类系统非常有效，因为它不需要任何用户交互数据。它可以立即推荐新项目，这有助于解决“新项目”问题。不过，它也有局限性。它倾向于推荐与用户已经看过的内容非常相似的项目，这可能导致用户难以发现新事物，即所谓的“过滤气泡”现象。此外，其推荐质量完全取决于项目元数据的丰富程度和准确性。

在下一章中，我们将研究一种不同的方法：协同过滤。我们不再观察项目内容，而是通过分析用户行为来发现模式并进行推荐，这有助于克服目前看到的一些限制。

## 参考资料

- [Recommender Systems: An Introduction](https://link.springer.com/book/10.1007/978-3-319-29659-3) — Aggarwal, Charu C. (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29659-3](https://doi.org/10.1007/978-3-319-29659-3)
  一本全面指导推荐系统基础知识的书籍，包括基于内容的过滤和项目画像的原理。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本标准教科书，解释了信息检索中用于文本表示和比较的TF-IDF、向量空间模型和余弦相似度。
- [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — scikit-learn (2023)
  官方文档，详细介绍了Scikit-learn中TF-IDF向量化器的参数、功能和用法。
- [sklearn.metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — Scikit-learn Developers (2023)
  官方文档，解释了使用Scikit-learn计算向量之间相似度的余弦相似度实现。
