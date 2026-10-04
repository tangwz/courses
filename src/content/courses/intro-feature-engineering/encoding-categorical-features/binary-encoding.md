---
course: "intro-feature-engineering"
chapter: "encoding-categorical-features"
lesson: "binary-encoding"
sourceId: 1333
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-3-encoding-categorical-features/binary-encoding"
title: "二进制编码"
description: "与独热编码相比，使用二进制编码作为一种维度降低手段。"
order: 7
plots: []
sourceHash: "217cd4be8d8dd9e0c6187b0cc98596bb7452d32d158e6919f04f0e9a5bfe277d"
sourceCorrections: []
---

在处理类别特征时，特别是那些具有许多独特值（高基数）的特征，独热编码（One-Hot Encoding）可能导致特征数量大幅增加，这通常被称为“维度灾难”。二进制编码（Binary Encoding）提供了一种平衡方法：与独热编码相比，它生成的特征数量更少，但对于名义数据，它仍能比简单的序数编码更好地反映每个类别的独特性。

可以将其视为一个两步过程，它结合了序数编码和独热编码的某些特点，但得到的是一种更紧凑的表示形式。

### 二进制编码的原理

1. **整数映射：** 首先，为独特类别分配整数值，从1开始（或有时从0开始）。这类似于序数编码，但重要的是，分配的特定顺序并不表示类别之间存在任何等级关系；它仅仅是一个中间步骤。
2. **二进制转换：** 接着，每个整数被转换成其二进制表示。所需的二进制位数（比特）由分配的最大整数决定。例如，如果您有8个独特类别，您将分配整数1到8。最大整数8在二进制中需要4位（1000）。因此，所有二进制表示都会用前导零填充，以达到4位（例如，1变为0001，2变为0010，3变为0011）。
3. **拆分成列：** 最后，二进制字符串被拆分成独立的列。二进制字符串中的每个位置都成为一个新的数值特征。

### 一个例子

我们考虑一个名为 `DeviceType` 的特征，它有四个独特类别：'Laptop'、'Tablet'、'Phone'、'Desktop'。

1. **整数映射：**

   - 'Laptop': 1
   - 'Tablet': 2
   - 'Phone': 3
   - 'Desktop': 4
2. **二进制转换：** 最大整数是4，其二进制表示为 $100$。我们需要3位（如果从1开始计数，需要 $ceil(log_2(4+1))$ 位；如果从0开始映射0-3，需要 $ceil(log_2(4))$ 位。让我们使用1-4的映射，这需要最多 $ceil(log_2(4)) = 2$ 位，但由于4是 $100$，我们需要3位）。为了与标准二进制编码库保持一致，我们重新映射为0-3：

   - 'Laptop': 0 -> 00
   - 'Tablet': 1 -> 01
   - 'Phone': 2 -> 10
   - 'Desktop': 3 -> 11
   - *更正*：对于 $k=4$ 个类别，如果映射有效率地开始（例如0到3），我们需要 $ceil(log_2(k)) = ceil(log_2(4)) = 2$ 位。让我们用从1开始使用整数的常见方法重做，如果0保留或不使用，则需要 $ceil(log_2(k+1))$ 位，或者简单地需要 $ceil(log_2(最大整数))$ 位。映射1-4：最大整数是4。 $log_2(4) = 2$。但4的二进制是100。我们需要3位。让我们用一个稍微大一点的例子：5个类别（添加'TV'）。
   - 'Laptop': 1 -> 001
   - 'Tablet': 2 -> 010
   - 'Phone': 3 -> 011
   - 'Desktop': 4 -> 100
   - 'TV': 5 -> 101
     最大整数是5。 $ceil(log_2(5)) = 3$。所以我们需要3位。
3. **拆分成列：** 我们创建3个新特征，分别为 `DeviceType_bin_0`、`DeviceType_bin_1`、`DeviceType_bin_2`。

| Original | 整数 | 二进制 | DeviceType\_bin\_0 | DeviceType\_bin\_1 | DeviceType\_bin\_2 |
| --- | --- | --- | --- | --- | --- |
| 'Laptop' | 1 | 001 | 0 | 0 | 1 |
| 'Tablet' | 2 | 010 | 0 | 1 | 0 |
| 'Phone' | 3 | 011 | 0 | 1 | 1 |
| 'Desktop' | 4 | 100 | 1 | 0 | 0 |
| 'TV' | 5 | 101 | 1 | 0 | 1 |

现在我们只有3个数值列来表示 `DeviceType`，而不是5个列（例如独热编码）。对于一个拥有100个独特类别的特征，独热编码会创建100个特征，而二进制编码只会创建 $ceil(log_2(100)) = 7$ 个特征。

### 优点

- **维度降低：** 与独热编码相比，大幅减少了特征数量，特别是对于高基数变量。特征数量随类别数量（$k$）呈对数增长（$log_2(k)$），而非线性增长。
- **避免隐含顺序（多数情况）：** 尽管它使用了中间整数表示，但最终的二进制列不会带来简单的线性顺序，这可以避免像普通序数编码那样对模型产生过多误导。

### 缺点与注意事项

- **可解释性降低：** 产生的二进制特征是抽象的。与独热特征（它们清楚地表示特定类别的存在与否）不同，二进制列代表比特位置，并且缺乏与原始类别直接关联的含义。
- **模型误解的可能：** 某些模型仍可能在二进制模式中发现复杂的、非预期的关系或顺序。树型模型通常比基于距离或线性模型对此更不敏感。
- **处理未知类别：** 像许多编码器一样，标准二进制编码需要一种方法来处理在新数据中出现但在训练期间未见的类别。

### 实现说明

像 `category_encoders` 这样的库提供了方便的二进制编码（`BinaryEncoder`）实现，它们与 Pandas DataFrames 和 Scikit-learn 管道很好地集成。

```python
# category_encoders 使用示例
# 假设 'df' 是您的 DataFrame，'DeviceType' 是列名
# import category_encoders as ce

# encoder = ce.BinaryEncoder(cols=['DeviceType'])
# df_encoded = encoder.fit_transform(df)

# print(df_encoded.head())
```

### 何时考虑二进制编码

二进制编码在以下情况是一种有用的方法：

- 您有名义类别特征（没有固有顺序）。
- 特征的基数较高，使得独热编码由于内存限制或模型性能问题而变得不切实际。
- 相较于降低维度，单个编码特征的可解释性不那么要紧。

它在独热编码的高维度与更简单方法（如名义数据的序数编码）所带来的信息损失或人工排序之间取得了平衡。请比较它对您的特定模型和数据集可能产生的影响，与哈希编码或目标编码等其他方法进行衡量，特别是在处理非常高基数时。

## 参考资料

- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  提供了特征工程技术的全面视角，包含分类变量编码方法及其权衡。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=u315EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 864
  涵盖了实用数据预处理，包括通用机器学习背景下的分类特征编码。
- [BinaryEncoder - category_encoders documentation](https://contrib.scikit-learn.org/category_encoders/binary.html) — Will McGinnis and category_encoders contributors (2024)
  `category_encoders`库中BinaryEncoder类的官方文档，说明了其用法。
