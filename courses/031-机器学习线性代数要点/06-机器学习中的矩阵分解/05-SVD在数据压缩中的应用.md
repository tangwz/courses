# SVD在数据压缩中的应用

来源：[原文](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/svd-data-compression)

[返回章节目录](README.md) · [返回课程目录](../README.md)

奇异值分解（SVD）的一个有力应用在于数据压缩，尤其适用于可以自然表示为矩阵的数据，例如图像。其主要思路是SVD按重要性顺序组织矩阵中的信息，这使我们能够丢弃不那么重要的信息，同时保留原始数据的良好近似。

回顾$m \times n$矩阵$A$的SVD：
$A = U \Sigma V^T$
此处，$\Sigma$是一个$m \times n$的对角矩阵，包含从大到小排列的奇异值$\sigma_1, \sigma_2, \sigma_3, \dots$ ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$)。这些奇异值量化 (quantization)了$U$和$V$的列所表示的相应方向的重要性。较大的奇异值对应于在矩阵中捕获更多数据方差或“能量”的维度。

通过SVD进行数据压缩，其方式是为原始矩阵$A$创建一个*低秩近似*。我们不使用所有的奇异值和向量 (vector)，而是只保留前$k$个成分，其中$k$小于原始矩阵的秩。这个过程通常被称为*截断SVD*。

我们构造近似矩阵$A_k$如下：
$A_k = U_k \Sigma_k V_k^T$
其中：

- $U_k$由$U$的前$k$列组成(一个$m \times k$矩阵)。
- $\Sigma_k$是一个$k \times k$的对角矩阵，包含前$k$个奇异值($\sigma_1, \dots, \sigma_k$)。
- $V_k^T$由$V^T$的前$k$行组成(即$V$的前$k$列的转置，形成一个$k \times n$矩阵)。

Eckart-Young-Mirsky定理指出，在Frobenius范数(以及谱范数)意义下，矩阵$A_k$是$A$的最佳秩$k$近似。从根本上说，通过保留与最大奇异值相关的成分，我们保留了原始矩阵$A$最重要的部分。

### 为什么这是压缩？

存储原始矩阵$A$需要存储$m \times n$个数字。存储截断SVD成分$U_k$, $\Sigma_k$, 和$V_k^T$需要存储：

- $U_k$的$m \times k$个值
- $\Sigma_k$对角元素的$k$个值
- $V_k^T$的$k \times n$个值

总存储量为$mk + k + kn = k(m + n + 1)$个值。如果$k$远小于$m$和$n$，那么$k(m + n + 1)$可以远小于$m \times n$，从而实现压缩。

### 图像压缩示例

灰度数字图像可以表示为一个矩阵，其中每个元素对应特定位置的像素强度。假设我们有一个$512 \times 512$的灰度图像。这个矩阵$A$有$512 \times 512 = 262,144$个元素。

1. **应用SVD：** 计算$A = U \Sigma V^T$。$U$和$V$将是$512 \times 512$的，$\Sigma$也将是$512 \times 512$的。
2. **截断：** 选择$k$的值。例如，设$k=50$。构造$A_{50} = U_{50} \Sigma_{50} V_{50}^T$。
3. **存储：** 我们不存储原始的262,144个像素值，而是存储$U_{50}$ ($512 \times 50$)、$\Sigma_{50}$ ($50$个值)和$V_{50}^T$ ($50 \times 512$)。总存储量为$50 \times 512 + 50 + 50 \times 512 = 25600 + 50 + 25600 = 51,250$个值。这大约是原始存储大小的20%。
4. **重构：** 要查看压缩图像，需要将存储的成分相乘：$A_{50} = U_{50} \Sigma_{50} V_{50}^T$。结果矩阵$A_{50}$是原始图像$A$的近似。

$k$的选择决定了压缩比和图像质量之间的权衡。较小的$k$会带来更高的压缩率，但图像重构质量可能较低。较大的$k$会使重构图像与原始图像更接近，但压缩率较低。通常，相对较小的$k$可以捕获大部分视觉信息，因为典型图像的奇异值往往会迅速衰减。



[交互图表：图像矩阵的奇异值衰减示例](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/svd-data-compression#plot-n48jaq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "line",
      "x": [
        1,
        2,
        3,
        4,
        5,
        10,
        20,
        50,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        10000,
        8000,
        6000,
        4000,
        3000,
        1500,
        800,
        300,
        100,
        20,
        5,
        2,
        1
      ],
      "mode": "lines+markers",
      "marker": {
        "color": "#228be6"
      },
      "line": {
        "color": "#228be6"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "图像矩阵的奇异值衰减示例"
    },
    "xaxis": {
      "title": {
        "text": "奇异值索引 (k)"
      }
    },
    "yaxis": {
      "title": {
        "text": "奇异值大小 (σ_k)"
      },
      "type": "log"
    },
    "margin": {
      "l": 60,
      "r": 20,
      "t": 50,
      "b": 50
    },
    "width": 600,
    "height": 400
  }
}
```

</details>



> 图像矩阵的奇异值通常显示出陡峭的初始下降，随后是一长串较小的值。这表明，仅保留前几十个或几百个奇异值(对应于秩$k$近似)可以保留图像的大部分结构，同时丢弃与较小奇异值相关的成分，这些成分可能对应于更精细的细节或噪声。

对于通常具有独立通道(例如，红色、绿色、蓝色)的彩色图像，SVD压缩可以独立应用于表示每个颜色通道的矩阵。

### 有损压缩

值得注意的是，基于SVD的压缩是*有损的*。重构矩阵$A_k$与原始矩阵$A$不完全相同(除非$k$等于$A$的秩)。在截断步骤中，部分信息被永久丢弃。然而，由于SVD优先处理最重要的成分，$A$和$A_k$之间的*感知*差异可以很小，特别是在$k$值适中的情况下，这使得它在图像压缩等应用中非常有用，因为在这些应用中并不总是需要完美重构。

图像压缩是一个经典的例子，但该原理适用于任何可以表示为矩阵的数据，只要低秩近似是可以接受的，并且对存储或计算效率有益。这可能包括机器学习 (machine learning)中的大型特征矩阵、推荐系统中的用户-物品交互矩阵，或数值模拟数据。

## 参考资料

- [Numerical Linear Algebra](https://epubs.siam.org/doi/book/10.1137/1.9780898719574) — Lloyd N. Trefethen and David Bau III (1997)
  Publisher: SIAM; DOI: [10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
  涵盖SVD的理论和计算方面及其在低秩近似中的应用，包括Eckart-Young-Mirsky定理。
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Gilbert Strang (2019)
  Publisher: Wellesley-Cambridge Press
  在机器学习背景下，讨论SVD作为数据压缩和降维的基本工具。
- [Digital Image Processing](https://www.pearson.com/us/higher-education/product/Gonzalez-Digital-Image-Processing-4th-Edition/9780133356724.html) — Rafael C. Gonzalez and Richard E. Woods (2016)
  Publisher: Pearson
  在数字图像处理领域，详细解释和示例SVD在图像压缩中的应用。(第四版)
- [18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) — Gilbert Strang (2018)
  Publisher: MIT OpenCourseWare
  提供视频讲座和课程笔记，阐明SVD原理并演示其在图像压缩中的应用。

---

[上一节](04-SVD%E5%9C%A8%E9%99%8D%E7%BB%B4%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md) · [下一节](06-%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3%E4%B8%8E%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3%E7%9A%84%E5%85%B3%E7%B3%BB.md)
