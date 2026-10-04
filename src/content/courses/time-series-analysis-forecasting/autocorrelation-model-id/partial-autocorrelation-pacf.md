---
course: "time-series-analysis-forecasting"
chapter: "autocorrelation-model-id"
lesson: "partial-autocorrelation-pacf"
sourceId: 3405
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/partial-autocorrelation-pacf"
title: "偏自相关函数 (PACF)"
description: "了解偏自相关的含义，以及PACF如何衡量在剔除较短滞后效应后的关联性。"
order: 2
plots: []
sourceHash: "7ce923817199c534a14850c52b146b68382e1c575bc17b18e7abac1a3c6b916c"
sourceCorrections: []
---

自相关函数 (ACF) 能够提供时间序列观测值与其过去值关联方式的概览。然而，ACF不区分直接和间接关系。例如，ACF衡量的$y_t$和$y_{t-2}$之间的关联包含了$y_{t-1}$的影响，因为$y_t$通常与$y_{t-1}$相关联，而$y_{t-1}$又与$y_{t-2}$相关联。为了分离出$y_t$和特定滞后$y_{t-k}$之间的直接关系，剔除$y_{t-1}, y_{t-2}, \dots, y_{t-k+1}$等中间观测值的影响，需要一种不同的工具。偏自相关函数 (PACF) 正是用来衡量这种直接关系的。

### 什么是偏自相关？

可以将滞后$k$处的偏自相关视为$y_t$和$y_{t-k}$之间的关联，这种关联*不*由它们与中间滞后（$t-1, t-2, \dots, t-k+1$）处时间序列值的相互关联所解释。

可以设想通过以下方式计算它：

1. 使用中间滞后$y_{t-1}, \dots, y_{t-k+1}$来预测$y_t$。将此预测的误差设为$e_t$。
2. 使用相同的中间滞后$y_{t-1}, \dots, y_{t-k+1}$来预测$y_{t-k}$。将此预测的误差设为$e_{t-k}$。
3. 滞后$k$处的偏自相关是这两个预测误差之间的关联，即$Corr(e_t, e_{t-k})$。

这个过程有效地去除了与中间滞后相关的线性依赖，只留下$y_t$和$y_{t-k}$之间的直接关联。

> 在考虑$y_t$和$y_{t-2}$之间的关系时，ACF和PACF之间的区别。PACF在考虑了$y_{t-1}$之后，分离出直接联系。

### 为何使用PACF？识别AR模型

PACF的主要用途在于确定自回归 (autoregressive) (AR) 模型的阶数 ($p$)。回想一下，AR(p)模型将$y_t$表示为其前$p$个值的线性组合加上一个误差项：


$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t
$$


根据定义，在纯AR(p)过程当中，$y_t$对$y_{t-1}, \dots, y_{t-p}$有直接线性依赖。然而，一旦考虑了这$p$个滞后，则$y_t$与更远的滞后（如$y_{t-p-1}, y_{t-p-2}$等）之间就不应存在*直接*线性关系。在ACF中，在这些更远的滞后处看到的任何关联都是间接的，通过前$p$个滞后传递。

因此，对于平稳AR(p)过程，我们预期PACF图会显示：

- 直至滞后$p$都存在显著的偏自相关。
- 在滞后$p$之后急剧截断，对于滞后$k > p$的偏自相关在统计上不显著（接近零且在显著性边界内）。

这种独特的模式与AR(p)过程的ACF形成对比，AR(p)过程的ACF通常呈现出更缓慢地衰减至零。

### PACF图的解读

与ACF图类似，PACF图在y轴上显示不同滞后的偏自相关值，x轴上显示滞后数。它们通常也包含置信区间（常以阴影区域表示，通常为95%置信水平）。

- **显著滞后：** 超出置信区间的尖峰被认为在统计上显著。
- **截断：** 在某个滞后$p$之后，偏自相关值急剧降至不显著，这是AR(p)过程的特点。
- **衰减：** 与AR模型的ACF不同，MA（移动平均）模型的PACF倾向于逐渐衰减。

通过结合ACF图（我们将在下一节讨论如何生成）查看PACF图，您将获得关于时间序列潜在结构的有用线索，并可以做出明智的决定，确定哪种模型类型（AR、MA或ARMA）可能合适，以及需要考量哪些阶数参数 (parameter)。对于识别AR模型阶数，PACF尤其有用。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E.P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley
  时间序列分析的经典参考文献，包含自相关和偏自相关函数的理论背景，用于ARIMA模型识别。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解的在线教科书，提供了时间序列概念的实用解释，包括用于模型识别的ACF和PACF，附有示例和R代码。
- [Introduction to Time Series and Forecasting](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEODI06gzYw_8t9x-aguuK7qR16UDXQQcgIuFLenSUqY0-uSu39MKjNU0HUTPivcGYGVu_8uqKwWHZ7kTridUK1HaTvxZGkwzCjU43d2ZV_AFxju0Et5QUtjIJXUau0jg_SZtJikMTYyg3RG4Bri3KNqfXd1ds6hjlXlcLC06mq8n1wx-BnBW29xLjX6szn9Xm58Xf-bC9p6Ykk3Fw=) — Peter J. Brockwell, Richard A. Davis (2006)
  Publisher: Springer
  一本广泛使用的时序分析教科书，涵盖了PACF的理论和应用，以及其在识别AR过程中的作用。
