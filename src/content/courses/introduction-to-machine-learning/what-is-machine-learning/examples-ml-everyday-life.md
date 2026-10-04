---
course: "introduction-to-machine-learning"
chapter: "what-is-machine-learning"
lesson: "examples-ml-everyday-life"
sourceId: 1452
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-1-what-is-machine-learning/examples-ml-everyday-life"
title: "机器学习在日常生活中的实例"
description: "了解您日常生活中遇到的机器学习运用实例。"
order: 3
plots: []
sourceHash: "482d39032cbe18d18c7c7f124695a31e8c4a2283a118e16c55807039e9638e7e"
sourceCorrections: []
---

机器学习 (machine learning)听起来可能像只存在于研究实验室或科技巨头公司中的事物，但您却在不断地与它互动，往往没有意识到这一点。它已融入您日常使用的许多数字服务和产品中。让我们看几个常见实例，以便更好地理解它。

### 您的电子邮件收件箱：垃圾邮件过滤

垃圾邮件过滤是机器学习 (machine learning)最早的广泛应用之一。您还记得以前收件箱里总是充斥着大量不想要的邮件吗？得益于机器学习，现代电子邮件服务在拦截垃圾邮件方面做得好得多。

- **工作原理：** 该系统从大量已被标记 (token)为“垃圾邮件”或“非垃圾邮件”（即正常邮件）的邮件中进行学习。它识别出模式，例如特定的词语或短语、发件人信誉、邮件结构，甚至是倾向于出现在垃圾邮件中的图像特征。当新邮件到达时，过滤器运用这些学到的模式来预测它是否为垃圾邮件。
- **“学习”过程：** 每次您手动将邮件标记为垃圾邮件，或将邮件从垃圾邮件文件夹中取出时，您都在提供反馈，这有助于系统调整其判断并随着时间推移提高准确性。这是分类的一个典型实例，属于监督学习 (supervised learning)的一种。

> 一个机器学习模型分析收到的邮件特征，将其分类为垃圾邮件或正常邮件。

### 获取推荐：产品、电影和音乐

您是否曾注意到Netflix等流媒体服务如何推荐您可能喜欢的电影，或者Amazon等在线零售商如何向您呈现与您浏览历史相关的产品？这些推荐引擎都由机器学习 (machine learning)驱动。

- **工作原理：** 这些系统分析您的过往行为（您观看过、购买过、喜欢过或评价过的），并将其与数百万其他用户的行为进行比较。它们寻找模式和相似之处。例如，如果您观看了电影A、B和C，而其他观看过A和B的用户也经常观看电影D，那么系统可能会向您推荐电影D。它们还会分析项目特征（如类型、演员、产品类别、品牌）。
- **“学习”过程：** 系统会根据您的持续互动和用户群不断变化的偏好来更新其推荐。它学习用户和项目之间复杂的关系，以预测您会觉得有吸引力或有用的内容。这通常涉及监督学习 (supervised learning)（预测评分）和无监督学习 (unsupervised learning)（将相似用户或项目分组）的技术。

### 网页搜索

像Google这样的搜索引擎使用精密的机器学习 (machine learning)模型来理解您的查询，并对数十亿网页进行排名，以便几乎即时地为您提供最相关的结果。

- **工作原理：** 机器学习模型分析您的搜索词，考虑同义词、上下文 (context)和潜在意图。它们还会根据数百个因素评估网页的质量和相关性，这些因素包括页面内容、指向该页面的链接数量和质量、加载速度以及用户交互数据（例如用户在访问页面后是否快速点击返回）。
- **“学习”过程：** 搜索引擎不断从用户行为中学习。当您点击搜索结果时，您隐式地告诉搜索引擎该结果与您的查询相关。这种反馈有助于模型调整其排名算法，从而为未来的搜索提供更好的结果。

### 语音助手和翻译

Apple的Siri、Amazon Alexa、Google Assistant和Google Translate等设备和服务都严重依赖于机器学习 (machine learning)。

- **工作原理：**
  - **语音识别：** 机器学习模型通过大量口语数据进行训练，将您声音的声波转换为文本。
  - **自然语言处理 (NLP)：** 其他模型随后分析此文本，以理解您的命令或问题（您的“意图”）。
  - **翻译：** 像Google Translate这样的服务使用在不同语言的平行文本上训练的机器学习模型，学习如何翻译句子，捕捉语法和意义上的特征。
- **“学习”过程：** 这些系统随着处理更多的语音命令、文本和翻译实例而不断改进，学习不同的口音、措辞和语言模式。

### 其他常见实例

- **图像识别：** 在社交媒体上自动识别照片中的朋友，在图片中识别物体以供搜索，甚至用您的面部解锁手机。
- **欺诈检测：** 银行和信用卡公司使用机器学习 (machine learning)，通过学习与合法和欺诈活动相关的模式来检测潜在的欺诈性交易。
- **导航应用：** Google Maps或Waze等应用使用机器学习，根据其他用户的实时数据和历史交通模式来预测交通状况和估计到达时间。

这些仅仅是几个实例，但它们阐明了机器学习如何使计算机能够执行使用传统方法难以或不可能明确编程的任务。这些系统不是遵循僵硬的指令，而是从数据中学习以进行预测和决策，并且通常会随着时间推移变得更准确、更有用。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  本书为读者提供了对各种机器学习模型（包括分类、回归和重采样方法）清晰的介绍，这些方法是垃圾邮件过滤和推荐系统等应用的基础。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  该资源解释了广泛的机器学习概念和算法，并通过实际例子加以说明，这些例子与章节中讨论的日常应用（如分类、推荐系统和自然语言处理）相符。
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng, Geoff Ladwig, Aarti Bagul, Eddy Shyu (2022)
  Publisher: DeepLearning.AI, Stanford Online
  一门广受认可的在线课程，通过清晰的解释和实际例子介绍核心机器学习概念和算法，非常适合有兴趣了解机器学习如何驱动日常应用的初学者。
