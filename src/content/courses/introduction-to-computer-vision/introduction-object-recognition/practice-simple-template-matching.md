---
course: "introduction-to-computer-vision"
chapter: "introduction-object-recognition"
lesson: "practice-simple-template-matching"
sourceId: 1302
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition/practice-simple-template-matching"
title: "实践：简单模板匹配"
description: "使用OpenCV进行图像基本模板匹配的实际操作。"
order: 7
plots: []
sourceHash: "28783e4d5ce5006ae523cbd72d4755e794abf55f21613ff7b066df8535675bf5"
sourceCorrections: []
---

模板匹配涉及在较大的源图像中滑动较小的模板图像以寻找最佳匹配。此实践练习将引导您使用Python和OpenCV库实现简单模板匹配。您将加载两张图像，执行匹配，并显示结果。

请确保您已按照第1章的说明配置好开发环境，包括Python和OpenCV。您还需要两张图片：

1. 一张较大的**源图像**，您将在此图中进行搜索。
2. 一张较小的**模板图像**，它表示您要在源图像中寻找的物体或模式。

在本例中，假设您有一张名为`main_scene.jpg`的源图像和一张名为`object_template.png`的模板图像。您可以使用自己的图片，但请确保模板图像比源图像小得多，并且是源图像中一个明显的组成部分。

### 使用OpenCV实现模板匹配

OpenCV提供了一个便利的函数`cv2.matchTemplate()`，它处理滑动模板并在每个位置计算相似度分数的中心逻辑。此函数返回一个结果图，它是一张灰度图像，其中每个像素值表示模板与源图像中以该像素为中心的区域的匹配程度。

让我们分步进行：

1. **导入库：** 我们需要`cv2`用于OpenCV函数，`numpy`用于数值操作。在某些环境（如Jupyter notebooks）中，我们也可能使用`matplotlib.pyplot`来方便显示，但`cv2.imshow()`也很常用。

   ```python
   import cv2
   import numpy as np
   # 可选：用于Jupyter等环境的显示
   # from matplotlib import pyplot as plt
   ```
2. **加载图像：** 加载源图像和模板图像。通常，在灰度图像上执行模板匹配效果更好，因为这通过去除颜色信息来降低复杂性，并专注于亮度模式。

   ```python
   # 加载源图像和模板图像
   # 使用cv2.IMREAD_GRAYSCALE直接将它们作为灰度图加载
   source_img_bgr = cv2.imread('main_scene.jpg')
   source_img_gray = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2GRAY)
   template_img_gray = cv2.imread('object_template.png', cv2.IMREAD_GRAYSCALE)

   # 检查图像是否正确加载
   if source_img_bgr is None or template_img_gray is None:
       print("加载图像出错。请检查文件路径。")
       exit()

   # 获取模板图像的宽度和高度
   # 之后绘制边界框需要此信息
   w, h = template_img_gray.shape[::-1] # shape返回(height, width)，所以我们反转它
   ```

   我们首先加载彩色源图像（`source_img_bgr`），因为我们希望在彩色图像上绘制结果矩形以便更好地显示。然后，我们创建一个灰度版本（`source_img_gray`）用于匹配过程。模板直接作为灰度图加载。我们还存储了模板的宽度（`w`）和高度（`h`）。
3. **执行模板匹配：** 调用`cv2.matchTemplate()`。您需要提供源图像（灰度）、模板图像（灰度）和比较方法。

   ```python
   # 执行模板匹配
   # cv2.TM_CCOEFF_NORMED通常是一个不错的选择
   # 其他方法包括：TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF
   method = cv2.TM_CCOEFF_NORMED
   result_map = cv2.matchTemplate(source_img_gray, template_img_gray, method)
   ```

   我们选择了`cv2.TM_CCOEFF_NORMED`（归一化 (normalization)相关系数）。此方法计算模板与图像块之间的相关性，并进行归一化以处理亮度变化。值范围从-1.0到1.0，其中1.0表示完美匹配。还存在其他方法，例如`cv2.TM_SQDIFF_NORMED`（归一化平方差），其中*最低*值表示最佳匹配。
4. **寻找最佳匹配位置：** `result_map`包含相似度分数。我们需要找到最高分数（或最低分数，取决于方法）的位置（像素坐标）。OpenCV的`cv2.minMaxLoc()`函数可以高效地完成此操作。

   ```python
   # 在结果图中找到最小值和最大值及其位置
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_map)

   # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
   if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
       top_left = min_loc
       match_value = min_val
   else:
       top_left = max_loc
       match_value = max_val

   print(f"最佳匹配左上角坐标: {top_left}")
   print(f"最佳匹配得分: {match_value:.4f}")
   ```

   `cv2.minMaxLoc()`返回最小值、最大值、最小值位置（作为`(x, y)`元组）和最大值位置。我们根据所使用的方法，选择`min_loc`或`max_loc`作为最佳匹配区域的`top_left`角。
5. **确定边界框坐标：** 找到的位置（`top_left`）是匹配区域的左上角。要绘制一个矩形，我们还需要右下角。我们可以使用模板的宽度（`w`）和高度（`h`）来计算。

   ```python
   # 计算边界框的右下角
   bottom_right = (top_left[0] + w, top_left[1] + h)
   ```
6. **显示结果：** 在*原始彩色源图像*上绘制一个矩形，以突出显示检测到的区域。然后，显示图像。

   ```python
   # 在源图像的匹配区域周围绘制一个矩形
   # 我们在彩色图像(source_img_bgr)上绘制以获得更好的显示效果
   cv2.rectangle(source_img_bgr, top_left, bottom_right, (0, 255, 0), 2) # 绿色矩形，粗细为2

   # 显示结果
   cv2.imshow('Matched Result', source_img_bgr)

   # 可选：显示结果图（显示匹配强度）
   # cv2.imshow('Result Map', result_map)

   # 等待按键，然后关闭窗口
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```

   我们使用`cv2.rectangle()`指定要绘制的图像、左上角、右下角、颜色（BGR格式——此处为绿色）和线条粗细。`cv2.imshow()`在窗口中显示图像。`cv2.waitKey(0)`无限期等待按键，`cv2.destroyAllWindows()`关闭显示窗口。

### 整合所有代码

以下是完整的Python脚本：

```python
import cv2
import numpy as np

# --- 配置 ---
SOURCE_IMAGE_PATH = 'main_scene.jpg'
TEMPLATE_IMAGE_PATH = 'object_template.png'
# 选择匹配方法
# 选项: cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#          cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
MATCHING_METHOD = cv2.TM_CCOEFF_NORMED
# -------------------

# 1. 加载图像
print("正在加载图像...")
source_img_bgr = cv2.imread(SOURCE_IMAGE_PATH)
# 检查源图像是否加载
if source_img_bgr is None:
    print(f"错误：无法从 {SOURCE_IMAGE_PATH} 加载源图像")
    exit()

# 将源图像转换为灰度图以进行匹配
source_img_gray = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2GRAY)

# 将模板作为灰度图加载
template_img_gray = cv2.imread(TEMPLATE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# 检查模板图像是否加载
if template_img_gray is None:
    print(f"错误：无法从 {TEMPLATE_IMAGE_PATH} 加载模板图像")
    exit()

# 获取模板尺寸（宽度，高度）
w, h = template_img_gray.shape[::-1]
print(f"模板尺寸（宽 x 高）：{w} x {h}")

# 2. 执行模板匹配
print(f"正在使用方法 {MATCHING_METHOD} 执行模板匹配...")
result_map = cv2.matchTemplate(source_img_gray, template_img_gray, MATCHING_METHOD)

# 3. 寻找最佳匹配位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_map)

# 根据方法确定左上角
if MATCHING_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    # 对于SQDIFF方法，最佳匹配是最小值
    top_left = min_loc
    match_value = min_val
    print(f"在 {top_left} 找到最佳匹配，得分为（越低越好）：{match_value:.4f}")
else:
    # 对于CCOEFF/CCORR方法，最佳匹配是最大值
    top_left = max_loc
    match_value = max_val
    print(f"在 {top_left} 找到最佳匹配，得分为（越高越好）：{match_value:.4f}")

# 4. 确定边界框
bottom_right = (top_left[0] + w, top_left[1] + h)

# 5. 显示结果
print("正在源图像上绘制边界框...")
# 在原始*彩色*图像上绘制矩形
cv2.rectangle(source_img_bgr, top_left, bottom_right, (0, 255, 0), 2) # 绿色矩形

# 显示带边界框的图像
cv2.imshow('Matched Result', source_img_bgr)
print("正在显示结果。按任意键退出。")

# 可选：显示结果图以查看匹配强度
# result_map_display = cv2.normalize(result_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# cv2.imshow('Result Map (Match Intensity)', result_map_display)

# 无限期等待直到按键
cv2.waitKey(0)

# 清理：关闭所有OpenCV窗口
cv2.destroyAllWindows()
print("窗口已关闭。正在退出。")
```

### 理解输出

当您运行此脚本时（假设您的图像已正确找到），您应该会看到一个弹出窗口，显示您的源图像，并在与模板最匹配的区域周围绘制了一个绿色矩形。控制台输出将显示此矩形的左上角坐标和匹配分数。

### 实验与观察

尝试对这段代码进行实验：

- 使用不同的源图像和模板图像。观察当模板中的物体在源图像中多次出现时它的工作情况（基本模板匹配只找到*最佳*的单个匹配）。
- 使用源图像中*不存在*的模板。观察匹配分数——它应该比存在良好匹配时显著降低（对于`TM_CCOEFF_NORMED`）或升高（对于`TM_SQDIFF_NORMED`）。
- 尝试不同的匹配方法（如`cv2.TM_SQDIFF_NORMED`、`cv2.TM_CCORR_NORMED`等），看看结果如何变化。请记住相应地调整选择`min_loc`或`max_loc`的逻辑。
- 稍微修改模板（例如，使用图像编辑器调整大小、轻微旋转），看看匹配的敏感度如何。

此练习展示了模板匹配的基本应用。在您实验时，可能会遇到它难以处理的情况，特别是当比例、旋转、光照或视角发生变化时。这直接说明了我们之前讨论的局限性，并强调了为什么对于更普遍的物体识别，更高级的技术（通常涉及机器学习 (machine learning)）是必要的。

## 参考资料

- [OpenCV: Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_template_matching.html) — OpenCV Development Team (2024)
  提供了关于cv2.matchTemplate()及其各种比较方法的详细信息，对实际实现至关重要。
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer; DOI: [10.1007/978-1-84882-935-4](https://doi.org/10.1007/978-1-84882-935-4)
  一本基础教科书，解释了计算机视觉中图像相关和模板匹配的理论基础。
- [Learning OpenCV 4 Computer Vision with Python 3](https://www.oreilly.com/library/view/learning-opencv-4/9781789531619/) — Joseph Howse, Joe Minichino (2020)
  Publisher: Packt Publishing
  提供了使用Python和OpenCV实现计算机视觉任务（包括模板匹配）的实践示例和方法。
