# 动手操作：图像数据FID计算

来源：[原文](https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-5-specialized-model-specific-metrics/practice-calculating-fid)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Fréchet Inception Distance (FID) 是评估合成图像质量和多样性的有用指标。动手计算一组真实图像和一组合成生成图像之间的 FID 分数，使用 Python 进行。

FID 通过比较预训练 (pre-training)神经网络 (neural network)（通常是 Inception v3 模型）提取的特征统计数据，来衡量两个图像分布之间的相似度。较低的 FID 分数表示合成图像的分布更接近真实图像的分布，表明其质量和多样性更优。

### 前提条件与设置

在我们开始之前，请确保您已安装所需的库。我们将主要使用 TensorFlow 加载 Inception v3 模型并执行计算，NumPy 进行数值运算，如果直接实现公式，可能还需要 SciPy 进行矩阵计算。

```bash
pip install tensorflow numpy scipy Pillow
```

您还需要两组图像：

1. 一个包含您的真实图像的目录。
2. 一个包含您的模型生成的合成图像的目录。

在本示例中，我们假设这些图像分别存储在 `path/to/real/images` 和 `path/to/synthetic/images` 中。

### 图像加载与预处理

Inception v3 模型需要特定大小的输入图像（通常是 299x299 像素），并以特定方式进行预处理（像素值缩放到 [-1, 1] 范围）。我们需要确保真实图像和合成图像都符合这些要求。

```python
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import warnings

# 抑制特定的 TensorFlow 警告，以获得更清晰的输出
warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# 定义 Inception v3 所需的图像大小
IMAGE_SIZE = (299, 299)

def preprocess_image(image_path):
    """为 Inception v3 加载并预处理图像。"""
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # 将像素值缩放到 [-1, 1] 范围，符合 InceptionV3 的预期
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"警告：由于错误 {e}，跳过文件 {image_path}")
        return None

def load_and_preprocess_images(dir_path, max_images=None):
    """从目录加载并预处理所有图像。"""
    image_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"从 {dir_path} 处理最多 {max_images} 张图像")

    processed_images = []
    for path in image_paths:
        processed = preprocess_image(path)
        if processed is not None:
            processed_images.append(processed)

    if not processed_images:
        raise ValueError(f"在目录中未找到有效图像或未处理任何图像：{dir_path}")

    return np.array(processed_images)

# --- 占位符路径：请替换为您的实际目录 ---
# 建议使用至少几千张图像以获得稳定的 FID。
# 为演示目的，我们可能会使用较少图像，但请注意结果会有所不同。
PATH_REAL_IMAGES = 'path/to/real/images' 
PATH_SYNTHETIC_IMAGES = 'path/to/synthetic/images' 
MAX_IMAGES_PER_SET = 100 # 使用少量图像进行快速演示；进行实际评估时请增加数量

print("正在加载和预处理真实图像...")
# 添加目录存在性错误处理
if not os.path.isdir(PATH_REAL_IMAGES):
    print(f"错误：未找到真实图像目录：{PATH_REAL_IMAGES}")
    print("请将 'path/to/real/images' 替换为正确的路径。")
    # 将 real_images 设置为 None 或进行适当处理
    real_images = None 
else:
    real_images = load_and_preprocess_images(PATH_REAL_IMAGES, MAX_IMAGES_PER_SET)

print("正在加载和预处理合成图像...")
if not os.path.isdir(PATH_SYNTHETIC_IMAGES):
    print(f"错误：未找到合成图像目录：{PATH_SYNTHETIC_IMAGES}")
    print("请将 'path/to/synthetic/images' 替换为正确的路径。")
    # 将 synthetic_images 设置为 None 或进行适当处理
    synthetic_images = None
else:
    synthetic_images = load_and_preprocess_images(PATH_SYNTHETIC_IMAGES, MAX_IMAGES_PER_SET)

# 在继续之前检查图像是否成功加载
if real_images is None or synthetic_images is None:
    print("\n由于图像数据缺失，中止执行。请检查路径和图像文件。")
    # 如果数据缺失，则退出或跳过 FID 计算
    # 对于脚本，您可以在导入 sys 后使用：sys.exit(1)
else:
    print(f"已加载 {len(real_images)} 张真实图像和 {len(synthetic_images)} 张合成图像。")
    print("预处理完成。")
```

请务必将 `path/to/real/images` 和 `path/to/synthetic/images` 替换为您的图像数据集的实际路径。我们还添加了一个 `MAX_IMAGES_PER_SET` 变量用于演示目的；为了获得可靠的 FID 分数，您应该使用更多图像（通常是数千张）。

### 使用 Inception V3 计算激活值

下一步是将这些预处理过的图像输入到 Inception v3 模型（在 ImageNet 上预训练 (pre-training)）中，并从其中一个较深层提取激活值。通常使用的层是分类输出之前的最终池化层，因为它的特征捕获了高级图像特性。

```python
from scipy.linalg import sqrtm # 用于矩阵平方根

def calculate_activations(images, model):
    """使用模型计算一批图像的激活值。"""
    if images is None or len(images) == 0:
        return np.array([])
    activations = model.predict(images)
    return activations

def calculate_fid(act1, act2):
    """计算两组激活值之间的 FID 分数。"""
    if act1.size == 0 or act2.size == 0:
        print("警告：一个或两个激活集为空。无法计算 FID。")
        return float('inf') # 或作为错误处理

    # 计算均值和协方差统计数据
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # 计算均值之间的平方差和
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # 计算协方差矩阵乘积的平方根
    # 有时可能需要添加一个小的 epsilon 以提高数值稳定性
    eps = 1e-6
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False) 

    # 检查并修正 sqrtm 产生的虚数
    if np.iscomplexobj(covmean):
        # print("警告：矩阵平方根中产生了复数。取实部。")
        covmean = covmean.real

    # 协方差矩阵的数值稳定性检查
    if not np.isfinite(sigma1).all() or not np.isfinite(sigma2).all():
         print("警告：在协方差矩阵中发现非有限值。FID 可能不稳定。")
         return float('inf') # 表示不稳定

    # 计算分数
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    # 检查是否存在负 FID（可能由于数值不稳定性而发生）
    if fid < 0:
        # print(f"警告：计算出负 FID ({fid})。截断为 0。")
        # 进行适当处理，例如，记录问题或截断
        fid = 0.0

    return fid

# 仅在图像成功加载后继续
if real_images is not None and synthetic_images is not None:
    # 加载在 ImageNet 上预训练的 InceptionV3 模型，不包括顶部分类层
    # 输出将是全局平均池化层的特征
    print("正在加载 InceptionV3 模型...")
    inception_model = tf.keras.applications.InceptionV3(
        include_top=False, 
        pooling='avg', # 全局平均池化层输出
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        weights='imagenet'
    )
    print("模型已加载。")

    print("正在计算真实图像的激活值...")
    activations_real = calculate_activations(real_images, inception_model)

    print("正在计算合成图像的激活值...")
    activations_synthetic = calculate_activations(synthetic_images, inception_model)

    if activations_real.size > 0 and activations_synthetic.size > 0:
        print("正在计算 FID 分数...")
        # 确保激活值是二维数组，以便进行协方差计算
        if activations_real.ndim == 1: activations_real = activations_real.reshape(-1, 1)
        if activations_synthetic.ndim == 1: activations_synthetic = activations_synthetic.reshape(-1, 1)

        fid_score = calculate_fid(activations_real, activations_synthetic)
        print(f"\n计算出的 FID 分数：{fid_score:.4f}")
    else:
        print("\n由于激活集为空，跳过 FID 计算。")
```

此代码加载 Inception v3 模型，使用 `predict` 方法计算两个图像集的激活值，然后使用我们的 `calculate_fid` 函数应用 FID 公式。我们使用 `scipy.linalg.sqrtm` 进行矩阵平方根计算，这通常是数值上最具挑战性的部分。我们已包含对空激活值和潜在数值问题（如复数或非有限值）的基本检查。

### 结果说明与注意事项

输出的 `fid_score` 代表 Fréchet Inception Distance。请记住：

- **值越低越好：** 分数越接近 0 表示真实和合成图像特征的分布越相似。
- **背景环境很重要：** FID 分数是相对的。请将您生成的图像的 FID 与基准模型或生成器的先前迭代进行比较。对于一个复杂的、当前最佳为 40 的数据集，FID 为 50 可能算不错，但如果典型分数低于 10，则表示不佳。
- **样本量：** FID 对所用图像数量很敏感。确保您使用来自真实和合成数据集的足够大且有代表性的样本（理想情况下是数千张图像），以获得稳定和有意义的结果。使用过少的图像（如我们 `MAX_IMAGES_PER_SET=100` 的示例）将导致分数不可靠。
- **计算成本：** 计算数千张图像的激活值可能很耗时，特别是在没有 GPU 加速的情况下。
- **特征提取器：** 尽管 Inception v3 是标准配置，但其他特征提取器也可使用，可能导致不同的 FID 分数。对于比较而言，使用相同提取器保持一致性很重要。

本实践练习展示了计算 FID 的核心步骤。通过应用此指标，您可以量化 (quantization)地评估您的生成模型捕获真实数据分布视觉特性的能力，这是评估合成图像质量的一个重要方面。请记住在将其应用于您自己的项目时，调整路径并考虑实际限制。

## 参考资料

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Volume: 30; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  介绍了用于评估生成模型的Fréchet Inception Distance（FID），提供了其数学基础和实际应用。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习的全面背景，包括生成模型（如GANs）及其评估方法。
- [Keras Applications: InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3) — TensorFlow Developers (2023)
  Publisher: Google
  Keras中InceptionV3模型的官方文档，详细说明其用法、参数和预处理要求。

---

[上一节](05-VAE%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md) · [下一节](../06-%E6%9E%84%E5%BB%BA%E5%85%A8%E9%9D%A2%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8A%A5%E5%91%8A/01-%E4%B8%BA%E4%BB%BB%E5%8A%A1%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E8%A1%A1%E9%87%8F%E6%A0%87%E5%87%86.md)
