---
course: "advanced-tensorflow"
chapter: "production-ml-pipelines-tfx"
lesson: "practice-simple-tfx-pipeline"
sourceId: 2915
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/practice-simple-tfx-pipeline"
title: "动手实践：构建一个简单的 TFX 流水线"
description: "使用标准 TFX 组件构建并运行一个基本的端到端机器学习流水线。"
order: 9
plots: []
sourceHash: "5de178cf7e579222a187be9b6e8b44c57f6c1076ba3932d505484575139f5539"
sourceCorrections: []
---

构建一个使用核心 TFX 组件的端到端机器学习 (machine learning)流水线，可以提供实际经验。这个演示展示了这些组件如何相互连接和传递数据（产物），自动化从数据引入到模型部署的工作流程。

本例将使用广泛采用的芝加哥出租车行程数据集的简化版本。我们的目的不是构建最准确的预测模型，而是说明构建和运行 TFX 流水线的机制。

### 前提条件与设置

在开始之前，请确保您的 Python 环境中已安装 TFX。您通常可以使用 pip 进行安装：

```bash
pip install tfx
```

根据您协调流水线的方式，您可能还需要 Apache Beam 等特定运行器。对于此本地示例，默认运行器应该足够。我们还假设您有一个本地目录，可以存储流水线产物和输入数据。

假设我们的原始出租车数据（例如 `data.csv`）位于已知目录 `./data` 中。我们的流水线定义和支持代码将位于一个 Python 脚本（例如 `taxi_pipeline.py`）中，TFX 将在指定的流水线根目录（例如 `./pipeline_output`）中生成输出。

### 定义流水线结构

TFX 流水线在 Python 中以编程方式定义。您导入所需的组件并将它们连接起来，指定输入和输出。流水线定义描述了所需的工作流程图。

让我们首先在 `taxi_pipeline.py` 脚本中设置基本结构。我们需要定义数据、流水线输出以及 `Transform` 和 `Trainer` 等组件所需的任何模块文件的路径。

```python
# taxi_pipeline.py
import os
import tfx
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.dsl.components.common importresolver
from tfx.dsl.experimental.latest_artifacts_resolver import LatestArtifactsResolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# 定义路径
_pipeline_name = 'taxi_simple'
_data_root = './data' # 包含 data.csv 的目录
_module_file = './taxi_utils.py' # 用于 Transform 和 Trainer 代码的文件
_pipeline_root = os.path.join('./pipeline_output', _pipeline_name)
_serving_model_dir = os.path.join(_pipeline_root, 'serving_model')

# 确保输出目录存在（可选，TFX 通常会处理）
os.makedirs(_pipeline_root, exist_ok=True)
os.makedirs(_serving_model_dir, exist_ok=True)
```

### 实现流水线组件

现在，让我们逐个实例化并连接 TFX 组件。

1. **数据引入 (`CsvExampleGen`)**：该组件从外部源读取数据。这里，我们使用 `CsvExampleGen` 来读取我们的 CSV 文件。它将数据转换为 `TFExample` 格式，这是 TensorFlow 训练的标准格式。

   ```python
   # 在 taxi_pipeline.py 中，继续...

   # CsvExampleGen 的输入规范
   output_config = example_gen_pb2.Output(
       split_config=example_gen_pb2.SplitConfig(splits=[
           example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
           example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
       ])
   )

   example_gen = CsvExampleGen(
       input_base=_data_root,
       output_config=output_config
   )
   ```

   我们指定输入位置 (`input_base`) 并配置训练和评估数据的输出拆分 (`output_config`)。`CsvExampleGen` 生成一个 `examples` 产物。
2. **数据验证 (`StatisticsGen`, `SchemaGen`, `ExampleValidator`)**：这些组件分析并验证引入的数据。

   - `StatisticsGen` 计算数据的统计信息。
   - `SchemaGen` 根据统计信息推断数据模式。
   - `ExampleValidator` 通过比较统计信息与模式来查找异常。

   ```python
   # 在 taxi_pipeline.py 中，继续...

   statistics_gen = StatisticsGen(
       examples=example_gen.outputs['examples']
   )

   schema_gen = SchemaGen(
       statistics=statistics_gen.outputs['statistics'],
       infer_feature_shape=True
   )

   example_validator = ExampleValidator(
       statistics=statistics_gen.outputs['statistics'],
       schema=schema_gen.outputs['schema']
   )
   ```

   请注意一个组件的输出产物（例如 `example_gen.outputs['examples']`）如何成为下一个组件的输入。
3. **特征工程 (`Transform`)**：该组件使用 Apache Beam 进行特征工程。它需要一个包含 `preprocessing_fn` 的独立 Python 文件 (`_module_file`)。此函数定义了在训练和提供服务期间要一致应用的转换（例如，缩放、独热编码）。

   ```python
   # 在 taxi_utils.py 中（独立文件）
   import tensorflow as tf
   import tensorflow_transform as tft

   # 定义特征和标签
   _NUMERIC_FEATURES = ['trip_miles', 'trip_seconds']
   _CATEGORICAL_FEATURES = ['pickup_community_area', 'dropoff_community_area']
   _LABEL_KEY = 'tips'

   def _transformed_name(key):
       return key + '_xf'

   def preprocessing_fn(inputs):
       """tf.transform 用于预处理的回调函数"""
       outputs = {}
       # 缩放数值特征
       for key in _NUMERIC_FEATURES:
           outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

       # 生成词汇表并将分类特征映射到整数
       for key in _CATEGORICAL_FEATURES:
           outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
               inputs[key], top_k=100 # 示例：使用前 100 个区域
           )

       # 保持标签不变（假设它对于回归/分类是数值型的）
       # 或者如果需要，应用转换（例如，对于分类使用 tft.bucketize）
       outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

       return outputs
   ```

   ```python
   # 在 taxi_pipeline.py 中，继续...

   transform = Transform(
       examples=example_gen.outputs['examples'],
       schema=schema_gen.outputs['schema'],
       module_file=_module_file # 指向 taxi_utils.py
   )
   ```

   `Transform` 消耗原始样本和模式，应用 `preprocessing_fn`，并生成 `transformed_examples` 和一个 `transform_graph` 产物，用于在提供服务时保持一致应用。
4. **模型训练 (`Trainer`)**：`Trainer` 组件训练一个 TensorFlow 模型。类似于 `Transform`，它通常使用一个模块文件 (`_module_file`)，其中包含定义模型架构、优化器、损失和训练逻辑的 `run_fn` 或 `trainer_fn`。

   ```python
   # 在 taxi_utils.py 中（独立文件），添加训练器函数
   import tensorflow as tf
   from tfx.components.trainer.fn_args_utils import FnArgs

   def _build_keras_model(tf_transform_output):
       """创建用于训练的 Keras 模型。"""
       feature_spec = tf_transform_output.transformed_feature_spec()
       # 从 feature_spec 中移除标签，用于输入层
       feature_spec.pop(_transformed_name(_LABEL_KEY))

       inputs = {
           key: tf.keras.layers.Input(shape=spec.shape, name=key, dtype=spec.dtype)
           for key, spec in feature_spec.items()
       }

       # 简单示例：连接数值特征和分类特征的嵌入层
       numeric_inputs = [_transformed_name(key) for key in _NUMERIC_FEATURES]
       categorical_inputs = [_transformed_name(key) for key in _CATEGORICAL_FEATURES]

       # 为分类特征创建嵌入（根据需要调整嵌入维度）
       embedded_cats = []
       for key in categorical_inputs:
           vocab_size = tf_transform_output.vocabulary_size_by_name(key.replace('_xf',''))
           embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=8)(inputs[key])
           embedded_cats.append(tf.keras.layers.Flatten()(embedding))

       # 连接所有已处理的特征
       features = tf.keras.layers.concatenate(
           [inputs[key] for key in numeric_inputs] + embedded_cats
       )

       # 简单 DNN
       x = tf.keras.layers.Dense(64, activation='relu')(features)
       x = tf.keras.layers.Dense(32, activation='relu')(x)
       output = tf.keras.layers.Dense(1)(x) # 回归输出（预测小费）

       model = tf.keras.Model(inputs=inputs, outputs=output)
       return model

   def run_fn(fn_args: FnArgs):
       """根据给定参数训练模型。"""
       tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

       train_dataset = # 从 fn_args.train_files 加载训练数据集的逻辑
       eval_dataset = # 从 fn_args.eval_files 加载评估数据集的逻辑

       model = _build_keras_model(tf_transform_output)

       model.compile(
           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
           loss='mean_squared_error', # 假设是回归问题
           metrics=[tf.keras.metrics.RootMeanSquaredError()]
       )

       model.fit(
           train_dataset,
           steps_per_epoch=fn_args.train_steps,
           validation_data=eval_dataset,
           validation_steps=fn_args.eval_steps
       )

       # 以 SavedModel 格式保存模型
       model.save(fn_args.serving_model_dir, save_format='tf')

   # 数据集加载逻辑的占位符（替换为实际实现）
   # 这通常涉及 tf.data.TFRecordDataset 并应用转换图
   def _input_fn(file_pattern, tf_transform_output, batch_size=64):
       # 示例占位符 - 需要完整实现
       # dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
       # dataset = dataset.map(lambda x: tf_transform_output.transform_raw_features(tf.io.parse_example(x,...)))
       # dataset = dataset.batch(batch_size).repeat()
       # return dataset
       return None # 替换为实际的数据集加载
   ```

   *注意：`run_fn`，特别是数据集加载 (`_input_fn`)，需要精心实现，涉及 `tf.data` 和应用 `transform_graph`。上面的代码提供了结构；完整的实现取决于具体的数据模式和 TF 版本。*

   ```python
   # 在 taxi_pipeline.py 中，继续...

   trainer = Trainer(
       module_file=_module_file, # 指向包含 run_fn 的 taxi_utils.py
       transformed_examples=transform.outputs['transformed_examples'],
       transform_graph=transform.outputs['transform_graph'],
       schema=schema_gen.outputs['schema'],
       train_args=trainer_pb2.TrainArgs(num_steps=1000), # 示例步数
       eval_args=trainer_pb2.EvalArgs(num_steps=500)   # 示例步数
   )
   ```

   `Trainer` 使用转换后的样本、转换图（以确保一致性）和模式来训练 `run_fn` 中定义的模型。它输出一个训练好的 `model` 产物。
5. **模型评估 (`Evaluator`)**：该组件对训练好的模型在评估数据集上的表现进行详细分析。它使用 TensorFlow 模型分析 (TFMA)。

   ```python
   # 在 taxi_pipeline.py 中，继续...
   from tfx.proto import evaluator_pb2
   import tensorflow_model_analysis as tfma

   eval_config = tfma.EvalConfig(
       model_specs=[tfma.ModelSpec(label_key='tips_xf')], # 使用转换后的标签名称
       slicing_specs=[tfma.SlicingSpec()], # 在整体数据集上评估
       metrics_specs=[
           tfma.MetricsSpec(metrics=[
               tfma.MetricConfig(class_name='ExampleCount'),
               tfma.MetricConfig(class_name='RootMeanSquaredError',
                                 threshold=tfma.MetricThreshold(
                                     value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 15.0}), # 示例阈值
                                     change_threshold=tfma.GenericChangeThreshold(
                                         direction=tfma.MetricDirection.LOWER_IS_BETTER,
                                         absolute={'value': -1e-10}))) # 要求相对于基线有改进
           ])
       ]
   )

   # 解析器，用于查找最新“认可”模型进行比较
   model_resolver = resolver.Resolver(
       strategy_class=LatestArtifactsResolver,
       model=Channel(type=Model),
       model_blessing=Channel(type=ModelBlessing)
   ).with_id('latest_blessed_model_resolver')

   evaluator = Evaluator(
       examples=example_gen.outputs['examples'], # 使用原始样本进行切片
       model=trainer.outputs['model'],
       baseline_model=model_resolver.outputs['model'], # 与之前的模型比较
       eval_config=eval_config,
       example_splits=['eval'] # 在 'eval' 拆分上评估
   )
   ```

   我们定义一个 `EvalConfig`，指定度量指标（如 RMSE）和阈值。`Evaluator` 将当前模型与基线（通常是使用 `Resolver` 找到的先前认可的模型）进行比较，并输出 `evaluation` 结果和一个 `blessing` 产物，指示模型是否通过了阈值。
6. **模型部署 (`Pusher`)**：根据评估结果，`Pusher` 组件有条件地将经过验证的模型部署到指定的提供服务位置。

   ```python
   # 在 taxi_pipeline.py 中，继续...

   pusher = Pusher(
       model=trainer.outputs['model'],
       model_blessing=evaluator.outputs['blessing'], # 仅在“认可”时推送
       push_destination=pusher_pb2.PushDestination(
           filesystem=pusher_pb2.PushDestination.Filesystem(
               base_directory=_serving_model_dir
           )
       )
   )
   ```

   `Pusher` 检查来自 `Evaluator` 的 `model_blessing` 产物。如果模型被“认可”，它会将模型产物 (SavedModel) 复制到 `push_destination`。在这里，我们推送到本地文件系统目录。

### 定义和运行流水线

定义好所有组件后，我们将它们组装成一个 TFX `Pipeline` 对象，并使用编排器来运行它。对于本地执行，`LocalDagRunner` 是合适的。

```python
# 在 taxi_pipeline.py 中，继续...
from tfx.orchestration import pipeline

# 定义流水线
components = [
    example_gen,
    statistics_gen,
    schema_gen,
    example_validator,
    transform,
    trainer,
    model_resolver, # 确保解析器在评估器之前运行
    evaluator,
    pusher,
]

pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True, # 对未更改的组件使用缓存
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
        os.path.join(_pipeline_root, 'metadata.sqlite')) # 在本地存储元数据
)

# 在本地运行流水线
LocalDagRunner().run(pipeline)
```

> 所描述的简单 TFX 流水线的一个典型工作流程图。箭头表示组件之间产物的流动。

要执行此流水线，只需运行 Python 脚本：

```bash
python taxi_pipeline.py
```

TFX 使用 `LocalDagRunner` 将根据定义的依赖关系按正确顺序执行每个组件。它将在 `_pipeline_root` 目录结构中生成产物（数据拆分、统计信息、模式、转换后的数据、模型检查点、评估结果以及最终推送的模型）。`metadata.sqlite` 文件跟踪所有执行、组件和产物，提供血缘信息并启用缓存。

### 检查结果

流水线完成后，查看 `pipeline_output` 目录。您将找到每个组件执行的子目录，其中包含其各自的输出产物。例如：

- `CsvExampleGen/examples/...`：包含 TFRecord 格式的引入数据。
- `StatisticsGen/statistics/...`：包含数据统计信息的可视化内容（例如，使用 Facets）。
- `SchemaGen/schema/...`：包含推断的模式 protobuf 文件。
- `Transform/transform_graph/...`：包含用于预处理的 TensorFlow 图。
- `Trainer/model/...`：包含 SavedModel 格式的训练模型。
- `Evaluator/evaluation/...`：包含可在浏览器中查看的 TFMA 结果。
- `Pusher/pushed_model/...`：如果模型被“认可”，则包含复制用于提供服务的最终模型。

检查这些产物有助于理解每个组件的作用并验证流水线的执行。

"这个动手示例为构建 TFX 流水线提供了一个具体的起点。尽管简单，但它说明了组件定义、产物流动和本地编排的核心原则。流水线通常涉及更复杂的数据、自定义组件、不同的编排器（如 Kubeflow Pipelines 或 Apache Airflow），以及更精密的模型架构和评估策略，它们都在这里建立的基础上进行构建。"

## 参考资料

- [TensorFlow Extended (TFX) Documentation](https://www.tensorflow.org/tfx) — Google (2024)
  提供了 TFX 组件和构建生产级机器学习管道的全面指南。
- [TensorFlow Transform (TFT) Guide](https://www.tensorflow.org/tfx/transform) — Google (2024)
  解释了如何使用 TensorFlow Transform 进行数据预处理，这是 TFX Transform 组件的核心部分。
- [TensorFlow Model Analysis (TFMA) Guide](https://www.tensorflow.org/tfx/model_analysis) — Google (2024)
  详细介绍了使用 TFMA 评估和理解机器学习模型的过程，对 TFX Evaluator 组件至关重要。
- [TFX: A TensorFlow-Based Production-Scale Machine Learning Platform](https://dl.acm.org/doi/10.1145/3394486.3403310) — Yuan, Vincent and Chen, Zhitao and Chen, Guanglei and Lu, Jianbo and Xie, Liang and Shan, Hongli and Xiao, Hao and Sun, Xiaoning and Li, Yiran and Wang, Yi and Shen, Rui and Luo, Junwei and Li, Yang (2020)
  Journal: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Pages: 3672–3681; DOI: [10.1145/3394486.3403310](https://doi.org/10.1145/3394486.3403310)
  一篇介绍 TFX 框架、其架构和在 Google 的实际应用的学术论文。
