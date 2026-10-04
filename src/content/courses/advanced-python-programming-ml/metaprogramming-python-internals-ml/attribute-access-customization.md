---
course: "advanced-python-programming-ml"
chapter: "metaprogramming-python-internals-ml"
lesson: "attribute-access-customization"
sourceId: 2319
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-3-metaprogramming-python-internals-ml/attribute-access-customization"
title: "属性访问自定义 (`__getattr__`, `__getattribute__`)"
description: "在机器学习对象中，实现 `__getattr__` 和 `__getattribute__` 以进行自定义属性处理和动态代理。"
order: 6
plots: []
sourceHash: "b5846da013ad9d8ccf3ae2ba9ff1a9e6f4a60f59b2c4405fcb5f0532ee3b4e8f"
sourceCorrections: []
---

Python 提供了定制属性访问的专门方法。特殊方法 `__getattr__` 和 `__getattribute__` 为属性访问机制本身提供了底层的接口。不同于描述符等机制（它们通常提供对特定属性的精细控制），这些方法允许您定义实例上*任何*属性查找的处理方式。这一能力是一个功能强大但可能复杂的工具，用于构建动态且响应迅速的机器学习 (machine learning)组件。

首先，理解 Python 通常如何查找属性是很重要的。当您访问 `obj.x` 时，Python 通常会检查：

1. `x` 是否是 `obj` 类或其父类上的数据描述符（例如属性）。
2. `x` 是否存在于 `obj.__dict__` 中（实例自身的属性）。
3. `x` 是否是 `obj` 类或其父类中找到的非数据描述符或其他属性（遵循方法解析顺序，即 MRO）。

如果此标准查找失败，Python 会进行最后一次尝试，如果定义了 `__getattr__` 方法，则调用它。如果在*此最终步骤之前*的任何时候查找成功，则*不会*调用 `__getattr__`。

### 使用 `__getattr__` 处理缺失属性

`__getattr__(self, name)` 方法仅在属性查找通过常规途径失败时被调用。其目的是提供一个备用机制，允许您在实例或其类继承结构中未直接找到属性时，动态计算或获取该属性。

**语法：**

```python
def __getattr__(self, name):
    # 'name' 是正在访问的属性的字符串名称
    # 计算或获取值的逻辑
    # 必须返回该值或抛出 AttributeError
    if name == 'some_dynamic_attribute':
        # 计算或获取值
        value = ...
        return value
    # 重要提示：对于未处理的名称，请抛出 AttributeError
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

**机器学习 (machine learning)中的使用场景：**

1. **动态特征生成：** 想象一个数据对象，您希望在不预先计算所有特征的情况下，即时访问特征的转换版本。`__getattr__` 可以拦截对特定转换的请求。

   ```python
   import math
   import pandas as pd
   import numpy as np

   class DynamicFeatures:
       def __init__(self, data_frame):
           # 使用 object.__setattr__ 以避免在定义了我们自己的 __setattr__ 时触发它
           object.__setattr__(self, '_data', data_frame.copy())

       def __getattr__(self, name):
           if name.startswith('log_'):
               original_feature = name[4:] # 移除 'log_' 前缀
               if original_feature in self._data.columns:
                   print(f"Dynamically computing log of {original_feature}")
                   # 动态计算并返回对数转换
                   # 确保对数值为正，并根据需要处理错误
                   series = self._data[original_feature]
                   # 使用 numpy 的对数函数以获得可能更好的性能和处理
                   log_transformed = np.log(series.astype(float).clip(lower=1e-9)) # 裁剪以避免 log(0)
                   return log_transformed
               else:
                   raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (original feature '{original_feature}' not found)")
           elif name.startswith('squared_'):
                original_feature = name[8:]
                if original_feature in self._data.columns:
                    print(f"Dynamically computing square of {original_feature}")
                    return self._data[original_feature] ** 2
                else:
                   raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (original feature '{original_feature}' not found)")

           # 重要提示：如果未处理，请抛出 AttributeError
           raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

       # 为了允许访问内部 '_data' 而不会无限触发 __getattr__，
       # 我们在 __init__ 中使用了 object.__setattr__。在 __getattr__ 内部访问 self._data
       # 是安全的，因为 '_data' 已经存在。

   # 使用示例
   df = pd.DataFrame({'feature_a': [1, 10, 100], 'feature_b': [-2, 20, 200]})
   dynamic_df = DynamicFeatures(df)

   print("访问 log_feature_a:")
   print(dynamic_df.log_feature_a)

   print("\n访问 squared_feature_b:")
   print(dynamic_df.squared_feature_b)

   try:
       print("\n访问 non_existent_feature:")
       print(dynamic_df.non_existent_feature)
   except AttributeError as e:
       print(e)
   ```

   ```text
   访问 log_feature_a:
   动态计算 feature_a 的对数
   0    0.000000
   1    2.302585
   2    4.605170
   Name: feature_a, dtype: float64

   访问 squared_feature_b:
   动态计算 feature_b 的平方
   0        4
   1      400
   2    40000
   Name: feature_b, dtype: int64

   访问 non_existent_feature:
   'DynamicFeatures' object has no attribute 'non_existent_feature'
   ```
2. **资源按需加载：** 在机器学习工作流程中，您可能会处理加载到内存中开销很大的大型模型或数据集。`__getattr__` 允许您延迟加载，直到实际需要该资源时再加载。

   ```python
   import time
   # 假设存在 joblib 以提供更真实的示例桩
   # import joblib

   class LazyModelLoader:
       def __init__(self, model_path_dict):
           # 使用 object.__setattr__ 安全地存储路径或重命名以避免冲突
           object.__setattr__(self, '_model_paths', model_path_dict)
           object.__setattr__(self, '_loaded_models', {}) # 已加载模型的缓存

       def __getattr__(self, name):
           # 检查是否是我们可以加载的已知模型名称
           if name in self._model_paths:
               # 检查是否已加载（在我们的缓存中）
               if name not in self._loaded_models:
                   print(f"正在按需加载模型 '{name}'，路径为 {self._model_paths[name]}...")
                   # 实际的模型加载逻辑将在此处
                   # 例如：self._loaded_models[name] = joblib.load(self._model_paths[name])
                   time.sleep(0.5) # 模拟加载时间
                   self._loaded_models[name] = f"已加载模型：{name.upper()}" # 占位符
               # 从缓存中返回已加载的模型
               return self._loaded_models[name]

           # 如果名称不是可加载模型的路径，则抛出 AttributeError
           raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

       # 如果需要，允许直接访问内部状态，由默认的 __getattribute__ 处理
       # 或者如果 __getattribute__ 被覆盖，也可以在此处显式处理。

   # 使用示例
   loader = LazyModelLoader({
       'classifier': '/path/to/classifier.pkl',
       'regressor': '/path/to/regressor.pkl'
   })

   # 模型尚未加载
   print("访问分类器...")
   model_c = loader.classifier # 触发 __getattr__，加载模型
   print(model_c)

   print("\n再次访问分类器...")
   model_c_again = loader.classifier # 访问缓存版本，不显示加载消息
   print(model_c_again)

   print("\n访问回归器...")
   model_r = loader.regressor # 触发 __getattr__，加载模型
   print(model_r)
   ```

   ```text
   访问分类器...
   正在按需加载模型 'classifier'，路径为 /path/to/classifier.pkl...
   已加载模型：CLASSIFIER

   再次访问分类器...
   已加载模型：CLASSIFIER

   访问回归器...
   正在按需加载模型 'regressor'，路径为 /path/to/regressor.pkl...
   已加载模型：REGRESSOR
   ```

实现 `__getattr__` 的一个要点是避免造成无限递归。如果您的 `__getattr__` 实现尝试访问 `self` 上不存在的属性（使用标准 `self.attribute_name` 语法），它将再次触发 `__getattr__`，导致循环。请始终确保您的 `__getattr__` 要么直接计算值，要么小心地访问已知存在的属性（如示例中的 `self._data` 或 `self._loaded_models`，它们在初始化时设置），要么抛出 `AttributeError`。

### 使用 `__getattribute__` 拦截所有访问

与 `__getattr__` 不同，`__getattribute__(self, name)` 方法侵入性强得多。对实例的*每次*属性查找都会调用它，无论属性是否存在。这提供了一个强大的机制，用于拦截并可能修改任何属性访问。

**语法：**

```python
def __getattribute__(self, name):
    # 'name' 是正在访问的属性的字符串名称
    # !!! 必须非常小心地实现以避免无限递归 !!!

    # 安全地访问原始属性值
    # 选项 1：使用 super()（在协作继承中更推荐）
    # value = super().__getattribute__(name)
    # 选项 2：直接使用 object 的 __getattribute__
    # value = object.__getattribute__(self, name)

    print(f"正在拦截对：{name} 的访问")
    try:
        value = object.__getattribute__(self, name) # 使用 object 的方法来防止递归
        # 返回前执行操作（日志记录、修改等）
        return value
    except AttributeError:
        # 处理属性确实不存在的情况
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

因为 `__getattribute__` 拦截*所有*访问，包括对 `__dict__` 或查找本身所需的其他方法的访问，所以如果不小心，极易导致无限递归。从 `__getattribute__` 内部安全获取实际属性值的最常用方法是使用 `super().__getattribute__(name)` 或 `object.__getattribute__(self, name)`。**切勿在 `__getattribute__` 内部使用 `self.any_attribute` 来获取 `any_attribute`**，因为这会再次递归调用 `__getattribute__`。

**机器学习 (machine learning)中的使用场景：**

1. **访问日志记录与监控：** 您可以跟踪对敏感配置参数 (parameter)或模型权重 (weight)的访问，这可能用于审计或调试复杂的交互。

   ```python
   import datetime

   class MonitoredConfig:
       def __init__(self, params):
           # 在初始化期间使用 object.__setattr__ 来绕过我们自己的 __getattribute__
           object.__setattr__(self, '_params', params)
           object.__setattr__(self, '_access_log', [])

       def __getattribute__(self, name):
           # 需要绕过对我们内部属性的拦截！
           if name in ('_params', '_access_log', 'get_log'): # 也允许方法访问
                return object.__getattribute__(self, name)

           timestamp = datetime.datetime.now().isoformat()
           print(f"日志：在 {timestamp} 访问 '{name}'")

           # 安全地获取日志并添加
           log = object.__getattribute__(self, '_access_log')
           log.append((name, timestamp))

           # 安全地从内部字典获取实际值
           params = object.__getattribute__(self, '_params')
           if name in params:
               return params[name]
           else:
               # 如果不是已知参数，尝试默认属性查找（例如，对于方法）
               # 我们已经在上面为 'get_log' 处理了这种情况。
               # 对于其他非参数、非方法的名称，抛出错误。
               raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

       def get_log(self):
           # 直接访问日志，使用内部名称绕过
           return self._access_log

   config = MonitoredConfig({'learning_rate': 0.01, 'optimizer': 'Adam', 'epochs': 100})

   lr = config.learning_rate # 触发 __getattribute__
   opt = config.optimizer    # 触发 __getattribute__

   print(f"配置的学习率：{lr}，优化器：{opt}")
   print("\n访问日志：")
   for entry in config.get_log(): # 访问 get_log 绕过日志记录
       print(entry)
   ```

   ```text
   日志：在 2023-10-27T10:30:00.123456 访问 'learning_rate'
   日志：在 2023-10-27T10:30:00.123500 访问 'optimizer'
   配置的学习率：0.01，优化器：Adam

   访问日志：
   ('learning_rate', '2023-10-27T10:30:00.123456')
   ('optimizer', '2023-10-27T10:30:00.123500')
   ```
2. **创建透明代理：** `__getattribute__` 对于创建代理对象很关键，这些代理对象将请求转发到另一个对象，可能会添加验证、缓存或单位转换等行为，而用户不知道他们正在与代理交互。这有助于封装复杂的模型对象或数据源。

   ```python
   class DataValidationProxy:
       def __init__(self, target_object):
           object.__setattr__(self, '_target', target_object)

       def __getattribute__(self, name):
           # 绕过内部 '_target'
           if name == '_target':
               return object.__getattribute__(self, name)

           print(f"代理：正在拦截对 '{name}' 的访问")
           # 安全地获取目标对象
           target = object.__getattribute__(self, '_target')

           # 从目标获取属性值
           value = getattr(target, name) # 在目标上使用标准的 getattr

           # 示例验证：检查数值数据是否在范围内
           if name == 'sensor_reading' and isinstance(value, (int, float)):
               if not (0 <= value <= 100):
                   print(f"代理：警告 - sensor_reading {value} 超出预期范围 [0, 100]")
           return value

       def __setattr__(self, name, value):
            # 也拦截设置以进行验证
            if name == '_target':
                object.__setattr__(self, name, value)
                return

            print(f"代理：正在拦截将 '{name}' 设置为 {value}")
            target = object.__getattribute__(self, '_target')

            # 示例验证：确保标签在已知集合中
            if name == 'predicted_label' and value not in ['cat', 'dog', 'other']:
                raise ValueError(f"无效标签 '{value}'。必须是 'cat'、'dog' 或 'other'。")

            setattr(target, name, value) # 设置到实际目标上

   class RawModelOutput:
       def __init__(self):
           self.sensor_reading = 105.5 # 超出范围的示例
           self.predicted_label = None
           self.confidence = 0.95

   raw_output = RawModelOutput()
   validated_output = DataValidationProxy(raw_output)

   # 通过代理访问触发验证检查
   reading = validated_output.sensor_reading
   print(f"获取的读数：{reading}")

   # 通过代理设置触发验证检查
   try:
       validated_output.predicted_label = 'cat' # 有效
       print(f"标签设置为：{validated_output.predicted_label}")
       validated_output.predicted_label = 'bird' # 无效
   except ValueError as e:
       print(e)

   # 检查原始对象状态
   print(f"原始对象标签：{raw_output.predicted_label}")
   ```

   ```text
   代理：正在拦截对 'sensor_reading' 的访问
   代理：警告 - sensor_reading 105.5 超出预期范围 [0, 100]
   获取的读数：105.5
   代理：正在拦截将 'predicted_label' 设置为 cat
   代理：正在拦截对 'predicted_label' 的访问
   标签设置为：cat
   代理：正在拦截将 'predicted_label' 设置为 bird
   无效标签 'bird'。必须是 'cat'、'dog' 或 'other'。
   原始对象标签：cat
   ```

### `__getattr__` 和 `__getattribute__` 的选择

选择完全取决于您的目标：

- 在需要以下情况时使用 **`__getattr__`**：

  - 为缺失属性提供默认值。
  - 仅当属性被请求*且未通过正常方式找到*时，才动态计算它们。
  - 实现资源的按需加载。
  - 它通常更安全，更不容易发生意外的无限递归。
- 在需要以下情况时使用 **`__getattribute__`**（务必极端小心）：

  - 拦截*每次*属性访问（读取），用于全面日志记录、安全检查或在返回值之前进行透明修改/验证等目的。
  - 实现代理模式，其中代理必须处理指向被代理对象的*所有*交互。
  - 请记住，在其实现中必须使用 `super().__getattribute__(name)` 或 `object.__getattribute__(self, name)` 来安全地获取属性值。

### 控制属性赋值和删除

与 `__getattr__` 和 `__getattribute__`（主要处理属性*读取*）相辅相成的是 `__setattr__(self, name, value)` 和 `__delattr__(self, name)`。

- `__setattr__(self, name, value)`：每当尝试属性赋值时调用（例如，`obj.x = 10`）。与 `__getattribute__` 类似，它拦截*所有*赋值。您必须在其实现中使用 `object.__setattr__(self, name, value)` 或 `super().__setattr__(name, value)` 来实际存储值，防止无限递归。这通常用于输入验证（如代理示例所示）、类型检查，或在属性更改时触发副作用（例如，将配置对象标记 (token)为“脏”）。
- `__delattr__(self, name)`：当尝试删除属性时调用（例如，`del obj.x`）。类似地，需要小心实现，使用 `object.__delattr__(self, name)` 或 `super().__delattr__(name)` 以避免递归。它允许您拦截删除，例如阻止删除关键属性或执行清理操作。

这四个方法（`__getattr__`、`__getattribute__`、`__setattr__`、`__delattr__`）构成了 Python 可定制属性访问控制的核心。

### 示例：验证和管理模型超参数 (parameter) (hyperparameter)

让我们完善 `HyperparameterConfig` 的设想，结合 `__setattr__` 进行验证，结合 `__getattr__` 潜在地访问派生值或默认值，同时使用 `__getattribute__` 清晰地管理访问。

```python
import math

class HyperparameterConfig:
    # 定义允许的超参数及其验证规则
    _allowed_params = {
        'learning_rate': lambda x: isinstance(x, (float, int)) and 0 < x < 1,
        'epochs': lambda x: isinstance(x, int) and x > 0,
        'batch_size': lambda x: isinstance(x, int) and x > 0,
        'optimizer': lambda x: x in ['Adam', 'SGD', 'RMSprop']
    }
    # 定义默认值
    _defaults = {
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32,
        'optimizer': 'Adam'
    }

    def __init__(self, **kwargs):
        # 使用 object.__setattr__ 进行内部状态初始化
        object.__setattr__(self, '_params', {})
        object.__setattr__(self, '_dataset_size', None) # 另一个受管理属性的示例

        # 应用默认值
        for key, value in self._defaults.items():
            self._params[key] = value

        # 使用提供的 kwargs 覆盖默认值，并使用我们的验证逻辑
        for key, value in kwargs.items():
            self.__setattr__(key, value) # 调用我们自定义的 __setattr__

    def __setattr__(self, name, value):
        # 允许直接设置内部属性
        if name in ('_params', '_dataset_size'):
            object.__setattr__(self, name, value)
            return

        if name in self._allowed_params:
            validator = self._allowed_params[name]
            if not validator(value):
                raise ValueError(f"超参数 '{name}' 的值 '{value}' 无效")
            # 验证通过，存储到内部字典
            self._params[name] = value
        else:
            # 处理尝试设置未知超参数的情况
            raise AttributeError(f"'{name}' 不是一个可识别的超参数。允许的参数有：{list(self._allowed_params.keys())}")

    def __getattr__(self, name):
        # 示例：如果 dataset_size 已知，计算总迭代次数
        if name == 'total_iterations':
            if self._dataset_size is not None and self._params.get('batch_size', 0) > 0:
                iters_per_epoch = math.ceil(self._dataset_size / self._params['batch_size'])
                return self._params.get('epochs', 0) * iters_per_epoch
            else:
                raise AttributeError("无法计算 'total_iterations'。请先设置 'dataset_size'。")

        # 如果属性没有被 __getattribute__ 找到，并且没有在这里动态生成，那么它确实不存在
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getattribute__(self, name):
        # 优先处理内部属性
        if name in ('_params', '_dataset_size', '_allowed_params', '_defaults'):
            return object.__getattribute__(self, name)

        # 检查它是否是存储在 _params 中的已知超参数
        params = object.__getattribute__(self, '_params')
        if name in params:
            return params[name]

        # 尝试标准属性查找（用于 __init__、__setattr__ 等方法）
        # 或者通过 __getattr__ 动态生成的属性
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # 如果标准查找失败，则显式调用 __getattr__ 作为最终的备用
            # 这种结构确保 __getattr__ 仅在需要时才被调用。
            return self.__getattr__(name)

# 使用示例
config = HyperparameterConfig(learning_rate=0.05, optimizer='SGD')

print(f"学习率：{config.learning_rate}，优化器：{config.optimizer}，默认迭代次数：{config.epochs}")

config.epochs = 50 # 通过 __setattr__ 进行有效设置
print(f"设置迭代次数：{config.epochs}")

try:
    config.batch_size = -10 # 通过 __setattr__ 进行无效设置
except ValueError as e:
    print(e)

try:
    config.dropout_rate = 0.5 # 未知超参数
except AttributeError as e:
    print(e)

# 访问动态属性
config._dataset_size = 50000 # 设置内部属性（绕过 __setattr__）
print(f"总迭代次数：{config.total_iterations}") # 通过 __getattribute__ 回调调用 __getattr__
```

```text
学习率：0.05，优化器：SGD，默认迭代次数：10
设置迭代次数：50
超参数 'batch_size' 的值 '-10' 无效
'dropout_rate' 不是一个可识别的超参数。允许的参数有：['learning_rate', 'epochs', 'batch_size', 'optimizer']
总迭代次数：78150
```

通过 `__getattr__`、`__getattribute__`、`__setattr__` 和 `__delattr__` 自定义属性访问，提供了对 Python 对象行为的深度控制。尽管功能强大，特别是对于框架开发、动态配置管理、验证层以及机器学习 (machine learning)场景中的资源按需处理，但这些方法需要仔细周密的实现。它们要求清晰地理解标准属性查找过程，并对内部状态访问进行细致处理，以避免无限递归，尤其是在使用高度拦截性的 `__getattribute__` 和 `__setattr__` 时。掌握这些技术有助于创建灵活、精密的组件，这些组件对于高级机器学习系统至关重要。

## 参考资料

- [The Python Language Reference - 3. Data model](https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  提供了__getattr__、__getattribute__、__setattr__和__delattr__规范和行为的权威官方文档。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: Chapter 21: Attribute Management
  对Python的对象模型和属性访问的特殊方法进行了全面的探讨，解释了底层机制。
- [Python Cookbook, 3rd Edition](https://www.oreilly.com/library/view/python-cookbook-3rd/9781449357337/) — David Beazley, Brian K. Jones (2013)
  Publisher: O'Reilly Media; Pages: Recipes 8.10: Customizing Attribute Access; 9.21: Bypassing __getattribute__ for Managed Attributes
  为自定义属性访问提供了实用而优雅的解决方案，包括委托、日志记录和__getattribute__的谨慎处理场景。
