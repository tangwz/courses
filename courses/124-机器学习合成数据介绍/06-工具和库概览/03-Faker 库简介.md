# Faker 库简介

来源：[原文](https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-6-tools-libraries-overview/intro-faker-library)

[返回章节目录](README.md) · [返回课程目录](../README.md)

"虽然 NumPy 等库有助于根据分布生成数值数据，Pandas 帮助组织数据，但您通常需要看起来像真实信息的合成数据。考虑一下姓名、地址、电子邮件地址、职位，甚至逼真的文本段落。生成真正的随机字符串或数字无法体现这类数据的*格式*和*合理性*。Faker 库在此就能派上用场。"

Faker 是一个 Python 库，专门用于生成虚假但看起来真实的数据。它不生成随机数字或简单字符串，而是提供“提供者”（providers），这些提供者生成真实数据集中常见的各种数据类型。这在您需要填充数据库进行测试、为演示创建模拟数据或在合成表格数据中生成占位符列时非常实用。

### Faker 使用入门

首先，您需要安装此库。与许多 Python 包一样，您可以使用 Python 包安装器 `pip` 来安装 Faker。打开您的终端或命令提示符并输入：

```bash
pip install Faker
```

安装完成后，使用 Faker 非常简单。您首先导入 `Faker` 类并创建其一个实例：

```python
# 导入 Faker 类
from faker import Faker

# 创建一个 Faker 实例
fake = Faker()

# 现在您可以使用 'fake' 对象生成数据
print(f"姓名: {fake.name()}")
print(f"地址: {fake.address()}")
print(f"电子邮件: {fake.email()}")
print(f"文本片段: {fake.text(max_nb_chars=100)}")
print(f"出生日期: {fake.date_of_birth(minimum_age=18, maximum_age=65)}")
```

运行此代码将生成看起来合理且真实的个人信息输出：

```
姓名: Michael Williams
地址: 880 Holmes Rapid Apt. 388
Lake Robert, OK 09191
电子邮件: sarahhernandez@example.com
文本片段: Service reach difficult security cell. Effort future section democratic director.
出生日期: 1978-05-21
```

每次运行它，您都会得到不同的模拟数据。请注意，生成的数据遵循预期的格式（例如，电子邮件地址包含“@”符号和域名，地址包含街道名称、城市、州和邮政编码）。

### 生成本地化数据

Faker 不仅限于基于英语的数据。它支持多种语言环境，让您能够生成适合特定语言和地区的数据。为此，您在创建 `Faker` 实例时传递一个语言环境代码字符串。

例如，要生成法语数据：

```python
# 导入 Faker 类
from faker import Faker

# 创建一个法语（法国）的 Faker 实例
fake_fr = Faker('fr_FR')

# 生成法语数据
print(f"姓名: {fake_fr.name()}")
print(f"地址: {fake_fr.address()}")
print(f"电子邮件: {fake_fr.email()}")
```

这将输出如下数据：

```
姓名: Monique Francois
地址: avenue Leclerc
66030 Robin-sur-Mer
电子邮件: andre64@example.com
```

此功能在创建用于国际应用或为不同区域设计的测试系统的合成数据集时非常宝贵。

### 了解提供者

Faker 是如何知道如何生成所有这些不同类型的数据的呢？它使用一套“提供者”（providers）系统。每个提供者都负责特定类别的数据。当您调用 `fake.name()` 或 `fake.address()` 等方法时，Faker 会使用相应的提供者（例如 `Person` 或 `Address`）来生成数据。

Faker 带有许多内置提供者，涵盖以下类别：

- `person`：姓名、前缀、后缀。
- `address`：街道名称、城市、国家、邮政编码。
- `internet`：电子邮件地址、域名、IP 地址、用户代理。
- `company`：公司名称、标语。
- `datetime`：日期、时间、时区。
- `lorem`：单词、句子、段落（占位符文本）。
- 还有许多其他提供者，用于财务、条形码、颜色等特定方面。

您通常不需要直接与提供者交互；只需在您的 `Faker` 实例上调用相应的方法即可（例如，`fake.company()`、`fake.ipv4()`）。

### 创建多个模拟记录

Faker 经常用于生成多行合成数据，例如创建虚假用户或客户列表。您可以使用循环轻松完成此操作：

```python
# 导入 Faker 类
from faker import Faker
import json # 用于美化输出

# 创建一个 Faker 实例
fake = Faker()

# 生成一个包含 5 条模拟用户记录的列表
fake_users = []
for _ in range(5):
    user = {
        'user_id': fake.uuid4(), # 唯一标识符
        'name': fake.name(),
        'email': fake.email(),
        'company': fake.company(),
        'join_date': fake.date_this_decade() # 过去十年内的一个日期
    }
    fake_users.append(user)

# 打印生成的列表（使用 json 以提高可读性）
print(json.dumps(fake_users, indent=2, default=str)) # 使用 default=str 来处理日期对象
```

此脚本生成一个包含五个字典的列表，每个字典都代表一个具有看起来真实数据的用户：

```json
[
  {
    "user_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "name": "Lisa Johnson",
    "email": "matthewbrown@example.org",
    "company": "Smith PLC",
    "join_date": "2021-08-15"
  },
  {
    "user_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
    "name": "Dr. Kevin Williams Jr.",
    "email": "jenniferharris@example.net",
    "company": "Wilson-Garcia",
    "join_date": "2023-02-28"
  },
  {
    "user_id": "c3d4e5f6-a7b8-9012-3456-7890abcdef01",
    "name": "Brenda Davis",
    "email": "christopher45@example.com",
    "company": "Jones, Miller and Clark",
    "join_date": "2020-11-01"
  },
  {
    "user_id": "d4e5f6a7-b8c9-0123-4567-890abcdef012",
    "name": "Scott Rodriguez",
    "email": "richardsonashley@example.org",
    "company": "Martinez Group",
    "join_date": "2024-01-10"
  },
  {
    "user_id": "e5f6a7b8-c9d0-1234-5678-90abcdef0123",
    "name": "Amanda Martin",
    "email": "karenmoore@example.com",
    "company": "Thomas LLC",
    "join_date": "2022-06-22"
  }
]
```

*(请注意：UUID 和日期在每次运行时都会不同)*

您可以轻松调整此代码以创建 Pandas DataFrame，使 Faker 成为生成带有逼真文本列的合成表格数据的一个实用工具。

### 在合成数据生成中的作用

Faker 提供了一种简单的方法来为特定字段生成*看起来*真实的合成数据。它在以下情况中特别实用：

- **填充文本列：** 为表格数据集生成逼真的姓名、地址、描述等。
- **创建测试数据：** 快速创建大量不同类型的数据以测试软件应用程序或数据库模式。
- **原型构建：** 在没有真实数据可用时，为演示或初始模型开发构建示例数据集。
- **匿名化支持：** 作为隐私保护过程的一部分，用逼真的模拟数据替换数据集中的敏感字段（如姓名或电子邮件）。

请记住，Faker 生成的数据点是*合理*但*独立*的。默认情况下，它不捕捉不同字段之间复杂的统计关系（例如，特定职位与某个年龄段或城市的相关性）。虽然您可以自定义 Faker 的行为，但生成统计上准确、高逼真度的数据集通常需要其他地方讨论的更高级建模技术。然而，对于为某些列添加真实感或创建基本测试数据，Faker 在合成数据工具包中是一个非常有价值且易于使用的工具。

## 参考资料

- [Faker Documentation](https://faker.readthedocs.io/en/master/) — Joke Verbeeck and the Faker Developers (2024)
  Faker Python库的权威资料，详细介绍了安装、配置和使用方法，包括其提供者、本地化功能以及生成逼真模拟数据的实际示例。
- [The Art of Software Testing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1hZJO562a-fsW3JIeNSJU-f-UF5DJwnQzqotpAXT83JXse3ZAta2Xswre7qhaLsPpumg3umVPaNz44260mqFFEjJbAIJY_e4VdMWukOEuXgDFQTJCKGgLTYUV957Q2Pym0Wno3ntcxFxIBenatoiKgNYgW5xntxB55DCNxCp9XwFwFFxN3DOaEx_cqVB23A==) — Glenford J. Myers, Corey Sandler, Tom Badgett, and Todd M. Thomas (2011)
  Publisher: John Wiley & Sons
  一本经典的软件测试教材，讨论了创建有效测试用例和生成相关测试数据的原则和技术，说明了像Faker这样的工具在测试工作流中的实际应用。

---

[上一节](02-%E7%94%A8%E4%BA%8E%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E6%93%8D%E4%BD%9C%E7%9A%84%E5%BA%93%20%28NumPy%2C%20Pandas%29.md) · [下一节](04-%E7%AE%80%E5%8D%95%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%BA%93%20%28Pillow%2C%20Scikit-image%29.md)
