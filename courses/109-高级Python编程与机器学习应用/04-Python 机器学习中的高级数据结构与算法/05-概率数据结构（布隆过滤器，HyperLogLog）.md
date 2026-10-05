# 概率数据结构（布隆过滤器，HyperLogLog）

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-4-advanced-data-structures-algorithms-ml/probabilistic-data-structures)

[返回章节目录](README.md) · [返回课程目录](../README.md)

处理海量数据集是机器学习 (machine learning)中的一个常见挑战，尤其是在内存消耗和处理时间方面。进行精确计算，比如确定集合成员或统计数十亿项中的不同元素数量，计算成本可能非常高。概率数据结构提供了一种有效的替代方案，它们能在内存和时间上大幅减少消耗，并提供具有数学限定误差的近似结果。两种广泛使用的概率结构是：用于近似成员测试的布隆过滤器和用于近似基数估计的HyperLogLog。

### 布隆过滤器：近似集合成员测试

布隆过滤器是一种节省空间的概率数据结构，其作用是判断一个元素是否*可能*是某个集合的成员。它的特点是会产生假阳性（错误地指示元素在集合中，而实际上不在），但绝不会产生假阴性（它总是能正确识别*确实*在集合中的元素）。因此，它们适用于假阳性可以偶尔出现，但不能遗漏实际成员的情况。

**工作原理：**

布隆过滤器由一个大小为 $m$ 的位数组（初始全为零）和 $k$ 个独立的哈希函数组成。

1. **添加元素：** 要向过滤器添加一个元素，该元素将由 $k$ 个哈希函数分别处理。每个哈希函数会在位数组范围 $[0, m-1]$ 内生成一个索引。这些 $k$ 个索引处的位将被设置为1。
2. **查询元素：** 要检查一个元素是否可能在集合中，该元素会通过相同的 $k$ 个哈希函数计算出 $k$ 个索引。如果数组中这些索引处的*所有*位都是1，则该元素被认为是*可能*存在。如果*任何一个*位是0，则该元素明确*不在*集合中（因为如果它在集合中，其对应的位在插入时就会被设置）。

出现假阳性的原因在于，查询索引处的位可能是在插入*其他*元素时被设置为1的。假阳性的概率取决于位数组的大小 ($m$)、哈希函数的数量 ($k$) 以及已插入元素的数量 ($n$)。

**参数 (parameter)选择：**

对于期望的容量 $n$（要插入的元素数量）和目标假阳性概率 $\epsilon$，最佳位数组大小 $m$（位数）和哈希函数数量 $k$ 可通过以下公式估算：


$$
m \approx -\frac{n \ln \epsilon}{(\ln 2)^2}
$$


$$
k \approx \frac{m}{n} \ln 2
$$


选择合适的参数对于平衡空间使用和准确性很重要。

**Python实现：**

虽然可以使用 `bitarray` 等库和哈希函数（例如 `mmh3`）来自己实现布隆过滤器，但在实际生产中，通常会使用 `pybloom_live` 这样经过优化的库。

```python
# 使用 pybloom_live 的示例（安装命令：pip install pybloom-live）
from pybloom_live import BloomFilter
import random
import string

# 配置：容量=10000，期望的假阳性率=0.01
capacity = 10000
error_rate = 0.01
bf = BloomFilter(capacity=capacity, error_rate=error_rate)

# 向过滤器添加元素
elements_in_set = set()
for i in range(capacity // 2): # 添加 5000 个元素
    element = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    bf.add(element)
    elements_in_set.add(element)

print(f"布隆过滤器大小: {bf.num_bits / 8 / 1024:.2f} KB")
print(f"哈希函数数量: {bf.num_hashes}")

# 测试成员资格
test_element_present = list(elements_in_set)[0]
test_element_absent = "this_element_is_not_in_the_set_hopefully"

print(f"'{test_element_present}' 在过滤器中吗? {test_element_present in bf}") # 应该为 True（无假阴性）
print(f"'{test_element_absent}' 在过滤器中吗? {test_element_absent in bf}")    # 应该为 False（假阳性概率低）

# 测试假阳性率（近似值）
false_positives = 0
test_count = 10000
for i in range(test_count):
    element = f"test_non_member_{i}"
    if element in bf and element not in elements_in_set:
        false_positives += 1

print(f"近似假阳性率: {false_positives / test_count:.4f} (目标值为 {error_rate})")
```

**机器学习 (machine learning)应用：**

- **特征哈希：** 通过将高基数类别特征哈希到更小、固定大小的向量 (vector)中，从而降低维度。布隆过滤器可以隐式或显式地用于此过程。
- **重复检测：** 在流式数据管道中识别已处理过的项目（例如文档、用户交互），而无需存储所有项目。
- **推荐系统：** 在推荐项目前，快速检查用户是否已与某个项目有过交互。
- **缓存：** 在查询速度较慢、成本更高的缓存或数据库之前，作为一种预先检查。

### HyperLogLog：基数估算

估算超大型数据集或数据流中的唯一项数量（基数），是另一个常见的问题。精确计数通常需要消耗过多内存。HyperLogLog (HLL) 算法提供了一种高效的方法，它能使用少量固定内存来近似估算基数，而与处理的元素总量无关。

**工作原理（直观解释）：**

HLL 依赖于哈希函数的统计特性。

1. **哈希：** 每个输入元素都会被哈希为一个固定大小的二进制字符串。
2. **观察：** HLL 观察二进制哈希值中前导零的模式。其直观思想是，如果出现一个哈希值以许多零开头（例如 `000...01...`），这是一个不常有的情况，这表示你可能已经遇到了许多不同的元素才能见到这样的值。观察到的最大前导零数量可以粗略估算为 $\log_2(N)$，其中 $N$ 是基数。
3. **分桶/寄存器：** 为了提高准确度，HLL 将哈希空间划分为 $m$ 个桶（即寄存器）。它对每个元素进行哈希，使用哈希值的前几位来选择一个桶，并将*该桶*中记录的最大前导零数量存储在相应的寄存器中。
4. **估算：** 最终的基数估算值是通过对 $m$ 个寄存器中存储的值进行一种广义调和平均，并结合校正因子来降低偏差而得出的。

HLL 的一个突出特点是，所需的内存只取决于寄存器数量 ($m$)，这决定了估算精度，而与实际被估算的基数大小无关。增加 $m$ 会提高估算准确度，但同时也会增加内存使用量。

**精度：**

HLL 的精度通常以相对标准误差（RSE）表示。对于 $m$ 个寄存器（其中 $m$ 通常是2的幂），RSE 近似为：


$$
\text{RSE} \approx \frac{1.04}{\sqrt{m}}
$$


例如，使用 $m = 2^{10} = 1024$ 个寄存器大约需要 1KB 内存，并带来大约 $\frac{1.04}{\sqrt{1024}} \approx 3.25\%$ 的 RSE。将 $m$ 翻倍（内存增加四倍）可以将误差减半。

**Python实现：**

像 `hyperloglog` 或 `datasketch` 这样的库提供了高效的 HLL 实现。

```python
# 使用 hyperloglog 库的示例（安装命令：pip install hyperloglog）
import hyperloglog
import random

# 创建一个 HyperLogLog 计数器
# p 对应于 m=2^p。p=10 表示 m=1024 个寄存器。误差约为 1.04/sqrt(2^p)
hll = hyperloglog.HyperLogLog(0.01) # 目标误差率 0.01（将据此选择 p 值）

# 模拟从数据流中添加元素
actual_distinct_elements = set()
total_elements = 500000
max_distinct_value = 100000 # 真实基数约为 100,000

print(f"使用 {2**hll.p} 个寄存器 (p={hll.p})")

for _ in range(total_elements):
    element = str(random.randint(1, max_distinct_value))
    hll.add(element)
    actual_distinct_elements.add(element)

# 获取基数估算值
estimated_cardinality = len(hll)
actual_cardinality = len(actual_distinct_elements)
error_percentage = abs(estimated_cardinality - actual_cardinality) / actual_cardinality * 100

print(f"实际不同元素数量: {actual_cardinality}")
print(f"估算的不同元素数量: {estimated_cardinality}")
print(f"误差: {error_percentage:.2f}%")
# 检查内存使用情况（通常很小，取决于库的实现）
# 注意：sys.getsizeof 对于复杂对象可能不完全准确
import sys
print(f"近似内存大小: {sys.getsizeof(hll.registers()) / 1024:.2f} KB")
```

**机器学习 (machine learning)应用：**

- **唯一特征值计数：** 在海量数据集中估算类别特征的唯一值数量，而无需存储所有唯一值，有助于进行特征工程决策。
- **词汇表 (vocabulary)大小估算：** 在自然语言处理预处理过程中，近似估算大型文本语料库的词汇表大小。
- **网络监控：** 在网络流量分析中，统计不同的源/目的IP地址或流量。
- **数据库查询优化：** 估算不同查询计划结果的基数。
- **在线广告：** 实时统计广告活动触达的唯一用户数量。

### 总结

布隆过滤器和 HyperLogLog 等概率数据结构，是在内存或时间受限环境下处理大规模数据时必不可少的工具。布隆过滤器提供高效的近似成员测试，且假阳性率可控；HyperLogLog 则使用极少量固定内存提供准确的基数估算。理解它们的工作原理、权衡（空间与准确性）以及应用范围，能帮助你在机器学习 (machine learning)工作流中构建更具扩展性且效率更高的数据处理流程和特征工程环节。尽管它们提供的是近似结果，但其资源效率往往使它们成为解决某些大规模问题的唯一可行方法。

## 参考资料

- [Space/time trade-offs in hash coding with allowable errors](https://dl.acm.org/doi/10.1145/362686.362692) — Burton H. Bloom (1970)
  Journal: Communications of the ACM; Publisher: Association for Computing Machinery (ACM); Volume: 13; Pages: 422-426; DOI: [10.1145/362686.362692](https://doi.org/10.1145/362686.362692)
  介绍布隆过滤器这一用于集合成员检测的概率数据结构的原始论文。
- [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](https://doi.org/10.46298/dmtcs.3545) — Philippe Flajolet, Éric Fusy, Olivier Gandouet, Frédéric Meunier (2007)
  Journal: Discrete Mathematics & Theoretical Computer Science, DMTCS Proceedings vol. AH, 2007 Conference on Analysis of Algorithms (AofA 07); Publisher: DMTCS; Volume: AH; Pages: 127-146; DOI: [10.46298/dmtcs.3545](https://doi.org/10.46298/dmtcs.3545)
  介绍 HyperLogLog 算法用于高效近似基数估计的奠基性论文。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2020)
  Publisher: Cambridge University Press
  一本全面的教材，涵盖多种数据挖掘算法，包括布隆过滤器和 HyperLogLog 的详细章节，适用于大规模数据应用。第3版。

---

[上一节](04-%E7%A9%BA%E9%97%B4%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%88%E5%9B%9B%E5%8F%89%E6%A0%91%E3%80%81%E5%85%AB%E5%8F%89%E6%A0%91%EF%BC%89.md) · [下一节](06-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E7%AE%97%E6%B3%95%E8%AE%BE%E8%AE%A1%E6%96%B9%E6%B3%95%EF%BC%88%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95%E3%80%81%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%EF%BC%89.md)
