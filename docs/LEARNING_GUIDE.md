# Genllm 学习指南

面向 LLM 推理新手的代码学习路径。

---

## 核心认知

LLM 推理本质就三件事：

1. **加载模型** — 从磁盘读取权重和配置（GGUF Parser）
2. **构建计算图** — 将模型架构表达为可执行的 DAG（Model + Graph）
3. **执行推理** — 按拓扑序遍历图节点，调度算子到设备执行（Executor + Backend）

抓住这条主线，其余都是支撑这三步的基础设施。

---

## 第一优先：理解推理全流程

### tests/main.cpp

这是整个框架的入口，67 行代码串联了所有核心组件。先搞懂每一步在做什么，再往深处钻。

```
DeviceManager::print_devices()          → 设备发现
GGUFParser("model.gguf")               → 解析模型文件
ModelFactory::CreateFromGGUF()          → 创建模型实例
model->build_graph()                    → 构建 DAG 计算图
GraphScheduler(graph, config)           → 调度器接管计算图
scheduler.schedule(devices)             → 分配层到设备
manager->load_weights()                 → 加载权重到内存池
Executor(scheduler)                     → 创建执行器
Tokenizer::from_gguf()                  → 加载分词器
executor.generate(prompt, ...)          → 执行推理，生成文本
```

---

## 第二优先：核心引擎（必须精读）

### include/core/tensor.hpp — Tensor 数据结构

计算图的基本单元，理解它的结构就理解了框架的数据流：

- `data` — 数据指针（指向内存池中的实际数据）
- `shape` — 张量形状（如 [batch, seq_len, hidden_dim]）
- `dtype` — 数据类型（FP32/BF16/FP16 等）
- `op` — 操作类型（MatMul/Add/Softmax 等）
- `device` — 所在设备（CPU/CUDA/Vulkan）
- `layer_id` — 层标识（用于调度分组）

### include/core/graph.h + src/core/graph.cpp — DAG 计算图

从"模型定义"到"实际执行"的桥梁：

- 以 Tensor 为节点、操作依赖为边构建有向无环图
- 拓扑排序（BFS 反向收集）确定执行顺序
- 按层分组以适配多设备调度
- DOT 格式导出可渲染为图片便于调试

关键算法：从输出节点反向 BFS，收集所有依赖，确保每个输入在消费前已完成计算。

### include/core/executor.h + src/core/executor.cpp — 推理执行器

推理的指挥中心，协调整个前向计算：

- **Prefill 阶段** — 并行处理整个 prompt 序列，计算量大但只需一次
- **Decode 阶段** — 逐 token 自回归生成，每步只算一个 token 但需反复调用
- Kernel dispatch — 将每个操作分发到对应设备的后端实现
- 采样策略 — argmax（贪心）和 top-p（nucleus）采样

这是框架中依赖最多（fan-out 最高）的文件，体现了它作为运行时核心的枢纽地位。

### include/core/gguf_parser.h + src/core/gguf_parser.cpp — GGUF 解析器

模型如何从磁盘进入内存：

- 解析 GGUF v3 二进制格式：魔数 + 版本号 + 元数据键值对 + 张量信息表
- 提取模型配置（架构名、层数、头数、隐藏维度等）
- 提取张量信息（名称、形状、数据类型、文件偏移量）
- 支持 int8/16/32/64、float32/64、string、array 等多种数据类型

---

## 第三优先：模型定义（理解图怎么建的）

### include/model/model.h — 模型抽象与工厂

- `ModelBase` — 定义模型的抽象接口（build_graph、vocab_size 等）
- `ModelFactory` — 工厂模式，按架构名映射到创建函数
- 静态初始化器在 main() 之前完成注册，避免手动 if-else 链

### src/model/Qwen3.cpp — Qwen3 模型实现（推荐精读）

展示完整的图构建流程：

```
从 GGUF 元数据读取配置（层数、头数、隐藏维度等）
    ↓
构建 embedding 层（token → 向量）
    ↓
构建 N 层 Transformer：
    ├── RMSNorm → GQA 注意力 → 残差连接
    └── RMSNorm → FFN (gate_proj + up_proj → SiLU → down_proj) → 残差连接
    ↓
构建 lm_head（向量 → logits）
    ↓
返回完整计算图
```

### include/model/op_factory.hpp — 图构建 DSL

提供图构建的静态工具方法：

- RoPE 缓存预计算 — 避免每次推理重复计算旋转位置编码
- 线性层输出推断 — 根据输入形状自动推断 MatMul 输出维度
- 步长计算 — 处理多维张量的内存布局

---

## 第四优先：算子实现（按需深入）

从 CPU 后端入手，比 CUDA 更容易理解算法本质。

### src/backend/cpu/attention.cpp — 注意力机制

Transformer 的核心，支持 4 种模式：

| 模式 | 特点 |
|------|------|
| 标准 Attention | Q×K^T → Softmax → ×V，最直观 |
| FlashAttention | 分块计算，减少内存访问，适合长序列 |
| PagedAttention | KV cache 分页管理，适合多请求并发 |
| Softmax | 注意力分数归一化，exp 求和取反 |

### src/backend/cpu/linear.cpp — 线性层（矩阵乘法）

推理的计算瓶颈：

- 使用 AVX2 SIMD 指令集，一条指令同时处理 8 个 float32
- FP16/BF16 权重加载到 256-bit 寄存器后做融合乘加（FMA）
- 理解这个文件就能理解"为什么推理需要 GPU"—— CPU 即使有 SIMD 也远不如 GPU 并行

### src/backend/cpu/normalization.cpp — RMSNorm

Transformer 的标配归一化：

- 比 LayerNorm 更简单（无均值中心化，无偏置）
- 计算：x / sqrt(mean(x^2) + eps) * weight
- 每个 Transformer 层的注意力和 FFN 前各用一次

### src/backend/cpu/rope.cpp — RoPE 位置编码

旋转位置编码，让模型理解 token 的位置关系：

- 将位置信息编码为旋转矩阵
- 对 Q 和 K 向量应用旋转变换
- 相对位置信息通过旋转角度差自然表达

---

## 学习路径总结

```
tests/main.cpp（全貌，67 行）
  │
  ├→ include/core/tensor.hpp（数据结构，理解 Tensor）
  │
  ├→ include/core/graph.h + src/core/graph.cpp（计算图，理解 DAG）
  │
  ├→ src/model/Qwen3.cpp（图构建，理解模型如何变成图）
  │
  ├→ include/core/executor.h + src/core/executor.cpp（执行器，理解推理流程）
  │
  └→ src/backend/cpu/attention.cpp（算子，理解注意力机制实现）
```

每一步都是对前一步的展开：main.cpp 告诉你"用了什么"，后续文件告诉你"怎么实现的"。

---

## 辅助资源

### Python 参考实现

`shell/` 目录下的 PyTorch 实现可用于交叉验证：

- `shell/model.py` — 完整的 Qwen3 推理，与 C++ 实现逐层对应
- `shell/ops.py` — 算子库（embedding、RMSNorm、RoPE、attention）

当 C++ 代码的数值结果有疑问时，用 Python 实现对比中间输出（embedding、attention、FFN），精确定位偏差来源。

### 可视化调试

计算图支持 DOT 格式导出：

```cpp
graph->export_dot("qwen3-graph.dot");  // 取消 main.cpp 中的注释
```

用 Graphviz 渲染后可以直观看到 DAG 结构和数据流向。

---

*基于知识图谱生成，参考 commit 72a0466*
