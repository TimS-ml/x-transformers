# x-transformers Colab 抄写练习计划

## 目标

通过逐模块手写抄写 `x-transformers` 的核心代码，深入理解现代 Transformer 架构的实现细节。
练习按依赖关系从底层到高层排列，每个 Session 对应一个 Colab notebook。

---

## 代码总量概览

| 模块 | 行数 | 难度 | 预计用时 |
|------|------|------|----------|
| attend.py | 625 | ★★☆ | 1.5h |
| autoregressive_wrapper.py | 1,020 | ★★☆ | 2h |
| x_transformers.py (分 6 部分) | 3,891 | ★★★ | 8-10h |
| 应用练习 (train_copy.py) | 180 | ★☆☆ | 0.5h |
| **核心模块合计** | **~5,716** | | **~14h** |

> 可选的高级模块 (continuous.py, xval.py, dpo.py 等) 约 5,900 行，按兴趣选做。

---

## Phase 1: 基础注意力引擎 — `attend.py` (625 行)

**目标**: 理解底层 attention 计算，包括 flash attention、causal masking、sparse attention。

### Session 1.1: Intermediates 与辅助函数 (~100 行)
- [ ] `Intermediates` dataclass — 存储 attention 中间值 (pre_softmax_attn, post_softmax_attn, cached_kv)
- [ ] 辅助函数: `log_prob_from_hard_attend()`, `selective_attn()`, `qk_l2_dist_squared()`
- [ ] Sparse attention: `one_hot_straight_through()`, `sparse_topk_attn()`
- [ ] Causal mask 创建: `create_causal_mask()`, `onnx_create_causal_mask()`

**练习要点**:
- 理解 `@dataclass` 在 PyTorch 模块中管理中间状态
- 理解 straight-through estimator 的梯度 trick
- 理解 causal mask 的构建方式

### Session 1.2: Attend 类核心 (~525 行)
- [ ] `__init__`: flash attention 配置、dropout、scale、causal 等参数
- [ ] `forward` 方法:
  - QKV 输入处理与 scale
  - Attention score 计算 (`einsum('b h i d, b h j d -> b h i j', q, k)`)
  - Mask 应用 (causal mask, input mask, context mask)
  - Softmax 与 dropout
  - Value 加权求和
  - Flash attention 路径 vs 普通路径
- [ ] Selective attention 机制
- [ ] Gumbel softmax attention (hard attention)

**练习要点**:
- 手写 einsum 表达式，理解每个维度的含义 (b=batch, h=heads, i=query_seq, j=key_seq, d=dim)
- 理解 flash attention 的 API 调用方式
- 理解 attention mask 的组合逻辑

---

## Phase 2: 自回归生成引擎 — `autoregressive_wrapper.py` (1,020 行)

**目标**: 理解自回归生成的完整流程，包括采样策略和 KV cache。

### Session 2.1: 采样策略与辅助函数 (~400 行)
- [ ] 基础工具: `exists()`, `default()`, `identity()`
- [ ] `eval_decorator` — 自动切换 eval/train 模式的装饰器
- [ ] `align_right()` — 右对齐序列 (处理 padding)
- [ ] `pad_at_dim()` — 按维度 padding
- [ ] 采样函数族:
  - `top_k()` — Top-K 采样
  - `top_p()` — Nucleus 采样
  - `top_a()` — Top-A 阈值采样
  - `min_p()` — Minimum-P 采样
  - `gumbel_sample()` — Gumbel 采样
- [ ] `contrastive_decode_fn()` — 对比解码
- [ ] `modify_cached_kv()` — KV cache 操作

**练习要点**:
- 理解各采样策略的数学原理和温度参数
- `torch.multinomial` 的使用
- 理解装饰器模式在 ML 代码中的应用

### Session 2.2: AutoregressiveWrapper 类 (~620 行)
- [ ] `__init__`: net 封装、pad_value、ignore_index 等
- [ ] `generate()` 方法 (核心!):
  - 序列初始化与 prompt 处理
  - KV cache 管理 (`cached_kv`)
  - 逐 token 生成循环
  - 采样策略应用
  - EOS 终止逻辑
  - Contrastive decoding
  - Beam search (可选)
- [ ] `forward()` 训练方法:
  - Input/target 的 shift 操作 (`x[:, :-1]`, `x[:, 1:]`)
  - Cross-entropy loss 计算
  - Label smoothing

**练习要点**:
- 理解 teacher forcing vs autoregressive generation 的区别
- KV cache 如何加速推理
- `F.cross_entropy` 的 ignore_index 参数处理 padding

---

## Phase 3: 核心 Transformer — `x_transformers.py` (3,891 行)

这是最核心也最庞大的模块，分为 6 个 sub-session。

### Session 3.1: Helper 函数与基础模块 (~290 行, L1-290)
- [ ] `LayerIntermediates` dataclass — 层间数据传递
- [ ] Helper 函数族: `exists()`, `default()`, `cast_tuple()`, `divisible_by()`
- [ ] 可调用类: `always()`, `not_equals()`, `equals()`
- [ ] Tensor 工具: `log()`, `max_neg_value()`, `l2norm()`, `softclamp()`, `masked_mean()`
- [ ] `pad_at_dim()`, `or_reduce()`, `orthog_project()`
- [ ] Cache 辅助: `get_cached_kvs()`
- [ ] 熵计算: `calc_entropy()`, `calc_z_loss()`
- [ ] 初始化: `init_zero_()`
- [ ] Kwargs 路由: `pick_and_pop()`, `group_dict_by_key()`, `groupby_prefix_and_trim()`
- [ ] `dropout_seq()` — 结构化 dropout
- [ ] 激活函数: `ReluSquared`, `SoLU`

**练习要点**:
- `groupby_prefix_and_trim` 是 x-transformers 的关键设计模式，用于将 `enc_*` / `dec_*` 前缀参数路由到对应模块
- `calc_z_loss` 来自 Switch Transformer / PaLM 的稳定化技巧

### Session 3.2: 位置编码 (~400 行, ~L290-690)
- [ ] `TokenEmbedding` — 带 scale 的 token embedding
- [ ] `AbsolutePositionalEmbedding` — 可学习绝对位置编码
- [ ] `ScaledSinusoidalEmbedding` — 带可学习 scale 的正弦位置编码
- [ ] `RotaryEmbedding` (RoPE) — 旋转位置编码 ★★★
  - `freqs_for()` — 频率计算
  - `forward()` — 生成 cos/sin 缓存
  - `rotate_half()`, `apply_rotary_pos_emb()` — 应用 RoPE
- [ ] `PolarEmbedding` — 极坐标编码
- [ ] 相对位置偏置:
  - `RelativePositionBias` — T5 风格
  - `DynamicPositionBias` — 动态版
  - `AlibiPositionalBias` — ALiBi (线性偏置)
  - `DataDependentAlibi` — 数据依赖的 ALiBi
- [ ] `CoPE` — Contextual Position Encoding

**练习要点**:
- RoPE 的数学推导: 频率 → cos/sin → 复数旋转
- ALiBi 的简洁设计: 不需要额外参数，只需几何级数的 slope
- 理解不同位置编码方案的优劣对比

### Session 3.3: 归一化与门控 (~350 行, ~L690-1040)
- [ ] 归一化层:
  - `LayerNorm`, `AdaptiveLayerNorm`
  - `RMSNorm`, `AdaptiveRMSNorm`, `SimpleRMSNorm`, `MultiheadRMSNorm`
  - `ScaleNorm`
- [ ] Scale 与门控:
  - `Scale`, `LayerScale`, `AdaptiveLayerScale`
  - `Residual`, `GRUGating`
  - `HyperConnection` — 超连接 (learnable residual mixing)
  - `DynamicLIMe` — 动态层间混合
- [ ] Token 操作:
  - `ShiftTokens` — 时间维度 shift (类似 1D 卷积)
  - `FoldAxially` — 轴向折叠
- [ ] 特殊激活:
  - `DynamicTanh`, `Derf`

**练习要点**:
- RMSNorm vs LayerNorm: 去掉 mean centering 的效率提升
- GRUGating: 用 GRU 单元控制残差连接
- HyperConnection: 可学习的层间权重混合

### Session 3.4: FeedForward 网络 (~100 行, ~L1040-1140)
- [ ] `FeedForward` 类:
  - 标准 FFN: Linear → Activation → Dropout → Linear
  - GLU 变体 (SwiGLU, GeGLU 等): Linear → [Gate × Activation] → Linear
  - 可选的 no_bias、zero_init_output
  - Post-activation 归一化

**练习要点**:
- GLU (Gated Linear Unit) 变体是现代 LLM 的标配
- 理解 `nn.SiLU()` (SwiGLU) 为何成为主流选择
- 维度变化: dim → inner_dim (通常 4x) → dim

### Session 3.5: Attention 类 (~780 行, ~L1140-1920) ★★★ 最核心
- [ ] `__init__` (60+ 参数):
  - 基础: dim, heads, dim_head, causal, dropout
  - Q/K/V 投影: to_q, to_k, to_v, to_out
  - 位置编码集成: rotary, rel_pos_bias, alibi 等
  - 高级特性: talking_heads, gate_values, flash_attn
  - QK normalization, softclamping
  - Selective attention, LASER attention
- [ ] `forward()` 方法:
  - Q/K/V 计算与 multi-head reshape
  - 位置编码应用 (RoPE, relative bias 等)
  - Cross-attention 处理
  - KV cache 读写
  - 调用 `self.attend()` (使用 attend.py)
  - Talking heads (pre/post softmax 变换)
  - Gate values (learned output gating)
  - 输出投影

**练习要点**:
- 这是整个 repo 最核心的类，集成了几乎所有 attention 变体
- `einops.rearrange` 的密集使用: `'b n (h d) -> b h n d'`
- KV cache 的拼接逻辑: `torch.cat((cached_k, k), dim=-2)`
- Multi-query attention (MQA) 和 Grouped-query attention (GQA) 的实现

### Session 3.6: AttentionLayers + TransformerWrapper + 高级封装 (~1,970 行, ~L1920-3891) ★★★
- [ ] `AttentionLayers` (~860 行):
  - Layer 类型调度: `'a'` (self-attn), `'c'` (cross-attn), `'f'` (FFN)
  - `default_block` 与 macaron/sandwich 配置
  - Pre-norm vs Post-norm 架构
  - 残差连接策略
  - Forward: 逐层执行 attention 和 FFN
  - KV cache 管理
  - HyperConnection 集成
- [ ] `Encoder` / `Decoder` / `PrefixDecoder` / `CrossAttender`:
  - 都是 `AttentionLayers` 的薄封装
  - Decoder 设置 `causal=True`
  - PrefixDecoder: 前 N 个 token 非 causal
- [ ] `TransformerWrapper` (~560 行):
  - Token embedding + 位置编码
  - Memory tokens (可学习的全局 tokens)
  - 调用 AttentionLayers
  - 输出投影到 logits
  - 可选: embedding dropout, post_emb_norm, recycling
- [ ] `ViTransformerWrapper`:
  - Vision Transformer: image → patches → transformer
- [ ] `XTransformer` (~60 行):
  - 完整的 encoder-decoder 架构
  - 集成 encoder TransformerWrapper + decoder TransformerWrapper
  - `generate()` 方法: 编码 → 自回归解码

**练习要点**:
- `AttentionLayers` 的 layer 调度机制是理解整个架构的关键
- `groupby_prefix_and_trim` 在 `XTransformer` 中的作用: 将 `enc_*` 和 `dec_*` 参数分别路由
- 理解 pre-norm (GPT-2 style) vs post-norm (原始 Transformer) 的区别

---

## Phase 4: 端到端应用练习

### Session 4.1: Copy Task — `train_copy.py` (~180 行)
- [ ] 搭建完整的训练 pipeline:
  - 数据生成 (`cycle()` generator)
  - 模型实例化 (`XTransformer`)
  - 训练循环 (forward → loss → backward → step)
  - 推理评估 (`model.generate()`)
- [ ] 在 Colab 中实际运行，观察 loss 下降和 copy 准确率提升

**练习要点**:
- 将前面抄写的所有模块串联起来
- 理解 encoder-decoder 模型在 seq2seq 任务上的完整工作流程
- 观察 attention pattern (可选: 可视化 attention weights)

### Session 4.2: 自定义实验 (可选)
- [ ] 修改 `train_copy.py`:
  - 试不同位置编码 (RoPE vs ALiBi)
  - 试不同 FFN (标准 vs SwiGLU)
  - 试不同注意力变体 (flash attention, sparse attention)
- [ ] 写一个 decoder-only 的字符级语言模型

---

## Phase 5: 高级模块 (可选，按兴趣选做)

### Session 5.1: continuous.py (467 行)
- [ ] `ContinuousTransformerWrapper` — 连续输入 (非离散 token) 的 Transformer
- [ ] `ContinuousAutoregressiveWrapper` — 连续值自回归生成
- 适合场景: 时间序列预测、音频、科学计算

### Session 5.2: nonautoregressive_wrapper.py (727 行)
- [ ] `NonAutoregressiveWrapper` — 并行解码
- [ ] `SelfCritic` — 自我批评奖励
- 适合场景: 机器翻译加速

### Session 5.3: xval.py (534 行)
- [ ] `XValTransformerWrapper` — 处理连续数值的 Transformer
- 适合场景: 数学推理、科学数据

### Session 5.4: dpo.py (381 行)
- [ ] `DPO` — Direct Preference Optimization
- 适合场景: RLHF/对齐训练

### Session 5.5: belief_state_wrapper.py (751 行)
- [ ] `BeliefStateWrapper` — 双向 belief state 建模
- 适合场景: 规划、推理

### Session 5.6: 其他特殊模块
- [ ] `gpt_vae.py` (441 行) — GPT + VAE 架构
- [ ] `free_transformer.py` (747 行) — Free Transformer
- [ ] `neo_mlp.py` (273 行) — 用 attention 实现 MLP
- [ ] `xl_autoregressive_wrapper.py` (334 行) — Transformer-XL 风格
- [ ] `entropy_based_tokenizer.py` (337 行) — 基于熵的分词器

---

## 推荐的 Colab 抄写方法

### 每个 Session 的 Notebook 结构

```
Cell 1: [Markdown] Session 标题 + 学习目标
Cell 2: [Code]    pip install einops einx loguru  # 依赖安装
Cell 3: [Code]    import torch, ... # 公共 imports
Cell 4: [Markdown] --- 模块 A 说明 ---
Cell 5: [Code]    # 抄写模块 A (对照源码手写)
Cell 6: [Code]    # 模块 A 测试: 构造简单输入，验证输出 shape
Cell 7: [Markdown] --- 模块 B 说明 ---
Cell 8: [Code]    # 抄写模块 B
Cell 9: [Code]    # 模块 B 测试
...
Cell N: [Markdown] Session 总结 + 关键 takeaway
```

### 抄写原则

1. **不要复制粘贴** — 逐行手打，边写边理解
2. **先读后写** — 先通读一遍源码，理解逻辑，再开始抄写
3. **写测试** — 每抄完一个类/函数，立即写一个简单测试验证
4. **加注释** — 用自己的话写注释，不要照搬原注释
5. **简化再扩展** — 第一遍可以跳过不常用的参数，先实现核心逻辑

### 建议的测试模式

```python
# 例: 测试 Attention 类
attn = Attention(dim=512, heads=8, dim_head=64)
x = torch.randn(2, 16, 512)  # (batch, seq_len, dim)
out = attn(x)
assert out.shape == (2, 16, 512), f"Expected (2, 16, 512), got {out.shape}"
print("✓ Attention basic test passed")

# 例: 测试 causal attention
attn_causal = Attention(dim=512, heads=8, causal=True)
out = attn_causal(x)
assert out.shape == (2, 16, 512)
print("✓ Causal Attention test passed")
```

---

## 依赖安装 (Colab 首个 Cell)

```python
!pip install torch einops einx loguru packaging
```

---

## 学习路线总结

```
Phase 1 (基础)        Phase 2 (生成)         Phase 3 (核心)              Phase 4 (应用)
┌──────────┐      ┌──────────────┐      ┌─────────────────────┐      ┌───────────┐
│ attend.py│ ───→ │ autoreg_     │ ───→ │ x_transformers.py   │ ───→ │train_copy │
│ (625行)  │      │ wrapper.py   │      │ (3891行, 6个session)│      │ (180行)   │
│ ~1.5h    │      │ (1020行)     │      │ ~8-10h              │      │ ~0.5h     │
└──────────┘      │ ~2h          │      └─────────────────────┘      └───────────┘
                  └──────────────┘              │
                                               ↓
                                        Phase 5 (高级, 可选)
                                        ┌─────────────────────┐
                                        │ continuous, xval,   │
                                        │ dpo, belief_state.. │
                                        │ (~5900行, 按兴趣选) │
                                        └─────────────────────┘
```

**核心路径预计总时长: ~14 小时 (可分 7-10 个 session 完成)**
