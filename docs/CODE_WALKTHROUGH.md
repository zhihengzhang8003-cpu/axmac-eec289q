# AxMAC Python 代码讲解

## 模块结构

```
axmac/
  exact_mac.py      — 格式定义 + 精确 MAC（INT & FP）
  approx_mac.py     — 近似 MAC：截断、rounding、ACA、DRUM
  power_model.py    — 功耗模型（解析式 + 数据相关开关活动）
  accuracy_eval.py  — 误差统计与 sweep
  dnn_inference.py  — 向量化推理注入（numpy + pytorch）
  sensitivity.py    — 层敏感度探测与预算分配（贡献 B）
  pareto.py         — 设计空间 sweep + Pareto front（贡献 A+B 评估）
experiments/
  cifar10_experiment.py   — CIFAR-10 端到端实验
  generate_figures.py     — 生成论文图
  redesign_experiments.py — 贡献 A/B 对比实验
tests/                    — pytest 套件（350 个测试）
rtl/                      — Verilog RTL + ModelSim testbench
```

---

## Step 1：数值格式定义 — `exact_mac.py:38`

整个项目的类型系统。所有 MAC 函数、功耗函数、精度评估函数都以这两个 dataclass 作为参数。

### IntFormat — `exact_mac.py:38`

```python
@dataclass(frozen=True)
class IntFormat:
    name: str
    bits: int

    @property
    def min_val(self) -> int:
        return -(1 << (self.bits - 1))     # INT8 → -128

    @property
    def max_val(self) -> int:
        return (1 << (self.bits - 1)) - 1  # INT8 → 127
```

`frozen=True`：实例不可变，可当 dict key。`power_model.py:70` 的 `_BASE_PJ` 表用
`(fmt.name, "mult")` 作为 key 查功耗数据。

三个实例在 `exact_mac.py:52-54`：`INT4`、`INT8`、`INT16`。

### FPFormat — `exact_mac.py:57`

```python
@dataclass(frozen=True)
class FPFormat:
    name: str
    exp_bits: int
    mant_bits: int
    has_inf: bool = True   # 关键字段
```

`has_inf` 是 FP 逻辑最重要的 flag。标准 IEEE 格式（FP32/FP16/BF16/E5M2）里全 1
指数 = inf 或 NaN。E4M3 牺牲 ±inf，让全 1 指数也能表示有限正常数，只保留一个
NaN（全 1 指数 + 全 1 mantissa）——多了一个 binade 的表示范围。

`has_inf` 在三处分支：

| 位置 | 作用 |
|---|---|
| `exact_mac.py:379` `_to_internal` | 解码：全 1 指数是 NaN/inf 还是 normal |
| `exact_mac.py:417` `_renormalize_and_pack` | 编码：最大合法指数是 `exp_all_ones` 还是 `exp_all_ones - 1` |
| `exact_mac.py:396` `_make_qnan` | NaN bit pattern：E4M3 用全 1 mantissa，其他用 MSB=1 |

六个实例在 `exact_mac.py:83-91`：

```python
FP32     = FPFormat("FP32", 8, 23)
FP16     = FPFormat("FP16", 5, 10)
BF16     = FPFormat("BF16", 8, 7)
FP8_E5M2 = FPFormat("E5M2", 5, 2)
FP8_E4M3 = FPFormat("E4M3", 4, 3, has_inf=False)  # 唯一 has_inf=False
```

---

## Step 2：Booth Radix-4 Partial Products — `exact_mac.py:112`

INT MAC 的乘法模拟硬件 Booth 编码，不是直接 `a * b`。

```python
# exact_mac.py:112
_BOOTH_DECODE = (0, 1, 1, 2, -2, -1, -1, 0)
```

下标是 3-bit 滑动窗口 `(b[i+1], b[i], b[i-1])`，值是系数 `{-2, -1, 0, 1, 2}`。
每次看 b 的 3 位（每隔 2 位滑动），部分积数量比 naive 方式减半。

```python
# exact_mac.py:115-130
def booth_radix4_pps(a, b, n):
    num_pps = (n + 1) // 2          # INT8 → 4 个部分积
    for i in range(num_pps):
        b_prev = 0 if i == 0 else (b >> (2*i - 1)) & 1
        b_lo   = (b >> (2*i))     & 1
        b_hi   = (b >> (2*i + 1)) & 1
        code   = (b_hi << 2) | (b_lo << 1) | b_prev
        pps.append((_BOOTH_DECODE[code] * a) << (2*i))
    return pps
```

Python 任意精度整数的算术右移自然 sign-extend，无需额外符号处理。

**部分积的用途（暴露给外部）：**
- `exact_mac.py:162`：`return_pps=True` 时返回 `IntMacResult(raw, pps)`
- `power_model.py:257` `pp_switching_activity`：对两个周期的 pps 算 Hamming distance，
  换算成数据相关的动态开关功耗

**近似路径的修改点（`approx_mac.py:167`）：**

```python
pps = booth_radix4_pps(a, b, fmt.bits)
product = _apply_rounding(sum(pps), K, rounding, rng)  # 精确路径没有这行
```

Booth 编码本身不变，只在 sum 之后、加到 acc 之前截断低 K 位。

---

## Step 3：三种 Rounding Mode — `approx_mac.py:80`

贡献 A 的核心实现。

```python
# approx_mac.py:80-106
def _apply_rounding(product, K, rounding, rng):
    if K <= 0:
        return product                       # K=0 → 精确
    if rounding == "round":
        product += 1 << (K - 1)             # 加修正常数 2^(K-1)
    elif rounding == "stochastic":
        product += rng.randrange(1 << K)    # 加随机 offset [0, 2^K)
    return product & ~((1 << K) - 1)        # 低 K 位清零
```

三种 mode 最后都做 `& ~mask`，区别只在清零前加了什么。

### 为什么 trunc 有问题

低 K 位真实值在 `[0, 2^K)`，清零后全丢，误差恒正，均值 ≈ `2^(K-1)`。
N 次 MAC 累加后总误差均值 = `N × c`，**线性增长**。
`accuracy_eval.py:59` 的 `bias` 字段测量这个值。

CIFAR-10 结果：`trunc K=4` 准确率 10.8%（接近随机），`round K=4` 保持 83.0%。

### round：加常数

被丢掉的低 K 位期望值是 `2^(K-1)`，提前加上再截断，误差变为 `[-2^(K-1), 2^(K-1))`，均值接近零。

**硬件代价：零。** `2^(K-1)` 是单 bit 常数，作为 carry-in 注入部分积加法树，
不需要额外加法器（Schulte & Swartzlander 1993）。`power_model.py:169`
`rounding_rng_pJ` 对 `round` 返回 0。

### stochastic：加随机 offset

每个 MAC 的 offset 独立同分布，N 次累加后总误差标准差 ∝ `√N`（不是 `trunc` 的 `N`）。
现代 FP8 训练硬件的标准做法（Gupta et al. ICML 2015）。

**硬件代价：贵。** 需要 per-MAC K-bit LFSR，`power_model.py:150` 建模：

```python
def rng_energy_pJ(K):
    return K * FF_ENERGY_PJ + _LFSR_FEEDBACK_TAPS * XOR_ENERGY_PJ
    # K × 0.012 pJ  +  2 × 0.003 pJ
```

### 三种 mode 对比

| mode | 误差均值 | 硬件额外代价 | 备注 |
|---|---|---|---|
| `trunc` | `~2^(K-1)` 正偏 | 0 | baseline，DNN 准确率大幅下降 |
| `round` | ≈ 0 | 0（常数 carry-in）| 本项目推荐 |
| `stochastic` | 0（期望）| K-bit LFSR | FP8 训练硬件 |

---

## Step 4：ACA 加法器 — `approx_mac.py:109`

近似加法器，处理乘积加到累加器这一步。

```python
# approx_mac.py:109-132
def aca_add(a, b, bits, window):
    if window >= bits:
        return _wrap(a_u + b_u, bits)   # 精确加法
    out = 0
    pos = 0
    while pos < bits:
        w = min(window, bits - pos)
        seg_mask = (1 << w) - 1
        a_seg = (a_u >> pos) & seg_mask
        b_seg = (b_u >> pos) & seg_mask
        out |= ((a_seg + b_seg) & seg_mask) << pos   # 段内加，进位不出段
        pos += w
    return _wrap(out, bits)
```

把 `bits` 宽的进位链切成 `window` 宽的独立段，段间进位直接丢弃。
进位链只需传播 `window` 位，延迟和面积按比例缩小。

**误差来源：** 段边界恰好发生进位时被丢弃，高位段少加 1。window 越窄误差越大。
`pareto.py:254` `_filter_usable` 过滤掉 NRMSE > 10× baseline 的无效配置。

---

## Step 5：完整 INT MAC — `approx_mac.py:135`

三个零件的组合：

```python
# approx_mac.py:135-174
def approx_mac_int(a, b, acc, fmt, *, K=0, aca_window=None, ...):
    pps     = booth_radix4_pps(a, b, fmt.bits)           # Booth 编码
    product = _apply_rounding(sum(pps), K, rounding, rng) # 截断
    window  = aca_window if aca_window is not None else acc_bits
    out     = aca_add(acc, product, acc_bits, window)     # ACA 加法
    return out
```

数据流：`a, b → pps → product → truncate → aca_add → out`

**K 和 aca_window 正交：**
- K=0, window=None → 精确 MAC（等价于 `exact_mac.mac_int`，pytest 回归保证）
- K>0, window=None → 只近似乘法器
- K=0, window<acc_bits → 只近似加法器
- K>0, window<acc_bits → 两者都近似

`pareto.py:439` `sweep_int_designs` 穷举这两个维度的所有组合。

---

## Step 6：FP 内部表示 — `exact_mac.py:373`

统一内部表示：`V = (-1)^s × M × 2^(E - mant_bits)`

`M` 含隐含 1 的完整 mantissa（非负整数），`E` 是未加 bias 的真实指数。
Normal 和 subnormal 共用同一套算术逻辑。

```python
# exact_mac.py:373-393
def _to_internal(bits, fmt):
    s, e, m = fp_unpack(bits, fmt)

    if e == fmt.exp_all_ones:
        if not fmt.has_inf:                      # E4M3 分支
            if m == (1 << fmt.mant_bits) - 1:   # 全 1 mantissa → NaN
                return s, m, 0, "nan"
            M = (1 << fmt.mant_bits) | m         # 否则是普通 normal
            return s, M, e - fmt.bias, "normal"
        return s, m, 0, "nan" if m != 0 else "inf"

    if e == 0:
        if m == 0: return s, 0, 0, "zero"
        return s, m, 1 - fmt.bias, "subnormal"  # 隐含位为 0，E 固定为 1-bias

    M = (1 << fmt.mant_bits) | m                # 加隐含 1
    return s, M, e - fmt.bias, "normal"
```

---

## Step 7：单次 RNE 归一化 — `exact_mac.py:403`

```python
def _renormalize_and_pack(sign, M, E, fmt) -> int:
```

输入 `(sign, M, E)`，表示 `V = M × 2^(E - mant_bits)`，塞进 `fmt` 的 bit pattern，做 RNE。

**为什么"单次"：** 朴素实现先 normalize 再 subnormal-shift，两次舍入产生
double-rounding（结果可能差一个 ULP）。这里一次 shift 同时处理，只做一次
guard/round/sticky 判断。

```python
# exact_mac.py:421-438  subnormal 路径
guard  = (M >> (shift - 1)) & 1
sticky = (M & ((1 << (shift - 1)) - 1)) != 0
keep   = M >> shift
if guard and (sticky or (keep & 1)):   # RNE 条件：guard=1 且（sticky 或 末位=1）
    keep += 1
```

**E4M3 overflow 处理（`exact_mac.py:457-464`）：**
无 inf，overflow 饱和到 ±max_normal，同时绕开 NaN 的 bit pattern（全 1 mantissa - 1）：

```python
if tentative_E > max_norm_E or (
    tentative_E == max_norm_E and (M & mant_mask) == mant_mask
):
    return fp_pack(sign, fmt.exp_all_ones, mant_mask - 1, fmt)
```

---

## Step 8：FP MAC 组合 — `exact_mac.py:579`

```python
# exact_mac.py:579-596
def mac_fp(a_bits, b_bits, acc_bits, fmt):
    prod_bits, mant_prod = fp_multiply(a_bits, b_bits, fmt)
    out = fp_add(acc_bits, prod_bits, fmt)
    return out
```

- `fp_multiply`（`exact_mac.py:471`）：mantissa 整数相乘，`E_prod = Ea + Eb - mant_bits`，
  送进 `_renormalize_and_pack`
- `fp_add`（`exact_mac.py:495`）：对齐指数（保留 3 位 guard bits），相加，再 renormalize

**近似版本只改一处（`approx_mac.py:280-283`）：**

```python
M_full = Ma * Mb
M_prod = _apply_rounding(M_full, K, rounding, rng)  # 截断 mantissa 乘积低 K 位
E_prod = Ea + Eb - fmt.mant_bits
prod_bits = _renormalize_and_pack(s_out, M_prod, E_prod, fmt)
```

后续 renormalize 和 add 保持精确。

---

## Step 9：向量化推理 — `dnn_inference.py`

把标量 MAC 展开成 numpy 矩阵运算，语义完全等价。

### 核心向量化 — `dnn_inference.py:93`

```python
prod  = a[:, :, None] * b[None, :, :]   # (M, K_dim, N)：所有 MAC 的乘积
prod  = truncate_products(prod, K, ...)  # 向量化 rounding
accum = prod.sum(axis=1)                # (M, N)：沿 reduction 轴求和
```

`truncate_products`（`dnn_inference.py:56`）是 `_apply_rounding` 的向量化版本，
对整个 `(M, K_dim, N)` 张量操作。

等价性由 `tests/test_dnn_inference.py` W2 回归测试保证：同一批输入跑标量循环和
向量化路径，逐元素比较。

### Conv2d — `dnn_inference.py:153`

```python
# im2col: (N, C, H, W) → (N*out_h*out_w, C*kh*kw)
cols, out_h, out_w = _im2col(x, kh, kw, stride, padding)
w_mat = w.reshape(c_out, c_in*kh*kw).T
flat_out = int_matmul_approx(cols, w_mat, ...)   # 复用 matmul 路径
```

每个输出位置对应一个 patch，铺成行；卷积核铺成列；整个 conv 变成一次 matmul。

### MLP 前向 — `dnn_inference.py:327`

```python
def tiny_mlp_forward(x, layers, *, fmt, K=0, rounding="trunc", rng=None):
    Ks = _per_layer_K(K, len(layers))   # 标量或 per-layer list
    for i, (w, b) in enumerate(layers):
        h = int_linear_approx(h, w, fmt=fmt, K=Ks[i], ...)
        if i != last:
            h = np.clip(h, 0, fmt.max_val)   # ReLU
```

`_per_layer_K`（`dnn_inference.py:310`）：K 可以是单个 int（全局统一）或 list
（per-layer），`sensitivity.py` 的分配结果从这里消费。

### DRUM 向量化 — `dnn_inference.py:364`

```python
msb   = np.floor(np.log2(abs_x[nonzero].astype(np.float64))).astype(np.int64)
shift = np.maximum(0, msb - (k - 1))
drum_abs = np.where(nonzero, (abs_x >> shift) << shift, 0)
```

用 `log2 + floor` 找每个元素的 MSB 位置，算各自 shift 量，做元素级位移。

---

## Step 10：层敏感度与预算分配 — `sensitivity.py`

**问题：** 全局统一 K 浪费结构。demo MLP 第一层（784→128）占 98.7% 的 MAC，
但全局 K 把相同的截断深度用在了第二层（128→10）。

### MAC 计数 — `sensitivity.py:48`

```python
def layer_mac_counts(layers) -> list[int]:
    return [w.shape[0] * w.shape[1] for w, _b in layers]
```

截断 MAC 多的层，能量收益按比例更大。

### 单层敏感度探测 — `sensitivity.py:98`

```python
def layer_sensitivity(x, layers, *, fmt, K_probe=4, metric="logit_nrmse"):
    y_ref = tiny_mlp_forward(x, layers, fmt=fmt, K=0)   # 精确基准
    for i in range(len(layers)):
        K_vec = [0] * len(layers)
        K_vec[i] = K_probe                              # 只扰动第 i 层
        y = tiny_mlp_forward(x, layers, fmt=fmt, K=K_vec)
        scores.append(output_divergence(y, y_ref, metric=metric))
```

逐层 ablation：偏差大 → 敏感，不能多截；偏差小 → 耐糙，可多截。

### 贪心预算分配 — `sensitivity.py:148`

```python
def allocate_K(sensitivities, mac_counts, *, total_budget, K_max):
    K = [0] * n
    for _ in range(total_budget):
        score = mac_counts[i] / (sensitivities[i] * (K[i] + 1) + eps)
        K[best_i] += 1   # 给 score 最高的层加 1
```

```
score = mac_counts[i] / (sensitivities[i] × (K[i] + 1))
```

- 分子 `mac_counts[i]`：MAC 多 → 能量收益大 → 优先截
- 分母 `sensitivities[i]`：越敏感 → score 越低 → 越不优先
- 分母 `(K[i] + 1)`：边际成本递增，防止全堆一层

**退化验证：** 所有层 sensitivity 和 mac_counts 相等时，退化为 `uniform_K`。

---

## Step 11：误差统计 — `accuracy_eval.py`

### 误差容器 — `accuracy_eval.py:52`

```python
@dataclass(frozen=True)
class ErrorStats:
    n_samples: int
    med: float        # mean|err|
    rmse: float
    max_abs_err: float
    nmed: float       # med / format 动态范围
    bias: float       # mean(err)；trunc 下显著正，round 接近零
```

### 逐 MAC 对比 — `accuracy_eval.py:163`

```python
for a, b, acc in zip(a_samples, b_samples, acc_samples):
    exact  = mac_int(a, b, acc, fmt, ...)
    approx = approx_mac_int(a, b, acc, fmt, K=K, ...)
    errors.append(float(exact - approx))   # 正值 = 截断偏低
```

`nmed = med / fmt.max_val²`：乘积最大可能值做分母，让 INT4 和 INT16 的误差在
同一量级下可比。

**Sweep 设计（`accuracy_eval.py:252`）：** a/b/acc 在循环外生成一次，所有 (K, W)
配置跑同一批样本（paired comparison），排除样本随机性的影响。

---

## Step 12：Pareto 分析 — `pareto.py`

### 统一参考信号 — `pareto.py:290`

```python
def float_reference(a, b):
    return np.einsum("nl,nl->n", a.astype(np.float64), b.astype(np.float64))
```

所有格式（INT4 到 FP32）都和同一批 float64 dot product 比较，误差落在同一轴上。

`error_nrmse = RMSE(approx - ref) / RMS(ref)`：唯一能跨格式比较的误差指标。
`error_nmed` 不能跨格式（分母 `max_val²` 在 INT4 和 INT16 之间差四个数量级）。

### DesignPoint — `pareto.py:73`

```python
@dataclass(frozen=True)
class DesignPoint:
    fmt_name: str
    K: int
    aca_window: int | None
    energy_pJ: float
    energy_breakdown: Energy   # multiplier + adder + rng 分解
    error_nrmse: float         # 主误差轴
    error_rel_bias: float      # 追踪贡献 A 的偏差修正效果
```

### Pareto front — `pareto.py:162`

```python
def dominates(a, b) -> bool:
    return ax <= bx and ay <= by and (ax < bx or ay < by)

def pareto_front(points):
    return [p for p in points if not any(dominates(q, p) for q in points if q != p)]
```

O(n²)，对几百个点足够。

三个 front 函数：
- `per_format_front`（`pareto.py:199`）：单格式内 (K, W) 效率曲线
- `per_format_fronts`（`pareto.py:214`）：所有格式的 dict
- `global_front`（`pareto.py:231`）：跨格式最优前沿，调用前检查 y_key 不是 `error_nmed`

**去重与过滤（`pareto.py:101-283`）：**
- `_canonicalise_windows`：W ≥ acc_bits 归一化为 None，去除重复 DesignPoint
- `_filter_usable`：NRMSE > 10× baseline 的配置过滤掉（ACA 窗过窄时误差爆炸）

---

## 数据流总结

```
实验层
  pareto.sweep_all_designs()
    → eval_int/fp_config()
        → approx_mac_int/fp()     对比 exact_mac
        → mac_int/fp_energy()     功耗估算
    → DesignPoint 列表
    → global_front()              Pareto 图（论文 Fig 3/4）

推理层
  dnn_inference.tiny_mlp_forward()
    → int_linear_approx()
    → int_conv2d_approx()
    → CIFAR-10 accuracy           论文 Fig 5（round K=4 → 83%，trunc K=4 → 10.8%）

贡献 B
  sensitivity.allocate_K_for_mlp()
    → layer_sensitivity()         逐层 ablation
    → allocate_K()                贪心 knapsack
    → per-layer K list
    → tiny_mlp_forward(K=list)    非均匀截断推理

RTL 层
  rtl/tb/tb_drum_multiplier.v     ModelSim 验证 DRUM 乘法器硬件实现
```
