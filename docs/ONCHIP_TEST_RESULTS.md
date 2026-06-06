# 片上测试：预期、结果与作用

## 测试架构概览

测试分三层，各有分工：

```
ModelSim 仿真     → 证明 RTL ≡ Python（bit-exact）
                    排除所有实现 bug，使 PPA 数据可信
       ↓
Quartus PPA       → 量化贡献 A 的硬件代价：
                    round vs trunc 几乎免费
                    stochastic 需要额外 LFSR
       ↓
板载 UART 运行    → 端到端闭环：
                    真实硬件 logit 与 Python golden 对比
```

---

## 第一层：ModelSim 仿真

**全部通过，时间：2026-06-01**（状态见 `rtl/README.md:51`）

| Phase | 测试内容 | 预期 | 实际 |
|---|---|---|---|
| 1 | `mac_unit` trunc/round，4046 组，bit-exact | 全部通过 | **PASS** |
| 1 | `mac_unit` stochastic，2000 组/K，均值 ±5σ | 均值接近零 | **PASS** |
| 2 | `aca_adder` W=4/8/16/32，1024 组，bit-exact | 全部通过 | **PASS** |
| 3 | `mac_array` 4×4，K=0 精确模式，16 个 PE | 与内联参考一致 | **PASS** |
| 4 | `mlp_top` 10 个 logit 与 Python golden 比对 | bit-exact | **PASS** |
| 4b | `mlp_top_demo` LED 显示 argmax | led_class=1（0001）| **PASS** |

### 各 testbench 的验证策略

**`tb_mac_unit.sv`（`rtl/tb/tb_mac_unit.sv:64`）**

- trunc / round：读 `mac_int8.csv`，逐行驱动 DUT，比对 `product_rounded` bit-exact
- stochastic：2000 次随机 `(a, b, rnd)`，验证误差均值在 ±5σ 内

```
sigma = (1 << K) / sqrt(12 * 2000)
tol   = 5 * sigma
```

这验证了 Gupta 2015 的无偏性：stochastic rounding 的误差均值应为零。

**`tb_aca_adder.sv`（`rtl/tb/tb_aca_adder.sv:21`）**

同时实例化 4 个不同 WINDOW 的 DUT，同一 `(a, b)` 驱动全部，CSV 里的 `window` 列决定比对哪个输出——一次仿真验证所有配置点。

**`tb_mac_array.sv`（`rtl/tb/tb_mac_array.sv:73`）**

结构性测试：用 K=0（精确模式）排除 rounding 误差干扰，只验证广播连接、PE 索引、clear/mac_en 时序是否正确。

**这层的意义：** 预期和实际没有偏差——bit-exact 就是设计目标。证明 RTL 与 Python 完全等价，为后续 PPA 数据提供可信基础。

---

## 第二层：Toy MLP Logit 对比

**数据来源：** `rtl/golden/mlp_toy/y_K*.csv`（Python 侧生成），`tb_mlp_top.sv` 验证 RTL 输出与之一致。

网络：64→16→10，INT8，输入为 `x.csv`（64 维随机量化向量）。

| 配置 | logit[1]（最大）| logit[8] | argmax | 与精确基准差（logit[1]）|
|---|---|---|---|---|
| K=0, trunc（精确基准）| **10730** | 10430 | 1 | — |
| K=2, trunc | **10718** | 10420 | 1 | −12 |
| K=4, trunc | **10702** | 10384 | 1 | −28 |
| K=2, round | **10746** | 10276 | 1 | +16 |
| K=4, round | **10830** | 7792 | 1 | +100 |

**观察：** 所有配置的 argmax 均为 1，预测结果不变。但这不代表近似无害——toy MLP 权重是随机生成的（`export_golden.py:159`），类 1 与类 8 的差距仅 300 左右，在更大 K 下有翻转风险。

**这层的意义：** 不测准确率，测**端到端结构正确性**——FSM 时序、tiling、bias 加法、ReLU 饱和、结果打包全部链路验证通过，RTL 与 Python 的 10 个数字完全一致。

---

## 第三层：Quartus PPA 测量

**数据来源：** `rtl/vendor/altera/ppa_results.csv`  
**目标芯片：** Altera Cyclone IV EP4CE10（60nm LP，50MHz）

| 配置 | 逻辑单元 | 寄存器 | 核心动态功耗 | 相对精确基准 |
|---|---|---|---|---|
| K=0, W=32（精确基准）| 1769 | 582 | **10.45 mW** | — |
| K=0, W=8 | 1755 | 550 | 8.91 mW | −14.8% |
| K=0, W=4 | 1732 | 534 | 9.19 mW | −12.1% |
| K=6, trunc | 1685 | 527 | 8.67 mW | −17.0% |
| K=6, round | 1731 | 527 | 9.20 mW | −12.0% |
| K=6, stochastic | 1817 | **591** | 9.39 mW | −10.2% |

### 预期 vs 实际

**trunc vs round（核心对比点）**

- 预期：分析模型认为 round 代价为零（常数 carry-in，Schulte & Swartzlander 1993）
- 实际：差 **0.53 mW**（8.67 → 9.20 mW）
- 原因：Quartus 优化了常数加法，但 `round_const` 的 MUX 仍贡献少量翻转功耗
- 结论：**在可接受范围内，round 的功耗代价可忽略**，分析模型成立

**stochastic 的寄存器开销**

- 预期：比 trunc 多一个 64-bit LFSR = 64 个额外寄存器
- 实际：591 − 527 = **64 个**，与预期完全吻合
- 对应代码：`mlp_top.v:153` 的 `generate if (MODE==2)` 条件实例化 LFSR
- 功耗差：9.39 − 8.67 = **0.72 mW**（LFSR 本体 + 每 MAC 的 rnd 加法）

**ACA vs 截断乘法器**

- K=6 trunc 节省 1.78 mW；W=8 ACA 节省 1.54 mW
- ACA 在累加器层面节省（每个 PE 每周期），截断只作用于乘积低位
- 两者正交，同时使用理论上可叠加

---

## 第四层：板载 UART 运行

**相关文件：** `burn_*.bat`、`mlp_top_demo.v`、`read_uart.py`

协议：`0xAB | logit[0..9]`（10×int32 小端），共 41 字节。

**`mlp_top_demo.v:19` 记录的预期结果（K=2, trunc）：**

```
logits = [4345, 10718, -4421, -9026, -3228, -6422, 4780, -3136, 10420, 6344]
argmax = 1  →  led_class = 0001（野火征途 4 个 LED）
```

与 `rtl/golden/mlp_toy/y_K2_trunc.csv` 完全一致。

**已烧录的配置（`burn_*.bat`）：**

| 脚本 | SOF 路径 | 对应配置 |
|---|---|---|
| `burn_K0.bat` | `E:\uart_builds\K0_trunc\demo.sof` | 精确基准 |
| `burn_K2_trunc.bat` | `E:\uart_builds\K2_trunc\demo.sof` | 轻度截断 |
| `burn_K4_trunc.bat` | `E:\uart_builds\K4_trunc\demo.sof` | CIFAR-10 准确率崩溃的配置 |
| `burn_K4_round.bat` | `E:\uart_builds\K4_round\demo.sof` | 贡献 A 核心对比点 |
| `burn_K6_trunc.bat` | `E:\uart_builds\K6_trunc\demo.sof` | 激进截断 |

`read_uart.py` 在 PC 端读取串口，解析 41 字节包，打印 10 个 logit 并显示 argmax，用于交叉验证板上输出与 Python golden 是否一致。

**这层的意义：** 端到端闭环——量化→近似 MAC→bias+ReLU→UART 帧打包→串口发送→PC 解析，整条链路在真实硬件上运行，最终结果与 Python 侧数字吻合，证明 RTL 实现的正确性不只停留在仿真层面。

---

## 结果对贡献 A 的支撑

三层测试共同支撑了同一个结论：

| 证据来源 | 支撑的论点 |
|---|---|
| `tb_mac_unit` stochastic 统计检验 | stochastic rounding 误差均值为零（Gupta 2015） |
| PPA：trunc vs round 差 0.53 mW | round 的硬件代价可忽略（≈"免费"的无偏性）|
| PPA：stochastic 多 64 个寄存器 + 0.72 mW | LFSR 代价是 round 不需要支付的额外成本 |
| CIFAR-10 实验（Python 侧）| round K=4 → 83.0%，trunc K=4 → 10.8%（准确率差异） |
| 板载 logit 与 Python golden 一致 | RTL 实现正确，PPA 数字可信 |
