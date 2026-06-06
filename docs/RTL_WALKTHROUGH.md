# AxMAC RTL 代码讲解

## 模块结构

```
rtl/
  src/
    drum_multiplier.v   — DRUM-k 近似乘法器（零均值误差）
    mac_unit.v          — 近似 MAC 单元（trunc / round / stochastic 三种模式）
    lfsr.v              — 随机数源（stochastic rounding 专用）
    aca_adder.v         — 分段进位近似加法器
    mac_array.v         — R×C 输出静止型 MAC 阵列
    mlp_top.v           — 玩具 64→16→10 MLP 顶层（FSM 控制）
    mlp_top_demo.v      — 板级 wrapper（LED 显示 + UART 输出）
    uart_tx.v           — 8N1 UART 发送器
    uart_framer.v       — logit 打包帧发送器
  tb/
    tb_drum_multiplier.sv — DRUM 乘法器 testbench（读 CSV 逐向量比对）
    tb_mac_unit.sv        — MAC 单元 testbench（确定性 + 随机统计检验）
    tb_aca_adder.sv       — ACA 加法器 testbench
    tb_mac_array.sv       — MAC 阵列结构性测试
    tb_mlp_top.sv         — MLP 端到端测试
    tb_mlp_top_demo.sv    — 板级 wrapper 仿真
    run_tests.py          — 统一测试驱动（Icarus / ModelSim 后端）
  golden/
    export_golden.py      — 生成所有 CSV golden 向量（Python 侧）
    gen_drum_golden.py    — 生成 drum_int8.csv
    mac_int8.csv          — mac_unit 的 bit-exact 参考
    aca.csv               — aca_adder 的参考
    mlp_toy/              — MLP 权重 + 参考输出（.mem + .csv）
  vendor/altera/
    run_K*.tcl            — Quartus 各配置点编译脚本
    ppa_sweep.tcl         — PPA 设计空间扫描脚本
```

RTL 层的核心逻辑和 Python 层完全对称：每个 `.v` 文件都是对应 `axmac/` Python 函数的硬件实现，golden CSV 是 Python 计算的结果，testbench 逐比特验证两者一致。

---

## Step 1：DRUM-k 乘法器 — `src/drum_multiplier.v`

这是 `approx_mac.drum_multiply`（`approx_mac.py:203`）的硬件实现。

### 信号流

```
a[7:0], b[7:0]
  ↓ 取绝对值（两补数取反加一）
abs_a, abs_b
  ↓ 找 MSB 位置（leading_bit 函数）
msb_a, msb_b
  ↓ 算右移量：shift = max(0, msb - K_DRUM + 1)
shift_a, shift_b
  ↓ 右移再左移（清除低位）
drum_abs_a, drum_abs_b
  ↓ 无符号乘法
mag_product
  ↓ 按 sign_out 决定是否取反
product[15:0]
```

### 关键实现细节

**LZC（Leading-bit Counter）— `drum_multiplier.v:53`：**

```verilog
function automatic [LOG2N-1:0] leading_bit;
    input [N_BITS-1:0] x;
    integer i;
    begin
        leading_bit = 0;
        for (i = 0; i < N_BITS; i = i + 1)
            if (x[i]) leading_bit = i[LOG2N-1:0];
    end
endfunction
```

遍历每一位，最后一个置 1 的位的下标就是 MSB。综合器会优先级编码这段逻辑，实际实现为编码器而不是真正的循环。

**移位量计算 — `drum_multiplier.v:74`：**

```verilog
wire signed [LOG2N:0] shift_a_signed = $signed({1'b0, msb_a}) - K1;
wire [LOG2N-1:0] shift_a = shift_a_signed[LOG2N] ? {LOG2N{1'b0}}
                                                   : shift_a_signed[LOG2N-1:0];
```

用有符号减法计算 `msb - (K_DRUM-1)`，符号位为 1 表示结果为负（即 MSB 本来就 < K_DRUM 位），此时饱和为 0（不移位）。

**DRUM 量化 — `drum_multiplier.v:86`：**

```verilog
wire [N_BITS-1:0] drum_abs_a = (abs_a >> shift_a) << shift_a;
```

右移再左移，效果是把低 `shift_a` 位清零，只保留最高 K_DRUM 位。与 Python 侧 `(ax >> shift) << shift`（`approx_mac.py:200`）完全对称。

**符号恢复 — `drum_multiplier.v:97`：**

```verilog
wire [2*N_BITS-1:0] neg_product = ~mag_product + 1'b1;
assign product = sign_out ? $signed(neg_product) : $signed(mag_product);
```

乘法只对幅值做，最后按 `sign_a XOR sign_b` 决定是否取反。这是 sign-magnitude 乘法器的标准结构。

**与 Python 的对应关系：**
- `drum_quantize_operand`（`approx_mac.py:183`）= LZC + shift + 清低位
- `drum_multiply`（`approx_mac.py:203`）= 量化双操作数 + 乘法 + 符号合并

---

## Step 2：近似 MAC 单元 — `src/mac_unit.v`

这是 `approx_mac._apply_rounding(a*b, K, mode)`（`approx_mac.py:80`）的硬件实现。

### 接口

```verilog
module mac_unit #(
    parameter integer A_BITS = 8,
    parameter integer B_BITS = 8
) (
    input  wire signed [A_BITS-1:0]        a,
    input  wire signed [B_BITS-1:0]        b,
    input  wire        [3:0]               K,      // 截断位数 0..15
    input  wire        [1:0]               mode,   // 00 trunc, 01 round, 10 stoch
    input  wire        [A_BITS+B_BITS-1:0] rnd,    // stoch 模式用的随机 offset
    output wire signed [A_BITS+B_BITS-1:0] product_rounded
);
```

`rnd` 来自外部（LFSR 或 testbench 注入），使这个模块保持纯组合逻辑，testbench 可以精确控制随机输入做 bit-exact 测试。

### 实现逻辑 — `mac_unit.v:49`

```verilog
wire signed [P_BITS-1:0] product_full = $signed(a) * $signed(b);

// mask = (1 << K) - 1，K=0 时 mask=0，~mask=全1，AND 无效果（精确）
wire [P_BITS-1:0] one_shifted = 1'b1 << K;
wire [P_BITS-1:0] mask        = one_shifted - 1;

// round 的修正常数：2^(K-1)
wire [P_BITS-1:0] round_const = (K == 0) ? 0 : (1'b1 << (K - 1));

// stochastic 的 offset：rnd 的低 K 位
wire [P_BITS-1:0] stoch_offset = rnd & mask;
```

三种 offset 通过 `case(mode)` 选择（`mac_unit.v:68`），加到乘积后清低 K 位：

```verilog
assign product_rounded = product_offset & ~mask;
```

**与 Python 的对应：**
- `product_full` = `sum(booth_radix4_pps(a,b,n))`（结果相同，实现不同——RTL 由综合器选择乘法器架构）
- `product_offset & ~mask` = `product & ~((1<<K)-1)`（`approx_mac.py:106`）
- `round_const` = `1 << (K-1)`（`approx_mac.py:103`）
- `stoch_offset` = `rng.randrange(1<<K)` 由外部 LFSR 提供

**K=0 的退化路径：** `mask=0`，`round_const=0`，`stoch_offset=0`，三种 mode 的 `offset` 均为 0，`product_rounded = product_full & ~0 = product_full`，精确乘积。

---

## Step 3：LFSR 随机源 — `src/lfsr.v`

stochastic rounding 需要每个 MAC 周期提供一个新的随机 offset，这正是本项目**贡献 A 的硬件代价**所在。

```verilog
module lfsr #(
    parameter integer WIDTH = 64,
    parameter [63:0]  SEED  = 64'hDEAD_BEEF_CAFE_BABE
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,         // 只在 MAC 活跃周期推进
    output reg  [WIDTH-1:0] state
);

// Fibonacci LFSR，多项式 x^64 + x^63 + x^61 + x^60 + 1
wire feedback = state[63] ^ state[62] ^ state[60] ^ state[59];

always @(posedge clk) begin
    if (!rst_n) state <= SEED[WIDTH-1:0];
    else if (en) state <= {state[WIDTH-2:0], feedback};
end
```

64-bit 最大长度 LFSR，周期 `2^64 - 1`，远超任何一次推理的 MAC 数量。

**`en` 的作用：** 只在 `array_mac_en=1` 的周期推进 LFSR，保证序列只在真正做 MAC 的周期消耗，非活跃周期（clear / drain / IDLE）不推进，使结果对给定调度可重现。

**`mlp_top.v:153` 的条件实例化：**

```verilog
generate
    if (MODE == 2) begin : g_lfsr          // 只有 stochastic 模式才实例化
        wire [63:0] lfsr_state;
        lfsr #(.WIDTH(64)) u_lfsr (...);
        assign array_rnd_flat = lfsr_state[C*P_BITS-1:0];
    end else begin : g_no_lfsr
        assign array_rnd_flat = {C*P_BITS{1'b0}};   // trunc/round：rnd 接 0
    end
endgenerate
```

`generate if (MODE==2)` 使 trunc / round 的比特流里不含 LFSR 逻辑——这正是贡献 A 量化的硬件代价差异：Quartus PowerPlay 对 MODE=1（round）和 MODE=2（stochastic）的功耗报告差值就是这个 LFSR 的代价。

---

## Step 4：ACA 加法器 — `src/aca_adder.v`

这是 `aca_add`（`approx_mac.py:109`）的硬件实现。

```verilog
module aca_adder #(
    parameter integer BITS   = 32,
    parameter integer WINDOW = 4
) (
    input  wire signed [BITS-1:0] a,
    input  wire signed [BITS-1:0] b,
    output wire signed [BITS-1:0] sum
);
```

**WINDOW 是编译期参数，不是运行时端口。** 这个设计决策和 Pareto 分析的方式对应：每个设计点对应一个独立的 RTL 实例，不同 W 的 ACA 在芯片上是物理上不同的加法器 IP，而不是一个可配置的模块。

```verilog
generate
    if (WINDOW >= BITS) begin : g_exact
        assign out = a + b;         // 精确加法
    end else begin : g_segmented
        for (i = 0; i < N_FULL; i = i + 1) begin : g_full
            // 每段独立相加，不带 carry-in
            assign out[i*WINDOW +: WINDOW] =
                a[i*WINDOW +: WINDOW] + b[i*WINDOW +: WINDOW];
        end
        if (TAIL > 0) begin : g_tail  // 最后不足 WINDOW 宽的尾段
            assign out[BITS-1 -: TAIL] =
                a[BITS-1 -: TAIL] + b[BITS-1 -: TAIL];
        end
    end
endgenerate
```

`g_full` 内每个 `assign` 是一个独立的 WINDOW 位加法器，段间无任何连接。硬件上进位链截断在段内，延迟从 32-bit 链缩短为 WINDOW 位链，关键路径缩短 `(BITS/WINDOW)` 倍。

**testbench 的 4 路并联实例化（`tb_aca_adder.sv:21`）：**

```verilog
aca_adder #(.BITS(32), .WINDOW(4))  dut_w4  (...);
aca_adder #(.BITS(32), .WINDOW(8))  dut_w8  (...);
aca_adder #(.BITS(32), .WINDOW(16)) dut_w16 (...);
aca_adder #(.BITS(32), .WINDOW(32)) dut_w32 (...);
```

四个 WINDOW 的 DUT 同时驱动相同 `(a, b)`，CSV 里的 `window` 列决定比对哪个输出——一次仿真验证全部配置点。

---

## Step 5：MAC 阵列 — `src/mac_array.v`

将 `mac_unit` 和 `aca_adder` 组合成 R×C 输出静止型阵列。

### 拓扑

```
         wgts[0]   wgts[1]   wgts[2]   wgts[3]
acts[0] [ PE(0,0)  PE(0,1)  PE(0,2)  PE(0,3) ]
acts[1] [ PE(1,0)  PE(1,1)  PE(1,2)  PE(1,3) ]
acts[2] [ PE(2,0)  PE(2,1)  PE(2,2)  PE(2,3) ]
acts[3] [ PE(3,0)  PE(3,1)  PE(3,2)  PE(3,3) ]
```

- 行广播：`acts[r]` 广播给同行所有 PE
- 列广播：`wgts[c]` 广播给同列所有 PE
- 每个 PE = `mac_unit` + `aca_adder` + 一个 `acc_r` 寄存器

### 每个 PE 的逻辑 — `mac_array.v:65`

```verilog
mac_unit #(...) u_mac (
    .a(a_r), .b(b_c), .K(K), .mode(mode), .rnd(rnd_rc),
    .product_rounded(prod)
);

// 符号扩展：P_BITS → ACC_BITS
wire signed [ACC_BITS-1:0] prod_ext = {{(ACC_BITS-P_BITS){prod[P_BITS-1]}}, prod};

reg signed [ACC_BITS-1:0] acc_r;
aca_adder #(.BITS(ACC_BITS), .WINDOW(WINDOW)) u_add (
    .a(acc_r), .b(prod_ext), .sum(sum_w)
);

always @(posedge clk) begin
    if (!rst_n)     acc_r <= 0;
    else if (clear) acc_r <= 0;
    else if (mac_en) acc_r <= sum_w;   // 累加
end
```

`clear` 和 `mac_en` 直接来自顶层 FSM 的组合逻辑（不经寄存器），确保 `clear=1` 的同一个 posedge 就清零，下一个 cycle 的 `mac_en=1` 时累加器已经是 0。

**端口扁平化：** 所有端口都是宽总线（`R*A_BITS` 宽），内部用 generate-for 切片。这是为了兼容旧版仿真器（不支持解包数组端口）。

---

## Step 6：MLP 顶层 FSM — `src/mlp_top.v`

将 mac_array 封装成完整的两层 MLP 推理，用 FSM 控制数据流。

### 网络结构

64 输入 → 16 隐藏（ReLU）→ 10 输出，权重和激活均为 INT8。  
`C=4`：每次处理 4 个输出神经元（一个 tile）。

### 存储器

```verilog
// mlp_top.v:72
reg [A_BITS-1:0] act_mem [0:L0_IN-1];    // 输入 X，只读
reg [A_BITS-1:0] h_mem   [0:L0_OUT-1];   // 隐藏层激活，layer-0 drain 写入
reg [A_BITS-1:0] w0_mem  [0:L0_IN*L0_OUT-1];
reg [A_BITS-1:0] w1_mem  [0:L1_IN*L1_OUT-1];
reg [A_BITS-1:0] b0_mem  [0:L0_OUT-1];
reg [A_BITS-1:0] b1_mem  [0:L1_OUT-1];
```

`act_mem` 和 `h_mem` 分开是关键：layer-0 有多个 tile，第 2 个 tile 运行时仍然需要读原始 `act_mem`，如果复用同一块内存写入激活就会破坏后续 tile 的输入。

### FSM 状态机 — `mlp_top.v:101`

```
IDLE → L0_INIT → L0_STR(L0_IN cycles) → L0_DRN
                                              ↓ tile_cnt < L0_NTILES
                                         L0_INIT（下一 tile）
                                              ↓ tile_cnt == L0_NTILES-1
                                         L1_INIT → L1_STR(L1_IN cycles) → L1_DRN
                                                                                ↓
                                                                            DONE
```

- `L0_INIT / L1_INIT`：单周期 clear，`array_clear=1` 清零 PE 累加器
- `L0_STR / L1_STR`：`mac_en=1`，`k_cnt` 每周期递增，流入一列输入激活和对应权重
- `L0_DRN / L1_DRN`：读出 4 个 PE 的累加结果，加 bias，layer-0 做 ReLU+饱和写入 `h_mem`，layer-1 直接写 `y_reg`

**`array_clear` 是组合信号 — `mlp_top.v:118`：**

```verilog
wire array_clear  = (state == S_L0_INIT) || (state == S_L1_INIT);
wire array_mac_en = (state == S_L0_STR)  || (state == S_L1_STR);
```

这两个信号是纯组合逻辑，直接由 `state` 驱动，不经过寄存器。如果改成寄存器赋值（`<=`），INIT 状态的 clear 会延迟一个周期，导致第一个 MAC（k_cnt=0）在累加器尚未清零时就执行，产生错误结果。注释里明确记录了这个 timing 约束。

### ReLU + 饱和 — `mlp_top.v:206`

```verilog
assign drained_l0_relu[gc] =
    (drained_l0[gc] <= 0)      ? {A_BITS{1'b0}} :   // 负数 → 0
    (drained_l0[gc] > A_MAX_S) ? A_MAX_S :           // 溢出 → +127
                                  drained_l0[gc][A_BITS-1:0];
```

与 Python 侧 `np.clip(h, 0, fmt.max_val)`（`dnn_inference.py:355`）语义相同。

### 推理时序

默认参数（C=4, L0_IN=64, L0_OUT=16, L1_OUT=10）下：
- Layer-0：4 tiles × (1 clear + 64 MAC + 1 drain) = 264 cycles
- Layer-1：3 tiles × (1 clear + 16 MAC + 1 drain) = 54 cycles
- 合计约 **320 cycles**，50 MHz 下约 **6.4 μs**

---

## Step 7：板级 Wrapper — `src/mlp_top_demo.v`

在野火征途 Pro（EP4CE10）开发板上跑 demo 的顶层模块。

```verilog
mlp_top #(.K_PARAM(K_PARAM), .MODE(MODE), .ACA_W(ACA_W), ...) u_mlp (
    .start(1'b1),   // 拉高 start：推理完成后停在 DONE 状态，结果保持
    ...
);
```

**组合 argmax — `mlp_top_demo.v:73`：**

```verilog
always @* begin
    amax = 4'd0;
    best = $signed(result_flat[0 +: ACC_BITS]);
    for (i = 1; i < L1_OUT; i = i + 1) begin
        cand = $signed(result_flat[i*ACC_BITS +: ACC_BITS]);
        if (cand > best) begin best = cand; amax = i[3:0]; end
    end
end
assign led_class = (LED_ACTIVE_LOW != 0) ? ~amax : amax;
```

遍历 10 个 logit，找最大值的下标，输出到 4 个 LED（野火 LED 低电平点亮，所以取反）。

---

## Step 8：UART 输出链 — `src/uart_tx.v` + `src/uart_framer.v`

### uart_tx — 8N1 发送器

```verilog
localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;  // 50M/115200 ≈ 434
```

4 状态 FSM：`IDLE → START（起始位）→ DATA（8 个数据位）→ STOP`。  
valid/ready 握手：`ready` 低电平期间表示正在发送，不接受新数据。

### uart_framer — logit 帧打包

推理完成（`done` 上升沿）时，打包 41 字节发送：

```
[0xAB] [logit[0] 低字节] [logit[0] 次低字节] ... [logit[9] 高字节]
```

```verilog
// uart_framer.v:36
if (!sending && done_rise) begin
    buf_r[0] <= HEADER;                              // 0xAB 帧头
    for (k = 0; k < NBYTES; k = k + 1)
        buf_r[k+1] <= result_flat[k*8 +: 8];        // 10 × 32bit = 40 字节
    sending <= 1'b1;
end
```

PC 端的 `read_uart.py` 读这个协议，解析出 10 个有符号 32-bit logit，做 argmax 验证。

---

## Step 9：Golden 向量生成 — `golden/export_golden.py`

这是 Python 层和 RTL 层的桥梁：Python 计算参考结果，RTL testbench 对比。

### 四类 CSV

**1. `mac_int8.csv` — `export_golden.py:69`**

```python
for a, b in pairs:
    product_full = sum(booth_radix4_pps(a, b, INT8.bits))
    for K in range(7):
        for mode in ("trunc", "round"):
            rounded = _apply_rounding(product_full, K, mode, None)
            w.writerow([a, b, K, MODE_CODE[mode], product_full, rounded])
```

覆盖边界值（-128, -1, 0, 1, 127）× 边界值的笛卡儿积 + 步长 32 的网格 + 200 个随机对，确保 corner case 覆盖。mode 用数字编码（0/1/2）而不是字符串，因为 ModelSim 10.5b 的 `$sscanf` 不支持 `%[^,]` 格式。

**2. `mac_int8_stoch.csv` — `export_golden.py:89`**

stochastic 模式的 Python 侧用 `random.Random`，RTL 侧用 LFSR，两者随机序列不同，所以**不做 bit-exact 比对**，只做统计检验（均值 / RMSE）。

**3. `aca.csv` — `export_golden.py:112`**

```python
for window in (4, 8, 16, 32):
    for _ in range(256):
        s = aca_add(a, b, bits, window)
        w.writerow([a, b, bits, window, s])
```

直接调用 Python 侧 `aca_add`，结果作为 RTL 的 bit-exact 参考。

**4. `mlp_toy/` — `export_golden.py:151`**

```python
layers = [(w0, b0), (w1, b1)]
for K in (0, 2, 4):
    for mode in ("trunc", "round"):
        y = tiny_mlp_forward(x, layers, fmt=INT8, K=K, rounding=mode)
        _write_int_matrix(bundle_dir / f"y_K{K}_{mode}.csv", y)
```

同一套权重，Python 侧跑 6 种配置的参考输出（K=0/2/4 × trunc/round）；RTL testbench（`tb_mlp_top.sv`）用相同权重（`.mem` 文件），比对 `done=1` 时 `result_flat` 是否和 CSV 一致。

**`gen_drum_golden.py`** 单独生成 `drum_int8.csv`：200 个随机 INT8 对，调用 `drum_multiply(a, b, INT8, k=4)` 计算参考值。

---

## Step 10：Testbench 策略

### 三种验证类型

| testbench | DUT | 验证方式 | 关键指标 |
|---|---|---|---|
| `tb_drum_multiplier.sv` | `drum_multiplier` | 逐向量 bit-exact | 200 vectors, 0 fail |
| `tb_mac_unit.sv` | `mac_unit` | deterministic: bit-exact；stochastic: 统计 | 5σ 均值检验 |
| `tb_aca_adder.sv` | `aca_adder` | 逐向量 bit-exact（4 个 DUT 并联） | 1024 vectors |
| `tb_mac_array.sv` | `mac_array` | 结构性：K=0 exact 矩阵乘对比内联参考 | R×C=16 PE |
| `tb_mlp_top.sv` | `mlp_top` | 端到端：对比 Python golden CSV | 6 种配置 |

### stochastic 的统计检验 — `tb_mac_unit.sv:126`

```verilog
for (K_i = 1; K_i <= 6; K_i = K_i + 1) begin
    sum_err = 0;
    for (i = 0; i < 2000; i = i + 1) begin
        mode = 2'b10;   rnd = $urandom();
        // 等待 1ns 稳定
        err_i = $signed(product_rounded) - (a_i * b_i);
        sum_err = sum_err + err_i;
    end
    mean_err = sum_err / 2000.0;
    sigma    = (1 << K_i) / $sqrt(12.0 * 2000.0);
    tol      = 5.0 * sigma;
    if ((mean_err > tol) || (mean_err < -tol)) $fatal(1);
end
```

对每个 K 值，2000 次随机样本的误差均值必须在 ±5σ 内（σ 由均匀分布方差公式推导）。这验证了 Gupta 2015 的无偏性：stochastic rounding 的误差均值应为零。

### mac_array 的结构性测试 — `tb_mac_array.sv:73`

```verilog
// 1. 生成随机激活和权重，计算参考 dot product（纯整数，不调用 DUT）
ref_acc[r][c] += acts_seq[t][r] * wgts_seq[t][c];

// 2. 逐周期驱动 DUT
for (t = 0; t < K_DIM; t++) @(posedge clk);

// 3. 比对每个 PE 的累加值
got_rc = $signed(acc_flat[(r*C + c)*ACC_BITS +: ACC_BITS]);
if (got_rc !== ref_acc[r][c]) $fatal(1);
```

这是**结构性测试**：验证 R×C 的广播连接、generate-for 的 PE 索引、`clear`/`mac_en` 时序是否正确。rounding 正确性已由 `tb_mac_unit` 覆盖，所以这里用 K=0（精确模式）排除 rounding 误差的干扰。

---

## Step 11：测试驱动 — `tb/run_tests.py`

统一的测试入口，支持 Icarus Verilog 和 ModelSim 两个后端。

```python
TESTS = {
    "mac_unit":  {"sources": [mac_unit.v, tb_mac_unit.sv], "toplevel": "tb_mac_unit"},
    "aca_adder": {"sources": [aca_adder.v, tb_aca_adder.sv], "toplevel": "tb_aca_adder"},
    "mac_array": {"sources": [mac_unit.v, aca_adder.v, mac_array.v, tb_mac_array.sv], ...},
    "mlp_top":   {"sources": [..., lfsr.v, mlp_top.v, tb_mlp_top.sv], ...},
    ...
}
```

**ModelSim 路径问题（`run_tests.py:22`）：**

```
Path caveat: ModelSim-Altera 10.5b 拒绝进入含有全角冒号 U+FF1A 的路径。
本项目路径 "EEC 289Q 002 SQ 2026：Deep Learning Hardware" 正好命中这个 bug。
解决方法：用 PowerShell 创建 junction 到 ASCII 路径，再从 junction 内运行。
```

脚本里特意用 `.absolute()` 而不是 `.resolve()`，因为 `resolve()` 会跟随 junction 回到原始 Unicode 路径，破坏这个 workaround。

**Icarus 后端：**

```python
cmd = ["iverilog", "-g2012", "-o", vvp_path, "-s", spec["toplevel"]] + sources
subprocess.call(cmd)
subprocess.call(["vvp", str(vvp_path)], cwd=str(TB_DIR))  # cwd=tb/ 使相对路径生效
```

Icarus 没有 Unicode 路径限制，直接用。`cwd=TB_DIR` 保证 testbench 里 `"../golden/..."` 的相对路径能正确解析。

---

## Step 12：Quartus 设计空间扫描 — `vendor/altera/`

每个 `run_K*.tcl` 对应 Pareto 曲线上的一个设计点，用 Quartus 综合 + 布局布线，通过 PowerPlay 拿到功耗数字。

```
run_K0_trunc.tcl  — K=0, mode=trunc（精确 baseline）
run_K2_trunc.tcl  — K=2, mode=trunc
run_K4_trunc.tcl  — K=4, mode=trunc（DNN 准确率大幅下降的配置）
run_K4_round.tcl  — K=4, mode=round（DNN 准确率保持的配置，贡献 A 的核心对比点）
run_K6_trunc.tcl  — K=6, mode=trunc（激进截断）
```

`ppa_sweep.tcl`：自动化地跑全部配置，收集 PPA（Power/Performance/Area）数字，生成论文里的 Table II。

---

## RTL ↔ Python 对应关系总结

| RTL 模块 | Python 对应 | 验证方式 |
|---|---|---|
| `drum_multiplier.v` | `approx_mac.drum_multiply` | `tb_drum_multiplier` ← `gen_drum_golden.py` |
| `mac_unit.v`（trunc/round）| `approx_mac._apply_rounding` | `tb_mac_unit` ← `export_golden.py:mac_int8.csv` |
| `mac_unit.v`（stochastic）| `approx_mac._apply_rounding(..., "stochastic")` | `tb_mac_unit` 统计检验（非 bit-exact）|
| `lfsr.v` | `random.Random.randrange` | 统计等价，序列不同 |
| `aca_adder.v` | `approx_mac.aca_add` | `tb_aca_adder` ← `export_golden.py:aca.csv` |
| `mac_array.v` | `dnn_inference.int_matmul_approx` | `tb_mac_array` 结构性测试 |
| `mlp_top.v` | `dnn_inference.tiny_mlp_forward` | `tb_mlp_top` ← `export_golden.py:mlp_toy/*.csv` |
| `uart_framer.v` + `uart_tx.v` | `read_uart.py` | 板载运行时验证 |
