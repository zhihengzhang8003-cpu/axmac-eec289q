# 参考文献库（AxMAC 项目）

本目录收录项目中每个公式 / 模型 / 指标的论文出处。原则：**代码、proposal、README、REDESIGN 中出现的每一个公式都必须能在此表找到背景论文**。

下载来源仅限合法开放渠道（arXiv、作者主页、大学院系页、机构开放仓库）。付费墙论文未下载，列出 DOI —— 可用 UC Davis 图书馆账号（IEEE Xplore / ACM DL / Oxford Academic）获取。

> 完整逐条清单（含每篇的获取网站、本地文件名、支撑模块）见同目录 **`参考文献清单.txt`**。本 README 为索引表。
>
> 统计：共 **26 篇** —— 已下载 PDF **19 篇**，需图书馆获取 **7 篇**。PDF 已 gitignore（版权原因不入库）。

## 已下载（19 篇）

| 本地文件 | 引用 | 支撑的模块 / 公式 |
|---|---|---|
| `MacSorley_1961_HighSpeedArith.pdf` | O. L. MacSorley, "High-Speed Arithmetic in Binary Computers," *Proc. IRE*, 49(1):67–91, 1961. DOI:10.1109/JRPROC.1961.287779 | `exact_mac.py` — Booth 基-4 乘法器 `booth_radix4_pps` |
| `Kalamkar_2019_bfloat16.pdf` | D. Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," arXiv:1905.12322, 2019. | `exact_mac.py` — BF16 格式 |
| `Micikevicius_2022_FP8Formats.pdf` | P. Micikevicius et al., "FP8 Formats for Deep Learning," arXiv:2209.05433, 2022. | `exact_mac.py` — FP8 E4M3 / E5M2 格式（FP8 现代化） |
| `Kahng_2012_ACA.pdf` | A. B. Kahng, S. Kang, "Accuracy-Configurable Adder for Approximate Arithmetic Designs," *DAC*, 2012. DOI:10.1145/2228360.2228509 | `approx_mac.py` — ACA 近似加法器 `aca_add`（**baseline**） |
| `Gupta_2015_LimitedPrecision.pdf` | S. Gupta, A. Agrawal, K. Gopalakrishnan, P. Narayanan, "Deep Learning with Limited Numerical Precision," *ICML*, 2015. arXiv:1502.02551 | **贡献 A** — `stochastic` 舍入模式（去偏技术来源） |
| `Croci_2022_StochasticRounding.pdf` | M. Croci, M. Fasi, N. J. Higham, T. Mary, M. Mikaitis, "Stochastic rounding: implementation, error analysis and applications," *R. Soc. Open Sci.*, 9(3):211631, 2022. DOI:10.1098/rsos.211631 | **贡献 A** — `stochastic` 模式误差分析与硬件代价 |
| `Chandrakasan_1992_LowPowerCMOS.pdf` | A. P. Chandrakasan, S. Sheng, R. W. Brodersen, "Low-Power CMOS Digital Design," *IEEE JSSC*, 27(4):473–484, 1992. DOI:10.1109/4.126534 | `power_model.py` — 动态功耗 α·C·V²·f |
| `Najm_1994_PowerEstSurvey.pdf` | F. N. Najm, "A Survey of Power Estimation Techniques in VLSI Circuits," *IEEE T-VLSI*, 2(4):446–455, 1994. DOI:10.1109/92.335013 | `power_model.py` — 开关活动功耗估计 |
| `Liang_2013_ApproxAdderMetrics.pdf` | J. Liang, J. Han, F. Lombardi, "New Metrics for the Reliability of Approximate and Probabilistic Adders," *IEEE T-Computers*, 62(9):1760–1771, 2013. DOI:10.1109/TC.2012.146 | `accuracy_eval.py` — MED / NMED 误差指标定义 |
| `Jacob_2018_IntegerQuant.pdf` | B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018. arXiv:1712.05877 | `dnn_inference.py` — 整数量化 |
| `Bengio_2013_STE.pdf` | Y. Bengio, N. Léonard, A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons," arXiv:1308.3432, 2013. | `dnn_inference.py` — 直通梯度估计（STE 反向） |
| `Mrazek_2019_ALWANN.pdf` | V. Mrazek, Z. Vasicek, L. Sekanina, M. A. Hanif, M. Shafique, "ALWANN: Automatic Layer-Wise Approximation of DNN Accelerators without Retraining," *ICCAD*, 2019. arXiv:1907.07229 | **贡献 B** — 层级非均匀近似分配（直接对标） |
| `Dong_2019_HAWQ.pdf` | Z. Dong, Z. Yao, A. Gholami, M. Mahoney, K. Keutzer, "HAWQ: Hessian AWare Quantization with Mixed-Precision," *ICCV*, 2019. arXiv:1905.03696 | **贡献 B** — 逐层敏感度分析 |
| `Yao_2021_HAWQv3.pdf` | Z. Yao, Z. Dong, Z. Zheng, A. Gholami, et al., "HAWQ-V3: Dyadic Neural Network Quantization," *ICML*, 2021. arXiv:2011.10680 | **贡献 B** — 整数化混合精度敏感度分配 |
| `Wang_2019_HAQ.pdf` | K. Wang, Z. Liu, Y. Lin, J. Lin, S. Han, "HAQ: Hardware-Aware Automated Quantization with Mixed Precision," *CVPR*, 2019. arXiv:1811.08886 | **贡献 B** — 混合精度预算分配 |
| `Frantar_2023_GPTQ.pdf` | E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," *ICLR*, 2023. arXiv:2210.17323 | 引言 — FP8 / LLM 量化动机 |
| `Xiao_2023_SmoothQuant.pdf` | G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, S. Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," *ICML*, 2023. arXiv:2211.10438 | 引言 — LLM 量化动机 |
| `Lin_2024_AWQ.pdf` | J. Lin, J. Tang, H. Tang, S. Yang, et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," *MLSys*, 2024. arXiv:2306.00978 | 引言 — LLM 量化动机 |
| `Armeniakos_2022_ApproxDNNSurvey.pdf` | G. Armeniakos, G. Zervakis, D. Soudris, J. Henkel, "Hardware Approximate Techniques for Deep Neural Network Accelerators: A Survey," *ACM Computing Surveys*, 55(4):83, 2022. arXiv:2203.08737 | 引言 — 近似 DNN 加速器最新综述 |

## 需图书馆获取（7 篇，付费墙 / 暂不可用）

| 引用 | DOI / 链接 | 支撑的模块 / 公式 |
|---|---|---|
| A. D. Booth, "A Signed Binary Multiplication Technique," *Q. J. Mech. Appl. Math.*, 4(2):236–240, 1951. | 10.1093/qjmam/4.2.236（Oxford Academic） | `exact_mac.py` — Booth 编码原始文献 |
| H. R. Mahdiani, A. Ahmadi, S. M. Fakhraie, C. Lucas, "Bio-Inspired Imprecise Computational Blocks...," *IEEE TCAS-I*, 57(4):850–862, 2010. | 10.1109/TCSI.2009.2027626（IEEE） | `approx_mac.py` — 截断乘法器（**贡献 A 的 baseline**） |
| M. J. Schulte, E. E. Swartzlander, "Truncated Multiplication with Correction Constant," *VLSI Signal Processing VI*, 1993. | 10.1109/VLSISP.1993.404467（IEEE） | **贡献 A** — 截断 + 补偿常数的原始思路（`round` 模式） |
| M. Horowitz, "Computing's Energy Problem (and what we can do about it)," *ISSCC*, 2014. | 10.1109/ISSCC.2014.6757323（IEEE） | `power_model.py` — 45 nm 基准每操作能耗 |
| S. Mittal, "A Survey of Techniques for Approximate Computing," *ACM Computing Surveys*, 48(4):62, 2016. | 10.1145/2893356（ACM） | 引言 / 近似计算综述背景 |
| S. Hashemi, R. I. Bahar, S. Reda, "DRUM: A Dynamic Range Unbiased Multiplier for Approximate Applications," *ICCAD*, 2015. | 10.1109/ICCAD.2015.7372600（IEEE） | **贡献 A** — 无偏乘法器使误差在累加中抵消（核心论据） |
| K. Chellapilla, S. Puri, P. Simard, "High Performance Convolutional Neural Networks for Document Processing," *IWFHR*, 2006. | HAL: inria-00112631（HAL 暂返回 500/HTML，可稍后重试） | `dnn_inference.py` — im2col 卷积 |

## 备注

- IEEE 754 浮点（`exact_mac.py` 的 FP 编解码）依据 **IEEE Std 754-2019** 标准文档，非论文。
- **承重文献**：Mahdiani / Kahng 是项目复现的 baseline；Schulte / DRUM / Gupta / Croci 支撑改进 A；ALWANN / HAWQ / HAWQ-V3 / HAQ 支撑改进 B；Liang-Han-Lombardi 为误差指标正名；Micikevicius 支撑 FP8 现代化。
- 卷号 / 页码为整理所得，正式提交前建议按 DOI 核对。
