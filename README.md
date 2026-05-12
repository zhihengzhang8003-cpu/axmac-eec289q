# AxMAC: Approximate MAC for Multi-Precision DNN Inference

Python-Based Design and Evaluation of an Approximate MAC (AxMAC) for Multi-Precision DNN Inference.
EEC 289Q 002 SQ 2026 — Deep Learning Hardware course project.

Authors: Jiabo Zhang, Yuxuan Wang.

## Layout

```
project/
  axmac/
    exact_mac.py        # Bit-accurate INT4/8/16 + FP16/BF16/FP32 MAC (Week 1)
    approx_mac.py       # Truncated multiplier + ACA adder, parameterized by K (Week 2)
    power_model.py      # Switching-activity dynamic power model (Week 3)
    accuracy_eval.py    # RMSE/MED/max-error sweeps (Week 4)
    dnn_inference.py    # Custom torch.autograd.Function for AxMAC; CIFAR-10/MNIST (Week 5)
    pareto.py           # Full (precision, K) sweep + Pareto plots (Week 6)
  tests/                # pytest unit tests
  main.py               # End-to-end driver
  requirements.txt
```

## Supported formats

| Format | Bits | Layout                       | Source                |
|--------|------|------------------------------|-----------------------|
| INT4   | 4    | sign + 3 magnitude (two's c.)| custom                |
| INT8   | 8    | sign + 7                     | custom                |
| INT16  | 16   | sign + 15                    | custom                |
| FP16   | 16   | s1 e5 m10                    | IEEE 754 binary16     |
| BF16   | 16   | s1 e8 m7                     | bfloat16 (truncated)  |
| FP32   | 32   | s1 e8 m23                    | IEEE 754 binary32     |

## Running tests

The project directory contains a full-width colon and breaks Python startup
when used as the working directory (cwd is mistaken for `sys.prefix`).
Always run pytest from a different cwd, passing the absolute test path:

```powershell
$env:PYTHONPATH = 'E:\AIDev\EEC 289Q 002 SQ 2026：Deep Learning Hardware\project'
Set-Location $env:USERPROFILE
python -m pytest "E:\AIDev\EEC 289Q 002 SQ 2026：Deep Learning Hardware\project\tests"
```

## Dependencies

- Python 3.12+ (tested on 3.14)
- numpy >= 2.0
- pytest (dev)
- torch + torchvision (Week 5+)
