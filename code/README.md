# Diagonal Unitary Synthesis — Baseline Reproduction

复现开题报告中的三套基线（精确综合 / 前辈 MCTS / 相位重要性图路径搜索）。

## Conda 环境（`bishe-diag`）

本项目使用独立 conda 环境，不污染 base：

```bash
# 若尚未创建
conda create -y -n bishe-diag python=3.12 numpy=2.0 pytest

conda activate bishe-diag
cd code
pip install -e .
pytest -q
```

也可：

```bash
conda env create -f environment.yml
conda activate bishe-diag
cd code && pip install -e . && pytest -q
```

## 快速运行

```bash
conda activate bishe-diag
cd code

python experiments/run_exact.py --qubits 5,8
python experiments/run_mcts.py --qubits 5 --reduction 0.1
python experiments/run_path_search.py --qubits 8 --reduction 0.1

python experiments/compare_baselines.py --qubits 5,8 --reductions 0.05,0.1,0.2 --trials 3 --out results/baselines.csv
```

更大比特（10/12）更慢；可用 `--skip-mcts-above 10` 跳过大规模 MCTS。

## 模块

| 路径 | 作用 |
|------|------|
| `diagonal_synth/walsh.py` | λ ↔ α（λ = Hα / 2^n，α = Hλ） |
| `diagonal_synth/metrics.py` | D(U_C,U_T)、效用比、CNOT 预算 |
| `diagonal_synth/gadgets.py` | Hamming 邻接、Gray 路径 |
| `diagonal_synth/exact.py` | 精确综合 |
| `diagonal_synth/mcts.py` | 前辈 MCTS |
| `diagonal_synth/path_search.py` | arXiv Algorithm 1–6 + top-k 支撑 |
| `diagonal_synth/optimize.py` | 搜索后梯度精修 α |

## 协议约定

- 随机实例：**先采样 α ~ Uniform(-π, π)**，再 `λ = Hα / 2^n`（与 Table 3 量级一致）
- 精确基线 CNOT 数按论文取 **2^n**
- 预算 `C = floor(2^n * (1 - ReCNOT))`
- 效用比 = CNOT 节省比例 / 误差

## 复现笔记（第一阶段）

**验收**：量级对齐，不抠死 Table 3 每一格。

本地冒烟（`bishe-diag`）：

| 方法 | n | ReCNOT | Error | Utility | 备注 |
|------|---|--------|-------|---------|------|
| path_search | 8 | 10% | ~0.027 | ~3.8 | Table 3 ≈ 0.035 / ~3.2 |
| path_search | 10 | 10% | ~0.030 | ~3.3 | Table 3 ≈ 0.032 |
| exact | 5/8 | 0 | ~0 | — | 往返误差 < 1e-12 |
| mcts | 5–8 | 10–20% | 更高 | 更低 | 可跑通；弱于 path_search |

**超参默认**

- MCTS：`β1=β2=β3=1.0`，`max_expand` 随预算缩放
- path_search：`γ=10`；ω 扫 0.01–0.5；支撑集取 top-(C+1) \|α\|（论文相位重要性叠加结论）

**已知偏差来源**

- 无前辈原始代码，MCTS 扩展/剪枝细节可能不同
- Walsh 公式在论文正文与实验量级之间以 Table 1/3 为准锁定
- 严格逐格对齐 Table 3、n=15 留中期再收紧

## 参考

- 《NISQ时期下针对对角线酉矩阵的近似综合算法研究》
- Zhang et al., Approximate Quantum Circuit Synthesis for Diagonal Unitary, arXiv:2412.01869
