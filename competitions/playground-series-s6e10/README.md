# Predicting Airline Satisfaction

比赛：[Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)。目标为预测满意度正类概率，以 ROC-AUC 评估；官方数据中 `True=1`。

## 当前正式版本 v12

已提交版本的 public ROC-AUC 为 **0.96118**，2026-10-19 05:28 UTC 已核实排名 **676/2302**。这是历史快照，尚未达到前 500。

[v12 说明](optimization/V12.md)包含冻结配方、验证局限、官方成绩和 CSV 精确回放入口。回放需要本机保留的完整产物，使用一个 CPU 数学线程，无需训练或 GPU。

完整 v12 训练及后续实验代码保留在[研究分支](https://github.com/siddhartha-yz/kaggle/tree/codex/s6e10-local-confirmation/competitions/playground-series-s6e10/optimization)。main 提供产物校验和回放；研究归档中的完整环境与数据依赖仍是原始数据重训的前提。

## 早期 CatBoost 基线

`baseline.py`、`config.json`、`requirements.txt` 和 `tests/test_baseline.py` 保留早期单模型学习实现。它采用固定分层 holdout 和 early stopping；v12 的训练过程见研究归档。

在隔离环境准备 [requirements.txt](requirements.txt) 中的依赖，数据放在被 Git 忽略的 `data/`。从本目录执行：

```bash
python -m unittest discover -s tests -p test_baseline.py
python baseline.py --data-dir data --positive-label True --validate-only
python baseline.py --data-dir data --positive-label True --output-dir outputs/cpu-baseline-001
```

数据目录需要 `train.csv`、`test.csv`、`sample_submission.csv`。使用其他数据时应明确其正类含义。训练默认 CPU，拒绝覆盖已有输出目录；只在 `SUCCESS.json` 存在时视为产物完整。测试使用临时合成数据；依赖不足时会跳过可选的极小训练测试。

程序检查表头、标签、概率、ID 唯一性与对齐，ID 和目标不进入特征。holdout 同时用于选轮次，分数有选择偏差。代码不处理登录、下载数据、接受条款或比赛提交。

数据、模型、预测和凭据留在本机，不入库。
