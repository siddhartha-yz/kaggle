# Predicting Airline Satisfaction

## 当前实际实验状态（2026-10-17）

官方 CatBoost 基线已提交并完成评分，public ROC-AUC **0.95816**。后续 8 项开发实验已结束，冻结 CatBoost depth8（1308 棵）+ LightGBM lgb63cats（639 棵）的 50/50 融合已在用户电脑 CPU 上通过第二划分稳定性复核：融合 **0.9590578562**，同划分固定基线 **0.9583800203**。此复核复用了官方训练样本，不是独立测试；本机已完成冻结配方全量重训与 299844 行候选 CSV 的独立校验；本机未登录 Kaggle、未正式提交；父云会话已提交并确认 public AUC **0.95855**（1209/2177，当时排名）。后续固定三组件融合未通过预设稳定性标准，保留已提交候选。

实际云端代码、冻结配置、哈希核验和本机记录见 [`optimization/`](optimization/)。它与本目录早期 `baseline.py` 脚手架不同。依赖仅安装于独立迁移 venv，全局环境与 GPU 驱动未改动。

## 初版基线脚手架历史说明

下文保留提交 `52f8cd1` 的初版说明；其中“未训练／未安装／未提交”描述的是当时的初版交付状态，当前实际进展以本页上节和 optimization 记录为准。

- 比赛：[Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- 目标：预测航空旅客满意度的正类概率，以 ROC-AUC 评估。
- 状态：已准备代码与合成测试；未下载真实数据、训练真实模型、报名或提交比赛。没有真实实验成绩。
- 当前交付用于迁移云端；本地不安装依赖、不启动真实训练。代码不含登录、下载、报名或自动提交功能。

## 环境与依赖

本机 GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，总显存 12227 MiB（约 12 GiB），驱动 595.91.07。检查时空闲 10795 MiB，此数值随进程变化；没有操作其他 GPU 进程。

已检查系统 Python 3.14.4 和现有 uv Python 3.12.14、3.11.16、3.11.10，均未发现 NumPy、pandas、scikit-learn、CatBoost 或 LightGBM。指定比赛数据目录内没有准备好的 CSV。未安装任何软件。

后续训练需在隔离的 Python 3.11/3.12 环境中准备 [requirements.txt](requirements.txt)：NumPy、pandas、scikit-learn、CatBoost。兼容范围尚未经本机训练验证，不是完整锁文件。每次运行记录实际包版本、Python 版本、配置、Git 提交及数据 SHA-256。当前无已装树模型可复用，因此选择原生处理类别特征、可选 NVIDIA GPU 的 CatBoost；未添加 LightGBM 分支。

## 文件

- `baseline.py`：标准库 CSV 校验、分层 holdout、CatBoost early stopping、指标与提交文件生成。
- `config.json`：固定种子 42、20% 验证集、最多 1000 轮、50 轮 early stopping、4 个 CPU 线程；默认 CPU。
- `tests/test_baseline.py`：临时合成 CSV 测试；依赖齐全时额外运行极小 CPU 训练集成测试。
- `data/`：已合法取得的 `train.csv`、`test.csv`、`sample_submission.csv`（Git 忽略）。
- `outputs/<run>/`：配置、元数据、指标、模型、验证预测、submission（Git 忽略）。

## 使用步骤

以下命令从仓库根目录执行。`python3` 应指向准备好的环境；校验和基础单元测试仅需标准库。

```bash
cd competitions/playground-series-s6e10
python3 -m unittest discover -s tests -v
python3 baseline.py --data-dir data --validate-only
```

先由用户确认规则、数据使用权限、Evaluation 正类定义，并在运行环境中准备三份 CSV。程序不会接受条款或获取数据。数据不在默认目录时使用 `--data-dir /absolute/path/to/data`，目录内必须包含三份规定名称的文件。

从 sample submission 推断目标列，默认 ID 列为 `id`，不同时在配置中明确设置 `id_column`。标签为精确字符串 `0`/`1` 时默认正类为 `1`；文本标签必须明确指定，例如 `--positive-label satisfied`，不可自行猜测含义。空单元格作为特征缺失值；其他文本保留原样，不自动解释为缺失值。

环境与数据准备好、确认允许训练后，执行一次 CPU 基线：

```bash
python3 baseline.py --data-dir data --output-dir outputs/cpu-baseline-001
```

可选 GPU 命令供后续云端使用，需兼容的 NVIDIA 驱动与 CatBoost 构建，并先确认 GPU 使用安排：

```bash
python3 baseline.py --data-dir data --device GPU --output-dir outputs/gpu-baseline-001
```

`gpu_devices` 默认 `0`，可在配置中调整。GPU 浮点归约可能导致同种子下的细微差异，CPU 是默认复现路径。尚未验证本机 GPU 与实际 CatBoost 环境的训练兼容性。

程序拒绝覆盖已有运行目录，输出必须位于本比赛的 `outputs/` 内。不要复用旧运行名。只有 `SUCCESS.json` 存在才表示所有产物已写完；中断目录可能包含不完整产物。缺文件、标签不合法或缺依赖时返回非零状态，不写成功标记。

## 验证与防泄漏

CSV 读取拒绝重复/空表头、坏行、空表、空/重复 ID、缺失标签、非二分类标签、列差异及训练/测试 ID 重叠。sample submission 必须恰含 ID 和目标两列，概率占位值须有限且在 `[0,1]`，其 ID 集合与行数必须匹配 test。

ID 和目标不进入特征。固定种子分层 holdout，验证两部分均有两类。字段类型仅在训练折推断；数值缺失交给 CatBoost，类别用无统计量的字符串编码，允许未见类别。训练折数值列在验证/测试中若出现无法解析的文本则报错，禁止静默转换。无全量填充、类别目标编码或验证集拟合。

early stopping 使用验证集 AUC，保存最佳轮次模型，并直接用它预测 test，不额外全量重训。holdout 同时用于选轮次，分数存在选择偏差，不是独立最终评估。若有分组/时间关系，应调整划分策略；程序不会自动发现语义泄漏。多次调参前建议独立验证集或交叉验证。

输出概率需有限、在 `[0,1]` 且数量正确；submission 按 sample 的 ID 顺序重新对齐，保留 ID 前导零和原列顺序，再读回检查。`validation.csv` 记录验证 ID、二值标签和概率；`metadata.json` 记录数据哈希、类型与代码来源。保存的 `config.json` 可直接作为下次 `--config` 输入。

## 测试与限制

本地执行结果：19 项测试中 17 项通过；预处理和真实 CatBoost 小型 CPU 集成测试共 2 项因缺依赖跳过。`--validate-only` 对缺失真实数据明确报错。未产生模型、真实指标或 submission。

合成数据仅验证代码行为，不能作为比赛成绩。标准库测试覆盖结构、标签、概率、ID 对齐及 CLI 退出状态。显示 `skipped` 的训练路径尚未验证，不能据此声称训练成功。测试均用临时合成文件，不用真实数据、网络或 GPU。

暂不包含真实数据成绩、交叉验证、调参、全量重训或比赛提交。迁移云端后的最小准备：配置上述依赖，由用户确认规则及正类含义并取得数据，再执行完整合成测试和 `--validate-only`。

实现参考：[CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit)、[scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
