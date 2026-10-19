# Kaggle

## 当前稳定版本：S6E10 v12

v12 已正式提交，public ROC-AUC **0.96118**；2026-10-19 05:28 UTC 实际排名 **676/2302**，尚未达到前 500。已整理的配方、验证证据和 CSV 精确回放入口见 [v12 说明](competitions/playground-series-s6e10/optimization/V12.md)。数据、模型与预测留在本机，不入库；后续研究保留在研究分支。

以下保留早期项目说明与历史进度，不代表当前成绩或提交额度。

按比赛组织的 Kaggle 学习与实验仓库，记录问题定义、验证方案、可复现代码及实际实验结果。

## 当前项目

- [Predicting Airline Satisfaction — Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- 项目目录：[`competitions/playground-series-s6e10/`](competitions/playground-series-s6e10/)
- 目标：建立航空旅客满意度预测基线，按 ROC-AUC 评估预测概率。
- 状态：官方 CatBoost 基线已真实提交，public ROC-AUC 0.95816。8 项开发实验完成；冻结 CatBoost/LightGBM 50/50 融合通过本机 CPU 第二划分稳定性检查（AUC 0.9590578562，对照 0.9583800203），已完成全量重训与候选 CSV 校验，父会话已提交冻结融合并确认 public AUC 0.95855（本轮已用 1 次）；后续单项互补融合未通过预设确认标准，保留已提交版本。第二划分复用训练样本，不能视为独立测试。代码与证据见 [`optimization/`](competitions/playground-series-s6e10/optimization/)。

## 目录约定

每个比赛使用 `competitions/<competition-slug>/`，在比赛 README 中维护数据说明、验证方法、基线计划和实验记录。后续代码和笔记本在相应比赛目录下添加。

数据集、模型权重、生成的预测文件和本地运行产物留在本机。不要提交密钥、`.env`、`kaggle.json` 或个人凭据。根目录 `.gitignore` 已覆盖常见路径与格式；每次提交仍需检查暂存文件。

最新补充：固定采样 LightGBM 对照已通过两个划分的预设配对标准，三组件冻结候选已完成全量训练及 CSV 校验，并于 18:43 UTC 交父会话处理。尚无该候选 public 成绩；正式提交总额仍由父会话协调。详见 optimization 的 bagging 记录。
