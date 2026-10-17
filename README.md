# Kaggle

按比赛组织的 Kaggle 学习与实验仓库，记录问题定义、验证方案、可复现代码及实际实验结果。

## 当前项目

- [Predicting Airline Satisfaction — Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- 项目目录：[`competitions/playground-series-s6e10/`](competitions/playground-series-s6e10/)
- 目标：建立航空旅客满意度预测基线，按 ROC-AUC 评估预测概率。
- 状态：官方 CatBoost 基线已真实提交，public ROC-AUC 0.95816。8 项开发实验完成；冻结 CatBoost/LightGBM 50/50 融合通过本机 CPU 第二划分稳定性检查（AUC 0.9590578562，对照 0.9583800203），已完成全量重训与候选 CSV 校验，父会话已提交冻结融合并确认 public AUC 0.95855（本轮已用 1 次）；后续单项互补融合未通过预设确认标准，保留已提交版本。第二划分复用训练样本，不能视为独立测试。代码与证据见 [`optimization/`](competitions/playground-series-s6e10/optimization/)。

## 目录约定

每个比赛使用 `competitions/<competition-slug>/`，在比赛 README 中维护数据说明、验证方法、基线计划和实验记录。后续代码和笔记本在相应比赛目录下添加。

数据集、模型权重、生成的预测文件和本地运行产物留在本机。不要提交密钥、`.env`、`kaggle.json` 或个人凭据。根目录 `.gitignore` 已覆盖常见路径与格式；每次提交仍需检查暂存文件。
