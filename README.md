# Kaggle

按比赛组织的 Kaggle 学习与实验仓库，记录问题定义、验证方案、可复现代码及实际实验结果。

## 当前项目

- [Predicting Airline Satisfaction — Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- 项目目录：[`competitions/playground-series-s6e10/`](competitions/playground-series-s6e10/)
- 目标：建立航空旅客满意度预测基线，按 ROC-AUC 评估预测概率。
- 状态：仅初始化目录与计划，尚未下载数据、训练模型或提交比赛；无实验成绩。

## 目录约定

每个比赛使用 `competitions/<competition-slug>/`，在比赛 README 中维护数据说明、验证方法、基线计划和实验记录。后续代码和笔记本在相应比赛目录下添加。

数据集、模型权重、生成的预测文件和本地运行产物留在本机。不要提交密钥、`.env`、`kaggle.json` 或个人凭据。根目录 `.gitignore` 已覆盖常见路径与格式；每次提交仍需检查暂存文件。
